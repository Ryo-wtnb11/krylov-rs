//! Hermitian eigenvalue problems by thick-restart Lanczos.

use num_traits::{Float, One, Zero};

use crate::dense::{Work, arrowhead_to_tridiag};
use crate::error::Error;
use crate::lanczos::{Fact, Lanczos};
use crate::scalar::sealed::{Ops as _, RealOps as _};
use crate::scalar::{MatRef, Real, RealScalar};
use crate::traits::{Basis, LinearMap, Vector, Workspace};

/// Which eigenvalues to target (KrylovKit `:LM`, `:LR`, `:SR`, `:LI`, `:SI`).
///
/// The imaginary-part selectors need complex arithmetic: the Hermitian
/// solvers ([`eigsolve`], `block_eigsolve`, `svdsolve`) and the real-vector
/// Arnoldi solvers reject them with [`Error::InvalidArgument`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Which {
    /// Largest magnitude.
    LargestMagnitude,
    /// Largest (most positive) real part.
    LargestReal,
    /// Smallest (most negative) real part.
    SmallestReal,
    /// Largest (most positive) imaginary part.
    LargestImag,
    /// Smallest (most negative) imaginary part.
    SmallestImag,
}

impl Which {
    pub(crate) fn is_imag(self) -> bool {
        matches!(self, Which::LargestImag | Which::SmallestImag)
    }
}

/// The `InvalidArgument` of a Hermitian solver given `LargestImag`/`SmallestImag`.
pub(crate) fn real_spectrum<E>(which: Which) -> Result<(), Error<E>> {
    if which.is_imag() {
        return Err(Error::InvalidArgument(
            "LargestImag/SmallestImag select complex eigenvalues; Hermitian solvers have real ones",
        ));
    }
    Ok(())
}

/// Convergence information of [`eigsolve`].
#[derive(Debug)]
#[non_exhaustive]
pub struct ConvergenceInfo<V: Vector> {
    /// Number of leading returned eigenpairs whose residual norm is `<= tol`.
    pub converged: usize,
    /// Residual vectors `A·xᵢ - λᵢ·xᵢ` of the returned eigenpairs.
    pub residual: Vec<V>,
    /// Norms of the residual vectors.
    pub normres: Vec<Real<V>>,
    /// Number of Krylov subspaces built (restarts + 1).
    pub numiter: usize,
    /// Number of linear-map applications.
    pub numops: usize,
    /// Largest `|Im α|` discarded from the Rayleigh quotient; a value well
    /// above rounding level means the map is not Hermitian.
    pub max_imag: Real<V>,
}

/// Computes `howmany` eigenpairs of the Hermitian map `op` selected by
/// `which`, starting from `x0`, with thick-restart Lanczos.
///
/// Follows KrylovKit v0.10.4 `eigsolve(A, x₀, howmany, which, ::Lanczos)`:
/// the result holds `max(howmany, converged)` pairs, or fewer when the
/// Krylov space is exhausted (invariant subspace); values are ordered by
/// `which`; non-convergence after `maxiter` subspaces is reported through
/// [`ConvergenceInfo::converged`]. Reaching an invariant subspace stops the
/// iteration (no random restart).
///
/// The workspace basis needs capacity `>= alg.get_krylovdim()`. Iterations and
/// restarts allocate no vectors; only the returned eigenvectors and residuals
/// are new vectors.
///
/// With `krylovdim == 1` there is no room to restart, and the first Rayleigh
/// quotient is returned (KrylovKit fails there; see [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md)).
#[allow(clippy::type_complexity)]
pub fn eigsolve<V, B, A>(
    op: &mut A,
    x0: &V,
    howmany: usize,
    which: Which,
    alg: &Lanczos,
    ws: &mut Workspace<V, B>,
) -> Result<(Vec<Real<V>>, Vec<V>, ConvergenceInfo<V>), Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
{
    let tol = alg.validate(ws)?;
    real_spectrum(which)?;
    let kd = alg.get_krylovdim();
    if howmany > kd {
        return Err(Error::HowmanyTooLarge {
            howmany,
            krylovdim: kd,
        });
    }
    let beta0 = x0.norm().map_err(Error::Vector)?;
    if !beta0.is_finite() {
        return Err(Error::NonFiniteStartVector);
    }
    if beta0 == Real::<V>::zero() {
        return Err(Error::ZeroStartVector);
    }

    let mut fact = Fact::<V::Scalar>::new(kd, alg.get_orth());
    fact.start(ws, op, Some(x0), Real::<V>::one() / beta0)?;
    let mut numiter = 1;

    let zero = Real::<V>::zero();
    let mut d = vec![zero; kd]; // eigenvalues, ordered by `which`
    let mut u = vec![zero; kd * kd]; // eigenvectors (K × K, leading dim K)
    let mut f = vec![zero; kd]; // residual couplings β·U[K-1, :]
    let mut e = vec![zero; kd];
    let mut perm = Vec::with_capacity(kd);
    let mut tmp = vec![zero; kd * kd];
    let mut harrow = vec![zero; (kd + 1) * kd];
    let mut refl = vec![zero; kd];
    let mut q = Vec::with_capacity(kd * kd);
    let mut lw = Work::new(kd, crate::dense::Problem::Tridiag);
    let mut converged = 0;
    let mut kk = 1;

    loop {
        let k = fact.len();
        let beta = fact.normres();
        if k == kd || beta <= tol || (alg.get_eager() && k >= howmany) {
            kk = k;
            d[..k].copy_from_slice(&fact.alpha);
            e[..k - 1].copy_from_slice(&fact.beta[..k - 1]);
            Real::<V>::stegr(&mut d[..k], &mut e[..k], &mut u[..k * k], &mut lw)
                .map_err(|()| Error::DenseNoConvergence)?;
            sort_eig(which, &mut d[..k], &mut u[..k * k], &mut perm, &mut tmp);
            for i in 0..k {
                f[i] = u[(k - 1) + i * k] * beta;
            }
            converged = f[..k].iter().take_while(|x| x.abs() <= tol).count();
            if converged >= howmany || beta <= tol {
                break;
            }
        }
        if k < kd {
            fact.expand(ws, op)?;
        } else {
            // Thick restart: keep `keep` Ritz vectors (KrylovKit's choice,
            // `converged <= keep < krylovdim`). With `krylovdim == 1` nothing
            // can be kept, so the first subspace is final.
            let keep = (3 * kd + 2 * converged) / 5;
            if numiter == alg.get_maxiter() || keep == 0 {
                break;
            }
            fact.alpha.truncate(keep);
            fact.beta.truncate(keep);
            arrowhead_to_tridiag(
                &d[..keep],
                &f[..keep],
                &mut fact.alpha,
                &mut fact.beta,
                &mut u[..k * k],
                k,
                &mut harrow[..(keep + 1) * keep],
                &mut refl,
            );
            q.clear();
            q.extend(u[..k * keep].iter().map(|&x| V::Scalar::from_real(x)));
            ws.basis
                .rotate(MatRef::new(&q, k, keep), keep)
                .map_err(Error::Vector)?;
            // The residual direction is unchanged; its new norm is β[keep-1].
            let s = V::Scalar::from_real(fact.beta[keep - 1] / beta);
            ws.w.axpby(V::Scalar::zero(), &ws.v, s)
                .map_err(Error::Vector)?;
            numiter += 1;
        }
    }

    let k = kk;
    let nret = if converged > howmany {
        converged
    } else {
        howmany.min(k)
    };
    let mut vectors = Vec::with_capacity(nret);
    let mut residual = Vec::with_capacity(nret);
    q.clear();
    q.resize(k, V::Scalar::zero());
    for i in 0..nret {
        for (qj, &uj) in q.iter_mut().zip(&u[i * k..(i + 1) * k]) {
            *qj = V::Scalar::from_real(uj);
        }
        let mut x = x0.similar().map_err(Error::Vector)?;
        ws.basis
            .combine_into(&q, &mut x, V::Scalar::one(), V::Scalar::zero())
            .map_err(Error::Vector)?;
        vectors.push(x);
        let mut r = x0.similar().map_err(Error::Vector)?;
        r.axpby(q[k - 1], &ws.w, V::Scalar::zero())
            .map_err(Error::Vector)?;
        residual.push(r);
    }
    let info = ConvergenceInfo {
        converged,
        residual,
        normres: f[..nret].iter().map(|x| x.abs()).collect(),
        numiter,
        numops: fact.numops,
        max_imag: fact.max_imag,
    };
    Ok((d[..nret].to_vec(), vectors, info))
}

/// Stable reordering of ascending eigenpairs by `which`, like Julia's
/// `sortperm(D; by, rev)`.
pub(crate) fn sort_eig<R: RealScalar, T: Copy>(
    which: Which,
    d: &mut [R],
    u: &mut [T],
    perm: &mut Vec<usize>,
    tmp: &mut [T],
) {
    let k = d.len();
    perm.clear();
    perm.extend(0..k);
    let key = |x: R| match which {
        Which::LargestMagnitude => -x.abs(),
        Which::LargestReal => -x,
        // Imaginary-part selectors are rejected before any sort of real values.
        Which::SmallestReal | Which::LargestImag | Which::SmallestImag => x,
    };
    perm.sort_by(|&a, &b| {
        key(d[a])
            .partial_cmp(&key(d[b]))
            .unwrap_or(core::cmp::Ordering::Equal)
    });
    let tmp = &mut tmp[..k * k];
    tmp.copy_from_slice(u);
    for (new, &old) in perm.iter().enumerate() {
        u[new * k..(new + 1) * k].copy_from_slice(&tmp[old * k..(old + 1) * k]);
    }
    // d[new] ← d[perm[new]] in place, following cycles; perm is consumed.
    for i in 0..k {
        if perm[i] == usize::MAX {
            continue;
        }
        let first = d[i];
        let mut j = i;
        loop {
            let p = core::mem::replace(&mut perm[j], usize::MAX);
            if p == i {
                d[j] = first;
                break;
            }
            d[j] = d[p];
            j = p;
        }
    }
}
