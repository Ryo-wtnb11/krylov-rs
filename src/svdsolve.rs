//! Partial singular value decomposition by thick-restart
//! Golub–Kahan–Lanczos bidiagonalization.

use num_traits::{Float, One, Zero};

use crate::dense::{Work, arrowhead_to_bidiag, bidiag_svd, lower_rect_bidiag_svd};
use crate::eigsolve::{ConvergenceInfo, Which};
use crate::error::Error;
use crate::orth::{Orth, orthogonalize};
use crate::scalar::sealed::{Ops as _, RealOps as _};
use crate::scalar::{MatRef, Real, Scalar};
use crate::traits::{AdjointMap, Basis, Vector, Workspace};

/// Golub–Kahan–Lanczos settings for [`svdsolve`].
///
/// Mirrors KrylovKit's `GKL`: `krylovdim` (at least 2) is the
/// maximal Krylov dimension, `maxiter` the number of Krylov subspaces built
/// (restarts + 1), `tol` an absolute bound on the residual
/// norms (met when strictly smaller), `eager` (default `false`) evaluates the
/// projected problem after every expansion, and `orth` (default
/// [`Orth::Cgs2`]) orthogonalizes each new vector twice against the full
/// basis on both sides. This differs from KrylovKit, whose CGS2 GKL keeps the
/// right side local and makes one pass on the left (six reductions per step
/// instead of three; see [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md)).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gkl {
    krylovdim: usize,
    maxiter: usize,
    tol: f64,
    orth: Orth,
    eager: bool,
}

impl Gkl {
    /// Settings with the given problem-dependent parameters; the variant
    /// options start at their documented defaults. The values are validated
    /// when a solver is called.
    pub fn new(krylovdim: usize, maxiter: usize, tol: f64) -> Self {
        Self {
            krylovdim,
            maxiter,
            tol,
            orth: Orth::Cgs2,
            eager: false,
        }
    }
    /// Sets the orthogonalizer.
    pub fn orth(mut self, orth: Orth) -> Self {
        self.orth = orth;
        self
    }
    /// Sets eager evaluation.
    pub fn eager(mut self, eager: bool) -> Self {
        self.eager = eager;
        self
    }
    /// Maximal Krylov dimension.
    pub fn get_krylovdim(&self) -> usize {
        self.krylovdim
    }
    /// Maximal number of Krylov subspaces built.
    pub fn get_maxiter(&self) -> usize {
        self.maxiter
    }
    /// Absolute residual tolerance.
    pub fn get_tol(&self) -> f64 {
        self.tol
    }
    /// Orthogonalizer.
    pub fn get_orth(&self) -> Orth {
        self.orth
    }
    /// Eager evaluation.
    pub fn get_eager(&self) -> bool {
        self.eager
    }
}

/// Bidiagonal Rayleigh quotient of `A·V = U·B + r·e_kᵀ`, `Aᴴ·U = V·Bᴴ`:
/// `alpha` the diagonal, `beta[..k-1]` the subdiagonal and `beta[k-1] = ‖r‖`.
/// `r` lives in `ws_u.w`, the newest `u` in `ws_u.v`, the newest `v` in
/// `ws_v.v`.
struct Fact<S: Scalar> {
    alpha: Vec<S::Real>,
    beta: Vec<S::Real>,
    h: Vec<S>,
    scratch: Vec<S>,
    numops: usize,
    orth: Orth,
}

impl<S: Scalar> Fact<S> {
    /// `w ← w - B·Bᴴ·w` against `basis`; returns `‖w‖` and the summed
    /// coefficient of the newest basis vector.
    fn orth<V: Vector<Scalar = S>, B: Basis<V>>(
        &mut self,
        basis: &B,
        w: &mut V,
    ) -> Result<(S::Real, S), V::Error> {
        let k = basis.len();
        self.h.clear();
        self.h.resize(k, S::zero());
        self.scratch.clear();
        self.scratch.resize(k, S::zero());
        orthogonalize(basis, w, &mut self.h, &mut self.scratch, self.orth)?;
        let last = self.h.last().copied().unwrap_or(S::zero());
        Ok((w.norm()?, last))
    }

    /// Appends `u = src/nrm` (`src = None`: the residual `ws_u.w`) and
    /// computes `α = ‖(I - VVᴴ)Aᴴu‖`. Returns `None` when `α <= tol`: then
    /// `Aᴴu` lies in the span of the right basis and `(U, V)` is an invariant
    /// pair, `A·V = U·B̃` with the `(k+1) × k` lower bidiagonal `B̃`.
    #[allow(clippy::too_many_arguments)]
    fn push_u<V, W, BU, BV, A>(
        &mut self,
        op: &mut A,
        ws_u: &mut Workspace<W, BU>,
        ws_v: &mut Workspace<V, BV>,
        src: Option<&W>,
        nrm: S::Real,
        tol: S::Real,
    ) -> Result<Option<S::Real>, Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        W: Vector<Scalar = S, Error = V::Error>,
        BU: Basis<W>,
        BV: Basis<V>,
        A: AdjointMap<V, W>,
    {
        let inv = S::from_real(S::Real::one() / nrm);
        let src = src.unwrap_or(&ws_u.w);
        ws_u.basis.push(src, inv).map_err(Error::Vector)?;
        ws_u.v.axpby(inv, src, S::zero()).map_err(Error::Vector)?;
        op.apply_adjoint_into(&ws_u.v, &mut ws_v.w)
            .map_err(Error::Vector)?;
        self.numops += 1;
        let (alpha, _) = self.orth(&ws_v.basis, &mut ws_v.w).map_err(Error::Vector)?;
        if !alpha.is_finite() {
            return Err(Error::NonFiniteIterate);
        }
        if alpha <= tol {
            return Ok(None);
        }
        Ok(Some(alpha))
    }

    /// Appends `v = ws_v.w/α` and computes `r = (I - UUᴴ)A·v`; records `α`,
    /// `β = ‖r‖`. Returns the coefficient `⟨u_k, A·v⟩`, which equals `α`.
    fn push_v<V, W, BU, BV, A>(
        &mut self,
        op: &mut A,
        ws_u: &mut Workspace<W, BU>,
        ws_v: &mut Workspace<V, BV>,
        alpha: S::Real,
    ) -> Result<S, Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        W: Vector<Scalar = S, Error = V::Error>,
        BU: Basis<W>,
        BV: Basis<V>,
        A: AdjointMap<V, W>,
    {
        let inv = S::from_real(S::Real::one() / alpha);
        ws_v.basis.push(&ws_v.w, inv).map_err(Error::Vector)?;
        ws_v.v
            .axpby(inv, &ws_v.w, S::zero())
            .map_err(Error::Vector)?;
        op.apply_into(&ws_v.v, &mut ws_u.w).map_err(Error::Vector)?;
        self.numops += 1;
        let (beta, coupling) = self.orth(&ws_u.basis, &mut ws_u.w).map_err(Error::Vector)?;
        if !beta.is_finite() || !coupling.all_finite() {
            return Err(Error::NonFiniteIterate);
        }
        self.alpha.push(alpha);
        self.beta.push(beta);
        Ok(coupling)
    }
}

/// Computes the `howmany` largest ([`Which::LargestReal`]) or smallest
/// ([`Which::SmallestReal`]) singular values of `op: V → W` with their left
/// (`W`) and right (`V`) singular vectors, starting from `x0` in the
/// codomain. `V` and `W` may be one Rust vector type (the default of
/// [`AdjointMap`]) or two types with the same scalar and error types, as
/// KrylovKit's `GKLFactorization{TU, TV}` allows.
///
/// Follows KrylovKit v0.10.4 `svdsolve(A, x₀, howmany, which, ::GKL)`: thick
/// restart of the Golub–Kahan–Lanczos bidiagonalization, values in
/// descending (`LargestReal`) or ascending (`SmallestReal`) order, and
/// `max(howmany, converged)` triplets, where `converged` counts the leading
/// triplets whose residual norm is strictly below `tol`. The returned
/// tuple is `(σ, left, right, info)` with `A·rightᵢ = σᵢ·leftᵢ +
/// info.residual[i]` and `Aᴴ·leftᵢ = σᵢ·rightᵢ`; `info.max_imag` is always
/// zero.
///
/// The Krylov space can stop growing before `krylovdim` (no random restart):
/// when the residual norm drops to `tol` (invariant subspace, KrylovKit's
/// rule) or when the new right vector `(I - VVᴴ)Aᴴu` does (the same rule on
/// the adjoint side). Then the triplets of the current space are returned,
/// possibly fewer than `howmany`; none if already `Aᴴ·x0` vanishes. In the
/// adjoint-side case `(U, V)` is an invariant pair: the triplets come from
/// the `(k+1) × k` bidiagonal `B̃` with `A·V = U·B̃`, have zero residual
/// (so they count as converged whenever `tol > 0`), and
/// `Aᴴ·leftᵢ = σᵢ·rightᵢ` holds up to the dropped norm `<= tol` (KrylovKit
/// has no such stop and divides by it). A non-finite norm during the
/// iteration (e.g. NaN from the map) is [`Error::NonFiniteIterate`].
///
/// `SmallestReal` requires that the codomain dimension not exceed the domain
/// dimension. Otherwise call `svdsolve` on the adjoint map, with `x0` in the
/// domain, and swap the left and right vectors; this is ARPACK's rule for
/// its SVD driver (`EXAMPLES/SVD/dsvd.f`: "appropriate when m .ge. n.
/// Reverse the roles of A and A' in the case that m .le. n"). The process
/// lives in the codomain and approximates `AAᴴ`. For a tall map `AAᴴ` has
/// one zero eigenvalue per missing domain dimension, so these zeros come
/// first as Ritz values `σ ≈ 0` that no right vector satisfies. They are
/// reported with their true, non-small residual (at least `σ_min − σ`) and
/// never count as converged; KrylovKit reports them as converged. A matrix-free
/// map does not reveal which side is smaller, and no other Ritz extraction
/// fixes this. The `(k+1) × k` projection (PROPACK's) yields the Ritz values
/// of `AᴴA`, which equals running this method on the adjoint from `Aᴴ·x0`,
/// so it only moves the spurious zeros to wide maps. Harmonic or refined
/// Ritz values targeting zero converge to the same genuine zero eigenvalues
/// of `AAᴴ`.
///
/// `ws_u` (codomain, `W`) holds the left basis and must be in the space of
/// `x0`; `ws_v` (domain, `V`) holds the right basis and must be in the domain
/// of `op`. `info.residual` lives in the codomain. Both bases need
/// capacity `>= alg.get_krylovdim()`. Iterations and restarts allocate no
/// vectors; a call allocates the `3·n` returned vectors.
///
/// A failed start check `⟨u, A·v⟩ = ‖Aᴴu‖` (with `v = Aᴴu/‖Aᴴu‖`) is
/// [`Error::IncompatibleAdjoint`] (KrylovKit: `ArgumentError`; see
/// [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md)).
#[allow(clippy::type_complexity)]
pub fn svdsolve<V, W, BU, BV, A>(
    op: &mut A,
    x0: &W,
    howmany: usize,
    which: Which,
    alg: &Gkl,
    ws_u: &mut Workspace<W, BU>,
    ws_v: &mut Workspace<V, BV>,
) -> Result<(Vec<Real<V>>, Vec<W>, Vec<V>, ConvergenceInfo<W>), Error<V::Error>>
where
    V: Vector,
    W: Vector<Scalar = V::Scalar, Error = V::Error>,
    BU: Basis<W>,
    BV: Basis<V>,
    A: AdjointMap<V, W>,
{
    let kd = alg.krylovdim;
    if kd < 2 {
        // A thick restart keeps (3·krylovdim + 2·converged)/5 >= 1 vectors
        // only from krylovdim 2 on; with 1 it would silently ignore maxiter.
        return Err(Error::InvalidArgument("krylovdim must be at least 2"));
    }
    if alg.maxiter == 0 {
        return Err(Error::InvalidArgument("maxiter must be at least 1"));
    }
    if alg.tol.is_nan() || alg.tol < 0.0 {
        return Err(Error::InvalidArgument("tol must be non-negative"));
    }
    if !matches!(which, Which::LargestReal | Which::SmallestReal) {
        return Err(Error::InvalidArgument(
            "svdsolve targets LargestReal or SmallestReal singular values",
        ));
    }
    if howmany > kd {
        return Err(Error::HowmanyTooLarge {
            howmany,
            krylovdim: kd,
        });
    }
    for capacity in [ws_u.basis.capacity(), ws_v.basis.capacity()] {
        if capacity < kd {
            return Err(Error::InsufficientCapacity {
                needed: kd,
                capacity,
            });
        }
    }
    let tol = Real::<V>::from_f64(alg.tol);
    let beta0 = x0.norm().map_err(Error::Vector)?;
    if !beta0.is_finite() {
        return Err(Error::NonFiniteStartVector);
    }
    if beta0 == Real::<V>::zero() {
        return Err(Error::ZeroStartVector);
    }
    let ve = Error::Vector;
    let zero = Real::<V>::zero();
    let mut fact = Fact::<V::Scalar> {
        alpha: Vec::with_capacity(kd),
        beta: Vec::with_capacity(kd),
        h: Vec::with_capacity(kd),
        scratch: Vec::with_capacity(kd),
        numops: 0,
        orth: alg.orth,
    };
    ws_u.basis.truncate(0);
    ws_v.basis.truncate(0);
    let empty = |numops| ConvergenceInfo {
        converged: 0,
        residual: Vec::new(),
        normres: Vec::new(),
        numiter: 1,
        numops,
        max_imag: zero,
    };
    // KrylovKit `gkl.jl:initialize`, including its check that `A` and `Aᴴ`
    // belong together: ⟨u, A·v⟩ must reproduce α = ‖Aᴴu‖.
    let Some(alpha) = fact.push_u(op, ws_u, ws_v, Some(x0), beta0, tol)? else {
        return Ok((Vec::new(), Vec::new(), Vec::new(), empty(fact.numops)));
    };
    let coupling = fact.push_v(op, ws_u, ws_v, alpha)?;
    let sqrt_eps = Real::<V>::epsilon().sqrt();
    let mismatch = (coupling - V::Scalar::from_real(alpha)).modulus();
    // α and the coupling are finite here (push_u/push_v reject NaN/Inf),
    // so this comparison cannot pass vacuously.
    if mismatch > sqrt_eps * alpha.max(coupling.modulus()) {
        return Err(Error::IncompatibleAdjoint);
    }
    let mut numiter = 1;

    let s0 = zero;
    let mut sv = vec![s0; kd];
    let mut p = vec![s0; (kd + 1) * kd]; // left singular vectors of B (lp × K)
    let mut y = vec![s0; kd * kd]; // right singular vectors of B
    let mut eb = vec![s0; kd];
    let mut lw = Work::default();
    let mut f = vec![s0; kd];
    let mut h = vec![s0; (kd + 1) * kd];
    let mut refl = vec![s0; kd];
    let mut q = Vec::with_capacity(kd * kd);
    let mut converged = 0;
    let mut kk = 1;
    let mut exhausted = false;
    let mut lp = 1; // rows of p: K, or K + 1 after an exhausted adjoint side
    let mut rect_scratch = Vec::new();

    loop {
        let k = fact.alpha.len();
        let beta = fact.beta[k - 1];
        if k == kd || beta <= tol || exhausted || (alg.eager && k >= howmany) {
            kk = k;
            lp = if exhausted { k + 1 } else { k };
            let (p, y) = (&mut p[..lp * k], &mut y[..k * k]);
            let dense = if exhausted {
                // Exact invariant pair: the SVD of B̃ has no residual.
                rect_scratch.resize((k + 1) * (k + 1) + k * k + 2 * k, s0);
                lower_rect_bidiag_svd(
                    &fact.alpha,
                    &fact.beta,
                    p,
                    y,
                    &mut sv[..k],
                    &mut rect_scratch,
                    &mut lw,
                )
            } else {
                // B (lower bidiagonal) = P·diag(σ)·Yᵀ, as KrylovKit's bdsqr!('L').
                sv[..k].copy_from_slice(&fact.alpha[..k]);
                eb[..k - 1].copy_from_slice(&fact.beta[..k - 1]);
                bidiag_svd(true, &mut sv[..k], &mut eb[..k], p, y, &mut lw)
            };
            dense.map_err(|()| Error::DenseNoConvergence)?;
            if which == Which::SmallestReal {
                sv[..k].reverse();
                for (m, ld) in [(&mut *p, lp), (&mut *y, k)] {
                    for j in 0..k / 2 {
                        for i in 0..ld {
                            m.swap(i + j * ld, i + (k - 1 - j) * ld);
                        }
                    }
                }
            }
            for i in 0..k {
                f[i] = if exhausted {
                    s0
                } else {
                    y[(k - 1) + i * k] * beta
                };
            }
            converged = f[..k].iter().take_while(|x| x.abs() < tol).count();
            if converged >= howmany || beta <= tol || exhausted {
                break;
            }
        }
        if k < kd {
            match fact.push_u(op, ws_u, ws_v, None, beta, tol)? {
                Some(alpha) => {
                    fact.push_v(op, ws_u, ws_v, alpha)?;
                }
                None => exhausted = true,
            }
        } else {
            let keep = (3 * kd + 2 * converged) / 5;
            if numiter == alg.maxiter || keep == 0 {
                break;
            }
            fact.alpha.truncate(keep);
            fact.beta.truncate(keep);
            arrowhead_to_bidiag(
                &sv[..keep],
                &f[..keep],
                &mut fact.alpha,
                &mut fact.beta,
                &mut p[..k * k],
                &mut y[..k * k],
                k,
                &mut h[..(keep + 1) * keep],
                &mut refl,
            );
            q.clear();
            q.extend(p[..k * keep].iter().map(|&x| V::Scalar::from_real(x)));
            ws_u.basis
                .rotate(MatRef::new(&q, k, keep), keep)
                .map_err(ve)?;
            q.clear();
            q.extend(y[..k * keep].iter().map(|&x| V::Scalar::from_real(x)));
            ws_v.basis
                .rotate(MatRef::new(&q, k, keep), keep)
                .map_err(ve)?;
            // The residual direction is unchanged; its new norm is β[keep-1].
            let s = V::Scalar::from_real(fact.beta[keep - 1] / beta);
            ws_u.w.axpby(V::Scalar::zero(), &ws_u.v, s).map_err(ve)?;
            numiter += 1;
        }
    }

    let k = kk;
    let nret = if converged > howmany {
        converged
    } else {
        howmany.min(k)
    };
    let (mut left, mut right, mut residual) = (
        Vec::with_capacity(nret),
        Vec::with_capacity(nret),
        Vec::with_capacity(nret),
    );
    let one = V::Scalar::one();
    for i in 0..nret {
        q.clear();
        q.extend(
            p[i * lp..(i + 1) * lp]
                .iter()
                .map(|&x| V::Scalar::from_real(x)),
        );
        let mut l = x0.similar().map_err(ve)?;
        ws_u.basis
            .combine_into(&q, &mut l, one, V::Scalar::zero())
            .map_err(ve)?;
        left.push(l);
        q.clear();
        q.extend(
            y[i * k..(i + 1) * k]
                .iter()
                .map(|&x| V::Scalar::from_real(x)),
        );
        let mut rv = ws_v.v.similar().map_err(ve)?;
        ws_v.basis
            .combine_into(&q, &mut rv, one, V::Scalar::zero())
            .map_err(ve)?;
        right.push(rv);
        let mut res = x0.similar().map_err(ve)?;
        let c = if exhausted {
            V::Scalar::zero()
        } else {
            q[k - 1]
        };
        res.axpby(c, &ws_u.w, V::Scalar::zero()).map_err(ve)?;
        residual.push(res);
    }
    let info = ConvergenceInfo {
        converged,
        residual,
        normres: f[..nret].iter().map(|x| x.abs()).collect(),
        numiter,
        numops: fact.numops,
        max_imag: zero,
    };
    Ok((sv[..nret].to_vec(), left, right, info))
}
