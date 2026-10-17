//! Partial singular value decomposition by thick-restart
//! Golub–Kahan–Lanczos bidiagonalization.

use num_traits::{Float, One, Zero};

use crate::dense::{arrowhead_to_bidiag, bidiag_svd, lower_rect_bidiag_svd};
use crate::eigsolve::{ConvergenceInfo, Which};
use crate::error::Error;
use crate::orth::{Orth, orthogonalize};
use crate::scalar::sealed::{Ops as _, RealOps as _};
use crate::scalar::{MatRef, Real, Scalar};
use crate::traits::{AdjointMap, Basis, Vector, Workspace};

/// Golub–Kahan–Lanczos settings for [`svdsolve`].
///
/// Mirrors KrylovKit's `GKL`: `krylovdim` (default 30, at least 2) is the
/// maximal Krylov dimension, `maxiter` (default 100) the number of Krylov subspaces built
/// (restarts + 1), `tol` (default `1e-12`) an absolute bound on the residual
/// norms (met when strictly smaller), `eager` (default `false`) evaluates the
/// projected problem after every expansion, and `orth` (default
/// [`Orth::Cgs2`]) orthogonalizes each new vector twice against the full
/// basis on both sides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gkl {
    krylovdim: usize,
    maxiter: usize,
    tol: f64,
    orth: Orth,
    eager: bool,
}

impl Default for Gkl {
    fn default() -> Self {
        Self {
            krylovdim: 30,
            maxiter: 100,
            tol: 1e-12,
            orth: Orth::Cgs2,
            eager: false,
        }
    }
}

impl Gkl {
    /// Default settings.
    pub fn new() -> Self {
        Self::default()
    }
    /// Sets the maximal Krylov dimension.
    pub fn krylovdim(mut self, krylovdim: usize) -> Self {
        self.krylovdim = krylovdim;
        self
    }
    /// Sets the maximal number of Krylov subspaces built.
    pub fn maxiter(mut self, maxiter: usize) -> Self {
        self.maxiter = maxiter;
        self
    }
    /// Sets the absolute residual tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
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
    fn push_u<V, BU, BV, A>(
        &mut self,
        op: &mut A,
        ws_u: &mut Workspace<V, BU>,
        ws_v: &mut Workspace<V, BV>,
        src: Option<&V>,
        nrm: S::Real,
        tol: S::Real,
    ) -> Result<Option<S::Real>, Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        BU: Basis<V>,
        BV: Basis<V>,
        A: AdjointMap<V>,
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
    fn push_v<V, BU, BV, A>(
        &mut self,
        op: &mut A,
        ws_u: &mut Workspace<V, BU>,
        ws_v: &mut Workspace<V, BV>,
        alpha: S::Real,
    ) -> Result<S, Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        BU: Basis<V>,
        BV: Basis<V>,
        A: AdjointMap<V>,
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
/// ([`Which::SmallestReal`]) singular values of `op` with their left and
/// right singular vectors, starting from `x0` in the codomain.
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
/// For `SmallestReal` the codomain should not be larger than the domain:
/// the process lives in the codomain and approximates `AAᴴ`, whose zero
/// eigenvalues (one per missing domain dimension) then come first as Ritz
/// values `σ ≈ 0` that no right vector satisfies; they are reported with
/// their true, non-small residual and never converge. Apply `svdsolve` to
/// the adjoint of a tall map instead (KrylovKit reports such a zero as
/// converged).
///
/// `ws_u` holds the left basis and must be in the space of `x0`; `ws_v` holds
/// the right basis and must be in the domain of `op`. Both bases need
/// capacity `>= alg.get_krylovdim()`. Iterations and restarts allocate no
/// vectors; a call allocates the `3·n` returned vectors.
#[allow(clippy::type_complexity)]
pub fn svdsolve<V, BU, BV, A>(
    op: &mut A,
    x0: &V,
    howmany: usize,
    which: Which,
    alg: &Gkl,
    ws_u: &mut Workspace<V, BU>,
    ws_v: &mut Workspace<V, BV>,
) -> Result<(Vec<Real<V>>, Vec<V>, Vec<V>, ConvergenceInfo<V>), Error<V::Error>>
where
    V: Vector,
    BU: Basis<V>,
    BV: Basis<V>,
    A: AdjointMap<V>,
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
    if which == Which::LargestMagnitude {
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
    let mut bt = vec![s0; kd * kd];
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
                rect_scratch.resize((k + 1) * (k + 1) + k * k, s0);
                lower_rect_bidiag_svd(
                    &fact.alpha,
                    &fact.beta,
                    p,
                    y,
                    &mut sv[..k],
                    &mut rect_scratch,
                )
            } else {
                // Bᵀ is upper bidiagonal: Bᵀ = Y·diag(σ)·Pᵀ, so B = P·diag(σ)·Yᵀ.
                let bt = &mut bt[..k * k];
                bt.fill(s0);
                for i in 0..k {
                    bt[i + i * k] = fact.alpha[i];
                    if i + 1 < k {
                        bt[i + (i + 1) * k] = fact.beta[i];
                    }
                }
                bidiag_svd(bt, y, p, &mut sv[..k])
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
