//! Linear systems `(a₀ + a₁·A)·x = b` by conjugate gradients and restarted
//! GMRES.

use num_traits::{Float, One, Zero};

use crate::dense::{Givens, givens, upper_solve};
use crate::error::Error;
use crate::orth::{Orth, orthogonalize};
use crate::scalar::Real;
use crate::scalar::sealed::{Ops as _, RealOps as _};
use crate::traits::{Basis, LinearMap, Vector, Workspace};

/// Conjugate gradients for Hermitian positive definite `a₀ + a₁·A`.
///
/// Mirrors KrylovKit's `CG`: `maxiter` bounds the number of
/// iterations, and `tol` is an absolute bound on the
/// residual norm `‖b - a₀x - a₁A·x‖`, met when strictly smaller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cg {
    maxiter: usize,
    tol: f64,
}

impl Cg {
    /// Settings with the given parameters, validated when [`linsolve`] is
    /// called.
    pub fn new(maxiter: usize, tol: f64) -> Self {
        Self { maxiter, tol }
    }
    /// Maximal number of iterations.
    pub fn get_maxiter(&self) -> usize {
        self.maxiter
    }
    /// Absolute residual tolerance.
    pub fn get_tol(&self) -> f64 {
        self.tol
    }
}

/// How [`Gmres`] forms the residual that starts each restart cycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RestartResidual {
    /// `b - a₀x - a₁A·x`, computed explicitly: one extra map application
    /// per restart, at most `1/krylovdim` of the applications. The default:
    /// every cycle minimizes the true residual, also for vectors whose
    /// arithmetic truncates (README, "Low-rank and tensor-network vectors").
    #[default]
    Recomputed,
    /// The GMRES recurrence over the Krylov basis, as KrylovKit does: no
    /// extra application, but it assumes exact vector arithmetic and the
    /// Arnoldi relation. Not recommended for truncating vectors: there the
    /// restart residual differs from `b - A·x` by about `τ·‖b‖`, so each
    /// cycle minimizes a perturbed residual and only the final explicit
    /// check sees the true one.
    Recurrence,
}

/// Restarted GMRES for general `a₀ + a₁·A`.
///
/// Mirrors KrylovKit's `GMRES`: `krylovdim` is the restart
/// length, `maxiter` the number of restart cycles, `tol`
/// an absolute bound on the residual norm, met when
/// strictly smaller, and `orth` (default [`Orth::Cgs2`]) the Arnoldi
/// orthogonalizer. `restart_residual` (default
/// [`RestartResidual::Recomputed`]) differs from KrylovKit, which uses
/// [`RestartResidual::Recurrence`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gmres {
    krylovdim: usize,
    maxiter: usize,
    tol: f64,
    orth: Orth,
    restart_residual: RestartResidual,
}

impl Gmres {
    /// Settings with the given problem-dependent parameters; the variant
    /// options start at their documented defaults. The values are validated
    /// when a solver is called.
    pub fn new(krylovdim: usize, maxiter: usize, tol: f64) -> Self {
        Self {
            krylovdim,
            maxiter,
            tol,
            orth: Orth::Cgs2,
            restart_residual: RestartResidual::Recomputed,
        }
    }
    /// Sets the orthogonalizer.
    pub fn orth(mut self, orth: Orth) -> Self {
        self.orth = orth;
        self
    }
    /// Sets how the residual that starts a restart cycle is formed.
    pub fn restart_residual(mut self, restart_residual: RestartResidual) -> Self {
        self.restart_residual = restart_residual;
        self
    }
    /// Restart length.
    pub fn get_krylovdim(&self) -> usize {
        self.krylovdim
    }
    /// Maximal number of restart cycles.
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
    /// Restart residual.
    pub fn get_restart_residual(&self) -> RestartResidual {
        self.restart_residual
    }
}

/// Convergence information of [`linsolve`].
#[derive(Debug)]
#[non_exhaustive]
pub struct LinsolveInfo<V: Vector> {
    /// `1` if `normres < tol`, else `0`.
    pub converged: usize,
    /// Residual `b - a₀x - a₁A·x` of the returned solution.
    pub residual: V,
    /// Norm of [`residual`](Self::residual).
    pub normres: Real<V>,
    /// CG: iterations; GMRES: restart cycles.
    pub numiter: usize,
    /// Number of linear-map applications.
    pub numops: usize,
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for super::Cg {}
    impl Sealed for super::Gmres {}
}

/// A [`linsolve`] algorithm: [`Cg`] or [`Gmres`]. Sealed.
pub trait LinearSolver: sealed::Sealed {
    #[doc(hidden)]
    #[allow(clippy::type_complexity)]
    fn solve<V, B, A>(
        &self,
        op: &mut A,
        b: &V,
        x0: &V,
        a0: V::Scalar,
        a1: V::Scalar,
        ws: &mut Workspace<V, B>,
    ) -> Result<(V, LinsolveInfo<V>), Error<V::Error>>
    where
        V: Vector,
        B: Basis<V>,
        A: LinearMap<V>;
}

/// Solves `(a₀ + a₁·A)·x = b` starting from `x0`.
///
/// Follows KrylovKit v0.10.4 `linsolve(A, b, x₀, alg, a₀, a₁)`; `a₀` and
/// `a₁` are applied implicitly. With [`Cg`] the map `a₀ + a₁·A` must be
/// Hermitian positive definite and `a₀`, `a₁` real; with [`Gmres`] it may be
/// general. Non-convergence is reported through
/// [`LinsolveInfo::converged`].
///
/// Workspace: [`Gmres`] needs basis capacity `>= krylovdim`; [`Cg`] uses only
/// the two operator vectors (any capacity, e.g.
/// `Workspace::with_capacity(&b, 0)`). A warm call allocates exactly two
/// vectors, the solution and its residual.
///
/// `Error::NonFiniteStartVector` reports a non-finite `b` or `x0`; a
/// non-finite map output (from `A·x0` on) is `Error::NonFiniteIterate`.
#[allow(clippy::type_complexity)]
pub fn linsolve<V, B, A, L>(
    op: &mut A,
    b: &V,
    x0: &V,
    alg: &L,
    a0: V::Scalar,
    a1: V::Scalar,
    ws: &mut Workspace<V, B>,
) -> Result<(V, LinsolveInfo<V>), Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
    L: LinearSolver,
{
    alg.solve(op, b, x0, a0, a1, ws)
}

fn check_tol<R: Float + crate::scalar::sealed::RealOps, E>(tol: f64) -> Result<R, Error<E>> {
    if tol.is_nan() || tol < 0.0 {
        return Err(Error::InvalidArgument("tol must be non-negative"));
    }
    Ok(R::from_f64(tol))
}

/// `r ← b - a₀x - a₁A·x`, using `ws.w` for `A·x`; returns `‖r‖`.
fn residual_into<V, B, A>(
    op: &mut A,
    b: &V,
    x: &V,
    a0: V::Scalar,
    a1: V::Scalar,
    r: &mut V,
    ws: &mut Workspace<V, B>,
) -> Result<Real<V>, V::Error>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
{
    op.apply_into(x, &mut ws.w)?;
    r.copy_from(b)?;
    if a0 != V::Scalar::zero() {
        r.axpby(-a0, x, V::Scalar::one())?;
    }
    r.axpby(-a1, &ws.w, V::Scalar::one())?;
    r.norm()
}

/// Solution `x = x0`, residual and its norm.
#[allow(clippy::type_complexity)]
fn start<V, B, A>(
    op: &mut A,
    b: &V,
    x0: &V,
    a0: V::Scalar,
    a1: V::Scalar,
    ws: &mut Workspace<V, B>,
) -> Result<(V, V, Real<V>), Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
{
    let mut x = b.similar().map_err(Error::Vector)?;
    x.copy_from(x0).map_err(Error::Vector)?;
    let mut r = b.similar().map_err(Error::Vector)?;
    let normr = residual_into(op, b, x0, a0, a1, &mut r, ws).map_err(Error::Vector)?;
    if !normr.is_finite() {
        // Classifying costs two reductions, paid only on this error path.
        for v in [b, x0] {
            if !v.norm().map_err(Error::Vector)?.is_finite() {
                return Err(Error::NonFiniteStartVector);
            }
        }
        return Err(Error::NonFiniteIterate);
    }
    Ok((x, r, normr))
}

fn info<V: Vector>(
    converged: bool,
    residual: V,
    normres: Real<V>,
    numiter: usize,
    numops: usize,
) -> LinsolveInfo<V> {
    LinsolveInfo {
        converged: usize::from(converged),
        residual,
        normres,
        numiter,
        numops,
    }
}

impl LinearSolver for Cg {
    /// Port of KrylovKit v0.10.4 `src/linsolve/cg.jl:linsolve(…, ::CG)`.
    fn solve<V, B, A>(
        &self,
        op: &mut A,
        b: &V,
        x0: &V,
        a0: V::Scalar,
        a1: V::Scalar,
        ws: &mut Workspace<V, B>,
    ) -> Result<(V, LinsolveInfo<V>), Error<V::Error>>
    where
        V: Vector,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        let tol: Real<V> = check_tol(self.tol)?;
        if self.maxiter == 0 {
            return Err(Error::InvalidArgument("maxiter must be at least 1"));
        }
        let zero = Real::<V>::zero();
        if a0.im() != zero || a1.im() != zero {
            return Err(Error::InvalidArgument("CG requires real a0 and a1"));
        }
        let (mut x, mut r, mut normr) = start(op, b, x0, a0, a1, ws)?;
        let mut numops = 1;
        let mut numiter = 0;
        let mut rho = normr * normr;
        let mut beta = V::Scalar::zero();
        let ve = Error::Vector;
        while normr >= tol {
            if normr == zero {
                // Exact solution, but `tol = 0` cannot be met strictly; a
                // further step would divide by `⟨p, A·p⟩ = 0`.
                break;
            }
            if numiter == self.maxiter {
                return Ok((x, info(false, r, normr, numiter, numops)));
            }
            // p ← r + β·p (β = 0 overwrites p); q ← a₀p + a₁A·p.
            let (p, q) = (&mut ws.v, &mut ws.w);
            p.axpby(V::Scalar::one(), &r, beta).map_err(ve)?;
            op.apply_into(p, q).map_err(ve)?;
            numops += 1;
            if a0 != V::Scalar::zero() || a1 != V::Scalar::one() {
                q.axpby(a0, p, a1).map_err(ve)?;
            }
            let pq = p.inner(q).map_err(ve)?;
            if !pq.all_finite() {
                return Err(Error::NonFiniteIterate);
            }
            if pq.re() <= zero {
                return Err(Error::NotPositiveDefinite);
            }
            let alpha = V::Scalar::from_real(rho) / pq;
            x.axpby(alpha, p, V::Scalar::one()).map_err(ve)?;
            r.axpby(-alpha, q, V::Scalar::one()).map_err(ve)?;
            normr = r.norm().map_err(ve)?;
            if !normr.is_finite() {
                return Err(Error::NonFiniteIterate);
            }
            numiter += 1;
            if normr < tol && numiter > 1 {
                // Recompute against the accumulated rounding (KrylovKit also
                // skips this after the first iteration) and restart.
                normr = residual_into(op, b, &x, a0, a1, &mut r, ws).map_err(ve)?;
                numops += 1;
                if !normr.is_finite() {
                    return Err(Error::NonFiniteIterate);
                }
                rho = normr * normr;
                beta = V::Scalar::zero();
            } else {
                let rho_old = rho;
                rho = normr * normr;
                beta = V::Scalar::from_real(rho / rho_old);
            }
        }
        Ok((x, info(normr < tol, r, normr, numiter, numops)))
    }
}

impl LinearSolver for Gmres {
    /// Port of KrylovKit v0.10.4 `src/linsolve/gmres.jl:linsolve(…, ::GMRES)`.
    fn solve<V, B, A>(
        &self,
        op: &mut A,
        b: &V,
        x0: &V,
        a0: V::Scalar,
        a1: V::Scalar,
        ws: &mut Workspace<V, B>,
    ) -> Result<(V, LinsolveInfo<V>), Error<V::Error>>
    where
        V: Vector,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        let tol: Real<V> = check_tol(self.tol)?;
        let kd = self.krylovdim;
        if kd == 0 {
            return Err(Error::InvalidArgument("krylovdim must be at least 1"));
        }
        if self.maxiter == 0 {
            return Err(Error::InvalidArgument("maxiter must be at least 1"));
        }
        let capacity = ws.basis.capacity();
        if capacity < kd {
            return Err(Error::InsufficientCapacity {
                needed: kd,
                capacity,
            });
        }
        let ve = Error::Vector;
        let zero = Real::<V>::zero();
        let (mut x, mut r, mut beta) = start(op, b, x0, a0, a1, ws)?;
        let mut numops = 1;
        if beta < tol || beta == zero {
            return Ok((x, info(beta < tol, r, beta, 0, numops)));
        }

        let s0 = V::Scalar::zero();
        let mut rm = vec![s0; kd * kd]; // R, column-major, leading dim kd
        let mut gs = vec![Givens::<V::Scalar>::identity(); kd];
        let mut y = vec![s0; kd + 1];
        let mut h = Vec::with_capacity(kd);
        let mut scratch = Vec::with_capacity(kd);
        let mut c = vec![s0; kd + 1];
        let mut numiter = 0;

        // Arnoldi start from r/‖r‖.
        let mut normres = arnoldi_step(op, ws, Some(&r), beta, &mut h, &mut scratch, self.orth)?;
        numops += 1;
        loop {
            numiter += 1;
            y[0] = V::Scalar::from_real(beta);
            let mut k = 1;
            rm[0] = a0 + a1 * h[0];
            let (g, rkk) = givens(rm[0], a1 * V::Scalar::from_real(normres), 0, 1);
            (gs[0], rm[0]) = (g, rkk);
            y[1] = s0;
            gs[0].lmul(&mut y);
            beta = y[1].modulus();

            while rm[(k - 1) * (kd + 1)] != s0 && beta > tol && ws.basis.len() < kd {
                normres = arnoldi_step(op, ws, None, normres, &mut h, &mut scratch, self.orth)?;
                numops += 1;
                k = ws.basis.len();
                let col = &mut rm[(k - 1) * kd..k * kd];
                for (ri, &hi) in col.iter_mut().zip(&h[..k - 1]) {
                    *ri = a1 * hi;
                }
                col[k - 1] = a0 + a1 * h[k - 1];
                for g in &gs[..k - 1] {
                    g.lmul(col);
                }
                let sub = a1 * V::Scalar::from_real(normres);
                if col[k - 1].modulus().hypot2(sub.modulus()) < tol {
                    // Singular in the Krylov space: rotate all weight into y[k].
                    let (g, yk) = givens(s0, y[k - 1], k, k - 1);
                    (gs[k - 1], y[k]) = (g, yk);
                    y[k - 1] = s0;
                    col[k - 1] = s0;
                } else {
                    let (g, rkk) = givens(col[k - 1], sub, k - 1, k);
                    (gs[k - 1], col[k - 1]) = (g, rkk);
                    y[k] = s0;
                    gs[k - 1].lmul(&mut y);
                }
                beta = y[k].modulus();
            }

            let n = if rm[(k - 1) * (kd + 1)] == s0 && y[k - 1] == s0 {
                k - 1
            } else {
                k
            };
            upper_solve(&rm, kd, &mut y, n).map_err(|()| Error::SingularKrylovSpace)?;
            ws.basis
                .combine_into(&y[..k], &mut x, V::Scalar::one(), V::Scalar::one())
                .map_err(ve)?;

            let recurrence = self.restart_residual == RestartResidual::Recurrence;
            let nr = if recurrence && beta > tol && numiter < self.maxiter {
                // r = y[k]·V_{k+1}·(G₁ᴴ⋯G_kᴴ e_{k+1}) with v_{k+1} = w/‖w‖,
                // instead of KrylovKit's rotation of the basis itself.
                c[..=k].fill(s0);
                c[k] = V::Scalar::one();
                for g in gs[..k].iter().rev() {
                    g.lmul_adjoint(&mut c[..=k]);
                }
                let yk = y[k];
                ws.basis.combine_into(&c[..k], &mut r, yk, s0).map_err(ve)?;
                // c[k] = 0 after a singular step, where ‖w‖ may vanish.
                if c[k] != s0 {
                    let a = yk * c[k] * V::Scalar::from_real(Real::<V>::one() / normres);
                    r.axpby(a, &ws.w, V::Scalar::one()).map_err(ve)?;
                }
                r.norm().map_err(ve)?
            } else {
                beta = residual_into(op, b, &x, a0, a1, &mut r, ws).map_err(ve)?;
                numops += 1;
                if !beta.is_finite() {
                    return Err(Error::NonFiniteIterate);
                }
                if beta < tol {
                    return Ok((x, info(true, r, beta, numiter, numops)));
                }
                if numiter >= self.maxiter || beta == zero {
                    return Ok((x, info(false, r, beta, numiter, numops)));
                }
                beta
            };
            normres = arnoldi_step(op, ws, Some(&r), nr, &mut h, &mut scratch, self.orth)?;
            numops += 1;
        }
    }
}

/// Appends `src/nrm` (`src = None`: the residual `ws.w`) to the basis and
/// performs one Arnoldi step: `ws.w ← A·v` orthogonalized, with the
/// coefficients in `h` (`len = basis.len()`). Returns `‖ws.w‖`.
fn arnoldi_step<V, B, A>(
    op: &mut A,
    ws: &mut Workspace<V, B>,
    src: Option<&V>,
    nrm: Real<V>,
    h: &mut Vec<V::Scalar>,
    scratch: &mut Vec<V::Scalar>,
    orth: Orth,
) -> Result<Real<V>, Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
{
    let ve = Error::Vector;
    let inv = V::Scalar::from_real(Real::<V>::one() / nrm);
    if src.is_some() {
        ws.basis.truncate(0);
    }
    let src = src.unwrap_or(&ws.w);
    ws.basis.push(src, inv).map_err(ve)?;
    ws.v.axpby(inv, src, V::Scalar::zero()).map_err(ve)?;
    op.apply_into(&ws.v, &mut ws.w).map_err(ve)?;
    let k = ws.basis.len();
    h.clear();
    h.resize(k, V::Scalar::zero());
    scratch.clear();
    scratch.resize(k, V::Scalar::zero());
    orthogonalize(&ws.basis, &mut ws.w, h, scratch, orth).map_err(ve)?;
    let nrm = ws.w.norm().map_err(ve)?;
    if !nrm.is_finite() || !h.iter().all(|c| c.all_finite()) {
        return Err(Error::NonFiniteIterate);
    }
    Ok(nrm)
}
