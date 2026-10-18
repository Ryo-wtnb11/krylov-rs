//! `φ`-function combinations `Σ_j t^j·φ_j(t·A)·u_j` (in particular
//! `exp(t·A)·x`) by Lanczos (Hermitian `A`) or Arnoldi (general `A`) with
//! adaptive time steps.

use core::ops::Mul;

use num_traits::{Float, One, Zero};

use crate::arnoldi::{Arnoldi, Fact as ArnoldiFact};
use crate::dense::{ExpmWork, Work, expm_into, phi_last, phi_pair};
use crate::error::Error;
use crate::lanczos::{Fact, Lanczos, validate};
use crate::orth::Orth;
use crate::scalar::sealed::{Ops as _, RealOps as _};
use crate::scalar::{Real, Scalar};
use crate::traits::{Basis, LinearMap, Vector, Workspace};

/// Convergence information of [`expintegrator`] and [`exponentiate`].
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ExpInfo<R> {
    /// `1` if the accumulated error estimate is `<= tol·|t|`, else `0`.
    pub converged: usize,
    /// Accumulated (rough) error estimate of the result.
    pub normres: R,
    /// Number of Krylov subspaces built.
    pub numiter: usize,
    /// Number of linear-map applications.
    pub numops: usize,
    /// Largest `|Im α|` discarded from the Rayleigh quotients.
    pub max_imag: R,
}

/// Computes `y = exp(t·A)·x`.
///
/// This is [`expintegrator`] with `u = [x]` (KrylovKit v0.10.4
/// `exponentiate`): the Krylov space is built from `A·x` and
/// `y = x + t·φ₁(t·A)·A·x` is advanced in time steps. `alg` is [`Lanczos`]
/// for a Hermitian `op` or [`Arnoldi`] for a general one. Only the returned
/// vector is allocated.
#[allow(clippy::type_complexity)]
pub fn exponentiate<V, B, A, L>(
    op: &mut A,
    t: V::Scalar,
    x: &V,
    alg: &L,
    ws: &mut Workspace<V, B>,
) -> Result<(V, ExpInfo<Real<V>>), Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
    L: KrylovProcess,
{
    expintegrator(op, t, core::slice::from_ref(x), alg, ws)
}

/// Computes `y = φ₀(t·A)·u₀ + t·φ₁(t·A)·u₁ + t²·φ₂(t·A)·u₂ + …` with
/// `φ₀ = exp` and `φ_{j+1}(z) = (φ_j(z) - 1/j!)/z`.
///
/// `y = x(t)` solves `ẋ = A·x + Σ_{j≥0} tʲ/j!·u_{j+1}` with `x(0) = u₀`.
/// Follows KrylovKit v0.10.4 `expintegrator(A, t, u, ::Union{Lanczos,
/// Arnoldi})` with `p = max(len(u) - 1, 1)`: from the current `w₁ = y` it
/// forms `w_{j+1} = A·w_j + Σ_l (sτ₀)ˡ/l!·u_{j+l}` (`p` applications), builds
/// the Krylov space from `w_{p+1}` and advances `y` in time steps chosen so
/// that the local error estimate stays below `tol` per unit time; `maxiter`
/// bounds the number of Krylov subspaces (time steps), and the last one is
/// forced to reach `t`. `t` may be complex (use a complex vector type for
/// real-time evolution `t = -iτ`). `u` must be non-empty, `tol` positive and
/// `t` finite (KrylovKit allows `t = ±∞` for `p = 1`). When `w_{p+1} = 0` with `p ≥ 2`, `y` is the exact polynomial
/// in `t` and is returned without a Krylov space (KrylovKit divides by zero
/// there). A non-finite (overflowed or NaN) step error estimate rejects the
/// step, which KrylovKit accepts when the estimate is NaN; if it stays
/// non-finite at the smallest allowed step, or a step's update
/// coefficients or the Krylov start norm are not finite, the call fails with [`Error::NonFiniteIterate`] instead of returning a
/// non-finite `y`.
///
/// `alg` selects the process: [`Lanczos`] (Hermitian `op`) evaluates the
/// `φ` functions of the tridiagonal Rayleigh quotient through its
/// eigendecomposition; [`Arnoldi`] (general `op`) takes the dense
/// exponential of the `(k+p+1)²` augmented Hessenberg matrix, as KrylovKit.
/// Its scaling exponent `⌈log₂(‖A‖₁/5.4)⌉` (Julia's `exp!`) is found by
/// exact halving, identical unless libm's `log2` rounds across an integer
/// (see [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md)).
///
/// The workspace basis needs capacity `>= alg.get_krylovdim()`. The returned
/// vector and `p - 1` scratch vectors are allocated (`p` per call); the small
/// dense scratch is allocated once per call, independent of the number of
/// time steps and step-size trials.
#[allow(clippy::type_complexity)]
pub fn expintegrator<V, B, A, L>(
    op: &mut A,
    t: V::Scalar,
    u: &[V],
    alg: &L,
    ws: &mut Workspace<V, B>,
) -> Result<(V, ExpInfo<Real<V>>), Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
    L: KrylovProcess,
{
    sealed::Sealed::expintegrator(alg, op, t, u, ws)
}

mod sealed {
    use super::*;

    pub trait Sealed {
        #[allow(clippy::type_complexity)]
        fn expintegrator<V, B, A>(
            &self,
            op: &mut A,
            t: V::Scalar,
            u: &[V],
            ws: &mut Workspace<V, B>,
        ) -> Result<(V, ExpInfo<Real<V>>), Error<V::Error>>
        where
            V: Vector,
            B: Basis<V>,
            A: LinearMap<V>;
    }
}

/// The Krylov process of [`expintegrator`] and [`exponentiate`]:
/// [`Lanczos`] or [`Arnoldi`]. Sealed.
pub trait KrylovProcess: sealed::Sealed {}
impl KrylovProcess for Lanczos {}
impl KrylovProcess for Arnoldi {}

impl sealed::Sealed for Lanczos {
    fn expintegrator<V, B, A>(
        &self,
        op: &mut A,
        t: V::Scalar,
        u: &[V],
        ws: &mut Workspace<V, B>,
    ) -> Result<(V, ExpInfo<Real<V>>), Error<V::Error>>
    where
        V: Vector,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        let eta = self.validate(ws)?;
        let s = Settings {
            kd: self.get_krylovdim(),
            maxiter: self.get_maxiter(),
            orth: self.get_orth(),
            eager: self.get_eager(),
            eta,
        };
        integrate::<_, _, _, LanczosProc<V::Scalar>>(op, t, u, ws, s)
    }
}

impl sealed::Sealed for Arnoldi {
    fn expintegrator<V, B, A>(
        &self,
        op: &mut A,
        t: V::Scalar,
        u: &[V],
        ws: &mut Workspace<V, B>,
    ) -> Result<(V, ExpInfo<Real<V>>), Error<V::Error>>
    where
        V: Vector,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        let eta = validate(self.get_krylovdim(), self.get_maxiter(), self.get_tol(), ws)?;
        let s = Settings {
            kd: self.get_krylovdim(),
            maxiter: self.get_maxiter(),
            orth: self.get_orth(),
            eager: self.get_eager(),
            eta,
        };
        integrate::<_, _, _, ArnoldiProc<V::Scalar>>(op, t, u, ws, s)
    }
}

struct Settings<R> {
    kd: usize,
    maxiter: usize,
    orth: Orth,
    eager: bool,
    eta: R,
}

/// What the step controller needs from a Krylov process: the factorization,
/// and the projected `φ` column `φ_p(sΔτ·H)e₁` and error-estimate entry
/// `e_kᵀφ_{p+1}(sΔτ·H)e₁` of its Rayleigh quotient `H`.
trait Process<S: Scalar>: Sized {
    fn new(kd: usize, orth: Orth, p: usize) -> Self;
    /// Starts a new Krylov space from `scale·ws.w`.
    fn start<V, B, A>(
        &mut self,
        ws: &mut Workspace<V, B>,
        op: &mut A,
        scale: S::Real,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
        A: LinearMap<V>;
    fn expand<V, B, A>(
        &mut self,
        ws: &mut Workspace<V, B>,
        op: &mut A,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
        A: LinearMap<V>;
    fn len(&self) -> usize;
    fn normres(&self) -> S::Real;
    fn numops(&self) -> usize;
    fn max_imag(&self) -> S::Real;
    /// Prepares the current Rayleigh quotient for [`Process::err`] calls.
    fn project<E>(&mut self) -> Result<(), Error<E>>;
    /// Local error estimate `Δτᵖ·β·‖r‖·|e_kᵀ φ_{p+1}(sΔτ H) e₁|`.
    fn err<E>(&mut self, sgn: S, dtau: S::Real, beta: S::Real) -> Result<S::Real, Error<E>>;
    /// `y ← y + Σ_{j<p} (sΔτ)ʲ/j!·w_{j+1}
    ///        + β(sΔτ)ᵖ·(V·φ_p(sΔτ H)e₁ + (e_kᵀ φ_{p+1}(sΔτ H) e₁)·r)`
    /// for the `dtau` of the last [`Process::err`] call.
    fn advance<V, B>(
        &mut self,
        ws: &Workspace<V, B>,
        y: &mut V,
        w: &[V],
        sgn: S,
        dtau: S::Real,
        beta: S::Real,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>;
}

/// The step controller of KrylovKit `expintegrator`, shared by both processes.
#[allow(clippy::type_complexity)]
fn integrate<V, B, A, P>(
    op: &mut A,
    t: V::Scalar,
    u: &[V],
    ws: &mut Workspace<V, B>,
    s: Settings<Real<V>>,
) -> Result<(V, ExpInfo<Real<V>>), Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
    P: Process<V::Scalar>,
{
    // Here, not in `expintegrator`: a generic `L: KrylovProcess` bound
    // still reaches the sealed method directly.
    let Some(u0) = u.first() else {
        return Err(Error::InvalidArgument("u must not be empty"));
    };
    let eta = s.eta;
    if eta.is_nan() || eta <= Real::<V>::zero() {
        return Err(Error::InvalidArgument("tol must be positive"));
    }
    if !t.all_finite() {
        return Err(Error::InvalidArgument("t must be finite"));
    }
    for ui in u {
        if !ui.norm().map_err(Error::Vector)?.is_finite() {
            return Err(Error::NonFiniteStartVector);
        }
    }
    let p = (u.len() - 1).max(1);
    let kd = s.kd;
    let maxiter = s.maxiter;
    let zero = Real::<V>::zero();
    let one = Real::<V>::one();

    let mut y = u0.similar().map_err(Error::Vector)?;
    y.copy_from(u0).map_err(Error::Vector)?;
    let tau = t.modulus();
    let mut info = ExpInfo {
        converged: 1,
        normres: zero,
        numiter: 0,
        numops: 0,
        max_imag: zero,
    };
    if tau == zero {
        return Ok((y, info));
    }
    // w_2 … w_p; w_{p+1} lives in ws.w and becomes the Krylov start.
    let mut w = (1..p)
        .map(|_| u0.similar())
        .collect::<Result<Vec<_>, _>>()
        .map_err(Error::Vector)?;
    let sgn = t / V::Scalar::from_real(tau);
    let mut dtau = tau;
    let mut dtau_min = tau / Real::<V>::from_usize(maxiter);
    let maxerr = tau * eta;
    let gamma = Real::<V>::from_f64(0.8);
    let mut tau0 = zero;
    let mut totalerr = zero;

    let mut proc = P::new(kd, s.orth, p);
    let mut numops = 0;
    let mut numiter = 0;

    'rebuild: loop {
        let st0 = sgn * V::Scalar::from_real(tau0);
        let beta =
            rebuild(op, &y, u, &mut w, &mut ws.w, st0, &mut numops).map_err(Error::Vector)?;
        if !beta.is_finite() {
            // The Krylov start would be scaled by 0 or NaN.
            return Err(Error::NonFiniteIterate);
        }
        if p == 1 && beta < eta {
            // y is a fixed point (A·y + u₁ = 0) up to tolerance.
            info.normres = beta;
            break;
        }
        if beta == zero {
            // The φ_p term vanishes identically, so the polynomial part is exact.
            poly(&mut y, &w, sgn * V::Scalar::from_real(tau - tau0)).map_err(Error::Vector)?;
            info.converged = usize::from(totalerr <= maxerr);
            info.normres = totalerr;
            break;
        }
        proc.start(ws, op, one / beta)?;
        numiter += 1;

        loop {
            let k = proc.len();
            let normres = proc.normres();
            if k == kd {
                if numiter < maxiter {
                    dtau = dtau.min(tau - tau0);
                    dtau_min = (tau - tau0) / Real::<V>::from_usize(maxiter - numiter + 1);
                } else {
                    dtau = tau - tau0;
                }
                proc.project()?;
                let mut eps = nan_to_inf(proc.err(sgn, dtau, beta)?);
                let mut omega = eps / (dtau * eta);
                let mut q = Real::<V>::from_usize(k) / (one + one);
                while numiter < maxiter && omega >= one && dtau > dtau_min {
                    let (eps_prev, dtau_prev) = (eps, dtau);
                    dtau = (dtau * (gamma / omega).powf(one / (q + one))).max(dtau_min);
                    eps = nan_to_inf(proc.err(sgn, dtau, beta)?);
                    omega = eps / (dtau * eta);
                    q = zero.max((eps / eps_prev).ln() / (dtau / dtau_prev).ln() - one);
                }
                if !eps.is_finite() {
                    return Err(Error::NonFiniteIterate);
                }
                tau0 = if numiter < maxiter { tau0 + dtau } else { tau };
                totalerr += eps;
                proc.advance(ws, &mut y, &w, sgn, dtau, beta)?;
                if omega < gamma {
                    dtau *= (gamma / omega).powf(one / (q + one));
                }
            } else if normres <= (tau - tau0) * eta || s.eager {
                let dt = tau - tau0;
                proc.project()?;
                let eps = proc.err(sgn, dt, beta)?;
                if eps / (dt * eta) < one {
                    totalerr += eps;
                    proc.advance(ws, &mut y, &w, sgn, dt, beta)?;
                    tau0 = tau;
                }
            }
            if tau0 >= tau {
                info.converged = usize::from(totalerr <= maxerr);
                info.normres = totalerr;
                break 'rebuild;
            }
            if k == kd {
                continue 'rebuild;
            }
            proc.expand(ws, op)?;
        }
    }
    info.numiter = numiter;
    info.numops = numops + proc.numops();
    info.max_imag = proc.max_imag();
    Ok((y, info))
}

/// Lanczos: `φ` functions of the tridiagonal `T` through its
/// eigendecomposition `T = Z·Λ·Zᵀ` (stage 1).
struct LanczosProc<S: Scalar> {
    fact: Fact<S>,
    lam: Vec<S::Real>,
    e: Vec<S::Real>,
    z: Vec<S::Real>,
    lw: Work<S::Real, S::Real>,
    c: Vec<S>,
    s1: Vec<S>,
    s2: Vec<S>,
    p: usize,
}

impl<S: Scalar> Process<S> for LanczosProc<S> {
    fn new(kd: usize, orth: Orth, p: usize) -> Self {
        let zero = S::Real::zero();
        Self {
            fact: Fact::new(kd, orth),
            lam: vec![zero; kd],
            e: vec![zero; kd],
            z: vec![zero; kd * kd],
            lw: Work::default(),
            c: Vec::with_capacity(kd),
            s1: vec![S::zero(); kd],
            s2: vec![S::zero(); kd],
            p,
        }
    }
    fn start<V, B, A>(
        &mut self,
        ws: &mut Workspace<V, B>,
        op: &mut A,
        scale: S::Real,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        self.fact.start(ws, op, None, scale)
    }
    fn expand<V, B, A>(
        &mut self,
        ws: &mut Workspace<V, B>,
        op: &mut A,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        self.fact.expand(ws, op)
    }
    fn len(&self) -> usize {
        self.fact.len()
    }
    fn normres(&self) -> S::Real {
        self.fact.normres()
    }
    fn numops(&self) -> usize {
        self.fact.numops
    }
    fn max_imag(&self) -> S::Real {
        self.fact.max_imag
    }
    fn project<E>(&mut self) -> Result<(), Error<E>> {
        eig(
            &self.fact,
            &mut self.lam,
            &mut self.e,
            &mut self.z,
            &mut self.lw,
        )
    }
    fn err<E>(&mut self, sgn: S, dtau: S::Real, beta: S::Real) -> Result<S::Real, Error<E>> {
        let k = self.fact.len();
        Ok(err(
            sgn,
            dtau,
            beta,
            &self.fact,
            &self.lam[..k],
            &self.z[..k * k],
            &mut self.s1,
            &mut self.s2,
            self.p,
        ))
    }
    fn advance<V, B>(
        &mut self,
        ws: &Workspace<V, B>,
        y: &mut V,
        w: &[V],
        sgn: S,
        dtau: S::Real,
        beta: S::Real,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
    {
        let k = self.fact.len();
        advance(
            ws,
            y,
            w,
            &mut self.c,
            self.p,
            sgn,
            dtau,
            beta,
            &self.lam[..k],
            &self.z[..k * k],
        )
    }
}

/// Arnoldi: `exp` of the augmented matrix
/// `[[sΔτ·H, e₁, 0], [0, 0, I_p], [0, 0, 0]]` of size `k + p + 1`
/// (KrylovKit `expintegrator.jl`, `H[1, K+1] = 1`, `H[K+i, K+i+1] = 1`),
/// whose column `k + p` holds `φ_p(sΔτ·H)e₁` in its first `k` rows and
/// whose entry `[k, k + p + 1]` is `e_kᵀφ_{p+1}(sΔτ·H)e₁` (1-based).
struct ArnoldiProc<S: Scalar> {
    fact: ArnoldiFact<S>,
    p: usize,
    aug: Vec<S>,
    exph: ExpmWork<S>,
}

impl<S: Scalar> ArnoldiProc<S> {
    fn dim(&self) -> usize {
        self.fact.k + self.p + 1
    }
}

impl<S: Scalar> Process<S> for ArnoldiProc<S> {
    fn new(kd: usize, orth: Orth, p: usize) -> Self {
        Self {
            fact: ArnoldiFact::new(kd, orth),
            p,
            aug: vec![S::zero(); (kd + p + 1) * (kd + p + 1)],
            exph: ExpmWork::new(kd + p + 1),
        }
    }
    fn start<V, B, A>(
        &mut self,
        ws: &mut Workspace<V, B>,
        op: &mut A,
        scale: S::Real,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        ws.basis.truncate(0);
        self.fact.push(ws, op, None, scale)
    }
    fn expand<V, B, A>(
        &mut self,
        ws: &mut Workspace<V, B>,
        op: &mut A,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        let scale = S::Real::one() / self.fact.normres();
        self.fact.push(ws, op, None, scale)
    }
    fn len(&self) -> usize {
        self.fact.k
    }
    fn normres(&self) -> S::Real {
        self.fact.normres()
    }
    fn numops(&self) -> usize {
        self.fact.numops
    }
    fn max_imag(&self) -> S::Real {
        S::Real::zero()
    }
    fn project<E>(&mut self) -> Result<(), Error<E>> {
        Ok(())
    }
    fn err<E>(&mut self, sgn: S, dtau: S::Real, beta: S::Real) -> Result<S::Real, Error<E>> {
        let (k, p, n, ld) = (self.fact.k, self.p, self.dim(), self.fact.ld);
        let x = sgn * S::from_real(dtau);
        let aug = &mut self.aug[..n * n];
        aug.fill(S::zero());
        for j in 0..k {
            for i in 0..k {
                aug[i + j * n] = self.fact.h[i + j * ld] * x;
            }
        }
        aug[k * n] = S::one();
        for i in 1..=p {
            aug[(k + i - 1) + (k + i) * n] = S::one();
        }
        expm_into(aug, n, &mut self.exph).map_err(|()| Error::DenseNoConvergence)?;
        let e = self.exph.result()[(k - 1) + (k + p) * n];
        Ok(powu(dtau, p) * beta * self.fact.normres() * e.modulus())
    }
    fn advance<V, B>(
        &mut self,
        ws: &Workspace<V, B>,
        y: &mut V,
        w: &[V],
        sgn: S,
        dtau: S::Real,
        beta: S::Real,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
    {
        let (k, p, n) = (self.fact.k, self.p, self.dim());
        let x = sgn * S::from_real(dtau);
        let col = (k + p - 1) * n;
        let c = &self.exph.result()[col..col + k];
        let corr = self.exph.result()[(k - 1) + (k + p) * n];
        let a = S::from_real(beta) * powu(x, p);
        check_step(a, c, corr)?;
        poly(y, w, x).map_err(Error::Vector)?;
        ws.basis
            .combine_into(c, y, a, S::one())
            .map_err(Error::Vector)?;
        y.axpby(a * corr, &ws.w, S::one()).map_err(Error::Vector)
    }
}

/// Forms `w_{j+1} = A·w_j + Σ_{l=0}^{p-j} (sτ₀)ˡ/l!·u_{j+l}` for `j = 1..=p`
/// with `w₁ = y`, `w_2 … w_p` in `w` and `w_{p+1}` in `last`; returns
/// `‖w_{p+1}‖`. Absent `u_i` (exponentiate's `u₁`) are zero.
fn rebuild<V: Vector, A: LinearMap<V>>(
    op: &mut A,
    y: &V,
    u: &[V],
    w: &mut [V],
    last: &mut V,
    st0: V::Scalar,
    numops: &mut usize,
) -> Result<Real<V>, V::Error> {
    let p = w.len() + 1;
    for j in 1..=p {
        let (head, tail) = w.split_at_mut(j - 1);
        let src = if j == 1 { y } else { &head[j - 2] };
        let dst = if j < p { &mut tail[0] } else { &mut *last };
        op.apply_into(src, dst)?;
        *numops += 1;
        let mut coef = V::Scalar::one();
        for (l, ul) in u.iter().skip(j).enumerate() {
            dst.axpby(coef, ul, V::Scalar::one())?;
            coef = coef * st0 / V::Scalar::from_real(Real::<V>::from_usize(l + 1));
        }
    }
    last.norm()
}

/// `y ← y + Σ_{j=1}^{p-1} xʲ/j!·w_{j+1}`.
fn poly<V: Vector>(y: &mut V, w: &[V], x: V::Scalar) -> Result<(), V::Error> {
    let mut coef = V::Scalar::one();
    for (j, wj) in w.iter().enumerate() {
        coef = coef * x / V::Scalar::from_real(Real::<V>::from_usize(j + 1));
        y.axpby(coef, wj, V::Scalar::one())?;
    }
    Ok(())
}

/// KrylovKit rejects a step when `ω >= 1`, which a NaN estimate passes;
/// an indeterminate estimate is rejected like an overflowed one.
fn nan_to_inf<R: Float>(e: R) -> R {
    if e.is_nan() { R::infinity() } else { e }
}

/// `xᵖ` by repeated multiplication (exact for `p = 1`).
fn powu<T: Copy + One + Mul<Output = T>>(x: T, p: usize) -> T {
    (0..p).fold(T::one(), |acc, _| acc * x)
}

fn eig<S: Scalar, E>(
    fact: &Fact<S>,
    lam: &mut [S::Real],
    e: &mut [S::Real],
    z: &mut [S::Real],
    lw: &mut Work<S::Real, S::Real>,
) -> Result<(), Error<E>> {
    let k = fact.len();
    lam[..k].copy_from_slice(&fact.alpha);
    e[..k - 1].copy_from_slice(&fact.beta[..k - 1]);
    S::Real::stegr(&mut lam[..k], &mut e[..k], &mut z[..k * k], lw)
        .map_err(|()| Error::DenseNoConvergence)
}

/// Local error estimate `Δτᵖ·β·‖r‖·|e_kᵀ φ_{p+1}(sΔτ T) e₁|` (KrylovKit's
/// `ϵ`, there read from the augmented matrix exponential).
#[allow(clippy::too_many_arguments)]
fn err<S: Scalar>(
    sgn: S,
    dtau: S::Real,
    beta: S::Real,
    fact: &Fact<S>,
    lam: &[S::Real],
    z: &[S::Real],
    s1: &mut [S],
    s2: &mut [S],
    p: usize,
) -> S::Real {
    let k = lam.len();
    let x = sgn * S::from_real(dtau);
    let e = phi_last(x, p + 1, &fact.alpha, &fact.beta, s1, s2).unwrap_or_else(|| {
        let mut e = S::zero();
        for (i, &l) in lam.iter().enumerate() {
            let (_, phi) = phi_pair(x * S::from_real(l), p);
            e += phi.mul_real(z[(k - 1) + i * k] * z[i * k]);
        }
        e
    });
    powu(dtau, p) * beta * fact.normres() * e.modulus()
}

/// `y ← y + Σ_{j<p} (sΔτ)ʲ/j!·w_{j+1}
///        + β(sΔτ)ᵖ·(V·φ_p(sΔτ T)e₁ + (e_kᵀ φ_{p+1}(sΔτ T) e₁)·r)`.
#[allow(clippy::too_many_arguments)]
fn advance<V: Vector, B: Basis<V>>(
    ws: &Workspace<V, B>,
    y: &mut V,
    w: &[V],
    c: &mut Vec<V::Scalar>,
    p: usize,
    sgn: V::Scalar,
    dtau: Real<V>,
    beta: Real<V>,
    lam: &[Real<V>],
    z: &[Real<V>],
) -> Result<(), Error<V::Error>> {
    let k = lam.len();
    let x = sgn * V::Scalar::from_real(dtau);
    c.clear();
    c.resize(k, V::Scalar::zero());
    let mut corr = V::Scalar::zero();
    for (i, &l) in lam.iter().enumerate() {
        let (phi_p, phi_q) = phi_pair(x * V::Scalar::from_real(l), p);
        let w1 = phi_p.mul_real(z[i * k]);
        for (cj, &zj) in c.iter_mut().zip(&z[i * k..(i + 1) * k]) {
            *cj += w1.mul_real(zj);
        }
        corr += phi_q.mul_real(z[(k - 1) + i * k] * z[i * k]);
    }
    let a = V::Scalar::from_real(beta) * powu(x, p);
    check_step(a, c, corr)?;
    poly(y, w, x).map_err(Error::Vector)?;
    ws.basis
        .combine_into(c, y, a, V::Scalar::one())
        .map_err(Error::Vector)?;
    y.axpby(a * corr, &ws.w, V::Scalar::one())
        .map_err(Error::Vector)
}

/// Rejects a step whose `k + 1` update coefficients `a·c_j`, `a·corr` are
/// not finite. Checked here, not through `y.norm()`: a user norm may
/// overflow (`√⟨y, y⟩`) for a finite `y`.
fn check_step<S: Scalar, E>(a: S, c: &[S], corr: S) -> Result<(), Error<E>> {
    if c.iter().chain([&corr]).all(|&v| (a * v).all_finite()) {
        Ok(())
    } else {
        Err(Error::NonFiniteIterate)
    }
}
