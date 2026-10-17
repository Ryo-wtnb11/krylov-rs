//! `φ`-function combinations `Σ_j t^j·φ_j(t·A)·u_j` (in particular
//! `exp(t·A)·x`) for Hermitian `A` by Lanczos with adaptive time steps.

use core::ops::Mul;

use num_traits::{Float, One, Zero};

use crate::dense::{Work, phi_last, phi_upto};
use crate::error::Error;
use crate::lanczos::{Fact, Lanczos};
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

/// Computes `y = exp(t·A)·x` for a Hermitian map `op`.
///
/// This is [`expintegrator`] with `u = [x]` (KrylovKit v0.10.4
/// `exponentiate`): the Krylov space is built from `A·x` and
/// `y = x + t·φ₁(t·A)·A·x` is advanced in time steps. Only the returned
/// vector is allocated.
#[allow(clippy::type_complexity)]
pub fn exponentiate<V, B, A>(
    op: &mut A,
    t: V::Scalar,
    x: &V,
    alg: &Lanczos,
    ws: &mut Workspace<V, B>,
) -> Result<(V, ExpInfo<Real<V>>), Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
{
    expintegrator(op, t, core::slice::from_ref(x), alg, ws)
}

/// Computes `y = φ₀(t·A)·u₀ + t·φ₁(t·A)·u₁ + t²·φ₂(t·A)·u₂ + …` for a
/// Hermitian map `op`, with `φ₀ = exp` and `φ_{j+1}(z) = (φ_j(z) - 1/j!)/z`.
///
/// `y = x(t)` solves `ẋ = A·x + Σ_{j≥0} tʲ/j!·u_{j+1}` with `x(0) = u₀`.
/// Follows KrylovKit v0.10.4 `expintegrator(A, t, u, ::Lanczos)` with
/// `p = max(len(u) - 1, 1)`: from the current `w₁ = y` it forms
/// `w_{j+1} = A·w_j + Σ_l (sτ₀)ˡ/l!·u_{j+l}` (`p` applications), builds the
/// Krylov space from `w_{p+1}` and advances `y` in time steps chosen so that
/// the local error estimate stays below `tol` per unit time; `maxiter` bounds
/// the number of Krylov subspaces (time steps), and the last one is forced to
/// reach `t`. `t` may be complex (use a complex vector type for real-time
/// evolution `t = -iτ`). `u` must be non-empty, `tol` positive and `t` finite.
/// When `w_{p+1} = 0` with `p ≥ 2`, `y` is the exact polynomial in `t` and is
/// returned without a Krylov space (KrylovKit divides by zero there).
///
/// The workspace basis needs capacity `>= alg.get_krylovdim()`. The returned
/// vector and `p - 1` scratch vectors are allocated (`p` per call).
#[allow(clippy::type_complexity)]
pub fn expintegrator<V, B, A>(
    op: &mut A,
    t: V::Scalar,
    u: &[V],
    alg: &Lanczos,
    ws: &mut Workspace<V, B>,
) -> Result<(V, ExpInfo<Real<V>>), Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
{
    let Some(u0) = u.first() else {
        return Err(Error::InvalidArgument("u must not be empty"));
    };
    let eta = alg.validate(ws)?;
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
    let kd = alg.get_krylovdim();
    let maxiter = alg.get_maxiter();
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
    // w_2 … w_p; w_{p+1} lives in ws.w and becomes the Lanczos start.
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

    let mut fact = Fact::<V::Scalar>::new(kd, alg.get_orth());
    let mut numiter = 0;
    let mut lam = vec![zero; kd];
    let mut e = vec![zero; kd];
    let mut z = vec![zero; kd * kd];
    let mut lw = Work::default();
    let mut c = Vec::with_capacity(kd);
    let mut s1 = vec![V::Scalar::zero(); kd];
    let mut s2 = vec![V::Scalar::zero(); kd];
    let mut phis = vec![V::Scalar::zero(); p + 2];

    'rebuild: loop {
        let st0 = sgn * V::Scalar::from_real(tau0);
        let beta =
            rebuild(op, &y, u, &mut w, &mut ws.w, st0, &mut fact.numops).map_err(Error::Vector)?;
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
        fact.start(ws, op, None, one / beta)
            .map_err(Error::Vector)?;
        numiter += 1;

        loop {
            let k = fact.len();
            let normres = fact.normres();
            if k == kd {
                if numiter < maxiter {
                    dtau = dtau.min(tau - tau0);
                    dtau_min = (tau - tau0) / Real::<V>::from_usize(maxiter - numiter + 1);
                } else {
                    dtau = tau - tau0;
                }
                eig(&fact, &mut lam, &mut e, &mut z, &mut lw)?;
                let (lam, z) = (&lam[..k], &z[..k * k]);
                let mut eps = err(sgn, dtau, beta, &fact, lam, z, &mut s1, &mut s2, &mut phis);
                let mut omega = eps / (dtau * eta);
                let mut q = Real::<V>::from_usize(k) / (one + one);
                while numiter < maxiter && omega >= one && dtau > dtau_min {
                    let (eps_prev, dtau_prev) = (eps, dtau);
                    dtau = (dtau * (gamma / omega).powf(one / (q + one))).max(dtau_min);
                    eps = err(sgn, dtau, beta, &fact, lam, z, &mut s1, &mut s2, &mut phis);
                    omega = eps / (dtau * eta);
                    q = zero.max((eps / eps_prev).ln() / (dtau / dtau_prev).ln() - one);
                }
                tau0 = if numiter < maxiter { tau0 + dtau } else { tau };
                totalerr += eps;
                advance(ws, &mut y, &w, &mut c, &mut phis, sgn, dtau, beta, lam, z)
                    .map_err(Error::Vector)?;
                if omega < gamma {
                    dtau *= (gamma / omega).powf(one / (q + one));
                }
            } else if normres <= (tau - tau0) * eta || alg.get_eager() {
                let dt = tau - tau0;
                eig(&fact, &mut lam, &mut e, &mut z, &mut lw)?;
                let (lam, z) = (&lam[..k], &z[..k * k]);
                let eps = err(sgn, dt, beta, &fact, lam, z, &mut s1, &mut s2, &mut phis);
                if eps / (dt * eta) < one {
                    totalerr += eps;
                    advance(ws, &mut y, &w, &mut c, &mut phis, sgn, dt, beta, lam, z)
                        .map_err(Error::Vector)?;
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
            fact.expand(ws, op).map_err(Error::Vector)?;
        }
    }
    info.numiter = numiter;
    info.numops = fact.numops;
    info.max_imag = fact.max_imag;
    Ok((y, info))
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
/// `ϵ`, there read from the augmented matrix exponential); `p + 2 = phis.len()`.
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
    phis: &mut [S],
) -> S::Real {
    let k = lam.len();
    let p = phis.len() - 2;
    let x = sgn * S::from_real(dtau);
    let e = phi_last(x, p + 1, &fact.alpha, &fact.beta, s1, s2).unwrap_or_else(|| {
        let mut e = S::zero();
        for (i, &l) in lam.iter().enumerate() {
            phi_upto(x * S::from_real(l), phis);
            e += phis[p + 1] * S::from_real(z[(k - 1) + i * k] * z[i * k]);
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
    phis: &mut [V::Scalar],
    sgn: V::Scalar,
    dtau: Real<V>,
    beta: Real<V>,
    lam: &[Real<V>],
    z: &[Real<V>],
) -> Result<(), V::Error> {
    let k = lam.len();
    let p = phis.len() - 2;
    let x = sgn * V::Scalar::from_real(dtau);
    poly(y, w, x)?;
    c.clear();
    c.resize(k, V::Scalar::zero());
    let mut corr = V::Scalar::zero();
    for (i, &l) in lam.iter().enumerate() {
        phi_upto(x * V::Scalar::from_real(l), phis);
        let w1 = phis[p] * V::Scalar::from_real(z[i * k]);
        for (cj, &zj) in c.iter_mut().zip(&z[i * k..(i + 1) * k]) {
            *cj += w1 * V::Scalar::from_real(zj);
        }
        corr += phis[p + 1] * V::Scalar::from_real(z[(k - 1) + i * k] * z[i * k]);
    }
    let a = V::Scalar::from_real(beta) * powu(x, p);
    ws.basis.combine_into(c, y, a, V::Scalar::one())?;
    y.axpby(a * corr, &ws.w, V::Scalar::one())
}
