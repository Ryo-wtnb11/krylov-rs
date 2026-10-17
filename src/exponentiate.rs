//! `exp(t·A)·x` for Hermitian `A` by Lanczos with adaptive time steps.

use num_traits::{Float, One, Zero};

use crate::dense::{phi2_last, phi12, tridiag_eigh};
use crate::error::Error;
use crate::lanczos::{Fact, Lanczos};
use crate::scalar::sealed::{Ops as _, RealOps as _};
use crate::scalar::{Real, Scalar};
use crate::traits::{Basis, LinearMap, Vector, Workspace};

/// Convergence information of [`exponentiate`].
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
/// Follows KrylovKit v0.10.4 `expintegrator(A, t, (x,), ::Lanczos)`: the
/// Krylov space is built from `A·x` and `y = x + t·φ₁(t·A)·A·x` is advanced in
/// time steps chosen so that the local error estimate stays below `tol` per
/// unit time; `maxiter` bounds the number of Krylov subspaces (time steps),
/// and the last one is forced to reach `t`. `t` may be complex (use a complex
/// vector type for real-time evolution `t = -iτ`). `tol` must be positive and
/// `t` finite.
///
/// The workspace basis needs capacity `>= alg.get_krylovdim()`. Only the
/// returned vector is allocated.
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
    let eta = alg.validate(ws)?;
    if eta.is_nan() || eta <= Real::<V>::zero() {
        return Err(Error::InvalidArgument("tol must be positive"));
    }
    if !t.all_finite() {
        return Err(Error::InvalidArgument("t must be finite"));
    }
    if !x.norm().map_err(Error::Vector)?.is_finite() {
        return Err(Error::NonFiniteStartVector);
    }
    let kd = alg.get_krylovdim();
    let maxiter = alg.get_maxiter();
    let zero = Real::<V>::zero();
    let one = Real::<V>::one();

    let mut y = x.similar().map_err(Error::Vector)?;
    y.copy_from(x).map_err(Error::Vector)?;
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
    let sgn = t / V::Scalar::from_real(tau);
    let mut dtau = tau;
    let mut dtau_min = tau / Real::<V>::from_usize(maxiter);
    let maxerr = tau * eta;
    let gamma = Real::<V>::from_f64(0.8);
    let mut tau0 = zero;
    let mut totalerr = zero;

    let mut fact = Fact::<V::Scalar>::new(kd, alg.get_orth());
    op.apply_into(&y, &mut ws.w).map_err(Error::Vector)?;
    fact.numops = 1;
    let mut beta = ws.w.norm().map_err(Error::Vector)?;
    if beta < eta {
        info.normres = beta;
        info.numops = fact.numops;
        return Ok((y, info));
    }
    fact.start(ws, op, None, one / beta)
        .map_err(Error::Vector)?;
    let mut numiter = 1;

    let mut lam = vec![zero; kd];
    let mut e = vec![zero; kd];
    let mut z = vec![zero; kd * kd];
    let mut c = Vec::with_capacity(kd);
    let mut s1 = vec![V::Scalar::zero(); kd];
    let mut s2 = vec![V::Scalar::zero(); kd];

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
            eig(&fact, &mut lam, &mut e, &mut z)?;
            let (lam, z) = (&lam[..k], &z[..k * k]);
            let mut eps = err(sgn, dtau, beta, &fact, lam, z, &mut s1, &mut s2);
            let mut omega = eps / (dtau * eta);
            let mut q = Real::<V>::from_usize(k) / (one + one);
            while numiter < maxiter && omega >= one && dtau > dtau_min {
                let (eps_prev, dtau_prev) = (eps, dtau);
                dtau = (dtau * (gamma / omega).powf(one / (q + one))).max(dtau_min);
                eps = err(sgn, dtau, beta, &fact, lam, z, &mut s1, &mut s2);
                omega = eps / (dtau * eta);
                q = zero.max((eps / eps_prev).ln() / (dtau / dtau_prev).ln() - one);
            }
            tau0 = if numiter < maxiter { tau0 + dtau } else { tau };
            totalerr += eps;
            advance(ws, &mut y, &mut c, sgn, dtau, beta, lam, z).map_err(Error::Vector)?;
            if omega < gamma {
                dtau *= (gamma / omega).powf(one / (q + one));
            }
        } else if normres <= (tau - tau0) * eta || alg.get_eager() {
            let dt = tau - tau0;
            eig(&fact, &mut lam, &mut e, &mut z)?;
            let (lam, z) = (&lam[..k], &z[..k * k]);
            let eps = err(sgn, dt, beta, &fact, lam, z, &mut s1, &mut s2);
            if eps / (dt * eta) < one {
                totalerr += eps;
                advance(ws, &mut y, &mut c, sgn, dt, beta, lam, z).map_err(Error::Vector)?;
                tau0 = tau;
            }
        }
        if tau0 >= tau {
            info.converged = usize::from(totalerr <= maxerr);
            info.normres = totalerr;
            break;
        }
        if k < kd {
            fact.expand(ws, op).map_err(Error::Vector)?;
        } else {
            op.apply_into(&y, &mut ws.w).map_err(Error::Vector)?;
            fact.numops += 1;
            beta = ws.w.norm().map_err(Error::Vector)?;
            if beta < eta {
                // y is a fixed point (A·y = 0) up to tolerance.
                info.normres = beta;
                break;
            }
            fact.start(ws, op, None, one / beta)
                .map_err(Error::Vector)?;
            numiter += 1;
        }
    }
    info.numiter = numiter;
    info.numops = fact.numops;
    info.max_imag = fact.max_imag;
    Ok((y, info))
}

fn eig<S: Scalar, E>(
    fact: &Fact<S>,
    lam: &mut [S::Real],
    e: &mut [S::Real],
    z: &mut [S::Real],
) -> Result<(), Error<E>> {
    let k = fact.len();
    lam[..k].copy_from_slice(&fact.alpha);
    e[..k - 1].copy_from_slice(&fact.beta[..k - 1]);
    tridiag_eigh(&mut lam[..k], &mut e[..k], &mut z[..k * k])
        .map_err(|()| Error::DenseNoConvergence)
}

/// Local error estimate `Δτ·β·‖r‖·|e_kᵀ φ₂(sΔτ T) e₁|` (KrylovKit's `ϵ` for
/// `p = 1`, there read from the augmented matrix exponential).
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
) -> S::Real {
    let k = lam.len();
    let x = sgn * S::from_real(dtau);
    let e = phi2_last(x, &fact.alpha, &fact.beta, s1, s2).unwrap_or_else(|| {
        let mut e = S::zero();
        for (i, &l) in lam.iter().enumerate() {
            let (_, p2) = phi12(x * S::from_real(l));
            e += p2 * S::from_real(z[(k - 1) + i * k] * z[i * k]);
        }
        e
    });
    dtau * beta * fact.normres() * e.modulus()
}

/// `y ← y + β·sΔτ·(V·φ₁(sΔτ T)e₁ + (e_kᵀ φ₂(sΔτ T) e₁)·r)`.
#[allow(clippy::too_many_arguments)]
fn advance<V: Vector, B: Basis<V>>(
    ws: &Workspace<V, B>,
    y: &mut V,
    c: &mut Vec<V::Scalar>,
    sgn: V::Scalar,
    dtau: Real<V>,
    beta: Real<V>,
    lam: &[Real<V>],
    z: &[Real<V>],
) -> Result<(), V::Error> {
    let k = lam.len();
    let x = sgn * V::Scalar::from_real(dtau);
    c.clear();
    c.resize(k, V::Scalar::zero());
    let mut corr = V::Scalar::zero();
    for (i, &l) in lam.iter().enumerate() {
        let (p1, p2) = phi12(x * V::Scalar::from_real(l));
        let w1 = p1 * V::Scalar::from_real(z[i * k]);
        for (cj, &zj) in c.iter_mut().zip(&z[i * k..(i + 1) * k]) {
            *cj += w1 * V::Scalar::from_real(zj);
        }
        corr += p2 * V::Scalar::from_real(z[(k - 1) + i * k] * z[i * k]);
    }
    let a = V::Scalar::from_real(beta) * x;
    ws.basis.combine_into(c, y, a, V::Scalar::one())?;
    y.axpby(a * corr, &ws.w, V::Scalar::one())
}
