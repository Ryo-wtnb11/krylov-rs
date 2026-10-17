//! φ functions of the exponential integrator.

use num_traits::{Float, One, Zero};

use crate::scalar::Scalar;
use crate::scalar::sealed::RealOps as _;

/// `φ₁(x)` and `φ₂(x)` with `φ₀ = exp`, `φ_{j+1}(x) = (φ_j(x) - 1/j!)/x`.
///
/// Uses the Taylor series for `|x| < 1` (no cancellation) and the recurrence
/// otherwise.
pub(crate) fn phi12<S: Scalar>(x: S) -> (S, S) {
    let one = S::one();
    if x.modulus() < S::Real::one() {
        // φ_j(x) = Σ_n xⁿ/(n+j)!; for |x| < 1 the tail after 22 terms is below 1/23!.
        let (mut p1, mut p2) = (S::zero(), S::zero());
        let mut t1 = one; // xⁿ/(n+1)!
        let mut t2 = S::from_real(S::Real::from_f64(0.5)); // xⁿ/(n+2)!
        for n in 0..22usize {
            p1 += t1;
            p2 += t2;
            t1 = t1 * x / S::from_real(S::Real::from_usize(n + 2));
            t2 = t2 * x / S::from_real(S::Real::from_usize(n + 3));
        }
        (p1, p2)
    } else {
        let p1 = (x.scalar_exp() - one) / x;
        let p2 = (p1 - one) / x;
        (p1, p2)
    }
}

/// `e_kᵀ φ₂(x·T) e₁` for the symmetric tridiagonal `T` (diagonal `alpha`,
/// off-diagonal `beta[..k-1]`) by its Taylor series `Σ_n xⁿ (Tⁿe₁)_k/(n+2)!`,
/// valid while `ρ = |x|·‖T‖_∞ < k/2` (`None` otherwise).
///
/// `(Tⁿe₁)_k` vanishes exactly for `n < k-1`, so the series keeps relative
/// accuracy for this tiny entry, which an eigendecomposition only resolves
/// to `ε·max|φ₂|`. `t` and `next` are scratch of length `k`.
pub(crate) fn phi2_last<S: Scalar>(
    x: S,
    alpha: &[S::Real],
    beta: &[S::Real],
    t: &mut [S],
    next: &mut [S],
) -> Option<S> {
    let k = alpha.len();
    let mut tnorm = S::Real::zero();
    for i in 0..k {
        let mut r = alpha[i].abs();
        if i > 0 {
            r += beta[i - 1].abs();
        }
        if i + 1 < k {
            r += beta[i].abs();
        }
        tnorm = tnorm.max(r);
    }
    let rho = x.modulus() * tnorm;
    let half_k = S::Real::from_usize(k) / (S::Real::one() + S::Real::one());
    if rho.is_nan() || rho >= half_k {
        return None;
    }
    let (t, next) = (&mut t[..k], &mut next[..k]);
    t.fill(S::zero());
    t[0] = S::from_real(S::Real::from_f64(0.5)); // e₁/2!
    let mut bound = S::Real::from_f64(0.5); // ρⁿ/(n+2)! ≥ ‖tₙ‖_∞
    let mut sum = S::zero();
    let eps = S::Real::epsilon();
    for n in 0..k + 200 {
        sum += t[k - 1];
        if n + 1 >= k && bound <= eps * sum.modulus() {
            break;
        }
        let f = x / S::from_real(S::Real::from_usize(n + 3));
        for i in 0..k {
            let mut v = t[i] * S::from_real(alpha[i]);
            if i > 0 {
                v += t[i - 1] * S::from_real(beta[i - 1]);
            }
            if i + 1 < k {
                v += t[i + 1] * S::from_real(beta[i]);
            }
            next[i] = v * f;
        }
        t.copy_from_slice(next);
        bound = bound * rho / S::Real::from_usize(n + 3);
    }
    Some(sum)
}
