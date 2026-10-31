//! φ functions of the exponential integrator.

use num_traits::{Float, One, Zero};

use crate::scalar::Scalar;
use crate::scalar::sealed::RealOps as _;

/// `(φ_p(x), φ_{p+1}(x))`, with `φ₀ = exp` and
/// `φ_{j+1}(x) = (φ_j(x) - 1/j!)/x`: the only two entries expintegrator reads.
///
/// For `|x| < p + 1` both come from the Taylor series
/// `φ_j(x) = Σ_n xⁿ/(n+j)!`, whose terms then decrease from the first; its
/// cancellation factor `Σ|terms|/|φ_p(x)| = φ_p(|x|)/|φ_p(x)|` is largest
/// as `x → -(p+1)` and stays below about 12 for `p ≤ 12`. Otherwise the
/// recurrence runs up from `exp(x)`; with `|x| ≥ p + 1 > j` each step
/// `φ_j - 1/j!` keeps most of `φ_j`. The error is a few ulps times the
/// conditioning of `φ_j` at `x` (graded against a double-double oracle in
/// `dense::tests`). A fixed radius 1
/// instead lost about `j!` ulps of `φ_j` just above it, and a wider
/// series radius loses digits to cancellation of terms near `e^{|x|}`.
///
/// When `e^{Re x}` overflows the result is infinite, even where `φ_j(x)` (about
/// `e^x/x^j`) would be representable; expintegrator then rejects the step
/// as non-finite and takes a shorter one.
pub(crate) fn phi_pair<S: Scalar>(x: S, p: usize) -> (S, S) {
    // An exactly real argument (real t on a Hermitian map) is evaluated in
    // real arithmetic: complex `exp`, products and quotients turn an
    // overflowed `∞ + 0i` into `∞ + NaN·i`, which a NaN-blind step
    // controller would accept.
    if x.im() == S::Real::zero() {
        let (a, b) = pair(x.re(), p);
        (S::from_real(a), S::from_real(b))
    } else {
        pair(x, p)
    }
}

fn pair<T: Scalar>(x: T, p: usize) -> (T, T) {
    let r = x.modulus();
    let mut inv_fact = T::Real::one(); // 1/p!
    for i in 1..=p {
        inv_fact /= T::Real::from_usize(i);
    }
    if r < T::Real::from_usize(p + 1) {
        let eps = T::Real::epsilon();
        let mut u = T::from_real(inv_fact); // xⁿ/(n+p)!
        let (mut sp, mut sq) = (T::zero(), T::zero());
        // The terms fall below ε·|φ_{p+1}| after O(p + 40) terms; the cap
        // only guards a sum that is exactly zero.
        for n in 0..p + 200 {
            let d = T::Real::from_usize(n + p + 1);
            sp += u;
            sq += u / T::from_real(d);
            u = u * x / T::from_real(d);
            if d > r && u.modulus() <= eps * sq.modulus() {
                break;
            }
        }
        (sp, sq)
    } else {
        let mut phi = x.scalar_exp();
        let mut f = T::Real::one(); // 1/j!
        for j in 0..p {
            phi = (phi - T::from_real(f)) / x;
            f /= T::Real::from_usize(j + 1);
        }
        (phi, (phi - T::from_real(f)) / x)
    }
}

/// `e_kᵀ φ_j(x·T) e₁` for the symmetric tridiagonal `T` (diagonal `alpha`,
/// off-diagonal `beta[..k-1]`) by its Taylor series `Σ_n xⁿ (Tⁿe₁)_k/(n+j)!`,
/// valid while `ρ = |x|·‖T‖_∞ < k/2` (`None` otherwise).
///
/// `(Tⁿe₁)_k` vanishes exactly for `n < k-1`, so the series keeps relative
/// accuracy for this tiny entry, which an eigendecomposition only resolves
/// to `ε·max|φ_j|`. `t` and `next` are scratch of length `k`.
pub(crate) fn phi_last<S: Scalar>(
    x: S,
    j: usize,
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
    let mut inv_fact = S::Real::one();
    for i in 1..=j {
        inv_fact /= S::Real::from_usize(i);
    }
    let (t, next) = (&mut t[..k], &mut next[..k]);
    t.fill(S::zero());
    t[0] = S::from_real(inv_fact); // e₁/j!
    let mut bound = inv_fact; // ρⁿ/(n+j)! ≥ ‖tₙ‖_∞
    let mut sum = S::zero();
    let eps = S::Real::epsilon();
    for n in 0..k + 200 {
        sum += t[k - 1];
        if n + 1 >= k && bound <= eps * sum.modulus() {
            break;
        }
        let f = x / S::from_real(S::Real::from_usize(n + j + 1));
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
        bound = bound * rho / S::Real::from_usize(n + j + 1);
    }
    Some(sum)
}
