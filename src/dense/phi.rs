//! φ functions of the exponential integrator.

use num_traits::{Float, One, Zero};

use crate::scalar::Scalar;
use crate::scalar::sealed::RealOps as _;

/// `out[j] = φ_j(x)` for `j < out.len()`, with `φ₀ = exp` and
/// `φ_{j+1}(x) = (φ_j(x) - 1/j!)/x`.
///
/// Uses the Taylor series for `|x| < 1` (no cancellation) and the recurrence
/// otherwise.
// ponytail: the recurrence loses about log10(e·j!) digits of φ_j just above
// |x| = 1; fine for the few φ's expintegrator needs, a per-j series radius
// would lift that if large p ever matters.
pub(crate) fn phi_upto<S: Scalar>(x: S, out: &mut [S]) {
    // An exactly real argument (real t on a Hermitian map) is evaluated in
    // real arithmetic: complex `exp`, products and quotients turn an
    // overflowed `∞ + 0i` into `∞ + NaN·i`, which a NaN-blind step
    // controller would accept.
    if x.im() == S::Real::zero() {
        phi_into(x.re(), out, S::from_real);
    } else {
        phi_into(x, out, |v| v);
    }
}

fn phi_into<T: Scalar, S>(x: T, out: &mut [S], to: impl Fn(T) -> S) {
    let mut inv_fact = T::Real::one(); // 1/j!
    if x.modulus() < T::Real::one() {
        // φ_j(x) = Σ_n xⁿ/(n+j)!; for |x| < 1 the tail after 22 terms is below 1/(22+j)!.
        for (j, o) in out.iter_mut().enumerate() {
            if j > 0 {
                inv_fact /= T::Real::from_usize(j);
            }
            let mut t = T::from_real(inv_fact); // xⁿ/(n+j)!
            let mut sum = T::zero();
            for n in 0..22usize {
                sum += t;
                t = t * x / T::from_real(T::Real::from_usize(n + j + 1));
            }
            *o = to(sum);
        }
    } else {
        let mut phi = x.scalar_exp();
        for (j, o) in out.iter_mut().enumerate() {
            *o = to(phi);
            phi = (phi - T::from_real(inv_fact)) / x;
            inv_fact /= T::Real::from_usize(j + 1);
        }
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
