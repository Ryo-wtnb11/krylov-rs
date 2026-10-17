//! Householder reflectors.

use num_traits::{Float, One, Zero};

use crate::scalar::Scalar;

/// Turns `x` into a Householder vector `u` (`u[0] = 1`) and returns `β` such
/// that `(I - β u uᴴ)·x_old = ‖x_old‖·e₀`; `None` when no reflection is needed.
///
/// Port of KrylovKit v0.10.4 `src/dense/reflector.jl:_householder!` with the
/// pivot at the first entry, plus the range safeguards of LAPACK `xLARFG`:
/// `u` and `β` do not change when `x` is scaled, so an `x` whose largest
/// component is below `√(min positive)` or above `√(max·ε)` is evaluated as
/// `2ᵏ·x` (exact), where the squared norm neither underflows nor overflows.
/// LAPACK's reflector itself is not used: it maps `x` to `-sign(Re x₀)·‖x‖·e₀`
/// and for complex `x` leaves a complex `β`, while every caller here relies on
/// `+‖x‖·e₀`.
pub(crate) fn householder<S: Scalar>(x: &mut [S]) -> Option<S> {
    let (first, rest) = x.split_first_mut()?;
    let biggest = |v: &S| v.re().abs().max(v.im().abs());
    let s = rest.iter().map(biggest).fold(biggest(first), S::Real::max);
    let small = s > S::Real::zero() && s < S::Real::min_positive_value().sqrt();
    let large = s.is_finite() && s > (S::Real::max_value() * S::Real::epsilon()).sqrt();
    let (f1, f2) = if small || large {
        pow2(-s.log2().round())
    } else {
        (S::Real::one(), S::Real::one())
    };
    // Multiplying by 1 is exact, so in-range input is evaluated as before.
    let sc = |v: S| v.mul_real(f1).mul_real(f2);
    let mut sigma = S::Real::zero();
    for r in rest.iter() {
        let r = sc(*r);
        sigma += r.re() * r.re() + r.im() * r.im();
    }
    let vi = sc(*first);
    let nu = (vi.re() * vi.re() + vi.im() * vi.im() + sigma).sqrt();
    if sigma == S::Real::zero() && vi == S::from_real(nu) {
        return None;
    }
    let nu_s = S::from_real(nu);
    let piv = if vi.re() < S::Real::zero() {
        vi - nu_s
    } else if vi.im() == S::Real::zero() {
        S::from_real(-(sigma / (vi.re() + nu)))
    } else {
        // ((vi - v̄i)·ν - σ)/(v̄i + ν) with numerator and denominator scaled
        // by 1/ν: a complex division of an O(s²) by an O(s) value forms O(s³)
        // products and leaves the range long before σ does.
        let inv_nu = S::from_real(nu.recip());
        ((vi - vi.conj()) - S::from_real(sigma) * inv_nu) / (vi.conj() * inv_nu + S::one())
    };
    // The pivot is O(σ/ν) when x is nearly aligned with e₀; complex
    // division forms |piv|², which then underflows although each quotient
    // is representable (Julia's `/` scales, num-complex's does not).
    let n2 = piv.re() * piv.re() + piv.im() * piv.im();
    let (g1, g2) = if S::COMPLEX && (n2 < S::Real::min_positive_value() || !n2.is_finite()) {
        pow2(-biggest(&piv).log2().round())
    } else {
        (S::Real::one(), S::Real::one())
    };
    let piv_g = piv.mul_real(g1).mul_real(g2);
    for r in rest.iter_mut() {
        *r = (sc(*r) / piv_g).mul_real(g1).mul_real(g2);
    }
    *first = S::one();
    Some(-piv.conj() / nu_s)
}

/// `2ᵏ` as two factors that are each representable (`2ᵏ` alone is not for
/// `|k|` beyond the exponent range, e.g. to lift an f32 subnormal).
fn pow2<R: Float>(k: R) -> (R, R) {
    let two = R::one() + R::one();
    let k = k.to_i32().unwrap_or(0);
    (two.powi(k / 2), two.powi(k - k / 2))
}

#[cfg(test)]
mod tests {
    use num_complex::{Complex, Complex32, Complex64};

    use super::*;

    /// Applies `I - β u uᴴ` to `x` and returns the largest deviation from
    /// `‖x‖·e₀` relative to `‖x‖` (computed in f64 from scaled copies).
    fn defect<S: Scalar>(x: &[S], to: impl Fn(S) -> Complex64) -> f64 {
        let mut u = x.to_vec();
        let xs: Vec<Complex64> = x.iter().map(|&v| to(v)).collect();
        let m = xs
            .iter()
            .map(|z| z.re.abs().max(z.im.abs()))
            .fold(0.0, f64::max);
        let xs: Vec<Complex64> = xs.iter().map(|z| z / m).collect();
        let nrm = xs.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let Some(beta) = householder(&mut u) else {
            // No reflection: x must already be ‖x‖·e₀.
            return xs
                .iter()
                .skip(1)
                .map(|z| z.norm())
                .fold((xs[0] - nrm).norm(), f64::max)
                / nrm;
        };
        let (b, u): (Complex64, Vec<Complex64>) = (to(beta), u.iter().map(|&v| to(v)).collect());
        assert!(
            b.is_finite() && u.iter().all(|z| z.is_finite()),
            "{b} {u:?}"
        );
        let mu: Complex64 = u
            .iter()
            .zip(&xs)
            .map(|(ui, xi)| ui.conj() * xi)
            .sum::<Complex64>()
            * b;
        let hx: Vec<Complex64> = u.iter().zip(&xs).map(|(ui, xi)| xi - mu * ui).collect();
        // Unitarity: |1 - β‖u‖²/2|... checked through ‖Hx‖ = ‖x‖ and Hx ∝ e₀.
        let mut d = (hx[0] - nrm).norm();
        for z in &hx[1..] {
            d = d.max(z.norm());
        }
        d / nrm
    }

    fn cases(tiny: f64, huge: f64) -> Vec<Vec<Complex64>> {
        let c = Complex64::new;
        vec![
            // Traced (#20): a lone complex entry far below √(min positive).
            vec![c(-2.3e-24, -2.6e-24)],
            vec![c(-2.3e-24, -2.6e-24), c(1e-25, 3e-26)],
            // Traced: x nearly aligned with e₀, so the pivot is O(σ/ν).
            vec![
                c(1.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(9.1e-13, 0.0),
                c(4.5e-13, 0.0),
                c(-2.2e-19, 0.0),
            ],
            vec![c(3.2e-12, 1.3e-25), c(-6.4e-32, 1.2e-32)],
            // Mixed scales near underflow and overflow, and ordinary input.
            vec![c(tiny, 0.0), c(-tiny, tiny), c(3.0 * tiny, 0.0)],
            vec![c(-tiny, 2.0 * tiny), c(tiny * 1e3, 0.0)],
            vec![c(huge, -huge), c(huge / 3.0, 0.0), c(0.0, huge)],
            vec![c(0.3, -0.4), c(1.2, 0.5), c(-0.7, 0.1)],
            vec![c(2.0, 0.0), c(0.0, 0.0)],
        ]
    }

    #[test]
    fn reflector_maps_x_to_its_norm_across_the_range() {
        let r = |v: Vec<Complex64>| v.iter().map(|z| z.re).collect::<Vec<_>>();
        for x in cases(1e-300, 1e300) {
            let d = defect(&x, |v: Complex64| v);
            assert!(d < 1e-14, "c64 {x:?}: {d}");
            let xr: Vec<f64> = r(x.clone());
            if xr.iter().any(|&v| v != 0.0) {
                let d = defect(&xr, |v: f64| Complex64::new(v, 0.0));
                assert!(d < 1e-14, "f64 {xr:?}: {d}");
            }
        }
        for x in cases(1e-40, 1e36) {
            let x32: Vec<Complex32> = x
                .iter()
                .map(|z| Complex::new(z.re as f32, z.im as f32))
                .collect();
            let up = |v: Complex32| Complex64::new(v.re.into(), v.im.into());
            if x32.iter().any(|z| *z != Complex32::new(0.0, 0.0)) {
                let d = defect(&x32, up);
                assert!(d < 1e-5, "c32 {x32:?}: {d}");
            }
            let xr: Vec<f32> = x32.iter().map(|z| z.re).collect();
            if xr.iter().any(|&v| v != 0.0) {
                let d = defect(&xr, |v: f32| Complex64::new(v.into(), 0.0));
                assert!(d < 1e-5, "f32 {xr:?}: {d}");
            }
        }
    }
}
