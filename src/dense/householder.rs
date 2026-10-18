//! Householder reflectors.

use num_traits::{Float, One, Zero};

use crate::scalar::Scalar;

/// Turns `x` into a Householder vector `u` (`u[0] = 1`) and returns
/// `(‖x_old‖, β)` such that `(I - β u uᴴ)·x_old = ‖x_old‖·e₀`; `β` is `None`
/// (and `x` unchanged) when no reflection is needed, i.e. `x` already is
/// `‖x‖·e₀`, up to components below `√(min positive)·‖x‖`. Callers write this
/// `‖x_old‖` into their reduced matrix rather than re-forming `√Σxᵢ²`, which
/// would underflow or overflow where the reflector does not.
///
/// Port of KrylovKit v0.10.4 `src/dense/reflector.jl:_householder!` with the
/// pivot at the first entry, plus the range safeguard of LAPACK `xLARFG`:
/// `u` and `β` do not change when `x` is scaled, so `x` is always evaluated
/// as `2ᵏ·x` with its largest component near 1. The scaling is exact, so
/// input that needed none gives bitwise the same result, and it keeps the
/// squares of every component that matters (down to `√(min positive)` of
/// the largest) out of the subnormal range. An unscaled `x` whose largest
/// component is merely small (`< 1e-16` in f32) already loses those digits.
/// LAPACK's reflector itself is not used: it maps `x` to `-sign(Re x₀)·‖x‖·e₀`
/// and for complex `x` leaves a complex `β`, while every caller here relies on
/// `+‖x‖·e₀`.
pub(crate) fn householder<S: Scalar>(x: &mut [S]) -> (S::Real, Option<S>) {
    let Some((first, rest)) = x.split_first_mut() else {
        return (S::Real::zero(), None);
    };
    let biggest = |v: &S| v.re().abs().max(v.im().abs());
    let s = rest.iter().map(biggest).fold(biggest(first), S::Real::max);
    let (f1, f2) = if s > S::Real::zero() && s.is_finite() {
        pow2(-s.log2().round())
    } else {
        (S::Real::one(), S::Real::one())
    };
    let sc = |v: S| v.mul_real(f1).mul_real(f2);
    let nrm = {
        let all = core::iter::once(&*first).chain(rest.iter());
        let mut t = S::Real::zero();
        for v in all {
            let v = sc(*v);
            t += v.re() * v.re() + v.im() * v.im();
        }
        t.sqrt() / f1 / f2
    };
    let mut sigma = S::Real::zero();
    for r in rest.iter() {
        let r = sc(*r);
        sigma += r.re() * r.re() + r.im() * r.im();
    }
    let vi = sc(*first);
    let nu = (vi.re() * vi.re() + vi.im() * vi.im() + sigma).sqrt();
    if sigma == S::Real::zero() && vi == S::from_real(nu) {
        return (nrm, None);
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
    if biggest(&piv) < S::Real::min_positive_value() {
        // x is ν·e₀ up to components below √(min positive)·ν, far under
        // rounding: the identity is the reflector. The exact one would need
        // u ~ ν/‖x₁‖ and β ~ ‖x₁‖²/ν², which leave the range (and the
        // pivot has already lost its digits).
        return (nrm, None);
    }
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
    (nrm, Some(-piv.conj() / nu_s))
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

    /// `(max |H·x/‖x‖ - e₀|, max |HᴴH - I|)` for the reflector of `x`, both
    /// evaluated in f64 from the returned `u`, `β` (the identity for `None`).
    fn defects<S: Scalar>(x: &[S], to: impl Fn(S) -> Complex64) -> (f64, f64) {
        let n = x.len();
        let mut u = x.to_vec();
        let xs: Vec<Complex64> = x.iter().map(|&v| to(v)).collect();
        let m = xs.iter().map(|z| z.norm()).fold(0.0, f64::max);
        let xs: Vec<Complex64> = xs.iter().map(|z| z / m).collect();
        let nrm = xs.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let (norm, beta) = householder(&mut u);
        let norm_err = (to(S::from_real(norm)).re / m - nrm).abs() / nrm;
        let (b, u): (Complex64, Vec<Complex64>) = match beta {
            None => (
                Complex64::new(0.0, 0.0),
                xs.iter().map(|_| Complex64::new(0.0, 0.0)).collect(),
            ),
            Some(beta) => (to(beta), u.iter().map(|&v| to(v)).collect()),
        };
        assert!(
            b.is_finite() && u.iter().all(|z| z.is_finite()),
            "{b} {u:?}"
        );
        // H = I - β u uᴴ, column-major.
        let h = |i: usize, j: usize| {
            Complex64::new(if i == j { 1.0 } else { 0.0 }, 0.0) - b * u[i] * u[j].conj()
        };
        let mut unit = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                let g: Complex64 = (0..n).map(|k| h(k, i).conj() * h(k, j)).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                unit = unit.max((g - want).norm());
            }
        }
        let mut map = norm_err;
        for i in 0..n {
            let hx: Complex64 = (0..n).map(|k| h(i, k) * xs[k]).sum();
            let want = if i == 0 { nrm } else { 0.0 };
            map = map.max((hx - want).norm() / nrm);
        }
        (map, unit)
    }

    fn cases(tiny: f64, huge: f64, band: &[(f64, f64)]) -> Vec<Vec<Complex64>> {
        let c = Complex64::new;
        let mut v = vec![
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
            // Traced (svdsolve restart): σ is subnormal in f32, so the
            // pivot -σ/(x₀+ν) underflows to 0.
            vec![c(1.0, 0.0), c(3e-23, 0.0), c(-2e-23, 0.0)],
            // Mixed scales near underflow and overflow, and ordinary input.
            vec![c(tiny, 0.0), c(-tiny, tiny), c(3.0 * tiny, 0.0)],
            vec![c(-tiny, 2.0 * tiny), c(tiny * 1e3, 0.0)],
            vec![c(huge, -huge), c(huge / 3.0, 0.0), c(0.0, huge)],
            vec![c(0.3, -0.4), c(1.2, 0.5), c(-0.7, 0.1)],
            vec![c(-0.3, 0.2), c(1e-9, 0.0), c(0.0, -2e-9)],
            vec![c(2.0, 0.0), c(0.0, 0.0)],
        ];
        // Largest component just above √(min positive), so the smaller
        // ones' squares are subnormal unless x is scaled (#20).
        for &(a, b) in band {
            v.push(vec![c(a, 0.0), c(b, 0.0)]);
            v.push(vec![c(a, 0.0), c(b, -b), c(-b, 0.0)]);
            v.push(vec![c(-a, 0.5 * a), c(b, 0.0)]);
        }
        v
    }

    fn check<S: Scalar>(name: &str, x: &[S], to: impl Fn(S) -> Complex64, eps: f64) {
        if x.iter().all(|v| *v == S::zero()) {
            return;
        }
        let (map, unit) = defects(x, to);
        assert!(
            map <= 16.0 * eps && unit <= 16.0 * eps,
            "{name} {x:?}: map {map:e}, unitarity {unit:e}"
        );
    }

    #[test]
    fn reflector_is_unitary_and_maps_x_to_its_norm_across_the_range() {
        let band64 = [(1e-150, 1e-160), (2e-154, 1e-163), (1e-148, 1e-156)];
        for x in cases(1e-300, 1e300, &band64) {
            check("c64", &x, |v: Complex64| v, f64::EPSILON);
            let xr: Vec<f64> = x.iter().map(|z| z.re).collect();
            check("f64", &xr, |v: f64| Complex64::new(v, 0.0), f64::EPSILON);
        }
        let band32 = [(2e-19, 1e-22), (1e-18, 1e-23), (1e-17, 3e-21)];
        let eps = f64::from(f32::EPSILON);
        for x in cases(1e-40, 1e36, &band32) {
            let x32: Vec<Complex32> = x
                .iter()
                .map(|z| Complex::new(z.re as f32, z.im as f32))
                .collect();
            check(
                "c32",
                &x32,
                |v: Complex32| Complex64::new(v.re.into(), v.im.into()),
                eps,
            );
            let xr: Vec<f32> = x32.iter().map(|z| z.re).collect();
            check("f32", &xr, |v: f32| Complex64::new(v.into(), 0.0), eps);
        }
    }
}
