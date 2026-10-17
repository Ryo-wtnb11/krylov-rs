//! Complex helpers for the dense kernels, built from real `+ − × ÷ sqrt`
//! only.
//!
//! num-complex's `/` forms `|b|²` (overflow/underflow for large or tiny
//! divisors) and its `sqrt` goes through `atan2`/`sin`/`cos` (libm, so not
//! bitwise reproducible across platforms); the LAPACK ports use these instead.

use num_complex::Complex;

use crate::scalar::RealScalar;

/// LAPACK `CABS1`: `|re| + |im|`.
pub(crate) fn cabs1<R: RealScalar>(z: Complex<R>) -> R {
    z.re.abs() + z.im.abs()
}

/// `|z|` without undue overflow.
pub(crate) fn cabs<R: RealScalar>(z: Complex<R>) -> R {
    z.re.hypot2(z.im)
}

/// Principal square root (the Fortran `SQRT` intrinsic), with the operand
/// rescaled by even powers of two so that `|re| + |z|` neither overflows nor
/// loses the subnormal range.
pub(crate) fn csqrt<R: RealScalar>(z: Complex<R>) -> Complex<R> {
    let zero = R::zero();
    if z.re == zero && z.im == zero {
        return Complex::new(zero, z.im);
    }
    let two = R::one() + R::one();
    let big = R::max_value() / (two * two);
    let eps = R::epsilon();
    let (mut a, mut b, mut out) = (z.re, z.im, R::one());
    if a.abs() >= big || b.abs() >= big {
        let q = R::one() / (two * two);
        a *= q;
        b *= q;
        out = two;
    } else if a.abs() <= R::min_positive_value() / eps && b.abs() <= R::min_positive_value() / eps {
        // Scale by the power of two 1/eps², whose square root 1/eps is exact.
        let up = R::one() / (eps * eps);
        a *= up;
        b *= up;
        out = eps;
    }
    let t = ((a.abs() + a.hypot2(b)) / two).sqrt();
    let w = if a >= zero {
        Complex::new(t, b / (two * t))
    } else {
        Complex::new(b.abs() / (two * t), t.copysign(b))
    };
    w * out
}

/// Robust complex division `x / y` (LAPACK 3.12 `ZLADIV` → `DLADIV`,
/// Baudin & Smith).
pub(crate) fn cdiv<R: RealScalar>(x: Complex<R>, y: Complex<R>) -> Complex<R> {
    let (a, b, c, d) = (x.re, x.im, y.re, y.im);
    let one = R::one();
    let two = one + one;
    let half = one / two;
    let (mut aa, mut bb, mut cc, mut dd) = (a, b, c, d);
    let ab = a.abs().max(b.abs());
    let cd = c.abs().max(d.abs());
    let mut s = one;
    let ov = R::max_value();
    let un = R::min_positive_value();
    let eps = R::epsilon() * half; // DLAMCH('Epsilon')
    let bs = two;
    let be = bs / (eps * eps);
    if ab >= half * ov {
        aa = half * aa;
        bb = half * bb;
        s = two * s;
    }
    if cd >= half * ov {
        cc = half * cc;
        dd = half * dd;
        s = half * s;
    }
    if ab <= un * bs / eps {
        aa *= be;
        bb *= be;
        s /= be;
    }
    if cd <= un * bs / eps {
        cc *= be;
        dd *= be;
        s *= be;
    }
    let (p, q) = if d.abs() <= c.abs() {
        ladiv1(aa, bb, cc, dd)
    } else {
        let (p, q) = ladiv1(bb, aa, dd, cc);
        (p, -q)
    };
    Complex::new(p * s, q * s)
}

fn ladiv1<R: RealScalar>(a: R, b: R, c: R, d: R) -> (R, R) {
    let r = d / c;
    let t = R::one() / (c + d * r);
    let p = ladiv2(a, b, c, d, r, t);
    let q = ladiv2(b, -a, c, d, r, t);
    (p, q)
}

fn ladiv2<R: RealScalar>(a: R, b: R, c: R, d: R, r: R, t: R) -> R {
    if r != R::zero() {
        let br = b * r;
        if br != R::zero() {
            (a + br) * t
        } else {
            a * t + (b * t) * r
        }
    } else {
        (a + d * (b / c)) * t
    }
}

/// `z / |z|` for `z ≠ 0`.
pub(crate) fn phase<R: RealScalar>(z: Complex<R>) -> Complex<R> {
    z / cabs(z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    #[test]
    fn csqrt_is_the_principal_root_across_the_range() {
        for s in [1.0, 1e300, 1e-300, 1e-310] {
            for z in [
                c(4.0, 0.0),
                c(-4.0, 0.0),
                c(-4.0, -0.0),
                c(3.0, 4.0),
                c(-3.0, 4.0),
                c(-3.0, -4.0),
                c(0.0, 2.0),
            ] {
                let z = z * s;
                let r = csqrt(z);
                assert!(
                    r.re >= 0.0 && r.re.is_finite() && r.im.is_finite(),
                    "{z}: {r}"
                );
                // r² = z, measured relative to |z|.
                let rel = (r / s.sqrt() * (r / s.sqrt()) - z / s).norm() / (z / s).norm();
                assert!(rel < 1e-15, "{z}: {r} rel {rel}");
            }
        }
        // Branch cut: −4 ± 0i maps to ±2i.
        assert_eq!(csqrt(c(-4.0, 0.0)), c(0.0, 2.0));
        assert_eq!(csqrt(c(-4.0, -0.0)), c(0.0, -2.0));
        assert_eq!(csqrt(c(0.0, 0.0)), c(0.0, 0.0));
    }

    #[test]
    fn cdiv_avoids_overflow_and_underflow() {
        for (x, y) in [
            (c(1.0, 2.0), c(3.0, -4.0)),
            (c(1e300, 1e300), c(1e300, -1e300)),
            (c(1e-300, 2e-300), c(3e-300, 1e-300)),
            (c(1.0, 0.0), c(1e-308, 1e-308)),
            (c(1e307, 1e-307), c(1e-307, 1e307)),
        ] {
            let q = cdiv(x, y);
            assert!(q.re.is_finite() && q.im.is_finite(), "{x}/{y} = {q}");
            // Check q·y = x after scaling both sides to unit size.
            let sy = y.norm();
            let err = (q * (y / sy) - x / sy).norm() / (x / sy).norm();
            assert!(err < 1e-15, "{x}/{y} = {q}: {err}");
        }
    }
}
