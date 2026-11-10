//! Exact power-of-two scaling shared by the solvers.

use num_traits::Float;

use crate::scalar::sealed::RealOps as _;
use crate::scalar::{RealScalar, Scalar};

/// `⌊log₂ x⌋` for finite `x > 0`, subnormals included.
pub(crate) fn ilogb<R: RealScalar>(x: R) -> i32 {
    let (m, e, _) = x.integer_decode();
    i32::from(e) + 63 - m.leading_zeros() as i32
}

/// `x·2ᵉ`, in factors of at most `2^±60` (representable even in `f32`), so
/// the result is exact whenever it is a normal number.
pub(crate) fn ldexp<S: Scalar>(mut x: S, mut e: i32) -> S {
    const STEP: i32 = 60;
    let two = S::Real::from_f64(2.0);
    while e > STEP {
        x = x.mul_real(two.powi(STEP));
        e -= STEP;
    }
    while e < -STEP {
        x = x.mul_real(two.powi(-STEP));
        e += STEP;
    }
    x.mul_real(two.powi(e))
}
