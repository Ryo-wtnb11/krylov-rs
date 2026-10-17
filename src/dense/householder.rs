//! Householder reflectors.

use num_traits::{Float, Zero};

use crate::scalar::Scalar;

/// Turns `x` into a Householder vector `u` (`u[0] = 1`) and returns `β` such
/// that `(I - β u uᴴ)·x_old = ‖x_old‖·e₀`; `None` when no reflection is needed.
///
/// Port of KrylovKit v0.10.4 `src/dense/reflector.jl:_householder!` with the
/// pivot at the first entry.
pub(crate) fn householder<S: Scalar>(x: &mut [S]) -> Option<S> {
    let (first, rest) = x.split_first_mut()?;
    let mut sigma = S::Real::zero();
    for r in rest.iter() {
        sigma += r.re() * r.re() + r.im() * r.im();
    }
    let vi = *first;
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
    for r in rest.iter_mut() {
        *r /= piv;
    }
    *first = S::one();
    Some(-piv.conj() / nu_s)
}
