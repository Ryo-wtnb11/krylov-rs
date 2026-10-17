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
    } else {
        ((vi - vi.conj()) * nu_s - S::from_real(sigma)) / (vi.conj() + nu_s)
    };
    for r in rest.iter_mut() {
        *r /= piv;
    }
    *first = S::one();
    Some(-piv.conj() / nu_s)
}
