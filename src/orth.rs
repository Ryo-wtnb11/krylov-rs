//! Gram–Schmidt orthogonalization of a new Krylov vector against the basis.

use num_traits::One;

use crate::traits::{Basis, Vector};

/// Orthogonalization scheme for new Krylov vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Orth {
    /// Classical Gram–Schmidt with one full reorthogonalization: two
    /// [`Basis::project`] calls and one norm per iteration (in
    /// [`Lanczos`](crate::Lanczos), the three-term recurrence replaces the
    /// first one).
    #[default]
    Cgs2,
    /// Modified Gram–Schmidt with one full reorthogonalization (KrylovKit's
    /// default): `2·len` sequential [`Basis::inner_at`] and
    /// [`Basis::axpy_at`] calls per iteration, one vector operation each.
    Mgs2,
}

/// `w ← (I - V·Vᴴ)·w` by `orth`, with the projection coefficients `Vᴴ·w_old`
/// summed over both passes into `h`. `h` must be zeroed and `s` (the
/// second CGS2 pass's coefficients) hold `basis.len()` entries.
pub(crate) fn orthogonalize<V: Vector, B: Basis<V>>(
    cx: &mut V::Context,
    basis: &B,
    w: &mut V,
    h: &mut [V::Scalar],
    s: &mut [V::Scalar],
    orth: Orth,
) -> Result<(), V::Error> {
    match orth {
        Orth::Cgs2 => {
            basis.project(cx, w, h)?;
            basis.combine_into(cx, h, w, -V::Scalar::one(), V::Scalar::one())?;
            basis.project(cx, w, s)?;
            basis.combine_into(cx, s, w, -V::Scalar::one(), V::Scalar::one())?;
            for (hi, si) in h.iter_mut().zip(s.iter()) {
                *hi += *si;
            }
        }
        Orth::Mgs2 => {
            for _ in 0..2 {
                for (i, hi) in h.iter_mut().enumerate() {
                    let c = basis.inner_at(cx, i, w)?;
                    basis.axpy_at(cx, i, -c, w)?;
                    *hi += c;
                }
            }
        }
    }
    Ok(())
}
