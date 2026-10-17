//! Gram–Schmidt orthogonalization of a new Krylov vector against the basis.

use num_traits::One;

use crate::traits::{Basis, Vector};

/// Orthogonalization scheme for new Krylov vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Orth {
    /// Classical Gram–Schmidt with one full reorthogonalization: two
    /// [`Basis::project`] calls and one norm per iteration.
    #[default]
    Cgs2,
    /// Modified Gram–Schmidt with one full reorthogonalization (KrylovKit's
    /// default): `2·len` sequential inner products per iteration.
    Mgs2,
}

/// `w ← (I - V·Vᴴ)·w` by `orth`, with the projection coefficients `Vᴴ·w_old`
/// summed over both passes into `h`. `h` must be zeroed and `scratch` hold
/// `basis.len()` entries.
pub(crate) fn orthogonalize<V: Vector, B: Basis<V>>(
    basis: &B,
    w: &mut V,
    h: &mut [V::Scalar],
    s: &mut [V::Scalar],
    orth: Orth,
) -> Result<(), V::Error> {
    match orth {
        Orth::Cgs2 => {
            basis.project(w, h)?;
            basis.combine_into(h, w, -V::Scalar::one(), V::Scalar::one())?;
            basis.project(w, s)?;
            basis.combine_into(s, w, -V::Scalar::one(), V::Scalar::one())?;
            for (hi, si) in h.iter_mut().zip(s.iter()) {
                *hi += *si;
            }
        }
        Orth::Mgs2 => {
            for _ in 0..2 {
                for (i, hi) in h.iter_mut().enumerate() {
                    let c = basis.inner_at(i, w, s)?;
                    basis.axpy_at(i, -c, w, s)?;
                    *hi += c;
                }
            }
        }
    }
    Ok(())
}
