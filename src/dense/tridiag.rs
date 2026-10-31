//! Lanczos thick-restart restoration (the tridiagonal eigenproblem itself is
//! LAPACK `xSTEGR` through `RealOps::stegr`).

use super::arrowhead_to_band;
use crate::scalar::RealScalar;

/// Restores Lanczos (tridiagonal) form after a thick restart.
///
/// Input: Ritz values `d[..keep]` and residual couplings `f[..keep]`, i.e. the
/// `(keep+1) × keep` arrowhead `[diag(d); fᵀ]`. Output: `alpha[..keep]`,
/// `beta[..keep]` of the equivalent tridiagonal form (with `beta[keep-1]` the
/// new residual norm), and `u` (`ldu × ncols`, column-major) right-multiplied
/// by the reflectors on its first `keep` columns. `h` is scratch of length
/// `(keep+1)·keep`.
///
/// Port of the restart block of KrylovKit v0.10.4
/// `src/eigsolve/lanczos.jl:eigsolve`: the reflectors are those of the block
/// restoration [`arrowhead_to_band`] with block size 1.
#[allow(clippy::too_many_arguments)]
pub(crate) fn arrowhead_to_tridiag<R: RealScalar>(
    d: &[R],
    f: &[R],
    alpha: &mut [R],
    beta: &mut [R],
    u: &mut [R],
    ldu: usize,
    h: &mut [R],
    refl: &mut [R],
) {
    let keep = d.len();
    let ld = keep + 1;
    h.fill(R::zero());
    for j in 0..keep {
        h[j + j * ld] = d[j];
        h[keep + j * ld] = f[j];
    }
    arrowhead_to_band(h, keep, 1, u, ldu, refl);
    for j in 0..keep {
        alpha[j] = h[j + j * ld];
        beta[j] = h[(j + 1) + j * ld];
    }
}
