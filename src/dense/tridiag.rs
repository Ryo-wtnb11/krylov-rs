//! Lanczos thick-restart restoration (the tridiagonal eigenproblem itself is
//! LAPACK `xSTEGR` through `RealOps::stegr`).

use super::householder;
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
/// `src/eigsolve/lanczos.jl:eigsolve`.
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
    for j in (0..keep).rev() {
        // Reflector from row j+1 (0-based: row j+1), columns 0..=j, pivot at j.
        // Stored reversed so the pivot is first, matching `householder`.
        let v = &mut refl[..=j];
        for (c, vc) in v.iter_mut().enumerate() {
            *vc = h[(j + 1) + (j - c) * ld];
        }
        let (nu, beta_h) = householder(v);
        h[(j + 1) + j * ld] = nu;
        for c in 0..j {
            h[(j + 1) + c * ld] = R::zero();
        }
        let Some(bh) = beta_h else {
            continue;
        };
        // index map: reflector entry t ↔ matrix index j - t.
        // lmul!(h, H): rows 0..=j of every column.
        for col in 0..keep {
            let mut mu = R::zero();
            for (t, vt) in v.iter().enumerate() {
                mu += *vt * h[(j - t) + col * ld];
            }
            mu *= bh;
            for (t, vt) in v.iter().enumerate() {
                h[(j - t) + col * ld] -= mu * *vt;
            }
        }
        // rmul!(H[0..=j, :], hᴴ): columns 0..=j of rows 0..=j.
        for row in 0..=j {
            let mut mu = R::zero();
            for (t, vt) in v.iter().enumerate() {
                mu += h[row + (j - t) * ld] * *vt;
            }
            mu *= bh;
            for (t, vt) in v.iter().enumerate() {
                h[row + (j - t) * ld] -= mu * *vt;
            }
        }
        // rmul!(U, hᴴ): columns 0..=j of every row.
        for row in 0..ldu {
            let mut mu = R::zero();
            for (t, vt) in v.iter().enumerate() {
                mu += u[row + (j - t) * ldu] * *vt;
            }
            mu *= bh;
            for (t, vt) in v.iter().enumerate() {
                u[row + (j - t) * ldu] -= mu * *vt;
            }
        }
    }
    for j in 0..keep {
        alpha[j] = h[j + j * ld];
        beta[j] = h[(j + 1) + j * ld];
    }
}
