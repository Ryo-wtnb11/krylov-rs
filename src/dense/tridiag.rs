//! Symmetric tridiagonal eigensolver and thick-restart restoration.

use super::householder;
use crate::scalar::RealScalar;

/// Eigendecomposition of the real symmetric tridiagonal matrix with diagonal
/// `d` and off-diagonal `e[..n-1]` by implicit QL with Wilkinson shifts.
///
/// On return `d` holds the eigenvalues in ascending order and `z` (`n × n`,
/// column-major, overwritten) the orthonormal eigenvectors as columns. `e` is
/// destroyed; it must have length `n`.
pub(crate) fn tridiag_eigh<R: RealScalar>(d: &mut [R], e: &mut [R], z: &mut [R]) -> Result<(), ()> {
    let n = d.len();
    debug_assert!(e.len() == n && z.len() == n * n);
    z.fill(R::zero());
    for i in 0..n {
        z[i + i * n] = R::one();
    }
    if n == 0 {
        return Ok(());
    }
    e[n - 1] = R::zero();
    let two = R::one() + R::one();
    let eps = R::epsilon();
    let mut budget = 30 * n;
    for l in 0..n {
        loop {
            let mut m = l;
            while m + 1 < n {
                let dd = d[m].abs() + d[m + 1].abs();
                if e[m].abs() <= eps * dd {
                    break;
                }
                m += 1;
            }
            if m == l {
                break;
            }
            if budget == 0 {
                return Err(());
            }
            budget -= 1;
            let mut g = (d[l + 1] - d[l]) / (two * e[l]);
            let mut r = g.hypot2(R::one());
            g = d[m] - d[l] + e[l] / (g + r.copysign(g));
            let (mut s, mut c, mut p) = (R::one(), R::one(), R::zero());
            let mut deflated = false;
            for i in (l..m).rev() {
                let f = s * e[i];
                let b = c * e[i];
                r = f.hypot2(g);
                e[i + 1] = r;
                if r == R::zero() {
                    d[i + 1] -= p;
                    e[m] = R::zero();
                    deflated = true;
                    break;
                }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + two * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                let (zi, zi1) = z.split_at_mut((i + 1) * n);
                for (a, b) in zi[i * n..].iter_mut().zip(zi1[..n].iter_mut()) {
                    let f = *b;
                    *b = s * *a + c * f;
                    *a = c * *a - s * f;
                }
            }
            if deflated {
                continue;
            }
            d[l] -= p;
            e[l] = g;
            e[m] = R::zero();
        }
    }
    // Selection sort keeps the column swaps explicit and is O(n²) like the
    // eigenvector update it follows.
    for i in 0..n {
        let k = (i..n).fold(i, |k, j| if d[j] < d[k] { j } else { k });
        if k != i {
            d.swap(i, k);
            for r in 0..n {
                z.swap(r + i * n, r + k * n);
            }
        }
    }
    Ok(())
}

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
        let nu = {
            let mut s = R::zero();
            for x in v.iter() {
                s += *x * *x;
            }
            s.sqrt()
        };
        let beta_h = householder(v);
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
