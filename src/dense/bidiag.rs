//! Bidiagonal SVD (LAPACK `xBDSQR`) and thick-restart restoration of GKL.

use super::{Work, householder};
use crate::scalar::RealScalar;

/// Rotation `(c, s)` with `c·y + s·z = hypot(y, z)` and `-s·y + c·z = 0`.
fn rot<R: RealScalar>(y: R, z: R) -> (R, R) {
    let r = y.hypot2(z);
    if r == R::zero() {
        (R::one(), R::zero())
    } else {
        (y / r, z / r)
    }
}

/// Columns `j, k` of the `n × n` column-major `m`:
/// `(m_j, m_k) ← (c·m_j + s·m_k, -s·m_j + c·m_k)`.
fn rot_cols<R: RealScalar>(m: &mut [R], n: usize, j: usize, k: usize, c: R, s: R) {
    for i in 0..n {
        let (a, b) = (m[i + j * n], m[i + k * n]);
        m[i + j * n] = c * a + s * b;
        m[i + k * n] = c * b - s * a;
    }
}

/// SVD `B = U·diag(d)·Vᵀ` of the real `n × n` lower (`lower`) or upper
/// bidiagonal matrix with diagonal `d` and off-diagonal `e[..n-1]`, by LAPACK
/// `xBDSQR` as KrylovKit v0.10.4 `src/dense/linalg.jl:bidiagsvd!` calls it.
///
/// `d` gets the singular values in descending order and `u`, `v` (`n × n`,
/// overwritten) the singular vectors as columns; `e` is destroyed.
pub(crate) fn bidiag_svd<R: RealScalar>(
    lower: bool,
    d: &mut [R],
    e: &mut [R],
    u: &mut [R],
    v: &mut [R],
    w: &mut Work<R, R>,
) -> Result<(), ()> {
    let n = d.len();
    for m in [&mut *u, &mut *v] {
        m[..n * n].fill(R::zero());
        for i in 0..n {
            m[i + i * n] = R::one();
        }
    }
    // v holds Vᵀ on return from LAPACK; transpose it in place.
    R::bdsqr(lower, d, e, v, u, w)?;
    for j in 0..n {
        for i in 0..j {
            v.swap(i + j * n, j + i * n);
        }
    }
    Ok(())
}

/// SVD `B̃ = P·diag(s)·Yᵀ` of the `(k+1) × k` lower bidiagonal matrix with
/// diagonal `alpha[..k]` and subdiagonal `beta[..k]` (`P` is `(k+1) × k`,
/// `Y` is `k × k`, both column-major and overwritten).
///
/// Givens rotations from the left reduce `B̃` to `Q·[R; 0]` with `R` upper
/// bidiagonal, whose SVD [`bidiag_svd`] computes; `P = Q[:, ..k]·U_R`.
/// `scratch` needs `(k+1)² + k² + 2k` entries.
pub(crate) fn lower_rect_bidiag_svd<R: RealScalar>(
    alpha: &[R],
    beta: &[R],
    p: &mut [R],
    y: &mut [R],
    s: &mut [R],
    scratch: &mut [R],
    w: &mut Work<R, R>,
) -> Result<(), ()> {
    let k = alpha.len();
    let k1 = k + 1;
    let (q, rest) = scratch.split_at_mut(k1 * k1);
    let (ur, rest) = rest.split_at_mut(k * k);
    let (re, _) = rest.split_at_mut(k.max(1));
    q.fill(R::zero());
    for i in 0..k1 {
        q[i + i * k1] = R::one();
    }
    // Row i holds only its diagonal entry `a`; row i+1 holds β_i and α_{i+1}.
    let mut a = alpha.first().copied().unwrap_or(R::zero());
    for i in 0..k {
        let (c, sn) = rot(a, beta[i]);
        s[i] = a.hypot2(beta[i]);
        if i + 1 < k {
            re[i] = sn * alpha[i + 1];
            a = c * alpha[i + 1];
        }
        rot_cols(q, k1, i, i + 1, c, sn);
    }
    bidiag_svd(false, &mut s[..k], re, ur, y, w)?;
    for j in 0..k {
        for row in 0..k1 {
            let mut acc = R::zero();
            for t in 0..k {
                acc += q[row + t * k1] * ur[t + j * k];
            }
            p[row + j * k1] = acc;
        }
    }
    Ok(())
}

/// Restores GKL (lower bidiagonal) form after a thick restart.
///
/// Input: singular values `sv[..keep]` and residual couplings `f[..keep]`,
/// i.e. the `(keep+1) × keep` matrix `[diag(sv); fᵀ]`. Output:
/// `alpha[..keep]`, `beta[..keep]` (diagonal and subdiagonal, `beta[keep-1]`
/// the new residual norm); `p` (left, `ld × ·`) and `y` (right, `ld × ·`)
/// right-multiplied by the reflectors on their first `keep` columns. `h` is
/// scratch of length `(keep+1)·keep`, `refl` of length `keep`.
///
/// Port of the restart block of KrylovKit v0.10.4
/// `src/eigsolve/svdsolve.jl:svdsolve` (two-sided Householder).
#[allow(clippy::too_many_arguments)]
pub(crate) fn arrowhead_to_bidiag<R: RealScalar>(
    sv: &[R],
    f: &[R],
    alpha: &mut [R],
    beta: &mut [R],
    p: &mut [R],
    y: &mut [R],
    ld: usize,
    h: &mut [R],
    refl: &mut [R],
) {
    let keep = sv.len();
    let lh = keep + 1;
    h.fill(R::zero());
    for j in 0..keep {
        h[j + j * lh] = sv[j];
        h[keep + j * lh] = f[j];
    }
    // `m ← m·(I - β u uᵀ)` on the columns `j - t` (t = 0..=j) of `rows` rows.
    let right = |m: &mut [R], ldm: usize, rows: usize, j: usize, u: &[R], bh: R| {
        for row in 0..rows {
            let mut mu = R::zero();
            for (t, ut) in u.iter().enumerate() {
                mu += m[row + (j - t) * ldm] * *ut;
            }
            mu *= bh;
            for (t, ut) in u.iter().enumerate() {
                m[row + (j - t) * ldm] -= mu * *ut;
            }
        }
    };
    for j in (0..keep).rev() {
        // Row j+1, columns 0..=j, pivot at column j: V side.
        let u = &mut refl[..=j];
        let mut nu = R::zero();
        for (t, ut) in u.iter_mut().enumerate() {
            *ut = h[(j + 1) + (j - t) * lh];
            nu += *ut * *ut;
        }
        let nu = nu.sqrt();
        let bh = householder(u);
        h[(j + 1) + j * lh] = nu;
        for c in 0..j {
            h[(j + 1) + c * lh] = R::zero();
        }
        if let Some(bh) = bh {
            right(h, lh, j + 1, j, u, bh); // rmul!(H[1:j, :], h')
            right(y, ld, ld, j, u, bh); // rmul!(V, h')
        }
        // Column j, rows 0..=j, pivot at row j: U side.
        let u = &mut refl[..=j];
        let mut nu = R::zero();
        for (t, ut) in u.iter_mut().enumerate() {
            *ut = h[(j - t) + j * lh];
            nu += *ut * *ut;
        }
        let nu = nu.sqrt();
        let bh = householder(u);
        h[j + j * lh] = nu;
        for r in 0..j {
            h[r + j * lh] = R::zero();
        }
        if let Some(bh) = bh {
            // lmul!(h, H[:, 1:j-1]) on rows j - t.
            for col in 0..j {
                let mut mu = R::zero();
                for (t, ut) in u.iter().enumerate() {
                    mu += *ut * h[(j - t) + col * lh];
                }
                mu *= bh;
                for (t, ut) in u.iter().enumerate() {
                    h[(j - t) + col * lh] -= mu * *ut;
                }
            }
            right(p, ld, ld, j, u, bh); // rmul!(U, h')
        }
    }
    for j in 0..keep {
        alpha[j] = h[j + j * lh];
        beta[j] = h[(j + 1) + j * lh];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    fn upper(d: &[f64], e: &[f64]) -> DMatrix<f64> {
        let n = d.len();
        DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                d[i]
            } else if i + 1 == j {
                e[i]
            } else {
                0.0
            }
        })
    }

    /// Checks `B = U·diag(s)·Vᵀ`, orthogonality, ordering and agreement of
    /// `s` with nalgebra.
    fn check(d: &[f64], e: &[f64]) {
        let n = d.len();
        let bm = upper(d, e);
        let (mut s, mut ee) = (d.to_vec(), e.to_vec());
        let (mut u, mut v) = (vec![0.0; n * n], vec![0.0; n * n]);
        bidiag_svd(false, &mut s, &mut ee, &mut u, &mut v, &mut Work::default()).unwrap();
        let um = DMatrix::from_column_slice(n, n, &u);
        let vm = DMatrix::from_column_slice(n, n, &v);
        let sm = DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(&s));
        let bnorm = bm.norm().max(f64::MIN_POSITIVE);
        assert!((&um * &sm * vm.transpose() - &bm).norm() <= 1e-14 * bnorm * n as f64);
        let id = DMatrix::<f64>::identity(n, n);
        assert!((um.transpose() * &um - &id).norm() <= 1e-14 * n as f64);
        assert!((vm.transpose() * &vm - &id).norm() <= 1e-14 * n as f64);
        assert!(s.windows(2).all(|w| w[0] >= w[1]) && s.iter().all(|&x| x >= 0.0));
        let mut exact: Vec<f64> = bm.singular_values().iter().copied().collect();
        exact.sort_by(|a, b| b.partial_cmp(a).unwrap());
        for (a, x) in s.iter().zip(&exact) {
            assert!((a - x).abs() <= 1e-14 * bnorm * n as f64, "{a} vs {x}");
        }
    }

    #[test]
    fn bidiag_svd_matches_dense() {
        let mut g = 7u64;
        for n in [1, 2, 3, 7, 25, 60] {
            let d: Vec<f64> = (0..n).map(|_| lcg(&mut g)).collect();
            let e: Vec<f64> = (0..n.max(1) - 1).map(|_| lcg(&mut g)).collect();
            check(&d, &e);
        }
    }

    #[test]
    fn bidiag_svd_degenerate_inputs() {
        // Zero diagonal entries (interior and last), zero superdiagonal,
        // graded and huge/tiny scales, and an exactly rank-deficient matrix.
        check(&[1.0, 0.0, 2.0, 3.0], &[0.5, 0.7, -0.2]);
        check(&[1.0, 2.0, 3.0, 0.0], &[0.5, 0.7, -0.2]);
        check(&[0.0, 0.0, 0.0], &[1.0, 1.0]);
        check(&[1.0, 2.0, 3.0], &[0.0, 0.0]);
        check(&[1e300, 1e-300, 1e150], &[1e299, 1e-10]);
        let graded: Vec<f64> = (0..12).map(|i| 10f64.powi(-2 * i)).collect();
        check(&graded, &graded[1..]);
        check(&[0.0; 5], &[0.0; 4]);
        check(&[2.0; 6], &[1e-170; 5]);
        let (mut s, mut e) = (vec![f64::NAN, 1.0], vec![1.0]);
        let (mut u, mut v) = (vec![0.0; 4], vec![0.0; 4]);
        assert!(bidiag_svd(false, &mut s, &mut e, &mut u, &mut v, &mut Work::default()).is_err());
    }

    #[test]
    fn lower_rectangular_svd_matches_dense() {
        let mut g = 3u64;
        for k in [1, 2, 5, 17] {
            let alpha: Vec<f64> = (0..k).map(|_| lcg(&mut g)).collect();
            let mut beta: Vec<f64> = (0..k).map(|_| lcg(&mut g)).collect();
            if k == 5 {
                beta[2] = 0.0;
            }
            let bm = DMatrix::from_fn(k + 1, k, |i, j| {
                if i == j {
                    alpha[j]
                } else if i == j + 1 {
                    beta[j]
                } else {
                    0.0
                }
            });
            let (mut p, mut y, mut s) = (vec![0.0; (k + 1) * k], vec![0.0; k * k], vec![0.0; k]);
            let mut scratch = vec![0.0; (k + 1) * (k + 1) + k * k + 2 * k];
            let mut w = Work::default();
            lower_rect_bidiag_svd(&alpha, &beta, &mut p, &mut y, &mut s, &mut scratch, &mut w)
                .unwrap();
            let pm = DMatrix::from_column_slice(k + 1, k, &p);
            let ym = DMatrix::from_column_slice(k, k, &y);
            let sm = DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(&s));
            assert!((&pm * sm * ym.transpose() - &bm).norm() < 1e-14 * k as f64);
            assert!(
                (pm.transpose() * &pm - DMatrix::<f64>::identity(k, k)).norm() < 1e-14 * k as f64
            );
            let mut exact: Vec<f64> = bm.singular_values().iter().copied().collect();
            exact.sort_by(|a, b| b.partial_cmp(a).unwrap());
            for (a, x) in s.iter().zip(&exact) {
                assert!((a - x).abs() < 1e-14 * k as f64);
            }
        }
    }

    #[test]
    fn bidiag_restoration_is_an_orthogonal_equivalence() {
        // [diag(sv); fᵀ] = Pᵀ-side and Y-side reflections of the result.
        let mut g = 11u64;
        let keep = 6;
        let sv: Vec<f64> = (0..keep).map(|i| 3.0 - 0.4 * i as f64).collect();
        let f: Vec<f64> = (0..keep).map(|_| lcg(&mut g)).collect();
        let ld = keep + 2;
        let id = |n| DMatrix::<f64>::identity(n, n);
        let (mut p, mut y) = (
            id(ld).iter().copied().collect::<Vec<_>>(),
            id(ld).iter().copied().collect::<Vec<_>>(),
        );
        let (mut alpha, mut beta) = (vec![0.0; keep], vec![0.0; keep]);
        let (mut h, mut refl) = (vec![0.0; (keep + 1) * keep], vec![0.0; keep]);
        arrowhead_to_bidiag(
            &sv, &f, &mut alpha, &mut beta, &mut p, &mut y, ld, &mut h, &mut refl,
        );
        // H = [diag(sv); fᵀ] (rows 0..=keep) and Bnew the restored lower
        // bidiagonal (keep+1)×keep; then H = Pᵀ... : P_k ᵀ H Y_k = Bnew with
        // P acting on rows 0..keep (row keep untouched) and Y on columns.
        let hm = DMatrix::from_fn(keep + 1, keep, |i, j| {
            if i == j {
                sv[j]
            } else if i == keep {
                f[j]
            } else {
                0.0
            }
        });
        let bnew = DMatrix::from_fn(keep + 1, keep, |i, j| {
            if i == j {
                alpha[j]
            } else if i == j + 1 {
                beta[j]
            } else {
                0.0
            }
        });
        let pm = DMatrix::from_column_slice(ld, ld, &p);
        let ym = DMatrix::from_column_slice(ld, ld, &y);
        let mut pk = DMatrix::<f64>::identity(keep + 1, keep + 1);
        pk.view_mut((0, 0), (keep, keep))
            .copy_from(&pm.view((0, 0), (keep, keep)));
        let yk = ym.view((0, 0), (keep, keep)).into_owned();
        assert!((pk.transpose() * &hm * &yk - &bnew).norm() < 1e-13);
        assert!((yk.transpose() * &yk - id(keep)).norm() < 1e-13);
        assert!(alpha.iter().chain(&beta).all(|&x| x >= 0.0));
        // The reflectors only touch the first `keep` columns.
        assert!((pm.column(keep + 1) - id(ld).column(keep + 1)).norm() == 0.0);
    }
}
