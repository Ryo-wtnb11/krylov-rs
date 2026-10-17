//! Small dense Hermitian eigenproblems and block thick-restart restoration.

use num_traits::{Float, Zero};

use super::householder;
use crate::scalar::Scalar;

/// Reduces the `n × n` Hermitian `a` (column-major, full storage) to real
/// symmetric tridiagonal form `T = Q·A·Qᴴ` by Householder similarities.
///
/// On return `d[..n]` and `e[..n-1]` hold the diagonal and the (non-negative)
/// off-diagonal of `T`; the reflectors stay below the subdiagonal of `a`
/// with their scales in `taus[..n-1]`, for [`apply_qh`]. `x` is scratch of
/// length `n`. With [`tridiag_eigh`](super::tridiag_eigh) this replaces
/// LAPACK's `eigen(Hermitian)` used by KrylovKit.
pub(crate) fn hermitian_tridiag<S: Scalar>(
    a: &mut [S],
    n: usize,
    d: &mut [S::Real],
    e: &mut [S::Real],
    taus: &mut [S],
    x: &mut [S],
) {
    for j in 0..n.saturating_sub(1) {
        let m = n - j - 1;
        let v = &mut x[..m];
        v.copy_from_slice(&a[j * n + j + 1..(j + 1) * n]);
        let mut nu = S::Real::zero();
        for vi in v.iter() {
            nu += vi.re() * vi.re() + vi.im() * vi.im();
        }
        e[j] = nu.sqrt();
        let beta = householder(v);
        taus[j] = beta.unwrap_or_else(S::zero);
        if let Some(b) = beta {
            // A ← H·A·Hᴴ on the trailing block, H = I - β·v·vᴴ on rows j+1..n.
            for c in j + 1..n {
                let col = &mut a[c * n + j + 1..(c + 1) * n];
                let mut mu = S::zero();
                for (vi, ai) in v.iter().zip(col.iter()) {
                    mu += vi.conj() * *ai;
                }
                mu *= b;
                for (vi, ai) in v.iter().zip(col.iter_mut()) {
                    *ai -= mu * *vi;
                }
            }
            for r in j + 1..n {
                let mut w = S::zero();
                for (t, vi) in v.iter().enumerate() {
                    w += a[(j + 1 + t) * n + r] * *vi;
                }
                w *= b.conj();
                for (t, vi) in v.iter().enumerate() {
                    a[(j + 1 + t) * n + r] -= w * vi.conj();
                }
            }
        }
        a[j * n + j + 2..(j + 1) * n].copy_from_slice(&v[1..]);
    }
    for j in 0..n {
        d[j] = a[j * n + j].re();
    }
}

/// `y ← Qᴴ·y` for the `Q` of [`hermitian_tridiag`] (`y` is `n × ncols`,
/// column-major), turning eigenvectors of `T` into eigenvectors of `A`.
pub(crate) fn apply_qh<S: Scalar>(a: &[S], n: usize, taus: &[S], y: &mut [S]) {
    for j in (0..n.saturating_sub(1)).rev() {
        let b = taus[j].conj();
        if b == S::zero() {
            continue;
        }
        let tail = &a[j * n + j + 2..(j + 1) * n]; // v[1..]; v[0] = 1
        for col in y.chunks_exact_mut(n) {
            let col = &mut col[j + 1..];
            let mut mu = col[0];
            for (vi, ci) in tail.iter().zip(&col[1..]) {
                mu += vi.conj() * *ci;
            }
            mu *= b;
            col[0] -= mu;
            for (vi, ci) in tail.iter().zip(col[1..].iter_mut()) {
                *ci -= mu * *vi;
            }
        }
    }
}

/// Restores block-band form after a block thick restart.
///
/// `h` (`(keep+bs) × keep`, column-major) holds `[diag(D); U_last]`, with
/// `U_last` the last `bs` rows of the Ritz vectors kept. Reflectors from the
/// right zero row `j+bs` left of column `j` (`j = keep-1, …, 0`), so the
/// residual couples only to the trailing columns, and they are applied as a
/// similarity to `h[..keep, ..keep]` and from the right to `u` (`ldu` rows,
/// at least `keep` columns). `refl` is scratch of length `keep`.
///
/// Port of the restart block of KrylovKit v0.10.4
/// `src/eigsolve/blocklanczos.jl:eigsolve` (`householder(H, j+bs, 1:j, j)`).
pub(crate) fn arrowhead_to_band<S: Scalar>(
    h: &mut [S],
    keep: usize,
    bs: usize,
    u: &mut [S],
    ldu: usize,
    refl: &mut [S],
) {
    let ld = keep + bs;
    for j in (0..keep).rev() {
        let row = j + bs;
        // Reflector on columns 0..=j from conj(h[row, ..=j]), stored reversed
        // so that the pivot (column j) is first.
        let v = &mut refl[..=j];
        let mut nu = S::Real::zero();
        for (t, vt) in v.iter_mut().enumerate() {
            *vt = h[row + (j - t) * ld].conj();
            nu += vt.re() * vt.re() + vt.im() * vt.im();
        }
        let nu = nu.sqrt();
        let beta = householder(v);
        h[row + j * ld] = S::from_real(nu);
        for c in 0..j {
            h[row + c * ld] = S::zero();
        }
        let Some(b) = beta else {
            continue;
        };
        // lmul!(h, H): rows 0..=j of every column.
        for col in 0..keep {
            let mut mu = S::zero();
            for (t, vt) in v.iter().enumerate() {
                mu += vt.conj() * h[(j - t) + col * ld];
            }
            mu *= b;
            for (t, vt) in v.iter().enumerate() {
                h[(j - t) + col * ld] -= mu * *vt;
            }
        }
        // rmul!(H[..row, :], hᴴ) and rmul!(U, hᴴ): columns 0..=j.
        let bc = b.conj();
        let right = |m: &mut [S], ldm: usize, rows: usize| {
            for r in 0..rows {
                let mut w = S::zero();
                for (t, vt) in v.iter().enumerate() {
                    w += m[r + (j - t) * ldm] * *vt;
                }
                w *= bc;
                for (t, vt) in v.iter().enumerate() {
                    m[r + (j - t) * ldm] -= w * vt.conj();
                }
            }
        };
        right(h, ld, row);
        right(u, ldu, ldu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dense::tridiag_eigh;
    use nalgebra::DMatrix;
    use num_complex::Complex64 as C;

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    fn unitary(s: &mut u64, n: usize) -> DMatrix<C> {
        DMatrix::from_fn(n, n, |_, _| C::new(lcg(s), lcg(s)))
            .qr()
            .q()
    }

    #[test]
    fn hermitian_eigendecomposition_matches_dense() {
        let mut s = 5;
        for n in [1, 2, 3, 8, 25] {
            // Generic and degenerate (each eigenvalue three times) spectra.
            for degenerate in [false, true] {
                let eigs: Vec<f64> = (0..n)
                    .map(|i| {
                        if degenerate {
                            (i / 3) as f64
                        } else {
                            lcg(&mut s)
                        }
                    })
                    .collect();
                let q = unitary(&mut s, n);
                let dm = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                    n,
                    eigs.iter().map(|&e| C::new(e, 0.0)),
                ));
                let am = &q * dm * q.adjoint();
                let am = (&am + am.adjoint()) * C::new(0.5, 0.0);
                let mut a: Vec<C> = am.iter().copied().collect();
                let (mut d, mut e, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n * n]);
                let (mut taus, mut x) = (vec![C::new(0.0, 0.0); n], vec![C::new(0.0, 0.0); n]);
                hermitian_tridiag(&mut a, n, &mut d, &mut e, &mut taus, &mut x);
                assert!(e[..n - 1].iter().all(|&v| v >= 0.0));
                tridiag_eigh(&mut d, &mut e, &mut z).unwrap();
                let mut u: Vec<C> = z.iter().map(|&v| C::new(v, 0.0)).collect();
                apply_qh(&a, n, &taus, &mut u);
                let um = DMatrix::from_column_slice(n, n, &u);
                let mut exact = eigs.clone();
                exact.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for (a, b) in d.iter().zip(&exact) {
                    assert!((a - b).abs() < 1e-13, "n={n}: {a} vs {b}");
                }
                let ortho = (um.adjoint() * &um - DMatrix::identity(n, n)).norm();
                assert!(ortho < 1e-13, "n={n}: {ortho}");
                let dd = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
                    n,
                    d.iter().map(|&v| C::new(v, 0.0)),
                ));
                assert!((&am * &um - &um * dd).norm() < 1e-12);
            }
        }
    }

    /// Eigenvalues of `scale·Q·diag(eigs)·Qᴴ` in the scalar type `S`,
    /// relative to `scale`, against the known spectrum.
    fn scaled_spectrum_error<S: Scalar>(
        to: fn(C) -> S,
        back: fn(S::Real) -> f64,
        scale: f64,
    ) -> f64 {
        let mut s = 21;
        let n = 15;
        let eigs: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 - 0.5).collect();
        let q = unitary(&mut s, n);
        let dm = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            n,
            eigs.iter().map(|&e| C::new(e, 0.0)),
        ));
        let am = &q * dm * q.adjoint();
        let mut a: Vec<S> = DMatrix::from_fn(n, n, |i, j| (am[(i, j)] + am[(j, i)].conj()) * 0.5)
            .iter()
            .map(|&z| to(z * scale))
            .collect();
        let zr = <S::Real as num_traits::Zero>::zero();
        let (mut d, mut e, mut z) = (vec![zr; n], vec![zr; n], vec![zr; n * n]);
        let (mut taus, mut x) = (vec![S::zero(); n], vec![S::zero(); n]);
        hermitian_tridiag(&mut a, n, &mut d, &mut e, &mut taus, &mut x);
        tridiag_eigh(&mut d, &mut e, &mut z).unwrap();
        d.iter()
            .zip(&eigs)
            .map(|(&got, &want)| (back(got) / scale - want).abs())
            .fold(0.0, f64::max)
    }

    #[test]
    fn complex_reduction_survives_extreme_scales() {
        // Squares of the entries stay in range; their cubes do not.
        for scale in [1e145, 1e-150] {
            let err = scaled_spectrum_error::<C>(|z| z, |r| r, scale);
            assert!(err < 1e-13, "Complex64 at {scale:e}: {err}");
        }
        for scale in [1e13, 1e-13] {
            let err = scaled_spectrum_error::<num_complex::Complex32>(
                |z| num_complex::Complex32::new(z.re as f32, z.im as f32),
                |r| r as f64,
                scale,
            );
            assert!(err < 1e-5, "Complex32 at {scale:e}: {err}");
        }
    }

    #[test]
    fn band_restoration_is_a_unitary_similarity() {
        let mut s = 9;
        let (keep, bs, ldu) = (6, 2, 9);
        let ld = keep + bs;
        let d: Vec<f64> = (0..keep).map(|i| i as f64 + lcg(&mut s)).collect();
        let f = DMatrix::from_fn(bs, keep, |_, _| C::new(lcg(&mut s), lcg(&mut s)));
        let mut h = vec![C::new(0.0, 0.0); ld * keep];
        for j in 0..keep {
            h[j + j * ld] = C::new(d[j], 0.0);
            for i in 0..bs {
                h[keep + i + j * ld] = f[(i, j)];
            }
        }
        let mut u: Vec<C> = DMatrix::<C>::identity(ldu, ldu).iter().copied().collect();
        let mut refl = vec![C::new(0.0, 0.0); keep];
        arrowhead_to_band(&mut h, keep, bs, &mut u, ldu, &mut refl);
        let y = DMatrix::from_fn(keep, keep, |i, j| u[i + j * ldu]);
        let hm = DMatrix::from_fn(ld, keep, |i, j| h[i + j * ld]);
        let dm = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            keep,
            d.iter().map(|&v| C::new(v, 0.0)),
        ));
        assert!((y.adjoint() * &y - DMatrix::identity(keep, keep)).norm() < 1e-13);
        assert!((y.adjoint() * dm * &y - hm.rows(0, keep)).norm() < 1e-13);
        let fy = &f * &y;
        assert!((&fy - hm.rows(keep, bs)).norm() < 1e-13);
        // Row i of the coupling vanishes left of column keep-bs+i, and the
        // band matrix has bandwidth bs.
        for i in 0..bs {
            for c in 0..keep - bs + i {
                assert_eq!(hm[(keep + i, c)], C::new(0.0, 0.0));
            }
            assert!(hm[(keep + i, keep - bs + i)].im == 0.0);
        }
        for j in 0..keep {
            for i in 0..keep {
                if i.abs_diff(j) > bs {
                    assert!(hm[(i, j)].norm() < 1e-13, "({i},{j})");
                }
            }
        }
    }
}
