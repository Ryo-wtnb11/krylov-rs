//! Small dense computations on the projected problem (host, `O(k²)`–`O(k³)`).

mod bidiag;
mod cplx;
mod givens;
mod heig;
mod householder;
mod phi;
mod schur;
mod tridiag;

pub(crate) use bidiag::{arrowhead_to_bidiag, bidiag_svd, lower_rect_bidiag_svd};
pub(crate) use givens::{Givens, givens, upper_solve};
pub(crate) use heig::{apply_qh, arrowhead_to_band, hermitian_tridiag};
pub(crate) use householder::householder;
pub(crate) use phi::{phi_last, phi_upto};
pub(crate) use schur::{complex_schur, real_schur};
pub(crate) use tridiag::{arrowhead_to_tridiag, tridiag_eigh};

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    fn tri_mat(d: &[f64], e: &[f64]) -> nalgebra::DMatrix<f64> {
        let n = d.len();
        nalgebra::DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                d[i]
            } else if i + 1 == j {
                e[i]
            } else if j + 1 == i {
                e[j]
            } else {
                0.0
            }
        })
    }

    #[test]
    fn tridiag_eigh_matches_dense_and_is_orthonormal() {
        let mut s = 1;
        for n in [1, 2, 3, 10, 40] {
            for split in [false, true] {
                let d: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
                let mut e: Vec<f64> = (0..n).map(|_| lcg(&mut s)).collect();
                if split && n > 3 {
                    e[n / 2] = 0.0; // decoupled blocks, repeated work on deflation
                    e[1] = 1e-300;
                }
                let t = tri_mat(&d, &e[..n.saturating_sub(1)]);
                let (mut dd, mut ee, mut z) = (d.clone(), e.clone(), vec![0.0; n * n]);
                tridiag_eigh(&mut dd, &mut ee, &mut z).unwrap();
                let mut exact: Vec<f64> = t
                    .clone()
                    .symmetric_eigen()
                    .eigenvalues
                    .iter()
                    .copied()
                    .collect();
                exact.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for (a, b) in dd.iter().zip(&exact) {
                    assert!((a - b).abs() < 1e-14, "n={n}: {a} vs {b}");
                }
                let zm = nalgebra::DMatrix::from_column_slice(n, n, &z);
                let ortho = (zm.transpose() * &zm - nalgebra::DMatrix::identity(n, n)).norm();
                assert!(ortho < 1e-13);
                let resid =
                    (&t * &zm - &zm * nalgebra::DMatrix::from_diagonal(&dd.clone().into())).norm();
                assert!(resid < 1e-13);
            }
        }
    }

    #[test]
    fn arrowhead_restoration_is_an_orthogonal_similarity() {
        let mut s = 7;
        let keep = 6;
        let ldu = 9;
        let d: Vec<f64> = (0..keep).map(|i| i as f64 + lcg(&mut s)).collect();
        let f: Vec<f64> = (0..keep).map(|_| lcg(&mut s)).collect();
        let mut u = vec![0.0; ldu * ldu];
        for i in 0..ldu {
            u[i + i * ldu] = 1.0;
        }
        let (mut a, mut b) = (vec![0.0; keep], vec![0.0; keep]);
        let mut h = vec![0.0; (keep + 1) * keep];
        let mut refl = vec![0.0; keep];
        arrowhead_to_tridiag(&d, &f, &mut a, &mut b, &mut u, ldu, &mut h, &mut refl);
        // Y = U[..keep, ..keep] maps the arrowhead to the tridiagonal form:
        // Yᵀ diag(d) Y = T and fᵀ Y = β_keep e_keepᵀ.
        let y = nalgebra::DMatrix::from_fn(keep, keep, |i, j| u[i + j * ldu]);
        let t = tri_mat(&a, &b[..keep - 1]);
        let dm = nalgebra::DMatrix::from_diagonal(&nalgebra::DVector::from_column_slice(&d));
        assert!((y.transpose() * dm * &y - t).norm() < 1e-13);
        let fy = nalgebra::RowDVector::from_row_slice(&f) * &y;
        for j in 0..keep - 1 {
            assert!(fy[j].abs() < 1e-14);
        }
        assert!((fy[keep - 1] - b[keep - 1]).abs() < 1e-14);
        let fnorm = f.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((b[keep - 1] - fnorm).abs() < 1e-14);
    }

    #[test]
    fn phi_functions_are_continuous_across_branches() {
        let c = |re, im| Complex64::new(re, im);
        let (mut below, mut above) = ([c(0.0, 0.0); 6], [c(0.0, 0.0); 6]);
        for x in [
            c(1.0 - 1e-12, 0.0),
            c(-1.0 + 1e-12, 0.0),
            c(0.0, 1.0 - 1e-12),
        ] {
            phi_upto(x, &mut below);
            phi_upto(x * (1.0 + 2e-12), &mut above);
            for (j, (b, a)) in below.iter().zip(&above).enumerate() {
                // The recurrence amplifies rounding by about e·j! just above |x| = 1.
                assert!((b - a).norm() < 1e-11 * [1.0, 1.0, 1.0, 2.0, 5.0, 20.0][j]);
            }
        }
        let mut phis = [c(0.0, 0.0); 6];
        phi_upto(c(0.0, 0.0), &mut phis);
        assert_eq!(
            phis,
            [1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0].map(|v| c(v, 0.0))
        );
        // Closed forms φ_j(x) = (eˣ - Σ_{i<j} xⁱ/i!)/xʲ on both branches.
        for x in [c(3.0, -2.0), c(0.6, 0.6)] {
            phi_upto(x, &mut phis);
            let (mut taylor, mut term) = (c(0.0, 0.0), c(1.0, 0.0));
            for (j, &got) in phis.iter().enumerate() {
                let want = (x.exp() - taylor) / x.powu(j as u32);
                assert!((got - want).norm() <= 1e-12 * want.norm(), "x={x} j={j}");
                taylor += term;
                term = term * x / (j as f64 + 1.0);
            }
        }
    }

    /// `e_kᵀ φ_j(xT) e₁`: the `(k, k+j)` entry of
    /// `exp([[xT, e₁, 0], [0, 0, I_{j-1}], [0, 0, 0]])`.
    fn phi_last_dense(x: Complex64, t: &nalgebra::DMatrix<f64>, j: usize) -> Complex64 {
        let k = t.nrows();
        let mut m = nalgebra::DMatrix::<Complex64>::zeros(k + j, k + j);
        for r in 0..k {
            for s in 0..k {
                m[(r, s)] = x * t[(r, s)];
            }
        }
        m[(0, k)] = Complex64::new(1.0, 0.0);
        for i in 1..j {
            m[(k + i - 1, k + i)] = Complex64::new(1.0, 0.0);
        }
        m.exp()[(k - 1, k + j - 1)]
    }

    #[test]
    fn phi_last_entry_series_matches_dense_and_is_relatively_accurate() {
        let mut s = 3;
        let k = 12;
        let a: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
        let b: Vec<f64> = (0..k).map(|_| 0.5 + lcg(&mut s).abs()).collect();
        let t = tri_mat(&a, &b[..k - 1]);
        let (mut s1, mut s2) = (
            vec![Complex64::new(0.0, 0.0); k],
            vec![Complex64::new(0.0, 0.0); k],
        );
        for j in 1..=4 {
            // Tiny x: the leading term x^(k-1)·Πβ/(k-1+j)! dominates; a dense
            // exponential only resolves this entry to ε·‖exp‖.
            let x = Complex64::new(-0.01, 0.0);
            let got = phi_last(x, j, &a, &b, &mut s1, &mut s2).unwrap();
            let fact: f64 = (1..k + j).map(|i| i as f64).product();
            let lead = x.powu(k as u32 - 1) * b[..k - 1].iter().product::<f64>() / fact;
            assert!(
                (got - lead).norm() <= 1e-2 * lead.norm(),
                "j={j}: {got} vs {lead}"
            );
            for x in [Complex64::new(0.0, -0.7), Complex64::new(1.5, 0.5)] {
                let got = phi_last(x, j, &a, &b, &mut s1, &mut s2).unwrap();
                let want = phi_last_dense(x, &t, j);
                // The dense exponential resolves this entry only to ε·‖exp‖.
                assert!(
                    (got - want).norm() <= 1e-10 * want.norm() + 1e-15,
                    "x={x} j={j}: {got} vs {want}"
                );
            }
            // Out of the series' range it declines.
            assert!(phi_last(Complex64::new(10.0, 0.0), j, &a, &b, &mut s1, &mut s2).is_none());
        }
    }
}
