//! Small dense computations on the projected problem (host, `O(k²)`–`O(k³)`).

mod householder;
mod phi;
mod tridiag;

pub(crate) use householder::householder;
pub(crate) use phi::{phi2_last, phi12};
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
        for x in [
            Complex64::new(1.0 - 1e-12, 0.0),
            Complex64::new(-1.0 + 1e-12, 0.0),
            Complex64::new(0.0, 1.0 - 1e-12),
        ] {
            let below = phi12(x);
            let above = phi12(x * (1.0 + 2e-12));
            assert!((below.0 - above.0).norm() < 1e-11);
            assert!((below.1 - above.1).norm() < 1e-11);
        }
        let (p1, p2) = phi12(Complex64::new(0.0, 0.0));
        assert_eq!(
            (p1, p2),
            (Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0))
        );
        let x = Complex64::new(3.0, -2.0);
        let (p1, p2) = phi12(x);
        assert!((p1 - (x.exp() - 1.0) / x).norm() < 1e-14);
        assert!((p2 - (x.exp() - 1.0 - x) / (x * x)).norm() < 1e-14);
    }

    #[test]
    fn phi2_last_entry_series_matches_dense_and_is_relatively_accurate() {
        let mut s = 3;
        let k = 12;
        let a: Vec<f64> = (0..k).map(|_| lcg(&mut s)).collect();
        let b: Vec<f64> = (0..k).map(|_| 0.5 + lcg(&mut s).abs()).collect();
        let t = tri_mat(&a, &b[..k - 1]);
        let (mut s1, mut s2) = (
            vec![Complex64::new(0.0, 0.0); k],
            vec![Complex64::new(0.0, 0.0); k],
        );
        // Tiny x: the leading term x^(k-1)·Πβ/(k+1)! dominates; a dense
        // exponential only resolves this entry to ε·‖exp‖.
        let x = Complex64::new(-0.01, 0.0);
        let got = phi2_last(x, &a, &b, &mut s1, &mut s2).unwrap();
        let fact: f64 = (1..=k + 1).map(|i| i as f64).product();
        let lead = x.powu(k as u32 - 1) * b[..k - 1].iter().product::<f64>() / fact;
        assert!((got - lead).norm() <= 1e-2 * lead.norm(), "{got} vs {lead}");
        for x in [Complex64::new(0.0, -0.7), Complex64::new(1.5, 0.5)] {
            let got = phi2_last(x, &a, &b, &mut s1, &mut s2).unwrap();
            // Reference: the (k, k+2) entry of exp([[xT, e1, 0], [0, 0, 1], [0, 0, 0]]).
            let mut m = nalgebra::DMatrix::<Complex64>::zeros(k + 2, k + 2);
            for i in 0..k {
                for j in 0..k {
                    m[(i, j)] = x * t[(i, j)];
                }
            }
            m[(0, k)] = Complex64::new(1.0, 0.0);
            m[(k, k + 1)] = Complex64::new(1.0, 0.0);
            let want = m.exp()[(k - 1, k + 1)];
            assert!(
                (got - want).norm() <= 1e-10 * want.norm(),
                "x={x}: {got} vs {want}"
            );
        }
        // Out of the series' range it declines.
        assert!(phi2_last(Complex64::new(10.0, 0.0), &a, &b, &mut s1, &mut s2).is_none());
    }
}
