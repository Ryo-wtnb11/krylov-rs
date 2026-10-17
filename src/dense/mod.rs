//! Small dense computations on the projected problem (host, `O(k²)`–`O(k³)`).

mod bidiag;
mod expm;
mod givens;
mod heig;
mod householder;
mod lapack;
mod phi;
mod schur;
mod tridiag;

pub(crate) use bidiag::{arrowhead_to_bidiag, bidiag_svd, lower_rect_bidiag_svd};
pub(crate) use expm::expm;
pub(crate) use givens::{Givens, givens, upper_solve};
pub(crate) use heig::arrowhead_to_band;
pub(crate) use householder::householder;
pub(crate) use lapack::Work;
pub(crate) use phi::{phi_last, phi_pair};
pub(crate) use schur::{hschur, permute_schur, schur_eigvals, schur_eigvecs};
pub(crate) use tridiag::arrowhead_to_tridiag;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::sealed::RealOps;
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
    fn stegr_matches_dense_and_is_orthonormal() {
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
                f64::stegr(&mut dd, &mut ee, &mut z, &mut Work::default()).unwrap();
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

    /// Projected entries near the f32 range limits (#20): the reflector norms
    /// written into the tridiagonal form used to be `√Σxᵢ²`, which is ∞ or 0
    /// there. The restoration commutes with scaling, so the f32 result over
    /// `scale` must match the f64 result of the unscaled input.
    #[test]
    fn arrowhead_restoration_is_finite_near_the_f32_range_limits() {
        let keep = 6;
        let run = |d: &[f64], f: &[f64], scale: f64| {
            let to = |v: &[f64]| v.iter().map(|&x| (x * scale) as f32).collect::<Vec<_>>();
            let (d32, f32_) = (to(d), to(f));
            let back = |v: &[f32]| v.iter().map(|&x| f64::from(x) / scale).collect::<Vec<_>>();
            let mut u = vec![0.0f32; keep * keep];
            let (mut a, mut b) = (vec![0.0f32; keep], vec![0.0f32; keep]);
            let mut h = vec![0.0f32; (keep + 1) * keep];
            let mut refl = vec![0.0f32; keep];
            arrowhead_to_tridiag(&d32, &f32_, &mut a, &mut b, &mut u, keep, &mut h, &mut refl);
            assert!(a.iter().chain(&b).all(|x| x.is_finite()), "scale {scale}");
            // The f64 oracle on the same (rounded) unscaled input.
            let (d64, f64_) = (back(&d32), back(&f32_));
            let mut u64_ = vec![0.0; keep * keep];
            let (mut a64, mut b64) = (vec![0.0; keep], vec![0.0; keep]);
            let mut h64 = vec![0.0; (keep + 1) * keep];
            let mut r64 = vec![0.0; keep];
            arrowhead_to_tridiag(
                &d64, &f64_, &mut a64, &mut b64, &mut u64_, keep, &mut h64, &mut r64,
            );
            let (a, b) = (back(&a), back(&b));
            for (got, want) in a.iter().zip(&a64).chain(b.iter().zip(&b64)) {
                assert!(
                    (got - want).abs() <= 1e-5 * 6.0,
                    "scale {scale}: {got} vs {want}"
                );
            }
        };
        let mut s = 7;
        let d: Vec<f64> = (0..keep).map(|i| i as f64 + lcg(&mut s)).collect();
        let f: Vec<f64> = (0..keep).map(|_| lcg(&mut s)).collect();
        for scale in [1.0, 1e19, 1e30, 1e36, 1e-21, 1e-30] {
            run(&d, &f, scale);
        }
    }

    /// Largest `|φ_j - oracle|/|oracle|` in ulps of `S` per `j = 1..=9`, over
    /// the 1024-bit oracle `tests/fixtures/phi/phi.txt` (generated by
    /// `phi.jl`; real rows only for real `S`). `φ_j` is read both as the
    /// first entry of `phi_pair(x, j)` and as the second of
    /// `phi_pair(x, j - 1)`; rows where
    /// `e^{Re x}` overflows `S` are skipped (see [`phi_pair`]).
    fn phi_ulps<S: crate::scalar::Scalar>(
        bits: u32,
        mk: impl Fn(f64, f64) -> Option<S>,
        back: impl Fn(S) -> Complex64,
    ) -> [f64; 10] {
        let eps =
            num_traits::ToPrimitive::to_f64(&<S::Real as num_traits::Float>::epsilon()).unwrap();
        let big =
            num_traits::ToPrimitive::to_f64(&<S::Real as num_traits::Float>::max_value()).unwrap();
        let mut worst = [0.0f64; 10];
        for line in include_str!("../../tests/fixtures/phi/phi.txt").lines() {
            let f: Vec<f64> = line.split(' ').map(|v| v.parse().unwrap()).collect();
            let (j, want) = (f[3] as usize, Complex64::new(f[4], f[5]));
            let Some(x) = mk(f[1], f[2]).filter(|_| f[0] as u32 == bits) else {
                continue;
            };
            if f[1].exp() > big {
                continue; // e^x overflows S: φ_j is ∞ by contract
            }
            for got in [phi_pair(x, j).0, phi_pair(x, j - 1).1] {
                let e = (back(got) - want).norm() / want.norm() / eps;
                worst[j] = worst[j].max(e);
            }
        }
        worst
    }

    #[test]
    fn phi_pair_matches_high_precision_oracle() {
        use num_complex::Complex32;
        let worst = [
            phi_ulps::<f64>(
                64,
                |r, i| (i == 0.0).then_some(r),
                |v| Complex64::new(v, 0.0),
            ),
            phi_ulps::<Complex64>(64, |r, i| Some(Complex64::new(r, i)), |v| v),
            phi_ulps::<f32>(
                32,
                |r, i| (i == 0.0).then_some(r as f32),
                |v| Complex64::new(v.into(), 0.0),
            ),
            phi_ulps::<Complex32>(
                32,
                |r, i| Some(Complex32::new(r as f32, i as f32)),
                |v| Complex64::new(v.re.into(), v.im.into()),
            ),
        ];
        for (name, w) in ["f64", "c64", "f32", "c32"].iter().zip(worst) {
            // A fixed series radius 1 reached 2·10⁵ ulps at j = 9 (#4).
            assert!(w[1..].iter().all(|&e| e <= 16.0), "{name}: {w:?}");
        }
        assert_eq!(phi_pair(0.0f64, 0), (1.0, 1.0));
        assert_eq!(phi_pair(0.0f64, 4), (1.0 / 24.0, 1.0 / 120.0));
    }

    #[test]
    fn exactly_real_phi_argument_is_evaluated_in_real_arithmetic() {
        let c = |re| Complex64::new(re, 0.0);
        for x in [-30.0, -1.5, -0.4, 0.7, 2.5, 4.5, 700.0, 800.0] {
            for p in 1..4 {
                let (a, b) = phi_pair(x, p);
                // Bitwise the real values, so an overflow stays ∞ + 0i, not ∞ + NaN·i.
                assert_eq!(phi_pair(c(x), p), (c(a), c(b)), "x = {x}, p = {p}");
                if x == 800.0 {
                    assert!(a.is_infinite() && b.is_infinite());
                }
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
