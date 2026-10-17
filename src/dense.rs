//! Small dense computations on the projected problem (host, `O(k²)`–`O(k³)`).

use num_traits::{Float, One, Zero};

use crate::scalar::sealed::RealOps as _;
use crate::scalar::{RealScalar, Scalar};

/// Turns `x` into a Householder vector `u` (`u[0] = 1`) and returns `β` such
/// that `(I - β u uᴴ)·x_old = ‖x_old‖·e₀`; `None` when no reflection is needed.
///
/// Port of KrylovKit v0.10.4 `src/dense/reflector.jl:_householder!` with the
/// pivot at the first entry.
pub(crate) fn householder<S: Scalar>(x: &mut [S]) -> Option<S> {
    let (first, rest) = x.split_first_mut()?;
    let mut sigma = S::Real::zero();
    for r in rest.iter() {
        sigma += r.re() * r.re() + r.im() * r.im();
    }
    let vi = *first;
    let nu = (vi.re() * vi.re() + vi.im() * vi.im() + sigma).sqrt();
    if sigma == S::Real::zero() && vi == S::from_real(nu) {
        return None;
    }
    let nu_s = S::from_real(nu);
    let piv = if vi.re() < S::Real::zero() {
        vi - nu_s
    } else {
        ((vi - vi.conj()) * nu_s - S::from_real(sigma)) / (vi.conj() + nu_s)
    };
    for r in rest.iter_mut() {
        *r /= piv;
    }
    *first = S::one();
    Some(-piv.conj() / nu_s)
}

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

/// `φ₁(x)` and `φ₂(x)` with `φ₀ = exp`, `φ_{j+1}(x) = (φ_j(x) - 1/j!)/x`.
///
/// Uses the Taylor series for `|x| < 1` (no cancellation) and the recurrence
/// otherwise.
pub(crate) fn phi12<S: Scalar>(x: S) -> (S, S) {
    let one = S::one();
    if x.modulus() < S::Real::one() {
        // φ_j(x) = Σ_n xⁿ/(n+j)!; for |x| < 1 the tail after 22 terms is below 1/23!.
        let (mut p1, mut p2) = (S::zero(), S::zero());
        let mut t1 = one; // xⁿ/(n+1)!
        let mut t2 = S::from_real(S::Real::from_f64(0.5)); // xⁿ/(n+2)!
        for n in 0..22usize {
            p1 += t1;
            p2 += t2;
            t1 = t1 * x / S::from_real(S::Real::from_usize(n + 2));
            t2 = t2 * x / S::from_real(S::Real::from_usize(n + 3));
        }
        (p1, p2)
    } else {
        let p1 = (x.scalar_exp() - one) / x;
        let p2 = (p1 - one) / x;
        (p1, p2)
    }
}

/// `e_kᵀ φ₂(x·T) e₁` for the symmetric tridiagonal `T` (diagonal `alpha`,
/// off-diagonal `beta[..k-1]`) by its Taylor series `Σ_n xⁿ (Tⁿe₁)_k/(n+2)!`,
/// valid while `ρ = |x|·‖T‖_∞ < k/2` (`None` otherwise).
///
/// `(Tⁿe₁)_k` vanishes exactly for `n < k-1`, so the series keeps relative
/// accuracy for this tiny entry, which an eigendecomposition only resolves
/// to `ε·max|φ₂|`. `t` and `next` are scratch of length `k`.
pub(crate) fn phi2_last<S: Scalar>(
    x: S,
    alpha: &[S::Real],
    beta: &[S::Real],
    t: &mut [S],
    next: &mut [S],
) -> Option<S> {
    let k = alpha.len();
    let mut tnorm = S::Real::zero();
    for i in 0..k {
        let mut r = alpha[i].abs();
        if i > 0 {
            r += beta[i - 1].abs();
        }
        if i + 1 < k {
            r += beta[i].abs();
        }
        tnorm = tnorm.max(r);
    }
    let rho = x.modulus() * tnorm;
    let half_k = S::Real::from_usize(k) / (S::Real::one() + S::Real::one());
    if rho.is_nan() || rho >= half_k {
        return None;
    }
    let (t, next) = (&mut t[..k], &mut next[..k]);
    t.fill(S::zero());
    t[0] = S::from_real(S::Real::from_f64(0.5)); // e₁/2!
    let mut bound = S::Real::from_f64(0.5); // ρⁿ/(n+2)! ≥ ‖tₙ‖_∞
    let mut sum = S::zero();
    let eps = S::Real::epsilon();
    for n in 0..k + 200 {
        sum += t[k - 1];
        if n + 1 >= k && bound <= eps * sum.modulus() {
            break;
        }
        let f = x / S::from_real(S::Real::from_usize(n + 3));
        for i in 0..k {
            let mut v = t[i] * S::from_real(alpha[i]);
            if i > 0 {
                v += t[i - 1] * S::from_real(beta[i - 1]);
            }
            if i + 1 < k {
                v += t[i + 1] * S::from_real(beta[i]);
            }
            next[i] = v * f;
        }
        t.copy_from_slice(next);
        bound = bound * rho / S::Real::from_usize(n + 3);
    }
    Some(sum)
}

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
