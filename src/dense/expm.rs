//! Dense matrix exponential: port of Julia 1.11 `LinearAlgebra.exp!`.
//!
//! KrylovKit's `expintegrator` evaluates `exp` of its augmented matrix with
//! Julia's `exp!` (`stdlib/v1.11/LinearAlgebra/src/dense.jl:exp!`): LAPACK
//! `xGEBAL('B')` balancing, a Padé approximant of degree 3, 5, 7, 9 or 13
//! chosen from `‖A‖₁` (Higham 2005), scaling and squaring for degree 13, an
//! `xGESV` solve, and undoing the balancing. The step controller reads one
//! tiny entry of the result, so this module follows that sequence instead of
//! another (equally valid) `exp` algorithm. Balancing and the solve are the
//! LAPACK routines Julia calls (see `dense/lapack.rs`); the driver around
//! them is ported here.

use num_traits::{Float, One, Zero};

use crate::scalar::{RealScalar, Scalar};

/// `exp(A)` of the `n × n` column-major `a`, or `Err` for non-finite input
/// or an exactly singular Padé denominator (Julia: `SingularException`).
///
/// Julia's `exp!` diverts Hermitian input to an eigendecomposition; that
/// branch is not ported because its consumer, the augmented matrix of the
/// exponential integrator, is never Hermitian.
#[allow(dead_code)] // ponytail: first consumer is the Arnoldi expintegrator (S3-3)
pub(crate) fn expm<S: Scalar>(a: &[S], n: usize) -> Result<Vec<S>, ()> {
    expm_impl(a, n).map(|(x, _, _)| x)
}

/// [`expm`] that also returns the Padé degree and the squaring count.
fn expm_impl<S: Scalar>(a: &[S], n: usize) -> Result<(Vec<S>, usize, usize), ()> {
    assert_eq!(a.len(), n * n, "expm: shape mismatch");
    if !a.iter().all(|x| x.all_finite()) {
        return Err(());
    }
    if n == 0 {
        return Ok((Vec::new(), 0, 0));
    }
    let mut a = a.to_vec();
    let bal = Balance::new(&mut a, n)?;
    let na = opnorm1(&a, n);
    // Julia compares `nA::T` with Float64 literals, i.e. in f64.
    let na64 = num_traits::ToPrimitive::to_f64(&na).unwrap_or(f64::NAN);
    let (mut x, degree, si) = if na64 <= 2.1 {
        let c: &[f64] = if na64 > 0.95 {
            &[
                17643225600.,
                8821612800.,
                2075673600.,
                302702400.,
                30270240.,
                2162160.,
                110880.,
                3960.,
                90.,
                1.,
            ]
        } else if na64 > 0.25 {
            &[
                17297280., 8648640., 1995840., 277200., 25200., 1512., 56., 1.,
            ]
        } else if na64 > 0.015 {
            &[30240., 15120., 3360., 420., 30., 1.]
        } else {
            &[120., 60., 12., 1.]
        };
        let c: Vec<S> = c.iter().map(|&v| real::<S>(v)).collect();
        let a2 = matmul(&a, &a, n);
        // U = C[2]·I + C[4]·A², V = C[1]·I + C[3]·A², then higher even
        // powers; U ← A·U; X = (V - U) \ (V + U).
        let mut p = a2.clone();
        let mut u: Vec<S> = p.iter().map(|&v| c[3] * v).collect();
        let mut v: Vec<S> = p.iter().map(|&v| c[2] * v).collect();
        add_identity(&mut u, n, c[1]);
        add_identity(&mut v, n, c[0]);
        for k in 2..c.len() / 2 {
            p = matmul(&p, &a2, n);
            for ((ui, vi), &pi) in u.iter_mut().zip(v.iter_mut()).zip(&p) {
                *ui += c[2 * k + 1] * pi;
                *vi += c[2 * k] * pi;
            }
        }
        let u = matmul(&a, &u, n);
        (pade_solve(&u, &v, n)?, c.len() - 1, 0)
    } else {
        // si = ceil(log2(nA/5.4)) when positive, by exact halving.
        let mut ratio = na64 / 5.4;
        let mut si = 0usize;
        while ratio > 1.0 {
            ratio *= 0.5;
            si += 1;
        }
        // Scale by 2^-si as `si` exact halvings (Julia's `2^si` is an Int
        // and overflows for si >= 63, i.e. ‖A‖₁ > 2.5e19).
        let mut factor = S::Real::one();
        for _ in 0..si {
            factor *= real_r::<S::Real>(0.5);
        }
        if si > 0 {
            let f = S::from_real(factor);
            for x in &mut a {
                *x *= f;
            }
        }
        let cc: Vec<S> = [
            64764752532480000.,
            32382376266240000.,
            7771770303897600.,
            1187353796428800.,
            129060195264000.,
            10559470521600.,
            670442572800.,
            33522128640.,
            1323241920.,
            40840800.,
            960960.,
            16380.,
            182.,
            1.,
        ]
        .iter()
        .map(|&v| real::<S>(v))
        .collect();
        let a2 = matmul(&a, &a, n);
        let a4 = matmul(&a2, &a2, n);
        let a6 = matmul(&a2, &a4, n);
        let comb = |c6: S, c4: S, c2: S| -> Vec<S> {
            a6.iter()
                .zip(&a4)
                .zip(&a2)
                .map(|((&x6, &x4), &x2)| c6 * x6 + c4 * x4 + c2 * x2)
                .collect()
        };
        // U = A·(A6·(c14 A6 + c12 A4 + c10 A2) + c8 A6 + c6 A4 + c4 A2 + c2 I)
        let t1 = comb(cc[13], cc[11], cc[9]);
        let mut t2 = comb(cc[7], cc[5], cc[3]);
        add_identity(&mut t2, n, cc[1]);
        let mut u0 = matmul(&a6, &t1, n);
        for (x, y) in u0.iter_mut().zip(&t2) {
            *x += *y;
        }
        let u = matmul(&a, &u0, n);
        // V = A6·(c13 A6 + c11 A4 + c9 A2) + c7 A6 + c5 A4 + c3 A2 + c1 I
        let t1 = comb(cc[12], cc[10], cc[8]);
        let mut t2 = comb(cc[6], cc[4], cc[2]);
        add_identity(&mut t2, n, cc[0]);
        let mut v = matmul(&a6, &t1, n);
        for (x, y) in v.iter_mut().zip(&t2) {
            *x += *y;
        }
        let mut x = pade_solve(&u, &v, n)?;
        for _ in 0..si {
            x = matmul(&x, &x, n);
        }
        (x, 13, si)
    };
    bal.undo(&mut x, n);
    Ok((x, degree, si))
}

fn real<S: Scalar>(v: f64) -> S {
    S::from_real(real_r(v))
}

fn real_r<R: RealScalar>(v: f64) -> R {
    R::from_f64(v)
}

fn add_identity<S: Scalar>(m: &mut [S], n: usize, c: S) {
    for i in 0..n {
        m[i + i * n] += c;
    }
}

/// Column-major `a·b`.
fn matmul<S: Scalar>(a: &[S], b: &[S], n: usize) -> Vec<S> {
    let mut c = vec![S::zero(); n * n];
    for j in 0..n {
        for k in 0..n {
            let bkj = b[k + j * n];
            if bkj == S::zero() {
                continue;
            }
            for i in 0..n {
                c[i + j * n] += a[i + k * n] * bkj;
            }
        }
    }
    c
}

/// `‖A‖₁`, the largest column sum of moduli.
fn opnorm1<S: Scalar>(a: &[S], n: usize) -> S::Real {
    let mut m = S::Real::zero();
    for j in 0..n {
        let mut s = S::Real::zero();
        for x in &a[j * n..(j + 1) * n] {
            s += x.modulus();
        }
        m = m.max(s);
    }
    m
}

/// `(V - U) \ (V + U)`.
fn pade_solve<S: Scalar>(u: &[S], v: &[S], n: usize) -> Result<Vec<S>, ()> {
    let mut den: Vec<S> = v.iter().zip(u).map(|(&v, &u)| v - u).collect();
    let mut num: Vec<S> = v.iter().zip(u).map(|(&v, &u)| v + u).collect();
    let mut ipiv = vec![0i32; n];
    S::gesv(&mut den, &mut num, n, n, &mut ipiv)?;
    Ok(num)
}

/// Balancing `D⁻¹·Pᵀ·A·P·D` from `xGEBAL('B')`, kept to undo it.
struct Balance<R> {
    ilo: usize,
    ihi: usize,
    scale: Vec<R>,
    perm: Vec<usize>,
}

impl<R: RealScalar> Balance<R> {
    /// LAPACK `xGEBAL('B')` on `a`; the permutation indices LAPACK stores in
    /// `scale` outside `ilo..=ihi` go to `perm` (0-based).
    fn new<S: Scalar<Real = R>>(a: &mut [S], n: usize) -> Result<Self, ()> {
        let mut scale = vec![R::one(); n];
        let (ilo, ihi) = S::gebal(a, n, &mut scale)?;
        let mut perm: Vec<usize> = (0..n).collect();
        for (j, p) in perm.iter_mut().enumerate() {
            if j < ilo || j > ihi {
                let idx = num_traits::ToPrimitive::to_usize(&scale[j]).ok_or(())?;
                if idx == 0 || idx > n {
                    return Err(());
                }
                *p = idx - 1;
            }
        }
        Ok(Self {
            ilo,
            ihi,
            scale,
            perm,
        })
    }

    /// Julia `exp!`: `X ← D·X·D⁻¹`, then the lower permutations in reverse
    /// and the upper ones in forward order.
    fn undo<S: Scalar<Real = R>>(&self, x: &mut [S], n: usize) {
        for j in self.ilo..=self.ihi {
            let sc = S::from_real(self.scale[j]);
            let inv = S::from_real(R::one() / self.scale[j]);
            for i in 0..n {
                x[j + i * n] *= sc;
            }
            for i in 0..n {
                x[i + j * n] *= inv;
            }
        }
        let rcswap = |x: &mut [S], i: usize, j: usize| {
            for k in 0..n {
                x.swap(k + i * n, k + j * n);
            }
            for k in 0..n {
                x.swap(i + k * n, j + k * n);
            }
        };
        for j in (0..self.ilo).rev() {
            rcswap(x, j, self.perm[j]);
        }
        for j in self.ihi + 1..n {
            rcswap(x, j, self.perm[j]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::sealed::Ops as _;
    use nalgebra::DMatrix;
    use num_complex::{Complex32, Complex64};

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    fn random(g: &mut u64, n: usize, complex: bool) -> DMatrix<Complex64> {
        DMatrix::from_fn(n, n, |_, _| {
            Complex64::new(lcg(g), if complex { lcg(g) } else { 0.0 })
        })
    }

    fn ours(a: &DMatrix<Complex64>, complex: bool) -> (DMatrix<Complex64>, usize, usize) {
        let n = a.nrows();
        if complex {
            let (x, d, s) = expm_impl(a.as_slice(), n).unwrap();
            (DMatrix::from_column_slice(n, n, &x), d, s)
        } else {
            let ar: Vec<f64> = a.iter().map(|z| z.re).collect();
            let (x, d, s) = expm_impl(&ar, n).unwrap();
            let x: Vec<Complex64> = x.iter().map(|&v| Complex64::new(v, 0.0)).collect();
            (DMatrix::from_column_slice(n, n, &x), d, s)
        }
    }

    fn rel(x: &DMatrix<Complex64>, y: &DMatrix<Complex64>) -> f64 {
        (x - y).norm() / y.norm()
    }

    /// `exp` by a long Taylor series in high working precision (f64 with
    /// scaling and squaring by an independent power), as a second oracle
    /// for `nalgebra::exp`.
    fn taylor(a: &DMatrix<Complex64>) -> DMatrix<Complex64> {
        let n = a.nrows();
        let norm = a.norm();
        let mut s = 0;
        while norm / 2f64.powi(s) > 0.1 {
            s += 1;
        }
        let b = a / Complex64::new(2f64.powi(s), 0.0);
        let mut term = DMatrix::<Complex64>::identity(n, n);
        let mut sum = term.clone();
        for k in 1..30 {
            term = &term * &b / Complex64::new(k as f64, 0.0);
            sum += &term;
        }
        for _ in 0..s {
            sum = &sum * &sum;
        }
        sum
    }

    #[test]
    fn every_degree_and_both_sides_of_each_threshold() {
        // A matrix with nonnegative entries needs no balancing scale beyond
        // powers of two and has ‖A‖₁ easy to place; scale it to land on
        // both sides of 0.015, 0.25, 0.95, 2.1 and 5.4.
        let mut g = 5u64;
        let n = 7;
        let base = DMatrix::from_fn(n, n, |_, _| Complex64::new(lcg(&mut g) + 0.6, 0.0));
        let mut seen = std::collections::BTreeSet::new();
        for target in [
            0.001, 0.0149, 0.0151, 0.2, 0.249, 0.251, 0.9, 0.949, 0.951, 2.0, 2.099, 2.101, 5.39,
            5.41, 11.0, 300.0,
        ] {
            for complex in [false, true] {
                let a0 = if complex {
                    base.map(|z| z * Complex64::new(0.6, 0.8))
                } else {
                    base.clone()
                };
                let mut tmp: Vec<Complex64> = a0.iter().copied().collect();
                Balance::new(&mut tmp, n).unwrap();
                let s = target / opnorm1(&tmp, n);
                let a = &a0 * Complex64::new(s, 0.0);
                let (x, d, si) = ours(&a, complex);
                seen.insert((d, si > 0));
                let exact = taylor(&a);
                assert!(
                    rel(&x, &exact) < 1e-13 * (1.0 + target),
                    "{target}: {}",
                    rel(&x, &exact)
                );
                assert!(rel(&x, &a.clone().exp()) < 1e-12 * (1.0 + target));
            }
        }
        let degrees: Vec<_> = seen.iter().copied().collect();
        assert_eq!(
            degrees,
            vec![
                (3, false),
                (5, false),
                (7, false),
                (9, false),
                (13, false),
                (13, true)
            ]
        );
    }

    #[test]
    fn random_real_and_complex() {
        let mut g = 9u64;
        for (n, scale) in [
            (1, 3.0),
            (2, 0.5),
            (5, 1.0),
            (12, 0.3),
            (12, 2.0),
            (20, 1.5),
        ] {
            for complex in [false, true] {
                let a = random(&mut g, n, complex) * Complex64::new(scale, 0.0);
                let (x, _, _) = ours(&a, complex);
                assert!(rel(&x, &taylor(&a)) < 1e-12, "{n} {scale} {complex}");
            }
        }
    }

    #[test]
    fn nilpotent_and_jordan_closed_forms() {
        // Strictly upper triangular N: exp(N) = Σ_{k<n} N^k/k!, and
        // exp(λI + N) = e^λ·exp(N), including large N (scaling path).
        let n = 6;
        let mut g = 13u64;
        for scale in [1.0, 40.0, 1e6] {
            let nm = DMatrix::from_fn(n, n, |i, j| {
                if j > i {
                    Complex64::new(scale * lcg(&mut g), 0.0)
                } else {
                    Complex64::new(0.0, 0.0)
                }
            });
            let mut exact = DMatrix::<Complex64>::identity(n, n);
            let mut term = exact.clone();
            for k in 1..n {
                term = &term * &nm / Complex64::new(k as f64, 0.0);
                exact += &term;
            }
            let (x, _, _) = ours(&nm, false);
            assert!(
                rel(&x, &exact) < 1e-15 * scale.max(10.0),
                "{scale}: {}",
                rel(&x, &exact)
            );
            let lam = Complex64::new(-0.7, 2.0);
            let j = &nm + DMatrix::<Complex64>::identity(n, n) * lam;
            let (x, _, _) = ours(&j, true);
            // Scaling and squaring of a non-normal matrix: error ~ ε·‖N‖.
            let r = rel(&x, &(exact * lam.exp()));
            assert!(r < 1e-15 * scale.max(10.0), "{scale}: {r}");
        }
    }

    #[test]
    fn normal_and_non_normal() {
        // Rotation generator: exp([0 θ; -θ 0]) = [cos θ, sin θ; -sin θ, cos θ].
        for th in [0.3, 3.0, 1e3] {
            let a =
                DMatrix::from_row_slice(2, 2, &[0.0, th, -th, 0.0]).map(|v| Complex64::new(v, 0.0));
            let (x, _, _) = ours(&a, false);
            let e = DMatrix::from_row_slice(2, 2, &[th.cos(), th.sin(), -th.sin(), th.cos()])
                .map(|v| Complex64::new(v, 0.0));
            assert!(rel(&x, &e) < 1e-15 * th.max(1.0) * 10.0, "{th}");
        }
        // Non-normal 2×2: exp([a b; 0 c]) = [e^a, b·(e^a - e^c)/(a - c); 0, e^c],
        // the divided difference formed as e^c·Σ (a-c)^k/(k+1)! to avoid the
        // cancellation of the closed form.
        for (a, b, c) in [
            (1.0f64, 1e4, 1.1f64),
            (-30.0, 50.0, -1.0),
            (0.5, 1e-8, -0.5),
        ] {
            let m = DMatrix::from_row_slice(2, 2, &[a, b, 0.0, c]).map(|v| Complex64::new(v, 0.0));
            let (x, _, _) = ours(&m, false);
            let h = a - c;
            let dd = if h.abs() < 1.0 {
                let (mut t, mut sum) = (1.0, 0.0);
                for k in 1..40 {
                    sum += t;
                    t *= h / (k + 1) as f64;
                }
                c.exp() * sum
            } else {
                (a.exp() - c.exp()) / h
            };
            let e = DMatrix::from_row_slice(2, 2, &[a.exp(), b * dd, 0.0, c.exp()])
                .map(|v| Complex64::new(v, 0.0));
            // Error ~ ε·‖A‖ for this non-normal matrix.
            assert!(
                rel(&x, &e) < 1e-15 * (10.0 + b),
                "{a} {b} {c}: {}",
                rel(&x, &e)
            );
        }
    }

    #[test]
    fn balancing_handles_bad_scaling_and_permutations() {
        // exp(D·A·D⁻¹) = D·exp(A)·D⁻¹ with D = diag(10^(3i)), and a
        // permuted block-triangular matrix with isolated eigenvalues.
        let mut g = 17u64;
        let n = 6;
        let a0 = random(&mut g, n, true) * Complex64::new(2.0, 0.0);
        let d = DMatrix::from_diagonal(&nalgebra::DVector::from_fn(n, |i, _| {
            Complex64::new(10f64.powi(3 * i as i32), 0.0)
        }));
        let dinv = d.map(|z| {
            if z.re == 0.0 {
                z
            } else {
                Complex64::new(1.0, 0.0) / z
            }
        });
        let a = &d * &a0 * &dinv;
        let (x, _, _) = ours(&a, true);
        let e = &d * taylor(&a0) * &dinv;
        for (xi, ei) in x.iter().zip(e.iter()) {
            assert!((xi - ei).norm() <= 1e-12 * ei.norm().max(1e-300) + 1e-300);
        }
        let mut t = random(&mut g, n, false);
        for j in 0..n {
            for i in 0..n {
                if (i > j && i >= 3) || (i == 5 && j < 5) || (j == 0 && i > 0) {
                    t[(i, j)] = Complex64::new(0.0, 0.0);
                }
            }
        }
        let p = [3, 0, 5, 1, 4, 2];
        let pt = DMatrix::from_fn(n, n, |i, j| t[(p[i], p[j])]);
        let mut tmp: Vec<Complex64> = pt.iter().copied().collect();
        let bal = Balance::new(&mut tmp, n).unwrap();
        assert!(bal.ilo > 0 || bal.ihi < n - 1, "permutation exercised");
        let (x, _, _) = ours(&pt, false);
        assert!(rel(&x, &taylor(&pt)) < 1e-13);
    }

    #[test]
    fn extreme_scales() {
        // ‖A‖ ~ 1e-300: exp(A) = I + A to rounding.
        let mut g = 19u64;
        let n = 4;
        let a = random(&mut g, n, true) * Complex64::new(1e-300, 0.0);
        let (x, d, _) = ours(&a, true);
        assert_eq!(d, 3);
        let e = DMatrix::<Complex64>::identity(n, n) + &a;
        assert!(rel(&x, &e) < 1e-16);
        assert!((&x - DMatrix::<Complex64>::identity(n, n) - &a).norm() <= 1e-15 * a.norm());
        // Nilpotent [0 a; 0 0] with a large: exp = I + A, but the reference
        // algorithm loses the unit diagonal (the LU solve leaves it at 1 - δ,
        // which 2^si squarings amplify). Julia 1.11 `exp!` gives exactly
        // these values for a = 1e10, and 1.27e-14 on the diagonal for 1e18;
        // matching it, not I + A, is the contract.
        let a =
            DMatrix::from_row_slice(2, 2, &[0.0, 1e10, 0.0, 0.0]).map(|v| Complex64::new(v, 0.0));
        let (x, d, si) = ours(&a, false);
        assert!(d == 13 && si == 31);
        assert!((x[(0, 0)].re - 0.9999997615814475).abs() < 1e-14);
        assert!((x[(0, 1)].re - 9.999997615814478e9).abs() < 1e-4);
        let a =
            DMatrix::from_row_slice(2, 2, &[0.0, 1e300, 0.0, 0.0]).map(|v| Complex64::new(v, 0.0));
        let (x, d, si) = ours(&a, false);
        assert!(d == 13 && si > 900, "Julia's Int 2^si would overflow here");
        assert!(x.iter().all(|z| z.re.is_finite()));
        // Decaying: exp(-1e300·diag(1, 2)) underflows to 0; growth overflows to
        // inf. Both are finite-input results, not errors.
        let a = DMatrix::from_row_slice(2, 2, &[-1e300, 1.0, 0.0, -2e300])
            .map(|v| Complex64::new(v, 0.0));
        let (x, _, _) = ours(&a, false);
        assert!(x.iter().all(|z| z.re == 0.0));
        let a =
            DMatrix::from_row_slice(2, 2, &[800.0, 1.0, 0.0, 1.0]).map(|v| Complex64::new(v, 0.0));
        let (x, _, _) = ours(&a, false);
        assert!(x[(0, 0)].re.is_infinite());
    }

    #[test]
    fn single_precision() {
        let mut g = 23u64;
        let a = random(&mut g, 6, true) * Complex64::new(1.3, 0.0);
        let a32: Vec<Complex32> = a
            .iter()
            .map(|z| Complex32::new(z.re as f32, z.im as f32))
            .collect();
        let x = expm(&a32, 6).unwrap();
        let x: Vec<Complex64> = x
            .iter()
            .map(|z| Complex64::new(z.re as f64, z.im as f64))
            .collect();
        let x = DMatrix::from_column_slice(6, 6, &x);
        let a64 = DMatrix::from_column_slice(
            6,
            6,
            &a32.iter()
                .map(|z| Complex64::new(z.re as f64, z.im as f64))
                .collect::<Vec<_>>(),
        );
        assert!(rel(&x, &taylor(&a64)) < 1e-5);
        let r: Vec<f32> = a32.iter().map(|z| z.re).collect();
        let x = expm(&r, 6).unwrap();
        let x = DMatrix::from_column_slice(
            6,
            6,
            &x.iter()
                .map(|&v| Complex64::new(v as f64, 0.0))
                .collect::<Vec<_>>(),
        );
        let ar = a64.map(|z| Complex64::new(z.re, 0.0));
        assert!(rel(&x, &taylor(&ar)) < 1e-5);
    }

    #[test]
    fn errors_and_lu() {
        assert!(expm(&[1.0, f64::NAN, 0.0, 1.0], 2).is_err());
        assert!(expm(&[f64::INFINITY], 1).is_err());
        assert!(expm::<f64>(&[], 0).unwrap().is_empty());
        // The xGESV hook against nalgebra, and on an exactly singular matrix.
        let mut g = 29u64;
        let a = random(&mut g, 7, true);
        let b = random(&mut g, 7, true);
        let (mut aa, mut bb): (Vec<Complex64>, Vec<Complex64>) =
            (a.iter().copied().collect(), b.iter().copied().collect());
        let mut ipiv = [0i32; 7];
        Complex64::gesv(&mut aa, &mut bb, 7, 7, &mut ipiv).unwrap();
        let x = a.clone().lu().solve(&b).unwrap();
        assert!(rel(&DMatrix::from_column_slice(7, 7, &bb), &x) < 1e-13);
        let mut s = vec![1.0, 2.0, 2.0, 4.0];
        let mut r = vec![1.0, 0.0, 0.0, 1.0];
        assert!(f64::gesv(&mut s, &mut r, 2, 2, &mut ipiv).is_err());
    }

    /// SplitMix64 mapped exactly to `k·2⁻⁵² - 1`, as in `generate.jl`.
    struct SplitMix(u64);
    impl SplitMix {
        fn unif(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            ((z ^ (z >> 31)) >> 11) as f64 * 2f64.powi(-52) - 1.0
        }
    }

    const FIXTURES: &str = include_str!("../../tests/fixtures/krylovkit/expm.txt");

    #[test]
    fn matches_julia_exp() {
        let mut cases = 0;
        let mut cur = std::collections::HashMap::new();
        for line in FIXTURES.lines().filter(|l| !l.starts_with('#')) {
            if line != "end" {
                let (k, v) = line.split_once(' ').unwrap_or((line, ""));
                cur.insert(k.to_string(), v.to_string());
                continue;
            }
            let c = std::mem::take(&mut cur);
            let num = |k: &str| -> f64 { c[k].parse().unwrap() };
            let floats = |k: &str| -> Vec<f64> {
                c[k].split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect()
            };
            let (n, complex) = (num("n") as usize, num("complex") == 1.0);
            let mut g = SplitMix(num("seed") as u64);
            let mut a = DMatrix::<Complex64>::zeros(n, n);
            for j in 0..n {
                for i in 0..n {
                    let re = g.unif();
                    let im = if complex { g.unif() } else { 0.0 };
                    a[(i, j)] = Complex64::new(re, im);
                }
            }
            match c["kind"].as_str() {
                "nilpotent" => a = a.upper_triangle() - DMatrix::from_diagonal(&a.diagonal()),
                "graded" => {
                    for j in 0..n {
                        for i in 0..n {
                            a[(i, j)] *= 2f64.powi(4 * (i as i32 - j as i32));
                        }
                    }
                }
                _ => {}
            }
            a *= Complex64::new(num("scale"), 0.0);
            let julia: Vec<Complex64> = floats("x_re")
                .into_iter()
                .zip(floats("x_im"))
                .map(|(re, im)| Complex64::new(re, im))
                .collect();
            let julia = DMatrix::from_column_slice(n, n, &julia);
            let (x, _, _) = ours(&a, complex);
            // Same algorithm, different rounding (provider kernels): agree to
            // a few ε·‖A‖ in norm, and entrywise relative to ‖X‖.
            let tol = 1e-15 * (10.0 + a.norm());
            let r = rel(&x, &julia);
            assert!(r < tol, "{}: {r}", c["case"]);
            let xmax = julia.iter().map(|z| z.norm()).fold(0.0, f64::max);
            for (p, q) in x.iter().zip(julia.iter()) {
                assert!((p - q).norm() <= tol * xmax, "{}", c["case"]);
            }
            cases += 1;
        }
        assert_eq!(cases, 9);
    }
}
