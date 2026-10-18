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
//! them is implemented here.

use num_traits::{Float, One, Zero};

use crate::scalar::{RealScalar, Scalar};

/// Caller-owned scratch and result of [`expm_into`], sized once for the
/// largest `n` of a call so that repeated evaluations (the step-size trials
/// of the exponential integrator) allocate nothing.
pub(crate) struct ExpmWork<S: Scalar> {
    x: Vec<S>,
    a2: Vec<S>,
    a4: Vec<S>,
    a6: Vec<S>,
    p: Vec<S>,
    u: Vec<S>,
    v: Vec<S>,
    t2: Vec<S>,
    tmp: Vec<S>,
    ipiv: Vec<i32>,
    scale: Vec<S::Real>,
    perm: Vec<usize>,
}

impl<S: Scalar> ExpmWork<S> {
    /// Scratch for matrices up to `nmax × nmax`.
    pub(crate) fn new(nmax: usize) -> Self {
        let m = || Vec::with_capacity(nmax * nmax);
        Self {
            x: m(),
            a2: m(),
            a4: m(),
            a6: m(),
            p: m(),
            u: m(),
            v: m(),
            t2: m(),
            tmp: m(),
            ipiv: Vec::with_capacity(nmax),
            scale: Vec::with_capacity(nmax),
            perm: Vec::with_capacity(nmax),
        }
    }

    /// The result of the last successful [`expm_into`] (`n × n`).
    pub(crate) fn result(&self) -> &[S] {
        &self.x
    }

    fn size(&mut self, n: usize) {
        for b in [
            &mut self.x,
            &mut self.a2,
            &mut self.a4,
            &mut self.a6,
            &mut self.p,
            &mut self.u,
            &mut self.v,
            &mut self.t2,
            &mut self.tmp,
        ] {
            b.clear();
            b.resize(n * n, S::zero());
        }
        self.ipiv.clear();
        self.ipiv.resize(n, 0);
        self.scale.clear();
        self.scale.resize(n, S::Real::one());
        self.perm.clear();
        self.perm.extend(0..n);
    }
}

/// `exp(A)` of the `n × n` column-major `a` into `w.result()`; `a` is used as
/// scratch (balanced and scaled in place). `Err` for non-finite input or an
/// exactly singular Padé denominator (Julia: `SingularException`).
///
/// Julia's `exp!` diverts Hermitian input to an eigendecomposition; that
/// branch is not ported because its consumer, the augmented matrix of the
/// exponential integrator, is never Hermitian.
pub(crate) fn expm_into<S: Scalar>(a: &mut [S], n: usize, w: &mut ExpmWork<S>) -> Result<(), ()> {
    expm_impl(a, n, w).map(|_| ())
}

/// [`expm_into`] that also returns the Padé degree and the squaring count.
fn expm_impl<S: Scalar>(a: &mut [S], n: usize, w: &mut ExpmWork<S>) -> Result<(usize, usize), ()> {
    assert_eq!(a.len(), n * n, "expm: shape mismatch");
    if !a.iter().all(|x| x.all_finite()) {
        return Err(());
    }
    w.size(n);
    if n == 0 {
        return Ok((0, 0));
    }
    let bal = Balance::new(a, n, &mut w.scale, &mut w.perm)?;
    let na = opnorm1(a, n);
    // Julia compares `nA::T` with Float64 literals, i.e. in f64.
    let na64 = num_traits::ToPrimitive::to_f64(&na).unwrap_or(f64::NAN);
    let (degree, si) = if na64 <= 2.1 {
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
        let (cs, len) = (c, c.len());
        let c = |i: usize| real::<S>(cs[i]);
        matmul_into(a, a, n, &mut w.a2);
        // U = C[2]·I + C[4]·A², V = C[1]·I + C[3]·A², then higher even
        // powers; U ← A·U; X = (V - U) \ (V + U).
        w.p.copy_from_slice(&w.a2);
        for ((ui, vi), &pi) in w.u.iter_mut().zip(w.v.iter_mut()).zip(&w.p) {
            *ui = c(3) * pi;
            *vi = c(2) * pi;
        }
        add_identity(&mut w.u, n, c(1));
        add_identity(&mut w.v, n, c(0));
        for k in 2..len / 2 {
            matmul_into(&w.p, &w.a2, n, &mut w.tmp);
            core::mem::swap(&mut w.p, &mut w.tmp);
            for ((ui, vi), &pi) in w.u.iter_mut().zip(w.v.iter_mut()).zip(&w.p) {
                *ui += c(2 * k + 1) * pi;
                *vi += c(2 * k) * pi;
            }
        }
        matmul_into(a, &w.u, n, &mut w.tmp);
        core::mem::swap(&mut w.u, &mut w.tmp);
        pade_solve(w, n)?;
        (len - 1, 0)
    } else {
        // si: number of exact halvings of nA/5.4 down to <= 1. Julia uses
        // ceil(log2(nA/5.4)); the two can differ where libm's log2 rounds
        // across an integer just above a power of two (design deviation 8).
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
            for x in a.iter_mut() {
                *x = x.mul_real(factor);
            }
        }
        const CC: [f64; 14] = [
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
        ];
        let cc = |i: usize| real::<S>(CC[i]);
        matmul_into(a, a, n, &mut w.a2);
        matmul_into(&w.a2, &w.a2, n, &mut w.a4);
        matmul_into(&w.a2, &w.a4, n, &mut w.a6);
        let comb = |out: &mut [S], a6: &[S], a4: &[S], a2: &[S], c6: S, c4: S, c2: S| {
            for (o, ((&x6, &x4), &x2)) in out.iter_mut().zip(a6.iter().zip(a4).zip(a2)) {
                *o = c6 * x6 + c4 * x4 + c2 * x2;
            }
        };
        // U = A·(A6·(c14 A6 + c12 A4 + c10 A2) + c8 A6 + c6 A4 + c4 A2 + c2 I)
        comb(&mut w.p, &w.a6, &w.a4, &w.a2, cc(13), cc(11), cc(9));
        comb(&mut w.t2, &w.a6, &w.a4, &w.a2, cc(7), cc(5), cc(3));
        add_identity(&mut w.t2, n, cc(1));
        matmul_into(&w.a6, &w.p, n, &mut w.tmp);
        for (x, y) in w.tmp.iter_mut().zip(&w.t2) {
            *x += *y;
        }
        matmul_into(a, &w.tmp, n, &mut w.u);
        // V = A6·(c13 A6 + c11 A4 + c9 A2) + c7 A6 + c5 A4 + c3 A2 + c1 I
        comb(&mut w.p, &w.a6, &w.a4, &w.a2, cc(12), cc(10), cc(8));
        comb(&mut w.t2, &w.a6, &w.a4, &w.a2, cc(6), cc(4), cc(2));
        add_identity(&mut w.t2, n, cc(0));
        matmul_into(&w.a6, &w.p, n, &mut w.v);
        for (x, y) in w.v.iter_mut().zip(&w.t2) {
            *x += *y;
        }
        pade_solve(w, n)?;
        for _ in 0..si {
            matmul_into(&w.x, &w.x, n, &mut w.tmp);
            core::mem::swap(&mut w.x, &mut w.tmp);
        }
        (13, si)
    };
    bal.undo(&mut w.x, n, &w.scale, &w.perm);
    Ok((degree, si))
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

/// Column-major `c ← a·b`.
fn matmul_into<S: Scalar>(a: &[S], b: &[S], n: usize, c: &mut [S]) {
    c.fill(S::zero());
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

/// `x ← (V - U) \ (V + U)`, with `tmp` as the LU scratch.
fn pade_solve<S: Scalar>(w: &mut ExpmWork<S>, n: usize) -> Result<(), ()> {
    for (((d, x), &v), &u) in w.tmp.iter_mut().zip(w.x.iter_mut()).zip(&w.v).zip(&w.u) {
        *d = v - u;
        *x = v + u;
    }
    S::gesv(&mut w.tmp, &mut w.x, n, n, &mut w.ipiv)
}

/// Balancing `D⁻¹·Pᵀ·A·P·D` from `xGEBAL('B')`, kept to undo it; the factors
/// and the permutation live in the caller's `scale` and `perm`.
struct Balance {
    ilo: usize,
    ihi: usize,
}

impl Balance {
    /// LAPACK `xGEBAL('B')` on `a`; the permutation indices LAPACK stores in
    /// `scale` outside `ilo..=ihi` go to `perm` (0-based).
    fn new<S: Scalar>(
        a: &mut [S],
        n: usize,
        scale: &mut [S::Real],
        perm: &mut [usize],
    ) -> Result<Self, ()> {
        let (ilo, ihi) = S::gebal(a, n, scale)?;
        for (j, p) in perm.iter_mut().enumerate() {
            if j < ilo || j > ihi {
                let idx = num_traits::ToPrimitive::to_usize(&scale[j]).ok_or(())?;
                if idx == 0 || idx > n {
                    return Err(());
                }
                *p = idx - 1;
            }
        }
        Ok(Self { ilo, ihi })
    }

    /// Julia `exp!`: `X ← D·X·D⁻¹`, then the lower permutations in reverse
    /// and the upper ones in forward order.
    fn undo<S: Scalar>(&self, x: &mut [S], n: usize, scale: &[S::Real], perm: &[usize]) {
        for j in self.ilo..=self.ihi {
            // Real factors scale component-wise, as Julia's `X[j,i] *= scj`
            // does, so an overflowed `Inf + 0i` entry stays `Inf + 0i`. The
            // GEBAL factors are powers of two, so 1/scale is exact.
            let (sc, inv) = (scale[j], S::Real::one() / scale[j]);
            for i in 0..n {
                x[j + i * n] = x[j + i * n].mul_real(sc);
            }
            for i in 0..n {
                x[i + j * n] = x[i + j * n].mul_real(inv);
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
        for (j, &pj) in perm[..self.ilo].iter().enumerate().rev() {
            rcswap(x, j, pj);
        }
        for (j, &pj) in perm.iter().enumerate().skip(self.ihi + 1) {
            rcswap(x, j, pj);
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
            let mut w = ExpmWork::new(n);
            let (d, s) = expm_impl(&mut a.as_slice().to_vec(), n, &mut w).unwrap();
            (DMatrix::from_column_slice(n, n, w.result()), d, s)
        } else {
            let ar: Vec<f64> = a.iter().map(|z| z.re).collect();
            let mut w = ExpmWork::new(n);
            let (d, s) = expm_impl(&mut ar.clone(), n, &mut w).unwrap();
            let x: Vec<Complex64> = w.result().iter().map(|&v| Complex64::new(v, 0.0)).collect();
            (DMatrix::from_column_slice(n, n, &x), d, s)
        }
    }

    /// `exp(A)` with a fresh workspace.
    fn expm<S: Scalar>(a: &[S], n: usize) -> Result<Vec<S>, ()> {
        let mut w = ExpmWork::new(n);
        expm_into(&mut a.to_vec(), n, &mut w)?;
        Ok(w.result().to_vec())
    }

    #[test]
    fn reused_workspace_matches_a_fresh_one() {
        // Sizes up and down and both branches through one workspace: the
        // result must not depend on what the workspace held before.
        let mut g = 31u64;
        let mut w = ExpmWork::new(9);
        for (n, scale) in [(9, 0.1), (3, 4.0), (9, 30.0), (1, 0.5), (6, 1.0)] {
            for complex in [false, true] {
                let a: Vec<Complex64> = random(&mut g, n, complex)
                    .map(|z| z * scale)
                    .iter()
                    .copied()
                    .collect();
                expm_into(&mut a.clone(), n, &mut w).unwrap();
                assert_eq!(w.result(), &expm(&a, n).unwrap()[..], "{n} {scale}");
            }
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
                Balance::new(&mut tmp, n, &mut vec![1.0; n], &mut vec![0; n]).unwrap();
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
        let bal = Balance::new(&mut tmp, n, &mut vec![1.0; n], &mut vec![0; n]).unwrap();
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
        // Complex: Julia 1.11 gives [Inf+0im Inf+0im; 0 e]. A NaN here would
        // pass the step controller's ω >= 1 test as "small error".
        let (x, _, _) = ours(&a, true);
        for z in [x[(0, 0)], x[(0, 1)]] {
            assert!(z.re == f64::INFINITY && z.im == 0.0, "{z}");
        }
        assert!(x[(1, 0)] == Complex64::new(0.0, 0.0));
        // Julia: 2.7182818284589936 (the squarings cost ~2e-13 of e).
        let e = x[(1, 1)] - Complex64::new(2.7182818284589936, 0.0);
        assert!(e.norm() < 1e-14, "{}", x[(1, 1)]);
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

    /// SplitMix64 mapped exactly to `k·2⁻⁵² - 1`, so the matrices below are
    /// reproducible bit for bit in Julia.
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

    /// Same Padé degree, scaling and balancing as Julia's `exp!` (what
    /// KrylovKit's expintegrator calls), so the result agrees with it to
    /// rounding (different provider kernels round differently). The first
    /// column of `exp(A)` was checked against Julia 1.11.6
    /// `LinearAlgebra.exp!` (KrylovKit.jl v0.10.4 environment) for one case
    /// per regime: degree 13 without squaring (real), scaling and squaring
    /// (complex), and a graded matrix that balancing rescales (complex);
    /// the degree and squaring count follow from Julia's thresholds on the
    /// balanced `‖A‖₁` (4.26, 55.7 and 4.42). The lower degrees are pinned
    /// against the Taylor oracle in `every_degree_and_both_sides_of_each_threshold`.
    #[test]
    fn matches_julia_exp() {
        type Case = (
            &'static str,
            u64,
            usize,
            bool,
            f64,
            (usize, usize),
            [(f64, f64); 10],
        );
        let cases: [Case; 3] = [
            (
                "deg13",
                35,
                8,
                false,
                0.8,
                (13, 0),
                [
                    (0.8268308284383482, 0.0),
                    (0.0342024372527508, 0.0),
                    (0.18922553985925525, 0.0),
                    (-0.584388336743765, 0.0),
                    (0.5180543591923297, 0.0),
                    (-0.2724009461299436, 0.0),
                    (0.21517588555629913, 0.0),
                    (-0.9545336416660247, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
            ),
            (
                "squaring",
                36,
                10,
                true,
                6.0,
                (13, 4),
                [
                    (-60706.297355832474, 143615.3181846314),
                    (76274.51236057113, -6520.500565855251),
                    (26893.768931602684, 65379.9593640823),
                    (-36339.57614142999, 5714.609618379751),
                    (55079.2351503049, -16204.550676512014),
                    (-31650.204430725073, 37821.88201428281),
                    (46729.3414446548, -76057.30142274681),
                    (-26894.26669948297, 34727.43712576238),
                    (-41251.63692230814, -45445.27938633388),
                    (100361.45940709075, -36380.72502209295),
                ],
            ),
            (
                "graded",
                38,
                7,
                true,
                0.7,
                (13, 0),
                [
                    (0.25310784063281944, -0.9183145509960402),
                    (-19.343441262341276, -11.51192593172199),
                    (127.55360257594381, 236.49677652028578),
                    (-710.541490422778, 1366.022614823692),
                    (-30623.84702932327, -56743.85846004804),
                    (535407.066308113, 665483.6344442582),
                    (636960.9481847256, 705131.2434061968),
                    (0.0, 0.0),
                    (0.0, 0.0),
                    (0.0, 0.0),
                ],
            ),
        ];
        for (name, seed, n, complex, scale, degree_squarings, col0) in cases {
            let mut g = SplitMix(seed);
            let mut a = DMatrix::<Complex64>::zeros(n, n);
            for j in 0..n {
                for i in 0..n {
                    let re = g.unif();
                    let im = if complex { g.unif() } else { 0.0 };
                    a[(i, j)] = Complex64::new(re, im);
                }
            }
            if name == "graded" {
                // Exact similarity by powers of two.
                for j in 0..n {
                    for i in 0..n {
                        a[(i, j)] *= 2f64.powi(4 * (i as i32 - j as i32));
                    }
                }
            }
            a *= Complex64::new(scale, 0.0);
            let (x, d, si) = ours(&a, complex);
            assert_eq!((d, si), degree_squarings, "{name}");
            let want: Vec<Complex64> = col0[..n]
                .iter()
                .map(|&(r, i)| Complex64::new(r, i))
                .collect();
            // Entrywise relative: these columns are well conditioned (the
            // graded one after balancing), so a norm-scaled bound would
            // leave its small entries unchecked.
            for (p, q) in x.column(0).iter().zip(&want) {
                assert!((p - q).norm() <= 1e-13 * q.norm(), "{name}: {p} vs {q}");
            }
        }
    }
}
