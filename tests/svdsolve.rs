//! `svdsolve` (GKL) against nalgebra's dense SVD and KrylovKit.jl counts.
//!
//! The vector type is shared by domain and codomain, so rectangular maps are
//! tested directly: codomain vectors have length `m`, domain vectors `n`.

mod common;

use common::*;
use krylov::{AdjointMap, Error, Gkl, LinearMap, Orth, Which, Workspace, svdsolve};
use nalgebra::{DMatrix, DVector};
use num_complex::{Complex32, Complex64};

/// Dense `m × n` map with its adjoint, counting both applications.
struct RectOp<S> {
    a: DMatrix<Complex64>,
    // Adjoint actually applied; `a.adjoint()` unless a test breaks it.
    adj: DMatrix<Complex64>,
    applications: usize,
    // Outputs become NaN once this many applications were made.
    nan_after: Option<usize>,
    _s: std::marker::PhantomData<S>,
}

impl<S: TestScalar> RectOp<S> {
    fn new(a: &DMatrix<Complex64>) -> Self {
        Self {
            a: a.clone(),
            adj: a.adjoint(),
            applications: 0,
            nan_after: None,
            _s: Default::default(),
        }
    }
}

fn apply<S: TestScalar>(m: &DMatrix<Complex64>, x: &DVec<S>, y: &mut DVec<S>) {
    assert_eq!(x.0.len(), m.ncols(), "input in the wrong space");
    assert_eq!(y.0.len(), m.nrows(), "output in the wrong space");
    let xv = DVector::from_iterator(x.0.len(), x.0.iter().map(|s| s.to_c64()));
    for (yi, v) in y.0.iter_mut().zip((m * xv).iter()) {
        *yi = S::from_c64(*v);
    }
}

impl<S: TestScalar> RectOp<S> {
    fn poison(&self, out: &mut DVec<S>) {
        if self.nan_after.is_some_and(|k| self.applications > k) {
            out.0[0] = S::from_c64(Complex64::new(f64::NAN, 0.0));
        }
    }
}

impl<S: TestScalar> LinearMap<DVec<S>> for RectOp<S> {
    fn apply_into(
        &mut self,
        _: &mut (),
        x: &DVec<S>,
        y: &mut DVec<S>,
    ) -> Result<(), std::convert::Infallible> {
        self.applications += 1;
        apply(&self.a, x, y);
        self.poison(y);
        Ok(())
    }
}

impl<S: TestScalar> AdjointMap<DVec<S>> for RectOp<S> {
    fn apply_adjoint_into(
        &mut self,
        _: &mut (),
        y: &DVec<S>,
        x: &mut DVec<S>,
    ) -> Result<(), std::convert::Infallible> {
        self.applications += 1;
        apply(&self.adj, y, x);
        self.poison(x);
        Ok(())
    }
}

fn rectangular(g: &mut SplitMix, m: usize, n: usize, complex: bool) -> DMatrix<Complex64> {
    let mut a = DMatrix::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            a[(i, j)] = g.scalar(complex);
        }
    }
    a
}

struct Out {
    values: Vec<f64>,
    normres: Vec<f64>,
    converged: usize,
    numiter: usize,
    numops: usize,
}

/// Runs `svdsolve` and checks what every result must satisfy: the triplet
/// relations `A·r = σ·l + res`, `Aᴴ·l = σ·r` (to rounding), orthonormality,
/// `‖res‖ = normres` and `numops` = applications.
fn run<S: TestScalar>(
    a: &DMatrix<Complex64>,
    x0: &[Complex64],
    howmany: usize,
    which: Which,
    alg: &Gkl,
) -> Out {
    let (m, n) = a.shape();
    let xv = to_dvec::<S>(x0);
    let domain = to_dvec::<S>(&vec![Complex64::default(); n]);
    let kd = alg.get_krylovdim();
    let mut ws_u = Workspace::with_capacity(&mut (), &xv, kd).unwrap();
    let mut ws_v = Workspace::with_capacity(&mut (), &domain, kd).unwrap();
    let mut op = RectOp::<S>::new(a);
    let (vals, left, right, info) = svdsolve(
        &mut (),
        &mut op,
        &xv,
        howmany,
        which,
        alg,
        &mut ws_u,
        &mut ws_v,
    )
    .unwrap();
    assert_eq!(info.numops, op.applications);
    assert_eq!(to_f64(info.max_imag), 0.0);
    let len = m.max(n);
    let triplet = rounding::<S>(len, op_norm(a));
    assert!(left.iter().all(|l| l.0.len() == m) && right.iter().all(|r| r.0.len() == n));
    within(
        orthogonality_error(&left),
        rounding::<S>(len, 1.0),
        "left orthonormality",
    );
    within(
        orthogonality_error(&right),
        rounding::<S>(len, 1.0),
        "right orthonormality",
    );
    for i in 0..vals.len() {
        let s = to_f64(vals[i]);
        let (l, r) = (
            DVector::from_vec(to_c64(&left[i])),
            DVector::from_vec(to_c64(&right[i])),
        );
        let res = DVector::from_vec(to_c64(&info.residual[i]));
        let c = Complex64::new(s, 0.0);
        within((a * &r - &l * c - &res).norm(), triplet, "A·r = σ·l + res");
        within((a.adjoint() * &l - &r * c).norm(), triplet, "Aᴴ·l = σ·r");
        within(
            (res.norm() - to_f64(info.normres[i])).abs(),
            triplet,
            "normres",
        );
    }
    Out {
        values: vals.iter().map(|&v| to_f64(v)).collect(),
        normres: info.normres.iter().map(|&v| to_f64(v)).collect(),
        converged: info.converged,
        numiter: info.numiter,
        numops: info.numops,
    }
}

fn dense_singular_values(a: &DMatrix<Complex64>) -> Vec<f64> {
    let mut s: Vec<f64> = a.clone().singular_values().iter().copied().collect();
    s.sort_by(|x, y| y.partial_cmp(x).unwrap());
    s
}

#[test]
fn largest_singular_values_match_dense() {
    for (m, n, complex) in [(80, 60, false), (50, 70, true), (64, 64, true)] {
        let mut g = SplitMix(301 + m as u64);
        let a = rectangular(&mut g, m, n, complex);
        let x0 = random_vec(&mut g, m, complex);
        let alg = Gkl::new(20, 100, 1e-10);
        let out = if complex {
            run::<Complex64>(&a, &x0, 3, Which::LargestReal, &alg)
        } else {
            run::<f64>(&a, &x0, 3, Which::LargestReal, &alg)
        };
        assert!(out.converged >= 3);
        let exact = dense_singular_values(&a);
        let bound = if complex {
            eig_bound::<Complex64>(alg.get_tol(), m.max(n), op_norm(&a))
        } else {
            eig_bound::<f64>(alg.get_tol(), m.max(n), op_norm(&a))
        };
        for (v, e) in out.values.iter().zip(&exact).take(out.converged) {
            within((v - e).abs(), bound, &format!("{m}×{n}: {v} vs {e}"));
        }
        assert!(out.values.windows(2).all(|w| w[0] >= w[1]));
        assert!(out.numiter > 1, "restarts exercised");
    }
}

#[test]
fn smallest_singular_values_including_zero() {
    // Square map with prescribed singular values 0, 1e-8 and a well
    // separated bulk; absolute accuracy ε·‖A‖ is all a Krylov method gives.
    for complex in [false, true] {
        let mut g = SplitMix(311 + complex as u64);
        let n = 50;
        let mut sv: Vec<f64> = (0..n).map(|i| 1.0 + 0.1 * i as f64).collect();
        sv[0] = 0.0;
        sv[1] = 1e-8;
        let u = unitary(&mut g, n, complex);
        let v = unitary(&mut g, n, complex);
        let d = DMatrix::from_diagonal(&DVector::from_iterator(
            n,
            sv.iter().map(|&s| Complex64::new(s, 0.0)),
        ));
        let a = &u * d * v.adjoint();
        let x0 = random_vec(&mut g, n, complex);
        let alg = Gkl::new(30, 300, 1e-10);
        let out = if complex {
            run::<Complex64>(&a, &x0, 3, Which::SmallestReal, &alg)
        } else {
            run::<f64>(&a, &x0, 3, Which::SmallestReal, &alg)
        };
        assert!(out.converged >= 3, "converged {}", out.converged);
        assert!(out.values.windows(2).all(|w| w[0] <= w[1]));
        let bound = if complex {
            eig_bound::<Complex64>(alg.get_tol(), n, op_norm(&a))
        } else {
            eig_bound::<f64>(alg.get_tol(), n, op_norm(&a))
        };
        for (v, e) in out.values.iter().zip([0.0, 1e-8, 1.2]) {
            within((v - e).abs(), bound, &format!("{v} vs {e}"));
        }
    }
}

#[test]
fn tall_and_wide_smallest_values() {
    // For a tall map the smallest singular values are computed on its
    // adjoint, which is wide (see the next test for why).
    for (m, n, complex) in [(60, 30, false), (30, 45, true)] {
        let mut g = SplitMix(321 + m as u64);
        let a = rectangular(&mut g, m, n, complex);
        let a = if m > n { a.adjoint() } else { a };
        let x0 = random_vec(&mut g, a.nrows(), complex);
        let alg = Gkl::new(24, 300, 1e-9);
        let out = if complex {
            run::<Complex64>(&a, &x0, 2, Which::SmallestReal, &alg)
        } else {
            run::<f64>(&a, &x0, 2, Which::SmallestReal, &alg)
        };
        assert!(out.converged >= 2, "{m}×{n}: converged {}", out.converged);
        let mut exact = dense_singular_values(&a);
        exact.reverse();
        let bound = if complex {
            eig_bound::<Complex64>(alg.get_tol(), m.max(n), op_norm(&a))
        } else {
            eig_bound::<f64>(alg.get_tol(), m.max(n), op_norm(&a))
        };
        for (v, e) in out.values.iter().zip(&exact).take(out.converged) {
            within((v - e).abs(), bound, &format!("{m}×{n}: {v} vs {e}"));
        }
    }
}

#[test]
fn tall_map_smallest_values_keep_an_unconverged_zero() {
    // With m > n the Krylov process on the codomain approximates AAᴴ, whose
    // m - n zero eigenvalues SmallestReal then targets (from any x0, rounding
    // seeds them). That zero belongs to no right singular vector
    // (‖A·r‖ >= σ_min for every unit r), so it stays first and is honestly
    // reported unconverged. (On this input with maxiter 100, KrylovKit 0.10.4
    // marks it converged although ‖A·r - σ·l‖ ≈ 2.7; at this test's maxiter 5
    // it does not.)
    let mut g = SplitMix(23);
    let a = rectangular(&mut g, 60, 30, false);
    let x0 = random_vec(&mut g, 60, false);
    let alg = Gkl::new(20, 5, 1e-9);
    let out = run::<f64>(&a, &x0, 2, Which::SmallestReal, &alg);
    assert_eq!(out.converged, 0);
    // A zero of AAᴴ: rounding only.
    within(
        out.values[0],
        rounding::<f64>(60, op_norm(&a)),
        "zero value",
    );
    let smin = *dense_singular_values(&a).last().unwrap();
    assert!(out.normres[0] >= 0.5 * smin, "{}", out.normres[0]);
}

#[test]
fn single_precision() {
    let mut g = SplitMix(331);
    let a = rectangular(&mut g, 40, 30, true);
    let x0 = random_vec(&mut g, 40, true);
    let alg = Gkl::new(16, 100, 1e-4).orth(Orth::Mgs2);
    let out = run::<Complex32>(&a, &x0, 2, Which::LargestReal, &alg);
    let exact = dense_singular_values(&a);
    assert!(out.converged >= 2);
    let bound = eig_bound::<Complex32>(alg.get_tol(), 40, op_norm(&a));
    for (v, e) in out.values.iter().zip(&exact).take(out.converged) {
        within((v - e).abs(), bound, "c32 value");
    }
    let ar = a.map(|z| Complex64::new(z.re, 0.0));
    let xr: Vec<Complex64> = x0.iter().map(|z| Complex64::new(z.re, 0.0)).collect();
    let out = run::<f32>(&ar, &xr, 2, Which::LargestReal, &alg.orth(Orth::Cgs2));
    let exact = dense_singular_values(&ar);
    let bound = eig_bound::<f32>(alg.get_tol(), 40, op_norm(&ar));
    for (v, e) in out.values.iter().zip(&exact).take(out.converged) {
        within((v - e).abs(), bound, "f32 value");
    }
}

#[test]
fn exhausted_spaces_return_what_they_have() {
    // A = diag(3, 2, 0, …): the Krylov space closes after a few steps (an
    // invariant subspace or Aᴴu ∈ span V); fewer than `howmany` triplets may
    // come back and every converged one is exact.
    let n = 8;
    let mut a = DMatrix::<Complex64>::zeros(n, n);
    a[(0, 0)] = Complex64::new(3.0, 0.0);
    a[(1, 1)] = Complex64::new(2.0, 0.0);
    let x0 = vec![Complex64::new(1.0, 0.0); n];
    let alg = Gkl::new(6, 100, 1e-12);
    let out = run::<f64>(&a, &x0, 5, Which::LargestReal, &alg);
    assert!(out.values.len() < 5);
    assert!(out.converged >= 2);
    // An exact invariant subspace: rounding only.
    let bound = rounding::<f64>(n, 3.0);
    within((out.values[0] - 3.0).abs(), bound, "σ = 3");
    within((out.values[1] - 2.0).abs(), bound, "σ = 2");
    assert_eq!(out.numiter, 1);
    // Aᴴ·x0 = 0: no triplet at all.
    let mut x0 = vec![Complex64::default(); n];
    x0[5] = Complex64::new(1.0, 0.0);
    let out = run::<f64>(&a, &x0, 1, Which::LargestReal, &alg);
    assert!(out.values.is_empty() && out.converged == 0 && out.numops == 1);
}

#[test]
fn warm_call_allocates_only_the_returned_triplets() {
    let mut g = SplitMix(341);
    let a = rectangular(&mut g, 30, 20, false);
    let x0 = to_dvec::<f64>(&random_vec(&mut g, 30, false));
    let domain = to_dvec::<f64>(&vec![Complex64::default(); 20]);
    let alg = Gkl::new(8, 100, 1e-10);
    let mut ws_u = Workspace::with_capacity(&mut (), &x0, 8).unwrap();
    let mut ws_v = Workspace::with_capacity(&mut (), &domain, 8).unwrap();
    let mut op = RectOp::<f64>::new(&a);
    let before = similar_calls();
    let (vals, _, _, info) = svdsolve(
        &mut (),
        &mut op,
        &x0,
        2,
        Which::LargestReal,
        &alg,
        &mut ws_u,
        &mut ws_v,
    )
    .unwrap();
    assert!(info.numiter > 1);
    assert_eq!(similar_calls() - before, 3 * vals.len());
}

#[test]
fn typed_errors() {
    let mut g = SplitMix(351);
    let a = rectangular(&mut g, 12, 9, true);
    let x0 = to_dvec::<Complex64>(&random_vec(&mut g, 12, true));
    let domain = to_dvec::<Complex64>(&[Complex64::default(); 9]);
    let mut ws_u = Workspace::with_capacity(&mut (), &x0, 6).unwrap();
    let mut ws_v = Workspace::with_capacity(&mut (), &domain, 6).unwrap();
    let alg = Gkl::new(6, 100, 1e-12);
    let mut op = RectOp::<Complex64>::new(&a);
    let mut call = |op: &mut RectOp<Complex64>, x: &DVec<Complex64>, h, w, alg: &Gkl| {
        svdsolve(&mut (), op, x, h, w, alg, &mut ws_u, &mut ws_v).map(|_| ())
    };
    // Transpose without conjugation is not the adjoint of a complex map.
    let mut bad = RectOp::<Complex64>::new(&a);
    bad.adj = a.transpose();
    assert_eq!(
        call(&mut bad, &x0, 1, Which::LargestReal, &alg),
        Err(Error::IncompatibleAdjoint)
    );
    let lm = call(&mut op, &x0, 1, Which::LargestMagnitude, &alg);
    assert!(matches!(lm, Err(Error::InvalidArgument(_))));
    assert_eq!(
        call(&mut op, &x0, 7, Which::LargestReal, &alg),
        Err(Error::HowmanyTooLarge {
            howmany: 7,
            krylovdim: 6
        })
    );
    assert_eq!(
        call(
            &mut op,
            &x0,
            1,
            Which::LargestReal,
            &Gkl::new(7, 100, 1e-12)
        ),
        Err(Error::InsufficientCapacity {
            needed: 7,
            capacity: 6
        })
    );
    let zero = to_dvec::<Complex64>(&[Complex64::default(); 12]);
    assert_eq!(
        call(&mut op, &zero, 1, Which::LargestReal, &alg),
        Err(Error::ZeroStartVector)
    );
    let mut nan = to_dvec::<Complex64>(&random_vec(&mut g, 12, true));
    nan.0[0] = Complex64::new(f64::NAN, 0.0);
    assert_eq!(
        call(&mut op, &nan, 1, Which::LargestReal, &alg),
        Err(Error::NonFiniteStartVector)
    );
    let neg = call(&mut op, &x0, 1, Which::LargestReal, &Gkl::new(6, 100, -1.0));
    assert!(matches!(neg, Err(Error::InvalidArgument(_))));
    // krylovdim 1 cannot restart.
    let one = call(
        &mut op,
        &x0,
        1,
        Which::LargestReal,
        &Gkl::new(1, 100, 1e-12),
    );
    assert!(matches!(one, Err(Error::InvalidArgument(_))));
    // A map that turns NaN after a few applications.
    let mut nan_op = RectOp::<Complex64>::new(&a);
    nan_op.nan_after = Some(5);
    assert_eq!(
        call(&mut nan_op, &x0, 1, Which::LargestReal, &alg),
        Err(Error::NonFiniteIterate)
    );
    // NaN from the very first application of A (the adjoint check must not
    // pass on NaN).
    let mut nan_op = RectOp::<Complex64>::new(&a);
    nan_op.nan_after = Some(1);
    assert_eq!(
        call(&mut nan_op, &x0, 1, Which::LargestReal, &alg),
        Err(Error::NonFiniteIterate)
    );
}

/// name, seed, m, n, complex, howmany, which, krylovdim, tol, eager,
/// KrylovKit `(nvals, converged, numiter, numops)`.
type KkCase = (
    &'static str,
    u64,
    usize,
    usize,
    bool,
    usize,
    Which,
    usize,
    f64,
    bool,
    Option<(usize, usize, usize, usize)>,
);

fn kk_case<S: TestScalar>(case: &KkCase) {
    let (name, seed, m, n, cplx, howmany, which, kd, tol, eager, kk) = *case;
    let mut g = SplitMix(seed);
    let a = rectangular(&mut g, m, n, cplx);
    let x0 = random_vec(&mut g, m, cplx);
    let alg = Gkl::new(kd, 100, tol).eager(eager);
    let out = run::<S>(&a, &x0, howmany, which, &alg);
    let mut want = dense_singular_values(&a);
    if which == Which::SmallestReal {
        want.reverse();
    }
    assert!(
        out.converged >= howmany,
        "{name}: converged {}",
        out.converged
    );
    let bound = eig_bound::<S>(tol, m.max(n), op_norm(&a));
    for ((v, e), r) in out.values.iter().zip(&want).zip(&out.normres) {
        within((v - e).abs(), bound, &format!("{name}: {v} vs {e}"));
        assert!(*r < tol, "{name}: residual {r}");
    }
    if let Some((nvals, converged, numiter, numops)) = kk {
        assert_eq!(out.values.len(), nvals, "{name}: number of values");
        assert_eq!(out.converged, converged, "{name}: converged");
        assert_eq!(out.numiter, numiter, "{name}: numiter");
        assert_eq!(out.numops, numops, "{name}: numops");
    }
}

/// KrylovKit.jl's GKL cases (CGS2, `maxiter = 100`) against nalgebra's
/// SVD of the same matrix; `run` checks `A·v = σu + r` and `Aᴴu = σv`. GKL
/// takes KrylovKit's restarts exactly; the counts are pinned on a tall LR
/// case that returns every converged triplet and a restart-heavy square SR
/// case, checked against KrylovKit.jl v0.10.4. Tall maps with
/// `SmallestReal` are not compared: KrylovKit then reports a spurious
/// converged σ ≈ 0 (`tall_map_smallest_values_keep_an_unconverged_zero`).
#[test]
fn svdsolve_matches_dense_svd_and_krylovkit_counts() {
    #[rustfmt::skip]
    let cases: [KkCase; 4] = [
        ("svd_real_lr", 21, 90, 70, false, 3, Which::LargestReal, 20, 1e-10, false, Some((4, 4, 4, 88))),
        ("svd_complex_lr", 22, 60, 80, true, 2, Which::LargestReal, 16, 1e-10, false, None),
        ("svd_real_sr_square", 23, 40, 40, false, 2, Which::SmallestReal, 20, 1e-9, false, Some((4, 4, 11, 200))),
        ("svd_complex_sr_wide_eager", 24, 40, 50, true, 2, Which::SmallestReal, 24, 1e-9, true, None),
    ];
    for case in &cases {
        if case.4 {
            kk_case::<Complex64>(case);
        } else {
            kk_case::<f64>(case);
        }
    }
}
