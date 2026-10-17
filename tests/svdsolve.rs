//! `svdsolve` (GKL) against nalgebra's dense SVD and KrylovKit.jl.
//!
//! The vector type is shared by domain and codomain, so rectangular maps are
//! tested directly: codomain vectors have length `m`, domain vectors `n`.

mod common;

use std::collections::HashMap;

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
    _s: std::marker::PhantomData<S>,
}

impl<S: TestScalar> RectOp<S> {
    fn new(a: &DMatrix<Complex64>) -> Self {
        Self {
            a: a.clone(),
            adj: a.adjoint(),
            applications: 0,
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

impl<S: TestScalar> LinearMap<DVec<S>> for RectOp<S> {
    fn apply_into(&mut self, x: &DVec<S>, y: &mut DVec<S>) -> Result<(), std::convert::Infallible> {
        self.applications += 1;
        apply(&self.a, x, y);
        Ok(())
    }
}

impl<S: TestScalar> AdjointMap<DVec<S>> for RectOp<S> {
    fn apply_adjoint_into(
        &mut self,
        y: &DVec<S>,
        x: &mut DVec<S>,
    ) -> Result<(), std::convert::Infallible> {
        self.applications += 1;
        apply(&self.adj, y, x);
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
    let mut ws_u = Workspace::with_capacity(&xv, kd).unwrap();
    let mut ws_v = Workspace::with_capacity(&domain, kd).unwrap();
    let mut op = RectOp::<S>::new(a);
    let (vals, left, right, info) =
        svdsolve(&mut op, &xv, howmany, which, alg, &mut ws_u, &mut ws_v).unwrap();
    assert_eq!(info.numops, op.applications);
    assert_eq!(to_f64(info.max_imag), 0.0);
    let single = std::mem::size_of::<S::Real>() == 4;
    let eps = if single { 1e-5 } else { 1e-12 };
    let anorm = a.norm();
    assert!(left.iter().all(|l| l.0.len() == m) && right.iter().all(|r| r.0.len() == n));
    assert!(orthogonality_error(&left) < eps * 10.0);
    assert!(orthogonality_error(&right) < eps * 10.0);
    for i in 0..vals.len() {
        let s = to_f64(vals[i]);
        let (l, r) = (
            DVector::from_vec(to_c64(&left[i])),
            DVector::from_vec(to_c64(&right[i])),
        );
        let res = DVector::from_vec(to_c64(&info.residual[i]));
        let c = Complex64::new(s, 0.0);
        assert!((a * &r - &l * c - &res).norm() <= eps * anorm);
        assert!((a.adjoint() * &l - &r * c).norm() <= eps * anorm);
        assert!((res.norm() - to_f64(info.normres[i])).abs() <= eps * anorm);
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
        let alg = Gkl::new().krylovdim(20).tol(1e-10);
        let out = if complex {
            run::<Complex64>(&a, &x0, 3, Which::LargestReal, &alg)
        } else {
            run::<f64>(&a, &x0, 3, Which::LargestReal, &alg)
        };
        assert!(out.converged >= 3);
        let exact = dense_singular_values(&a);
        for (v, e) in out.values.iter().zip(&exact) {
            assert!((v - e).abs() < 1e-9 * e, "{m}×{n}: {v} vs {e}");
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
        let alg = Gkl::new().krylovdim(30).tol(1e-10).maxiter(300);
        let out = if complex {
            run::<Complex64>(&a, &x0, 3, Which::SmallestReal, &alg)
        } else {
            run::<f64>(&a, &x0, 3, Which::SmallestReal, &alg)
        };
        assert!(out.converged >= 3, "converged {}", out.converged);
        assert!(out.values.windows(2).all(|w| w[0] <= w[1]));
        for (v, e) in out.values.iter().zip([0.0, 1e-8, 1.2]) {
            assert!((v - e).abs() < 1e-11, "{v} vs {e}");
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
        let alg = Gkl::new().krylovdim(24).tol(1e-9).maxiter(300);
        let out = if complex {
            run::<Complex64>(&a, &x0, 2, Which::SmallestReal, &alg)
        } else {
            run::<f64>(&a, &x0, 2, Which::SmallestReal, &alg)
        };
        assert!(out.converged >= 2, "{m}×{n}: converged {}", out.converged);
        let mut exact = dense_singular_values(&a);
        exact.reverse();
        for (v, e) in out.values.iter().zip(&exact) {
            assert!((v - e).abs() < 1e-9 * e.max(1.0), "{m}×{n}: {v} vs {e}");
        }
    }
}

#[test]
fn tall_map_smallest_values_keep_an_unconverged_zero() {
    // With m > n the Krylov process on the codomain approximates AAᴴ, whose
    // m - n zero eigenvalues SmallestReal then targets (from any x0, rounding
    // seeds them). That zero belongs to no right singular vector
    // (‖A·r‖ >= σ_min for every unit r), so it stays first and is honestly
    // reported unconverged. KrylovKit 0.10.4 reports it as converged on this
    // input although ‖A·r - σ·l‖ ≈ 2.7.
    let mut g = SplitMix(23);
    let a = rectangular(&mut g, 60, 30, false);
    let x0 = random_vec(&mut g, 60, false);
    let alg = Gkl::new().krylovdim(20).tol(1e-9).maxiter(5);
    let out = run::<f64>(&a, &x0, 2, Which::SmallestReal, &alg);
    assert_eq!(out.converged, 0);
    assert!(out.values[0] < 1e-12);
    let smin = *dense_singular_values(&a).last().unwrap();
    assert!(out.normres[0] >= 0.5 * smin, "{}", out.normres[0]);
}

#[test]
fn single_precision() {
    let mut g = SplitMix(331);
    let a = rectangular(&mut g, 40, 30, true);
    let x0 = random_vec(&mut g, 40, true);
    let alg = Gkl::new().krylovdim(16).tol(1e-4).orth(Orth::Mgs2);
    let out = run::<Complex32>(&a, &x0, 2, Which::LargestReal, &alg);
    let exact = dense_singular_values(&a);
    assert!(out.converged >= 2);
    for (v, e) in out.values.iter().zip(&exact) {
        assert!((v - e).abs() < 1e-4 * e);
    }
    let ar = a.map(|z| Complex64::new(z.re, 0.0));
    let xr: Vec<Complex64> = x0.iter().map(|z| Complex64::new(z.re, 0.0)).collect();
    let out = run::<f32>(&ar, &xr, 2, Which::LargestReal, &alg.orth(Orth::Cgs2));
    let exact = dense_singular_values(&ar);
    for (v, e) in out.values.iter().zip(&exact) {
        assert!((v - e).abs() < 1e-4 * e);
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
    let alg = Gkl::new().krylovdim(6).tol(1e-12);
    let out = run::<f64>(&a, &x0, 5, Which::LargestReal, &alg);
    assert!(out.values.len() < 5);
    assert!(out.converged >= 2);
    assert!((out.values[0] - 3.0).abs() < 1e-13 && (out.values[1] - 2.0).abs() < 1e-13);
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
    let alg = Gkl::new().krylovdim(8).tol(1e-10);
    let mut ws_u = Workspace::with_capacity(&x0, 8).unwrap();
    let mut ws_v = Workspace::with_capacity(&domain, 8).unwrap();
    let mut op = RectOp::<f64>::new(&a);
    let before = similar_calls();
    let (vals, _, _, info) = svdsolve(
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
    let mut ws_u = Workspace::with_capacity(&x0, 6).unwrap();
    let mut ws_v = Workspace::with_capacity(&domain, 6).unwrap();
    let alg = Gkl::new().krylovdim(6);
    let mut op = RectOp::<Complex64>::new(&a);
    let mut call = |op: &mut RectOp<Complex64>, x: &DVec<Complex64>, h, w, alg: &Gkl| {
        svdsolve(op, x, h, w, alg, &mut ws_u, &mut ws_v).map(|_| ())
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
            &Gkl::new().krylovdim(7)
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
    let neg = call(&mut op, &x0, 1, Which::LargestReal, &alg.tol(-1.0));
    assert!(matches!(neg, Err(Error::InvalidArgument(_))));
}

// KrylovKit fixtures (`fixtures/krylovkit/svdsolve.txt`, from `svdsolve.jl`).

const FIXTURES: &str = include_str!("fixtures/krylovkit/svdsolve.txt");

fn cases() -> Vec<HashMap<String, String>> {
    let mut out = Vec::new();
    let mut cur = HashMap::new();
    for line in FIXTURES.lines().filter(|l| !l.starts_with('#')) {
        if line == "end" {
            out.push(std::mem::take(&mut cur));
            continue;
        }
        let (k, v) = line.split_once(' ').unwrap_or((line, ""));
        cur.insert(k.to_string(), v.to_string());
    }
    out
}

fn num<T: std::str::FromStr>(c: &HashMap<String, String>, k: &str) -> T
where
    T::Err: std::fmt::Debug,
{
    c[k].parse().unwrap()
}

fn floats(c: &HashMap<String, String>, k: &str) -> Vec<f64> {
    c[k].split_whitespace()
        .map(|x| x.parse().unwrap())
        .collect()
}

fn fixture_case<S: TestScalar>(c: &HashMap<String, String>) {
    let cplx = num::<u8>(c, "complex") == 1;
    let (m, n): (usize, usize) = (num(c, "m"), num(c, "n"));
    let mut g = SplitMix(num(c, "seed"));
    let a = rectangular(&mut g, m, n, cplx);
    let x0 = random_vec(&mut g, m, cplx);
    let which = match c["which"].as_str() {
        "LR" => Which::LargestReal,
        "SR" => Which::SmallestReal,
        w => panic!("unknown selector {w}"),
    };
    let alg = Gkl::new()
        .krylovdim(num(c, "krylovdim"))
        .tol(num(c, "tol"))
        .eager(num::<u8>(c, "eager") == 1)
        .maxiter(100);
    let out = run::<S>(&a, &x0, num(c, "howmany"), which, &alg);
    let name = &c["case"];
    eprintln!(
        "{name}: krylov numiter {} numops {} converged {} | KrylovKit {} {} {}",
        out.numiter, out.numops, out.converged, c["numiter"], c["numops"], c["converged"]
    );
    let expect = floats(c, "values");
    assert_eq!(out.values.len(), expect.len(), "{name}: number of values");
    assert_eq!(out.converged, num::<usize>(c, "converged"), "{name}");
    let tol: f64 = num(c, "tol");
    for ((v, e), r) in out.values.iter().zip(&expect).zip(&out.normres) {
        assert!((v - e).abs() <= 1e-9 * e.max(1.0), "{name}: {v} vs {e}");
        assert!(*r < tol, "{name}: residual {r}");
    }
    assert_eq!(out.numiter, num::<usize>(c, "numiter"), "{name}: numiter");
    assert_eq!(out.numops, num::<usize>(c, "numops"), "{name}: numops");
}

#[test]
fn matches_krylovkit_fixtures() {
    let cases = cases();
    assert_eq!(cases.len(), 4);
    for c in &cases {
        if num::<u8>(c, "complex") == 1 {
            fixture_case::<Complex64>(c);
        } else {
            fixture_case::<f64>(c);
        }
    }
}
