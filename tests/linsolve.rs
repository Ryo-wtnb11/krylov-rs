//! `linsolve` (CG, GMRES) against nalgebra's dense LU and KrylovKit.jl.

mod common;

use std::collections::HashMap;

use common::*;
use krylov::{
    Cg, Error, Gmres, LinearSolver, LinsolveInfo, Orth, RestartResidual, Workspace, linsolve,
};
use nalgebra::{DMatrix, DVector};
use num_complex::{Complex32, Complex64};

/// General matrix filled column by column, as `general` in `linsolve.jl`.
fn general(g: &mut SplitMix, n: usize, complex: bool) -> DMatrix<Complex64> {
    let mut a = DMatrix::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = g.scalar(complex);
        }
    }
    a
}

fn shifted(a: &DMatrix<Complex64>, a0: Complex64, a1: Complex64) -> DMatrix<Complex64> {
    a * a1 + DMatrix::identity(a.nrows(), a.ncols()) * a0
}

fn dense_solve(m: &DMatrix<Complex64>, b: &[Complex64]) -> Vec<Complex64> {
    let x = m
        .clone()
        .lu()
        .solve(&DVector::from_column_slice(b))
        .unwrap();
    x.iter().copied().collect()
}

/// Runs `linsolve` and checks the bookkeeping every result must satisfy:
/// `numops` counts applications and `info.residual` is the true residual.
#[allow(clippy::too_many_arguments)]
fn run<S: TestScalar, L: LinearSolver>(
    a: &DMatrix<Complex64>,
    b: &[Complex64],
    x0: &[Complex64],
    alg: &L,
    a0: Complex64,
    a1: Complex64,
    capacity: usize,
) -> (Vec<Complex64>, Out) {
    let (bv, xv) = (to_dvec::<S>(b), to_dvec::<S>(x0));
    let mut op = DenseOp::<S>::new(a);
    let mut ws = Workspace::with_capacity(&bv, capacity).unwrap();
    let (x, info): (_, LinsolveInfo<DVec<S>>) = linsolve(
        &mut op,
        &bv,
        &xv,
        alg,
        S::from_c64(a0),
        S::from_c64(a1),
        &mut ws,
    )
    .unwrap();
    assert_eq!(info.numops, op.applications);
    let x = to_c64(&x);
    let m = shifted(a, a0, a1);
    let r: Vec<Complex64> = (DVector::from_column_slice(b) - &m * DVector::from_column_slice(&x))
        .iter()
        .copied()
        .collect();
    let scale = norm(b).max(1.0) * (m.norm() * norm(&x)).max(1.0);
    let eps = if std::mem::size_of::<S::Real>() == 4 {
        1e-5
    } else {
        1e-12
    };
    assert!(diff_norm(&r, &to_c64(&info.residual)) <= eps * scale);
    assert!((to_f64(info.normres) - norm(&to_c64(&info.residual))).abs() <= eps * scale);
    let out = Out {
        converged: info.converged,
        numiter: info.numiter,
        numops: info.numops,
        normres: to_f64(info.normres),
    };
    (x, out)
}

/// Scalar-independent summary of a [`LinsolveInfo`].
struct Out {
    converged: usize,
    numiter: usize,
    numops: usize,
    normres: f64,
}

fn rel_err(x: &[Complex64], exact: &[Complex64]) -> f64 {
    diff_norm(x, exact) / norm(exact)
}

#[test]
fn cg_matches_dense_solution_real_and_complex() {
    for complex in [false, true] {
        let mut g = SplitMix(41 + complex as u64);
        let n = 70;
        let eigs: Vec<f64> = (0..n)
            .map(|i| 1.0 + 9.0 * i as f64 / (n - 1) as f64)
            .collect();
        let a = with_spectrum(&mut g, &eigs, complex);
        let b = random_vec(&mut g, n, complex);
        let x0 = random_vec(&mut g, n, complex);
        let (one, zero) = (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
        let exact = dense_solve(&a, &b);
        let alg = Cg::new(500, 1e-11);
        let (x, info) = if complex {
            run::<Complex64, _>(&a, &b, &x0, &alg, zero, one, 0)
        } else {
            run::<f64, _>(&a, &b, &x0, &alg, zero, one, 0)
        };
        assert_eq!(info.converged, 1);
        assert!(info.normres < 1e-11);
        assert!(rel_err(&x, &exact) < 1e-10, "complex {complex}");
        assert_eq!(info.numops, info.numiter + 2, "one start + one recompute");
    }
}

#[test]
fn cg_applies_shift_and_scale_implicitly() {
    // Indefinite A, but 6 + 0.5·A is positive definite.
    let mut g = SplitMix(43);
    let n = 50;
    let eigs: Vec<f64> = (0..n)
        .map(|i| -10.0 + 20.0 * i as f64 / (n - 1) as f64)
        .collect();
    let a = with_spectrum(&mut g, &eigs, true);
    let b = random_vec(&mut g, n, true);
    let (a0, a1) = (Complex64::new(6.0, 0.0), Complex64::new(0.5, 0.0));
    let exact = dense_solve(&shifted(&a, a0, a1), &b);
    let alg = Cg::new(500, 1e-11);
    let (x, info) = run::<Complex64, _>(&a, &b, &vec![Complex64::default(); n], &alg, a0, a1, 0);
    assert_eq!(info.converged, 1);
    assert!(rel_err(&x, &exact) < 1e-10);
}

#[test]
fn gmres_matches_dense_solution_with_restarts() {
    for (complex, orth) in [(false, Orth::Cgs2), (true, Orth::Cgs2), (true, Orth::Mgs2)] {
        let mut g = SplitMix(51 + complex as u64);
        let n = 60;
        let a = general(&mut g, n, complex);
        let b = random_vec(&mut g, n, complex);
        let x0 = random_vec(&mut g, n, complex);
        let (a0, a1) = (Complex64::new(14.0, -1.0), Complex64::new(1.0, 0.5));
        let a0 = if complex {
            a0
        } else {
            Complex64::new(a0.re, 0.0)
        };
        let a1 = if complex {
            a1
        } else {
            Complex64::new(a1.re, 0.0)
        };
        let exact = dense_solve(&shifted(&a, a0, a1), &b);
        let alg = Gmres::new(8, 100, 1e-11).orth(orth);
        let (x, info) = if complex {
            run::<Complex64, _>(&a, &b, &x0, &alg, a0, a1, 8)
        } else {
            run::<f64, _>(&a, &b, &x0, &alg, a0, a1, 8)
        };
        assert_eq!(info.converged, 1);
        assert!(info.numiter > 1, "restarts exercised");
        assert!(rel_err(&x, &exact) < 1e-10, "complex {complex} {orth:?}");
    }
}

#[test]
fn gmres_solves_hermitian_indefinite_in_full_space() {
    let mut g = SplitMix(53);
    let n = 40;
    let a = hermitian(&mut g, n, true);
    let b = random_vec(&mut g, n, true);
    let (a0, a1) = (Complex64::new(0.21, 0.0), Complex64::new(1.0, 0.0));
    let exact = dense_solve(&shifted(&a, a0, a1), &b);
    let alg = Gmres::new(n, 100, 1e-10);
    let (x, info) = run::<Complex64, _>(&a, &b, &vec![Complex64::default(); n], &alg, a0, a1, n);
    assert_eq!(info.converged, 1);
    assert_eq!(info.numiter, 1);
    assert!(rel_err(&x, &exact) < 1e-8);
}

#[test]
fn single_precision_solves() {
    let mut g = SplitMix(61);
    let n = 40;
    let eigs: Vec<f64> = (0..n)
        .map(|i| 1.0 + 4.0 * i as f64 / (n - 1) as f64)
        .collect();
    let a = with_spectrum(&mut g, &eigs, true);
    let b = random_vec(&mut g, n, true);
    let x0 = vec![Complex64::default(); n];
    let (zero, one) = (Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0));
    let exact = dense_solve(&a, &b);
    let (x, info) = run::<Complex32, _>(&a, &b, &x0, &Cg::new(100, 1e-4), zero, one, 0);
    assert_eq!(info.converged, 1);
    assert!(rel_err(&x, &exact) < 1e-4);
    let ar: DMatrix<Complex64> = a.map(|z| Complex64::new(z.re, 0.0));
    let ar = (&ar + ar.transpose()) * Complex64::new(0.5, 0.0);
    let br: Vec<Complex64> = b.iter().map(|z| Complex64::new(z.re, 0.0)).collect();
    let alg = Gmres::new(10, 100, 1e-4);
    let (x, info) = run::<f32, _>(&ar, &br, &x0, &alg, Complex64::new(6.0, 0.0), one, 10);
    assert_eq!(info.converged, 1);
    assert!(
        rel_err(
            &x,
            &dense_solve(&shifted(&ar, Complex64::new(6.0, 0.0), one), &br)
        ) < 1e-4
    );
}

#[test]
fn exact_start_returns_without_iterations() {
    let mut g = SplitMix(71);
    let n = 20;
    let a = general(&mut g, n, false) + DMatrix::identity(n, n) * Complex64::new(10.0, 0.0);
    let x = random_vec(&mut g, n, false);
    let b: Vec<Complex64> = (&a * DVector::from_column_slice(&x))
        .iter()
        .copied()
        .collect();
    let one = Complex64::new(1.0, 0.0);
    let zero = Complex64::new(0.0, 0.0);
    let (_, info) = run::<f64, _>(&a, &b, &x, &Gmres::new(30, 100, 1e-10), zero, one, 30);
    assert_eq!((info.converged, info.numiter, info.numops), (1, 0, 1));
    let a = shifted(&a, zero, one) + a.transpose(); // symmetric, diagonally dominant
    let b: Vec<Complex64> = (&a * DVector::from_column_slice(&x))
        .iter()
        .copied()
        .collect();
    let (_, info) = run::<f64, _>(&a, &b, &x, &Cg::new(100, 1e-10), zero, one, 0);
    assert_eq!((info.converged, info.numiter, info.numops), (1, 0, 1));
}

#[test]
fn non_convergence_is_reported_not_an_error() {
    let mut g = SplitMix(73);
    let n = 60;
    let a = general(&mut g, n, true);
    let b = random_vec(&mut g, n, true);
    let x0 = vec![Complex64::default(); n];
    let (zero, one) = (Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0));
    let alg = Gmres::new(3, 2, 1e-12);
    let (_, info) = run::<Complex64, _>(&a, &b, &x0, &alg, zero, one, 3);
    assert_eq!((info.converged, info.numiter), (0, 2));
    let spd = with_spectrum(
        &mut g,
        &(1..=n).map(|i| i as f64 * i as f64).collect::<Vec<_>>(),
        true,
    );
    let (_, info) = run::<Complex64, _>(&spd, &b, &x0, &Cg::new(3, 1e-12), zero, one, 0);
    assert_eq!((info.converged, info.numiter), (0, 3));
}

#[test]
fn warm_calls_allocate_only_solution_and_residual() {
    let mut g = SplitMix(79);
    let n = 30;
    let a = hermitian(&mut g, n, false) + DMatrix::identity(n, n) * Complex64::new(8.0, 0.0);
    let b = to_dvec::<f64>(&random_vec(&mut g, n, false));
    let x0 = to_dvec::<f64>(&vec![Complex64::default(); n]);
    let mut op = DenseOp::<f64>::new(&a);
    let mut ws = Workspace::with_capacity(&b, 5).unwrap();
    let before = similar_calls();
    let (_, info) = linsolve(
        &mut op,
        &b,
        &x0,
        &Gmres::new(5, 100, 1e-12),
        0.0,
        1.0,
        &mut ws,
    )
    .unwrap();
    assert!(info.numiter > 1);
    assert_eq!(similar_calls() - before, 2);
    let before = similar_calls();
    let (_, info) = linsolve(&mut op, &b, &x0, &Cg::new(100, 1e-12), 0.0, 1.0, &mut ws).unwrap();
    assert_eq!(info.converged, 1);
    assert_eq!(similar_calls() - before, 2);
}

#[test]
fn typed_errors() {
    let mut g = SplitMix(83);
    let n = 12;
    let a = with_spectrum(
        &mut g,
        &(0..n).map(|i| i as f64 - 5.5).collect::<Vec<_>>(),
        true,
    );
    let b = to_dvec::<Complex64>(&random_vec(&mut g, n, true));
    let x0 = to_dvec::<Complex64>(&vec![Complex64::default(); n]);
    let mut op = DenseOp::<Complex64>::new(&a);
    let mut ws = Workspace::with_capacity(&b, 4).unwrap();
    let (one, zero) = (Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0));
    let cg = Cg::new(100, 1e-12);
    let e = linsolve(&mut op, &b, &x0, &cg, zero, one, &mut ws).map(|_| ());
    assert_eq!(e, Err(Error::NotPositiveDefinite));
    let e = linsolve(
        &mut op,
        &b,
        &x0,
        &cg,
        Complex64::new(0.0, 1.0),
        one,
        &mut ws,
    )
    .map(|_| ());
    assert!(matches!(e, Err(Error::InvalidArgument(_))));
    let e = linsolve(&mut op, &b, &x0, &Cg::new(100, -1.0), zero, one, &mut ws).map(|_| ());
    assert!(matches!(e, Err(Error::InvalidArgument(_))));
    let e = linsolve(
        &mut op,
        &b,
        &x0,
        &Gmres::new(5, 100, 1e-12),
        zero,
        one,
        &mut ws,
    )
    .map(|_| ());
    assert_eq!(
        e,
        Err(Error::InsufficientCapacity {
            needed: 5,
            capacity: 4
        })
    );
    let e = linsolve(
        &mut op,
        &b,
        &x0,
        &Gmres::new(0, 100, 1e-12),
        zero,
        one,
        &mut ws,
    )
    .map(|_| ());
    assert!(matches!(e, Err(Error::InvalidArgument(_))));
    let mut bad = to_dvec::<Complex64>(&random_vec(&mut g, n, true));
    bad.0[3] = Complex64::new(f64::NAN, 0.0);
    let e = linsolve(
        &mut op,
        &bad,
        &x0,
        &Gmres::new(4, 100, 1e-12),
        zero,
        one,
        &mut ws,
    )
    .map(|_| ());
    assert_eq!(e, Err(Error::NonFiniteStartVector));
    // The zero map with a₀ = 0: the projected problem is exactly singular.
    let mut zero_op = DenseOp::<Complex64>::new(&DMatrix::zeros(n, n));
    let e = linsolve(
        &mut zero_op,
        &b,
        &x0,
        &Gmres::new(4, 100, 1e-12),
        zero,
        one,
        &mut ws,
    );
    assert_eq!(e.map(|_| ()), Err(Error::SingularKrylovSpace));
}

// KrylovKit fixtures (`fixtures/krylovkit/linsolve.txt`, from `linsolve.jl`).

const FIXTURES: &str = include_str!("fixtures/krylovkit/linsolve.txt");

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

fn complex(c: &HashMap<String, String>, k: &str) -> Complex64 {
    let v = floats(c, k);
    Complex64::new(v[0], v[1])
}

fn fixture_case<S: TestScalar>(c: &HashMap<String, String>, rr: RestartResidual) {
    let cplx = num::<u8>(c, "complex") == 1;
    let n: usize = num(c, "n");
    let mut g = SplitMix(num(c, "seed"));
    let a = match c["matrix"].as_str() {
        "general" => general(&mut g, n, cplx),
        _ => hermitian(&mut g, n, cplx),
    };
    let b = random_vec(&mut g, n, cplx);
    let x0 = random_vec(&mut g, n, cplx);
    let (a0, a1) = (complex(c, "a0"), complex(c, "a1"));
    let (kd, maxiter, tol): (usize, usize, f64) =
        (num(c, "krylovdim"), num(c, "maxiter"), num(c, "tol"));
    let (x, info) = match c["kind"].as_str() {
        "cg" => run::<S, _>(&a, &b, &x0, &Cg::new(maxiter, tol), a0, a1, 0),
        _ => {
            let alg = Gmres::new(kd, maxiter, tol).restart_residual(rr);
            run::<S, _>(&a, &b, &x0, &alg, a0, a1, kd)
        }
    };
    let expect: Vec<Complex64> = floats(c, "x_re")
        .into_iter()
        .zip(floats(c, "x_im"))
        .map(|(re, im)| Complex64::new(re, im))
        .collect();
    let name = &c["case"];
    eprintln!(
        "{name} {rr:?}: krylov numiter {} numops {} normres {:?} | KrylovKit {} {} {}",
        info.numiter, info.numops, info.normres, c["numiter"], c["numops"], c["normres"]
    );
    assert_eq!(info.converged, num::<usize>(c, "converged"), "{name}");
    let rel = rel_err(&x, &expect);
    assert!(rel < 1e-9, "{name}: relative difference {rel}");
    let (ni, no): (usize, usize) = (num(c, "numiter"), num(c, "numops"));
    assert_eq!(info.numiter, ni, "{name}: numiter");
    // KrylovKit does not count the apply of CG's residual recompute (one per
    // run here) nor the apply that starts each GMRES restart cycle.
    // `Recomputed` adds one explicit residual per restart.
    let uncounted = match (c["kind"].as_str(), rr) {
        ("cg", _) => 1,
        (_, RestartResidual::Recurrence) => ni - 1,
        (_, RestartResidual::Recomputed) => 2 * (ni - 1),
    };
    assert_eq!(info.numops, no + uncounted, "{name} {rr:?}: numops");
}

#[test]
fn matches_krylovkit_fixtures() {
    let cases = cases();
    assert_eq!(cases.len(), 5);
    // `Recurrence` is KrylovKit's algorithm; `Recomputed` takes the same
    // cycles here and agrees within the tolerance.
    for rr in [RestartResidual::Recurrence, RestartResidual::Recomputed] {
        for c in &cases {
            if num::<u8>(c, "complex") == 1 {
                fixture_case::<Complex64>(c, rr);
            } else {
                fixture_case::<f64>(c, rr);
            }
        }
    }
}

#[test]
fn gmres_singular_krylov_space() {
    // A = projector onto the first half; b has a unit component outside the
    // range, so min ‖b - A·x‖ = 1. The first cycle detects the singular
    // direction and returns the least-squares solution; a restart from the
    // remaining residual (in the null space) has an exactly singular first
    // pivot, which KrylovKit also rejects (`SingularException`).
    let n = 10;
    let a = DMatrix::from_fn(n, n, |i, j| {
        Complex64::new(if i == j && i < n / 2 { 1.0 } else { 0.0 }, 0.0)
    });
    let mut b = vec![Complex64::default(); n];
    b[0] = Complex64::new(1.0, 0.0);
    b[n / 2] = Complex64::new(1.0, 0.0);
    let x0 = vec![Complex64::default(); n];
    let (zero, one) = (Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0));
    let alg = Gmres::new(4, 1, 1e-10);
    let (x, info) = run::<f64, _>(&a, &b, &x0, &alg, zero, one, 4);
    assert_eq!(info.converged, 0);
    assert!((info.normres - 1.0).abs() < 1e-12, "{}", info.normres);
    assert!((x[0].re - 1.0).abs() < 1e-12, "{:?}", x[0]); // x[n/2] lies in the null space
    let (bv, xv) = (to_dvec::<f64>(&b), to_dvec::<f64>(&x0));
    let mut ws = Workspace::with_capacity(&bv, 4).unwrap();
    let e = linsolve(
        &mut DenseOp::new(&a),
        &bv,
        &xv,
        &Gmres::new(4, 3, 1e-10),
        0.0,
        1.0,
        &mut ws,
    );
    assert_eq!(e.map(|_| ()), Err(Error::SingularKrylovSpace));
}

#[test]
fn gmres_with_subnormal_shift() {
    // (1e-310 + A)·x = e₁ for the rotation A = [0 1; -1 0]: x = [1e-310, 1]
    // up to O(1e-620). The first Givens pivot is the subnormal shift.
    let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, 0.0]).map(|v| Complex64::new(v, 0.0));
    let b = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)];
    let x0 = [Complex64::new(0.0, 0.0); 2];
    let alg = Gmres::new(2, 3, 1e-12);
    let (a0, a1) = (Complex64::new(1e-310, 0.0), Complex64::new(1.0, 0.0));
    let (x, info) = run::<f64, _>(&a, &b, &x0, &alg, a0, a1, 2);
    assert_eq!(info.converged, 1);
    assert!((x[1].re - 1.0).abs() < 1e-15, "{x:?}");
    assert!((x[0].re - 1e-310).abs() < 1e-323, "{x:?}");
}
