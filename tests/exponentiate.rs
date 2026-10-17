//! `exponentiate` against nalgebra's dense matrix exponential.

mod common;

use common::*;
use krylov::{Error, Lanczos, Workspace, exponentiate};
use nalgebra::DMatrix;
use num_complex::{Complex32, Complex64};

/// Runs exponentiate and returns (relative error vs dense expm, info).
fn run<S: TestScalar>(
    a: &DMatrix<Complex64>,
    t: Complex64,
    x: &[Complex64],
    alg: Lanczos,
) -> (f64, krylov::ExpInfo<S::Real>) {
    let xv = to_dvec::<S>(x);
    let mut op = DenseOp::<S>::new(a);
    let mut ws = Workspace::with_capacity(&xv, alg.get_krylovdim()).unwrap();
    let (y, info) = exponentiate(&mut op, S::from_c64(t), &xv, &alg, &mut ws).unwrap();
    assert_eq!(info.numops, op.applications);
    let exact = dense_expv(a, t, x);
    (diff_norm(&to_c64(&y), &exact) / norm(&exact), info)
}

#[test]
fn imaginary_time_real_symmetric() {
    let mut g = SplitMix(31);
    // Spectrum in about [-1.2, 1.2] so that exp(t·A) stays O(1) and the
    // absolute tolerance per unit time is attainable within maxiter.
    let a = hermitian(&mut g, 80, false) * Complex64::new(0.12, 0.0);
    let x = random_vec(&mut g, 80, false);
    for (t, kd) in [(-0.1, 30), (-1.0, 8), (-4.0, 10)] {
        let alg = Lanczos::new().krylovdim(kd).tol(1e-12);
        let (err, info) = run::<f64>(&a, Complex64::new(t, 0.0), &x, alg);
        assert_eq!(info.converged, 1, "t = {t}: {info:?}");
        assert!(err < 1e-10, "t = {t}: relative error {err}");
        if t == -4.0 {
            assert!(info.numiter > 1, "long time must take several steps");
        }
    }
}

#[test]
fn real_time_complex_hermitian() {
    let mut g = SplitMix(32);
    let a = hermitian(&mut g, 80, true);
    let x = random_vec(&mut g, 80, true);
    for (tau, kd) in [(0.5, 15), (5.0, 15), (20.0, 30)] {
        let alg = Lanczos::new().krylovdim(kd).tol(1e-12);
        let (err, info) = run::<Complex64>(&a, Complex64::new(0.0, -tau), &x, alg);
        assert_eq!(info.converged, 1, "tau = {tau}: {info:?}");
        assert!(err < 1e-9, "tau = {tau}: relative error {err}");
        if tau == 20.0 {
            assert!(info.numiter > 5);
        }
    }
    // A real symmetric map evolved in real time needs complex vectors.
    let a = hermitian(&mut g, 50, false);
    let x = random_vec(&mut g, 50, false);
    let (err, _) = run::<Complex64>(
        &a,
        Complex64::new(0.0, 3.0),
        &x,
        Lanczos::new().krylovdim(20),
    );
    assert!(err < 1e-10);
}

#[test]
fn complex_time_and_eager() {
    let mut g = SplitMix(33);
    let a = hermitian(&mut g, 60, true);
    let x = random_vec(&mut g, 60, true);
    let t = Complex64::new(-0.3, 1.2);
    let (err, lazy) = run::<Complex64>(&a, t, &x, Lanczos::new().krylovdim(40));
    assert!(err < 1e-10);
    let (err, eager) = run::<Complex64>(&a, t, &x, Lanczos::new().krylovdim(40).eager(true));
    assert!(err < 1e-10);
    assert!(eager.numops < lazy.numops);
}

#[test]
fn small_time_uses_series_branch() {
    // |t·λ| < 1 everywhere: φ functions come from their Taylor series.
    let mut g = SplitMix(34);
    let a = hermitian(&mut g, 40, false) * Complex64::new(0.05, 0.0);
    let x = random_vec(&mut g, 40, false);
    let (err, info) = run::<f64>(
        &a,
        Complex64::new(-1e-3, 0.0),
        &x,
        Lanczos::new().krylovdim(10),
    );
    assert_eq!(info.numiter, 1);
    assert!(err < 1e-14, "relative error {err}");
}

#[test]
fn fixed_point_and_zero_time() {
    // A·x = 0: x is returned after one application.
    let mut a = DMatrix::<Complex64>::zeros(3, 3);
    a[(1, 1)] = Complex64::new(2.0, 0.0);
    let x = [1.0, 0.0, -1.0].map(|v| Complex64::new(v, 0.0));
    let xv = to_dvec::<f64>(&x);
    let mut ws = Workspace::with_capacity(&xv, 3).unwrap();
    let alg = Lanczos::new().krylovdim(3);
    let (y, info) = exponentiate(&mut DenseOp::new(&a), -1.0, &xv, &alg, &mut ws).unwrap();
    assert_eq!(y, xv);
    assert_eq!((info.converged, info.numops, info.numiter), (1, 1, 0));

    let (y, info) = exponentiate(&mut DenseOp::new(&a), 0.0, &xv, &alg, &mut ws).unwrap();
    assert_eq!(y, xv);
    assert_eq!(info.numops, 0);

    // Small invariant subspace: the first Krylov space is exact.
    let mut g = SplitMix(35);
    let eigs: Vec<f64> = (0..30).map(|i| (i % 3) as f64).collect();
    let a = with_spectrum(&mut g, &eigs, false);
    let x = random_vec(&mut g, 30, false);
    let (err, info) = run::<f64>(
        &a,
        Complex64::new(-2.0, 0.0),
        &x,
        Lanczos::new().krylovdim(10),
    );
    assert!(err < 1e-12);
    assert_eq!(info.numiter, 1);
}

#[test]
fn single_precision() {
    let mut g = SplitMix(36);
    let a = hermitian(&mut g, 40, true);
    let x = random_vec(&mut g, 40, true);
    let alg = Lanczos::new().krylovdim(15).tol(1e-5);
    let (err, info) = run::<Complex32>(&a, Complex64::new(0.0, -2.0), &x, alg);
    assert_eq!(info.converged, 1);
    assert!(err < 1e-4, "relative error {err}");
    let a = hermitian(&mut g, 40, false);
    let x = random_vec(&mut g, 40, false);
    let (err, _) = run::<f32>(&a, Complex64::new(-0.5, 0.0), &x, alg);
    assert!(err < 1e-4, "relative error {err}");
}

#[test]
fn warm_call_allocates_only_the_result() {
    let mut g = SplitMix(37);
    let a = hermitian(&mut g, 60, true);
    let x = to_dvec::<Complex64>(&random_vec(&mut g, 60, true));
    let alg = Lanczos::new().krylovdim(10);
    let mut op = DenseOp::new(&a);
    let mut ws = Workspace::with_capacity(&x, 10).unwrap();
    for _ in 0..2 {
        let before = similar_calls();
        let (_, info) =
            exponentiate(&mut op, Complex64::new(0.0, -3.0), &x, &alg, &mut ws).unwrap();
        assert!(info.numiter > 1);
        assert_eq!(similar_calls() - before, 1);
    }
}

#[test]
fn errors() {
    let a = DMatrix::<Complex64>::identity(3, 3);
    let x = DVec(vec![1.0f64, 2.0, 3.0]);
    let mut op = DenseOp::<f64>::new(&a);
    let mut ws = Workspace::with_capacity(&x, 3).unwrap();
    let alg = Lanczos::new().krylovdim(3);
    let e = |r: Result<_, Error<_>>| r.map(|_| ()).unwrap_err();
    assert!(matches!(
        e(exponentiate(&mut op, f64::INFINITY, &x, &alg, &mut ws)),
        Error::InvalidArgument(_)
    ));
    assert!(matches!(
        e(exponentiate(&mut op, 1.0, &x, &alg.tol(0.0), &mut ws)),
        Error::InvalidArgument(_)
    ));
    assert_eq!(
        e(exponentiate(
            &mut op,
            1.0,
            &DVec(vec![f64::NAN; 3]),
            &alg,
            &mut ws
        )),
        Error::NonFiniteStartVector
    );
    assert_eq!(
        e(exponentiate(
            &mut op,
            1.0,
            &x,
            &Lanczos::new().krylovdim(5),
            &mut ws
        )),
        Error::InsufficientCapacity {
            needed: 5,
            capacity: 3
        }
    );
}
