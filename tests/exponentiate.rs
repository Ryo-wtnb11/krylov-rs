//! `exponentiate` and `expintegrator` (Lanczos and Arnoldi) against
//! nalgebra's dense matrix exponential and KrylovKit fixtures.

mod common;

use common::*;
use krylov::{Arnoldi, Error, KrylovProcess, Lanczos, Workspace, expintegrator, exponentiate};
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
        let alg = Lanczos::new(kd, 100, 1e-12);
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
        let alg = Lanczos::new(kd, 100, 1e-12);
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
        Lanczos::new(20, 100, 1e-12),
    );
    assert!(err < 1e-10);
}

#[test]
fn complex_time_and_eager() {
    let mut g = SplitMix(33);
    let a = hermitian(&mut g, 60, true);
    let x = random_vec(&mut g, 60, true);
    let t = Complex64::new(-0.3, 1.2);
    let (err, lazy) = run::<Complex64>(&a, t, &x, Lanczos::new(40, 100, 1e-12));
    assert!(err < 1e-10);
    let (err, eager) = run::<Complex64>(&a, t, &x, Lanczos::new(40, 100, 1e-12).eager(true));
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
        Lanczos::new(10, 100, 1e-12),
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
    let alg = Lanczos::new(3, 100, 1e-12);
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
        Lanczos::new(10, 100, 1e-12),
    );
    assert!(err < 1e-12);
    assert_eq!(info.numiter, 1);
}

#[test]
fn single_precision() {
    let mut g = SplitMix(36);
    let a = hermitian(&mut g, 40, true);
    let x = random_vec(&mut g, 40, true);
    let alg = Lanczos::new(15, 100, 1e-5);
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
    let alg = Lanczos::new(10, 100, 1e-12);
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
    let alg = Lanczos::new(3, 100, 1e-12);
    let e = |r: Result<_, Error<_>>| r.map(|_| ()).unwrap_err();
    assert!(matches!(
        e(exponentiate(&mut op, f64::INFINITY, &x, &alg, &mut ws)),
        Error::InvalidArgument(_)
    ));
    assert!(matches!(
        e(exponentiate(
            &mut op,
            1.0,
            &x,
            &Lanczos::new(3, 100, 0.0),
            &mut ws
        )),
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
            &Lanczos::new(5, 100, 1e-12),
            &mut ws
        )),
        Error::InsufficientCapacity {
            needed: 5,
            capacity: 3
        }
    );
}

/// Runs expintegrator and returns (relative error vs the dense oracle, info).
fn run_int<S: TestScalar>(
    a: &DMatrix<Complex64>,
    t: Complex64,
    u: &[Vec<Complex64>],
    alg: Lanczos,
) -> (f64, krylov::ExpInfo<S::Real>) {
    let uv: Vec<DVec<S>> = u.iter().map(|x| to_dvec::<S>(x)).collect();
    let mut op = DenseOp::<S>::new(a);
    let mut ws = Workspace::with_capacity(&uv[0], alg.get_krylovdim()).unwrap();
    let (y, info) = expintegrator(&mut op, S::from_c64(t), &uv, &alg, &mut ws).unwrap();
    assert_eq!(info.numops, op.applications);
    let exact = dense_phi_comb(a, t, u);
    (diff_norm(&to_c64(&y), &exact) / norm(&exact), info)
}

#[test]
fn phi_combinations_match_dense_oracle() {
    let mut g = SplitMix(41);
    let a = hermitian(&mut g, 70, false) * Complex64::new(0.15, 0.0);
    let u: Vec<_> = (0..5).map(|_| random_vec(&mut g, 70, false)).collect();
    for nu in 2..=5 {
        for (t, kd) in [(-0.05, 20), (-1.0, 10), (-3.0, 12)] {
            let alg = Lanczos::new(kd, 100, 1e-12);
            let (err, info) = run_int::<f64>(&a, Complex64::new(t, 0.0), &u[..nu], alg);
            assert_eq!(info.converged, 1, "nu={nu} t={t}: {info:?}");
            assert!(err < 1e-10, "nu={nu} t={t}: relative error {err}");
        }
    }
    let a = hermitian(&mut g, 70, true);
    let u: Vec<_> = (0..4).map(|_| random_vec(&mut g, 70, true)).collect();
    for nu in 2..=4 {
        for (t, kd) in [
            (Complex64::new(0.0, -4.0), 15),
            (Complex64::new(-0.2, 1.0), 25),
        ] {
            for eager in [false, true] {
                let alg = Lanczos::new(kd, 100, 1e-12).eager(eager);
                let (err, info) = run_int::<Complex64>(&a, t, &u[..nu], alg);
                assert_eq!(info.converged, 1, "nu={nu} t={t}: {info:?}");
                assert!(err < 1e-9, "nu={nu} t={t}: relative error {err}");
                if t.im == -4.0 && !eager {
                    assert!(info.numiter > 1);
                }
            }
        }
    }
}

#[test]
fn exponentiate_is_the_single_vector_case() {
    let mut g = SplitMix(42);
    let a = hermitian(&mut g, 50, true);
    let x = random_vec(&mut g, 50, true);
    let zero = vec![Complex64::new(0.0, 0.0); 50];
    let t = Complex64::new(0.0, -2.0);
    let alg = Lanczos::new(15, 100, 1e-12);
    let (e1, i1) = run_int::<Complex64>(&a, t, std::slice::from_ref(&x), alg);
    let (e2, i2) = run_int::<Complex64>(&a, t, &[x, zero], alg);
    assert!(e1 < 1e-10 && e2 < 1e-10);
    assert_eq!((i1.numiter, i1.numops), (i2.numiter, i2.numops));
}

#[test]
fn phi_combinations_single_precision() {
    let mut g = SplitMix(43);
    let a = hermitian(&mut g, 40, true);
    let u: Vec<_> = (0..3).map(|_| random_vec(&mut g, 40, true)).collect();
    let alg = Lanczos::new(15, 100, 1e-5);
    let (err, info) = run_int::<Complex32>(&a, Complex64::new(0.0, -2.0), &u, alg);
    assert_eq!(info.converged, 1);
    assert!(err < 1e-4, "relative error {err}");
    let a = hermitian(&mut g, 40, false) * Complex64::new(0.2, 0.0);
    let u: Vec<_> = (0..3).map(|_| random_vec(&mut g, 40, false)).collect();
    let (err, _) = run_int::<f32>(&a, Complex64::new(-0.5, 0.0), &u, alg);
    assert!(err < 1e-4, "relative error {err}");
}

#[test]
fn exact_polynomial_and_fixed_point() {
    // A = 0, u₂ = 0: w₃ = A·w₂ + u₂ vanishes, y = u₀ + t·u₁ exactly.
    let mut g = SplitMix(44);
    let n = 8;
    let a = DMatrix::<Complex64>::zeros(n, n);
    let u = [
        random_vec(&mut g, n, false),
        random_vec(&mut g, n, false),
        vec![Complex64::new(0.0, 0.0); n],
    ];
    let uv: Vec<DVec<f64>> = u.iter().map(|x| to_dvec(x)).collect();
    let mut ws = Workspace::with_capacity(&uv[0], 4).unwrap();
    let alg = Lanczos::new(4, 100, 1e-12);
    let (y, info) = expintegrator(&mut DenseOp::new(&a), -1.5, &uv, &alg, &mut ws).unwrap();
    for i in 0..n {
        assert!((y.0[i] - (uv[0].0[i] - 1.5 * uv[1].0[i])).abs() < 1e-15);
    }
    assert_eq!((info.converged, info.numiter, info.numops), (1, 0, 2));
    assert_eq!(info.normres, 0.0);

    // p = 1 with A·u₀ + u₁ = 0: u₀ is a fixed point.
    let a = hermitian(&mut g, n, false);
    let u0 = random_vec(&mut g, n, false);
    let u1: Vec<_> = (&a * nalgebra::DVector::from_column_slice(&u0) * Complex64::new(-1.0, 0.0))
        .iter()
        .copied()
        .collect();
    let uv = [to_dvec::<f64>(&u0), to_dvec::<f64>(&u1)];
    let (y, info) = expintegrator(&mut DenseOp::new(&a), 2.0, &uv, &alg, &mut ws).unwrap();
    assert_eq!(y, uv[0]);
    assert_eq!((info.converged, info.numiter, info.numops), (1, 0, 1));
}

#[test]
fn warm_expintegrator_allocates_p_vectors() {
    let mut g = SplitMix(45);
    let a = hermitian(&mut g, 60, true);
    let u: Vec<_> = (0..4)
        .map(|_| to_dvec::<Complex64>(&random_vec(&mut g, 60, true)))
        .collect();
    let alg = Lanczos::new(10, 100, 1e-12);
    let mut op = DenseOp::new(&a);
    let mut ws = Workspace::with_capacity(&u[0], 10).unwrap();
    for _ in 0..2 {
        let before = similar_calls();
        let (_, info) =
            expintegrator(&mut op, Complex64::new(0.0, -3.0), &u, &alg, &mut ws).unwrap();
        assert!(info.numiter > 1);
        assert_eq!(similar_calls() - before, 3);
    }
}

#[test]
fn expintegrator_errors() {
    let a = DMatrix::<Complex64>::identity(3, 3);
    let x = DVec(vec![1.0f64, 2.0, 3.0]);
    let mut op = DenseOp::<f64>::new(&a);
    let mut ws = Workspace::with_capacity(&x, 3).unwrap();
    let alg = Lanczos::new(3, 100, 1e-12);
    let e = |r: Result<_, Error<_>>| r.map(|_| ()).unwrap_err();
    assert!(matches!(
        e(expintegrator(&mut op, 1.0, &[], &alg, &mut ws)),
        Error::InvalidArgument(_)
    ));
    let u = [
        DVec(vec![1.0; 3]),
        DVec(vec![0.0; 3]),
        DVec(vec![f64::NAN; 3]),
    ];
    assert_eq!(
        e(expintegrator(&mut op, 1.0, &u, &alg, &mut ws)),
        Error::NonFiniteStartVector
    );
}

#[test]
fn matches_krylovkit_expintegrator_fixtures() {
    // Format of fixtures/krylovkit/expint.txt (see tests/krylovkit.rs).
    let mut cases = Vec::new();
    let mut cur = std::collections::HashMap::new();
    for line in include_str!("fixtures/krylovkit/expint.txt").lines() {
        if line.starts_with('#') {
            continue;
        }
        if line == "end" {
            cases.push(std::mem::take(&mut cur));
            continue;
        }
        let (k, v) = line.split_once(' ').unwrap_or((line, ""));
        cur.insert(k.to_string(), v.to_string());
    }
    assert!(cases.len() >= 4);
    let floats =
        |s: &str| -> Vec<f64> { s.split_whitespace().map(|x| x.parse().unwrap()).collect() };
    for c in &cases {
        let num = |k: &str| -> usize { c[k].parse().unwrap() };
        let complex = num("complex") == 1;
        let n = num("n");
        let mut g = SplitMix(c["seed"].parse().unwrap());
        let a = hermitian(&mut g, n, complex);
        let u: Vec<_> = (0..num("nu"))
            .map(|_| random_vec(&mut g, n, complex))
            .collect();
        let t = floats(&c["t"]);
        let t = Complex64::new(t[0], t[1]);
        let alg =
            Lanczos::new(num("krylovdim"), 100, c["tol"].parse().unwrap()).eager(num("eager") == 1);
        let uv: Vec<DVec<Complex64>> = u.iter().map(|x| to_dvec(x)).collect();
        let mut ws = Workspace::with_capacity(&uv[0], alg.get_krylovdim()).unwrap();
        let (y, info) = expintegrator(&mut DenseOp::new(&a), t, &uv, &alg, &mut ws).unwrap();
        let expect: Vec<Complex64> = floats(&c["y_re"])
            .into_iter()
            .zip(floats(&c["y_im"]))
            .map(|(re, im)| Complex64::new(re, im))
            .collect();
        let name = &c["case"];
        eprintln!(
            "{name}: krylov numiter {} numops {} err {:?} | KrylovKit {} {} {}",
            info.numiter, info.numops, info.normres, c["numiter"], c["numops"], c["normres"]
        );
        let rel = diff_norm(&to_c64(&y), &expect) / norm(&expect);
        assert!(rel < 1e-10, "{name}: relative difference {rel}");
        let oracle = dense_phi_comb(&a, t, &u);
        assert!(
            diff_norm(&expect, &oracle) / norm(&oracle) < 1e-10,
            "{name}: KrylovKit vs dense"
        );
        assert_eq!(info.converged, num("converged"), "{name}");
        let (ni, no) = (num("numiter"), num("numops"));
        assert!(
            info.numiter.abs_diff(ni) * 10 <= ni.max(10),
            "{name}: numiter {} vs {ni}",
            info.numiter
        );
        assert!(
            info.numops.abs_diff(no) * 10 <= no,
            "{name}: numops {} vs {no}",
            info.numops
        );
    }
}

/// Parses a `fixtures/krylovkit/*.txt` file (see tests/krylovkit.rs).
fn fixture_cases(text: &str) -> Vec<std::collections::HashMap<String, String>> {
    let mut cases = Vec::new();
    let mut cur = std::collections::HashMap::new();
    for line in text.lines().filter(|l| !l.starts_with('#')) {
        if line == "end" {
            cases.push(std::mem::take(&mut cur));
            continue;
        }
        let (k, v) = line.split_once(' ').unwrap_or((line, ""));
        cur.insert(k.to_string(), v.to_string());
    }
    cases
}

/// Runs expintegrator with any process; returns (y, relative error vs the
/// dense oracle, info).
fn run_alg<S: TestScalar, L: KrylovProcess>(
    a: &DMatrix<Complex64>,
    t: Complex64,
    u: &[Vec<Complex64>],
    alg: &L,
    capacity: usize,
) -> (Vec<Complex64>, f64, krylov::ExpInfo<S::Real>) {
    let uv: Vec<DVec<S>> = u.iter().map(|x| to_dvec::<S>(x)).collect();
    let mut op = DenseOp::<S>::new(a);
    let mut ws = Workspace::with_capacity(&uv[0], capacity).unwrap();
    let (y, info) = expintegrator(&mut op, S::from_c64(t), &uv, alg, &mut ws).unwrap();
    assert_eq!(info.numops, op.applications);
    let y = to_c64(&y);
    let exact = dense_phi_comb(a, t, u);
    let err = diff_norm(&y, &exact) / norm(&exact);
    (y, err, info)
}

#[test]
fn arnoldi_matches_krylovkit_fixtures() {
    let cases = fixture_cases(include_str!("fixtures/krylovkit/expint_arnoldi.txt"));
    assert_eq!(cases.len(), 10);
    let floats =
        |s: &str| -> Vec<f64> { s.split_whitespace().map(|x| x.parse().unwrap()).collect() };
    for c in &cases {
        let num = |k: &str| -> usize { c[k].parse().unwrap() };
        let name = &c["case"];
        let complex = num("complex") == 1;
        let n = num("n");
        let mut g = SplitMix(c["seed"].parse().unwrap());
        let a = outliers(&mut g, n, complex);
        let u: Vec<_> = (0..num("nu"))
            .map(|_| random_vec(&mut g, n, complex))
            .collect();
        let t = floats(&c["t"]);
        let t = Complex64::new(t[0], t[1]);
        let alg =
            Arnoldi::new(num("krylovdim"), 100, c["tol"].parse().unwrap()).eager(num("eager") == 1);
        let kd = alg.get_krylovdim();
        // Real vectors for real t and a real map, as in the fixture.
        let (y, oracle_err, info) = if complex {
            run_alg::<Complex64, _>(&a, t, &u, &alg, kd)
        } else {
            let (y, e, i) = run_alg::<f64, _>(&a, t, &u, &alg, kd);
            (y, e, i)
        };
        eprintln!(
            "{name}: krylov numiter {} numops {} converged {} err {:?} | KrylovKit {} {} {} {}",
            info.numiter,
            info.numops,
            info.converged,
            info.normres,
            c["numiter"],
            c["numops"],
            c["converged"],
            c["normres"]
        );
        let expect: Vec<Complex64> = floats(&c["y_re"])
            .into_iter()
            .zip(floats(&c["y_im"]))
            .map(|(re, im)| Complex64::new(re, im))
            .collect();
        let rel = diff_norm(&y, &expect) / norm(&expect);
        assert!(rel < 1e-10, "{name}: relative difference {rel}");
        let oracle = dense_phi_comb(&a, t, &u);
        let kk_err = diff_norm(&expect, &oracle) / norm(&oracle);
        assert!(kk_err < 1e-9, "{name}: KrylovKit vs dense {kk_err}");
        assert!(oracle_err < 1e-9, "{name}: krylov vs dense {oracle_err}");
        let kk_normres: f64 = c["normres"].parse().unwrap();
        assert!(
            (info.normres - kk_normres).abs() <= 1e-6 * kk_normres,
            "{name}: normres {} vs {kk_normres}",
            info.normres
        );
        // Exact KrylovKit counts (declared offset 0).
        assert_eq!(info.converged, num("converged"), "{name}: converged");
        assert_eq!(info.numiter, num("numiter"), "{name}: numiter");
        assert_eq!(info.numops, num("numops"), "{name}: numops");
        assert_eq!(info.max_imag, 0.0);
    }
}

/// `D + N` with `D` diagonal and `N` strictly upper triangular of size
/// `offdiag`: non-normal with a known spectrum, conjugated by a random
/// unitary so that the Krylov space sees a dense matrix.
fn non_normal(g: &mut SplitMix, n: usize, complex: bool, offdiag: f64) -> DMatrix<Complex64> {
    let mut t = DMatrix::<Complex64>::zeros(n, n);
    for j in 0..n {
        for i in 0..j {
            t[(i, j)] = g.scalar(complex) * offdiag / (n as f64).sqrt();
        }
        t[(j, j)] = g.scalar(complex);
    }
    let q = unitary(g, n, complex);
    &q * t * q.adjoint()
}

#[test]
fn arnoldi_matches_dense_oracle_on_non_normal_maps() {
    let mut g = SplitMix(71);
    for complex in [false, true] {
        for offdiag in [1.0, 5.0] {
            let a = non_normal(&mut g, 60, complex, offdiag);
            let u: Vec<_> = (0..4).map(|_| random_vec(&mut g, 60, complex)).collect();
            let ts: &[Complex64] = if complex {
                &[
                    Complex64::new(0.0, -2.0),
                    Complex64::new(-0.5, 1.5),
                    Complex64::new(-1.0, 0.0),
                ]
            } else {
                &[
                    Complex64::new(-0.1, 0.0),
                    Complex64::new(-1.5, 0.0),
                    Complex64::new(1.0, 0.0),
                ]
            };
            for nu in 1..=4 {
                for &t in ts {
                    for (kd, eager) in [(20, false), (10, true)] {
                        let alg = Arnoldi::new(kd, 100, 1e-12).eager(eager);
                        let (_, err, info) = if complex {
                            run_alg::<Complex64, _>(&a, t, &u[..nu], &alg, kd)
                        } else {
                            run_alg::<f64, _>(&a, t, &u[..nu], &alg, kd)
                        };
                        let tag =
                            format!("complex={complex} offdiag={offdiag} nu={nu} t={t} kd={kd}");
                        assert_eq!(info.converged, 1, "{tag}: {info:?}");
                        assert!(err < 1e-9, "{tag}: relative error {err}");
                        assert_eq!(info.max_imag, 0.0);
                    }
                }
            }
        }
    }
}

#[test]
fn arnoldi_agrees_with_lanczos_on_hermitian_maps() {
    let mut g = SplitMix(72);
    let a = hermitian(&mut g, 50, true);
    let u: Vec<_> = (0..3).map(|_| random_vec(&mut g, 50, true)).collect();
    let t = Complex64::new(0.0, -2.0);
    let (yl, el, _) = run_alg::<Complex64, _>(&a, t, &u, &Lanczos::new(15, 100, 1e-12), 15);
    let (ya, ea, _) = run_alg::<Complex64, _>(&a, t, &u, &Arnoldi::new(15, 100, 1e-12), 15);
    assert!(el < 1e-10 && ea < 1e-10);
    assert!(diff_norm(&yl, &ya) / norm(&yl) < 1e-10);
}

#[test]
fn arnoldi_single_precision() {
    let mut g = SplitMix(73);
    let a = non_normal(&mut g, 40, true, 1.0);
    let u: Vec<_> = (0..3).map(|_| random_vec(&mut g, 40, true)).collect();
    let alg = Arnoldi::new(15, 100, 1e-5);
    let (_, err, info) = run_alg::<Complex32, _>(&a, Complex64::new(0.0, -2.0), &u, &alg, 15);
    assert_eq!(info.converged, 1);
    assert!(err < 1e-4, "relative error {err}");
    let a = non_normal(&mut g, 40, false, 1.0);
    let u: Vec<_> = (0..2).map(|_| random_vec(&mut g, 40, false)).collect();
    let (_, err, info) = run_alg::<f32, _>(&a, Complex64::new(-0.5, 0.0), &u, &alg, 15);
    assert_eq!(info.converged, 1);
    assert!(err < 1e-4, "relative error {err}");
}

#[test]
fn arnoldi_exact_polynomial_and_fixed_point() {
    let mut g = SplitMix(74);
    let n = 8;
    let a = DMatrix::<Complex64>::zeros(n, n);
    let u = [
        random_vec(&mut g, n, false),
        random_vec(&mut g, n, false),
        vec![Complex64::new(0.0, 0.0); n],
    ];
    let uv: Vec<DVec<f64>> = u.iter().map(|x| to_dvec(x)).collect();
    let mut ws = Workspace::with_capacity(&uv[0], 4).unwrap();
    let alg = Arnoldi::new(4, 100, 1e-12);
    let (y, info) = expintegrator(&mut DenseOp::new(&a), -1.5, &uv, &alg, &mut ws).unwrap();
    for i in 0..n {
        assert!((y.0[i] - (uv[0].0[i] - 1.5 * uv[1].0[i])).abs() < 1e-15);
    }
    assert_eq!((info.converged, info.numiter, info.numops), (1, 0, 2));

    // Invariant subspace: a map with three distinct eigenvalues (non-normal)
    // gives an exact Krylov space of dimension 3 and one step.
    let mut t = DMatrix::<Complex64>::zeros(30, 30);
    for i in 0..30 {
        t[(i, i)] = Complex64::new((i % 3) as f64, 0.0);
    }
    let q = unitary(&mut g, 30, false);
    let s = DMatrix::from_fn(30, 30, |i, j| {
        Complex64::new(
            if i == j {
                2.0
            } else if i < j {
                0.3
            } else {
                0.0
            },
            0.0,
        )
    });
    let a = &q * &s * t * s.try_inverse().unwrap() * q.adjoint();
    let x = random_vec(&mut g, 30, false);
    let alg = Arnoldi::new(10, 100, 1e-12);
    let (_, err, info) = run_alg::<f64, _>(&a, Complex64::new(-1.0, 0.0), &[x], &alg, 10);
    assert!(err < 1e-10, "relative error {err}");
    assert_eq!(info.numiter, 1);
}

#[test]
fn arnoldi_warm_calls_allocate_only_the_result_and_scratch() {
    let mut g = SplitMix(75);
    let a = non_normal(&mut g, 60, true, 1.0);
    let u: Vec<_> = (0..4)
        .map(|_| to_dvec::<Complex64>(&random_vec(&mut g, 60, true)))
        .collect();
    let alg = Arnoldi::new(10, 100, 1e-12);
    let mut op = DenseOp::new(&a);
    let mut ws = Workspace::with_capacity(&u[0], 10).unwrap();
    for _ in 0..2 {
        let before = similar_calls();
        let (_, info) =
            exponentiate(&mut op, Complex64::new(0.0, -3.0), &u[0], &alg, &mut ws).unwrap();
        assert!(info.numiter > 1);
        assert_eq!(similar_calls() - before, 1);
        let before = similar_calls();
        let (_, info) =
            expintegrator(&mut op, Complex64::new(0.0, -3.0), &u, &alg, &mut ws).unwrap();
        assert!(info.numiter > 1);
        assert_eq!(similar_calls() - before, 3);
    }
}

#[test]
fn arnoldi_errors() {
    let a = DMatrix::<Complex64>::identity(3, 3);
    let x = DVec(vec![1.0f64, 2.0, 3.0]);
    let mut op = DenseOp::<f64>::new(&a);
    let mut ws = Workspace::with_capacity(&x, 3).unwrap();
    let alg = Arnoldi::new(3, 100, 1e-12);
    let e = |r: Result<_, Error<_>>| r.map(|_| ()).unwrap_err();
    assert!(matches!(
        e(expintegrator(&mut op, 1.0, &[], &alg, &mut ws)),
        Error::InvalidArgument(_)
    ));
    assert!(matches!(
        e(exponentiate(&mut op, f64::INFINITY, &x, &alg, &mut ws)),
        Error::InvalidArgument(_)
    ));
    for bad in [
        Arnoldi::new(3, 100, 0.0),
        Arnoldi::new(3, 100, f64::NAN),
        Arnoldi::new(0, 100, 1e-12),
        Arnoldi::new(3, 0, 1e-12),
    ] {
        assert!(matches!(
            e(exponentiate(&mut op, 1.0, &x, &bad, &mut ws)),
            Error::InvalidArgument(_)
        ));
    }
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
            &Arnoldi::new(5, 100, 1e-12),
            &mut ws
        )),
        Error::InsufficientCapacity {
            needed: 5,
            capacity: 3
        }
    );
    // A map producing NaN is a typed error, not a NaN result.
    let mut nan_op = |_: &DVec<f64>, y: &mut DVec<f64>| {
        y.0.fill(f64::NAN);
        Ok::<(), std::convert::Infallible>(())
    };
    assert_eq!(
        e(exponentiate(&mut nan_op, 1.0, &x, &alg, &mut ws)),
        Error::NonFiniteIterate
    );
}

/// The sealed entry point stays callable through a generic bound; it must
/// reject an empty `u` like the public function.
fn sealed_entry<L: KrylovProcess>(alg: &L) -> Error<std::convert::Infallible> {
    let a = DMatrix::<Complex64>::identity(3, 3);
    let x = DVec(vec![1.0f64; 3]);
    let mut ws = Workspace::with_capacity(&x, 3).unwrap();
    let empty: &[DVec<f64>] = &[];
    alg.expintegrator(&mut DenseOp::<f64>::new(&a), 1.0, empty, &mut ws)
        .map(|_| ())
        .unwrap_err()
}

#[test]
fn sealed_entry_rejects_empty_u() {
    for e in [
        sealed_entry(&Lanczos::new(3, 100, 1e-12)),
        sealed_entry(&Arnoldi::new(3, 100, 1e-12)),
    ] {
        assert!(matches!(e, Error::InvalidArgument(_)));
    }
}

/// `A = Q·diag(eigs)·Qᵀ` (real orthogonal `Q`) with `λ_max = 1`, a start
/// vector `x = 10⁻³·r` and the dense `exp(t·A)·x`.
fn growing(t: f64) -> (DMatrix<Complex64>, Vec<Complex64>, Vec<Complex64>) {
    let mut g = SplitMix(91);
    let eigs = [1.0, 0.5, 0.2, 0.0, -0.3, -0.6, -0.8, -1.0];
    let a = with_spectrum(&mut g, &eigs, false);
    let x: Vec<_> = random_vec(&mut g, eigs.len(), false)
        .iter()
        .map(|z| z * 1e-3)
        .collect();
    let exact = dense_expv(&a, Complex64::new(t, 0.0), &x);
    (a, x, exact)
}

#[test]
fn overflowing_step_is_rejected_and_shrunk() {
    // Single precision: e^{t·λ_max} = e^92 overflows f32 while
    // ‖exp(t·A)·x‖ ≈ 10³⁷ does not. The full step's estimate is ∞ (NaN for
    // complex vectors before the exact-real φ evaluation); it must be
    // rejected, and the shrunk steps reach the finite result.
    let t = 92.0;
    let (a, x, exact) = growing(t);
    let t = Complex64::new(t, 0.0);
    let u = std::slice::from_ref(&x);
    let check = |y: Vec<Complex64>, numiter: usize, what: &str| {
        assert!(
            y.iter().all(|z| z.re.is_finite() && z.im.is_finite()),
            "{what}: {y:?}"
        );
        let err = diff_norm(&y, &exact) / norm(&exact);
        assert!(err < 1e-3, "{what}: relative error {err}");
        assert!(numiter > 1, "{what}: the overflowing step must be split");
    };
    let lanczos = Lanczos::new(8, 100, 1e-6);
    let arnoldi = Arnoldi::new(8, 100, 1e-6);
    let (y, _, info) = run_alg::<Complex32, _>(&a, t, u, &lanczos, 8);
    check(y, info.numiter, "lanczos c32");
    let (y, _, info) = run_alg::<f32, _>(&a, t, u, &lanczos, 8);
    check(y, info.numiter, "lanczos f32");
    let (y, _, info) = run_alg::<Complex32, _>(&a, t, u, &arnoldi, 8);
    check(y, info.numiter, "arnoldi c32");
    let (y, _, info) = run_alg::<f32, _>(&a, t, u, &arnoldi, 8);
    check(y, info.numiter, "arnoldi f32");
}

#[test]
fn overflowing_result_is_a_typed_error() {
    // ‖exp(t·A)·x‖ ≈ e^800·10⁻³ overflows f64: no step size helps.
    let (a, x, _) = growing(800.0);
    fn go<S: TestScalar, L: KrylovProcess>(
        a: &DMatrix<Complex64>,
        x: &[Complex64],
        alg: &L,
    ) -> Error<std::convert::Infallible> {
        let xv = to_dvec::<S>(x);
        let mut ws = Workspace::with_capacity(&xv, 8).unwrap();
        let t = S::from_c64(Complex64::new(800.0, 0.0));
        match exponentiate(&mut DenseOp::<S>::new(a), t, &xv, alg, &mut ws) {
            Ok((y, info)) => panic!("{:?} {info:?}", &y.0[..2]),
            Err(e) => e,
        }
    }
    // maxiter = 1 forces the single step whose estimate is ∞; otherwise the
    // estimate at the smallest step, the iterate or the result overflows.
    for maxiter in [1, 100] {
        let lanczos = Lanczos::new(8, maxiter, 1e-12);
        let arnoldi = Arnoldi::new(8, maxiter, 1e-12);
        for e in [
            go::<Complex64, _>(&a, &x, &lanczos),
            go::<f64, _>(&a, &x, &lanczos),
            go::<Complex64, _>(&a, &x, &arnoldi),
            go::<f64, _>(&a, &x, &arnoldi),
        ] {
            assert_eq!(e, Error::NonFiniteIterate, "maxiter {maxiter}");
        }
    }
}

#[test]
fn large_finite_result_with_a_naive_norm_is_returned() {
    // y = e^{λt}·x is finite, but ‖y‖² overflows the vector's own norm.
    fn go<S: TestScalar>(lambda: f64, alg: &impl KrylovProcess, rtol: f64) {
        let x = Naive(vec![S::from_c64(Complex64::new(1e-10, 0.0)); 4]);
        let mut op = |v: &Naive<S>, y: &mut Naive<S>| {
            for (yi, vi) in y.0.iter_mut().zip(&v.0) {
                *yi = *vi * S::from_c64(Complex64::new(lambda, 0.0));
            }
            Ok::<(), std::convert::Infallible>(())
        };
        let mut ws = Workspace::with_capacity(&x, 8).unwrap();
        let (y, _) = exponentiate(&mut op, S::one(), &x, alg, &mut ws).unwrap();
        let want = 1e-10 * lambda.exp();
        for yi in &y.0 {
            let got = yi.to_c64();
            assert!(
                (got - want).norm() <= rtol * want,
                "λ = {lambda}: {got} vs {want}"
            );
        }
    }
    let (lanczos, arnoldi) = (Lanczos::new(8, 100, 1e-12), Arnoldi::new(8, 100, 1e-12));
    go::<f64>(400.0, &lanczos, 1e-12);
    go::<f64>(400.0, &arnoldi, 1e-12);
    go::<Complex64>(400.0, &lanczos, 1e-12);
    go::<f32>(80.0, &lanczos, 1e-3);
    go::<f32>(80.0, &arnoldi, 1e-3);
    go::<Complex32>(80.0, &lanczos, 1e-3);
}
