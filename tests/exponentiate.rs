//! `exponentiate` and `expintegrator` against nalgebra's dense matrix
//! exponential, and `expintegrator` against KrylovKit fixtures.

mod common;

use common::*;
use krylov::{Error, Lanczos, Workspace, expintegrator, exponentiate};
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

/// `Σ_j tʲ φ_j(tA) u_j` as the first `n` entries of `exp(t·M)·[u₀; e₁]` with
/// `M = [[A, U], [0, N]]`, `U = [u₁ … u_p]` and `N` the lower shift: the
/// augmented state `[x; c]` with `c_j(t) = t^{j-1}/(j-1)!` solves
/// `ẋ = A·x + Σ_j c_j·u_j`.
fn dense_phi_comb(a: &DMatrix<Complex64>, t: Complex64, u: &[Vec<Complex64>]) -> Vec<Complex64> {
    let n = a.nrows();
    let p = (u.len() - 1).max(1);
    let mut m = DMatrix::<Complex64>::zeros(n + p, n + p);
    m.view_mut((0, 0), (n, n)).copy_from(a);
    for (j, uj) in u.iter().enumerate().skip(1) {
        for i in 0..n {
            m[(i, n + j - 1)] = uj[i];
        }
    }
    for j in 1..p {
        m[(n + j, n + j - 1)] = Complex64::new(1.0, 0.0);
    }
    let mut z0 = nalgebra::DVector::<Complex64>::zeros(n + p);
    z0.rows_mut(0, n).copy_from_slice(&u[0]);
    z0[n] = Complex64::new(1.0, 0.0);
    let y = (m * t).exp() * z0;
    y.rows(0, n).iter().copied().collect()
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
            let alg = Lanczos::new().krylovdim(kd);
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
                let alg = Lanczos::new().krylovdim(kd).eager(eager);
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
    let alg = Lanczos::new().krylovdim(15);
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
    let alg = Lanczos::new().krylovdim(15).tol(1e-5);
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
    let alg = Lanczos::new().krylovdim(4);
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
    let alg = Lanczos::new().krylovdim(10);
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
    let alg = Lanczos::new().krylovdim(3);
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
        let alg = Lanczos::new()
            .krylovdim(num("krylovdim"))
            .tol(c["tol"].parse().unwrap())
            .eager(num("eager") == 1)
            .maxiter(100);
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
