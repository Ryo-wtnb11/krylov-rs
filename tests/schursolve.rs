//! Krylov–Schur (`schursolve`, `arnoldi_eigsolve`, `realeigsolve`) against
//! dense oracles and KrylovKit.

mod common;

use std::collections::HashMap;

use common::*;
use krylov::{
    Arnoldi, BlockLanczos, Error, Lanczos, Which, Workspace, arnoldi_eigsolve, block_eigsolve,
    eigsolve, realeigsolve, schursolve,
};
use nalgebra::DMatrix;
use num_complex::{Complex32, Complex64};

/// KrylovKit's `outliers` (see `fixtures/krylovkit/arnoldi.jl`).
fn outliers(g: &mut SplitMix, n: usize, complex: bool) -> DMatrix<Complex64> {
    let mut a = DMatrix::zeros(n, n);
    let s = (n as f64).sqrt();
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = g.scalar(complex) / s;
        }
    }
    let c = Complex64::new;
    if complex {
        a[(0, 0)] += c(0.0, 3.0);
        a[(1, 1)] += c(-2.5, 0.0);
        a[(2, 2)] += c(2.0, 1.0);
        a[(3, 3)] += c(-1.5, -1.5);
    } else {
        a[(0, 0)] += c(3.0, 0.0);
        a[(1, 1)] += c(-2.5, 0.0);
        a[(2, 2)] += c(2.0, 0.0);
        a[(3, 3)] += c(1.5, 0.0);
        a[(4, 4)] += c(1.5, 0.0);
        a[(3, 4)] += c(1.2, 0.0);
        a[(4, 3)] -= c(1.2, 0.0);
    }
    a
}

/// KrylovKit's `permuted_triangular`: real spectrum `diag(T0)`.
fn permuted_triangular(g: &mut SplitMix, n: usize) -> (DMatrix<Complex64>, Vec<f64>) {
    let mut t0 = DMatrix::<f64>::zeros(n, n);
    for j in 0..n {
        for i in 0..=j {
            t0[(i, j)] = if i == j { 4.0 * g.unif() } else { g.unif() };
        }
    }
    let keys: Vec<f64> = (0..n).map(|_| g.unif()).collect();
    let mut p: Vec<usize> = (0..n).collect();
    p.sort_by(|&a, &b| keys[a].partial_cmp(&keys[b]).unwrap());
    let a = DMatrix::from_fn(n, n, |i, j| Complex64::new(t0[(p[i], p[j])], 0.0));
    (a, (0..n).map(|i| t0[(i, i)]).collect())
}

fn dense_spectrum(a: &DMatrix<Complex64>) -> Vec<Complex64> {
    a.clone()
        .schur()
        .eigenvalues()
        .unwrap()
        .iter()
        .copied()
        .collect()
}

fn nearest(spec: &[Complex64], l: Complex64) -> f64 {
    spec.iter()
        .map(|x| (x - l).norm())
        .fold(f64::INFINITY, f64::min)
}

fn apply(a: &DMatrix<Complex64>, x: &[Complex64]) -> Vec<Complex64> {
    (a * nalgebra::DVector::from_column_slice(x))
        .iter()
        .copied()
        .collect()
}

struct Run {
    values: Vec<Complex64>,
    normres: Vec<f64>,
    converged: usize,
    numiter: usize,
    numops: usize,
}

/// `schursolve`, with the Schur-relation and orthonormality checks.
fn run_schur<S: TestScalar>(
    a: &DMatrix<Complex64>,
    x0: &[Complex64],
    howmany: usize,
    which: Which,
    alg: &Arnoldi,
) -> Run {
    let x0 = to_dvec::<S>(x0);
    let mut op = DenseOp::<S>::new(a);
    let mut ws = Workspace::with_capacity(&x0, alg.get_krylovdim()).unwrap();
    let (t, q, vals, info) = schursolve(&mut op, &x0, howmany, which, alg, &mut ws).unwrap();
    let n = q.len();
    assert_eq!((t.len(), vals.len(), info.normres.len()), (n * n, n, n));
    assert_eq!(info.numops, op.applications);
    let qs: Vec<Vec<Complex64>> = q.iter().map(to_c64).collect();
    assert!(orthogonality_error(&q) < 1e-10);
    for i in 0..n {
        // A·qᵢ − Σ_j qⱼ·T[j,i] is the reported residual.
        let mut r = apply(a, &qs[i]);
        for (j, qj) in qs.iter().enumerate() {
            let tji = t[j + i * n].to_c64();
            for (rk, qk) in r.iter_mut().zip(qj) {
                *rk -= qk * tji;
            }
        }
        let got = to_c64(&info.residual[i]);
        let nr = to_f64(info.normres[i]);
        assert!(
            diff_norm(&r, &got) <= 1e-8 * (1.0 + nr),
            "Schur residual {i}"
        );
        assert!((norm(&r) - nr).abs() <= 1e-8, "residual norm {i}");
    }
    Run {
        values: vals
            .iter()
            .map(|v| Complex64::new(to_f64(v.re), to_f64(v.im)))
            .collect(),
        normres: info.normres.iter().map(|&x| to_f64(x)).collect(),
        converged: info.converged,
        numiter: info.numiter,
        numops: info.numops,
    }
}

/// Eigenpair checks shared by `arnoldi_eigsolve` and `realeigsolve`.
fn check_pairs(
    a: &DMatrix<Complex64>,
    vals: &[Complex64],
    vecs: &[Vec<Complex64>],
    res: &[Vec<Complex64>],
    normres: &[f64],
) {
    for i in 0..vals.len() {
        assert!((norm(&vecs[i]) - 1.0).abs() < 1e-10, "unit eigenvector {i}");
        let mut r = apply(a, &vecs[i]);
        for (rk, xk) in r.iter_mut().zip(&vecs[i]) {
            *rk -= vals[i] * xk;
        }
        assert!(
            diff_norm(&r, &res[i]) <= 1e-8 * (1.0 + normres[i]),
            "residual {i}"
        );
        assert!((norm(&r) - normres[i]).abs() <= 1e-8, "residual norm {i}");
    }
}

fn run_eig(
    a: &DMatrix<Complex64>,
    x0: &[Complex64],
    howmany: usize,
    which: Which,
    alg: &Arnoldi,
) -> Run {
    let x0 = to_dvec::<Complex64>(x0);
    let mut op = DenseOp::<Complex64>::new(a);
    let mut ws = Workspace::with_capacity(&x0, alg.get_krylovdim()).unwrap();
    let (vals, vecs, info) = arnoldi_eigsolve(&mut op, &x0, howmany, which, alg, &mut ws).unwrap();
    assert_eq!(info.numops, op.applications);
    let vecs: Vec<_> = vecs.iter().map(to_c64).collect();
    let res: Vec<_> = info.residual.iter().map(to_c64).collect();
    check_pairs(a, &vals, &vecs, &res, &info.normres);
    Run {
        values: vals,
        normres: info.normres.clone(),
        converged: info.converged,
        numiter: info.numiter,
        numops: info.numops,
    }
}

fn run_real(
    a: &DMatrix<Complex64>,
    x0: &[Complex64],
    howmany: usize,
    which: Which,
    alg: &Arnoldi,
) -> Run {
    let x0 = to_dvec::<f64>(x0);
    let mut op = DenseOp::<f64>::new(a);
    let mut ws = Workspace::with_capacity(&x0, alg.get_krylovdim()).unwrap();
    let (vals, vecs, info) = realeigsolve(&mut op, &x0, howmany, which, alg, &mut ws).unwrap();
    assert_eq!(info.numops, op.applications);
    let vals: Vec<Complex64> = vals.iter().map(|&v| Complex64::new(v, 0.0)).collect();
    let vecs: Vec<_> = vecs.iter().map(to_c64).collect();
    let res: Vec<_> = info.residual.iter().map(to_c64).collect();
    check_pairs(a, &vals, &vecs, &res, &info.normres);
    Run {
        values: vals,
        normres: info.normres.clone(),
        converged: info.converged,
        numiter: info.numiter,
        numops: info.numops,
    }
}

// KrylovKit fixtures (`fixtures/krylovkit/arnoldi.txt`, from `arnoldi.jl`).

const FIXTURES: &str = include_str!("fixtures/krylovkit/arnoldi.txt");

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

#[test]
fn matches_krylovkit_fixtures() {
    let cases = cases();
    assert_eq!(cases.len(), 8);
    for c in &cases {
        let name = &c["case"];
        let complex = num::<u8>(c, "complex") == 1;
        let n: usize = num(c, "n");
        let mut g = SplitMix(num(c, "seed"));
        let (a, real_spec) = if c["kind"] == "real" {
            let (a, d) = permuted_triangular(&mut g, n);
            (a, Some(d))
        } else {
            (outliers(&mut g, n, complex), None)
        };
        let x0 = random_vec(&mut g, n, complex);
        let which = match c["which"].as_str() {
            "LM" => Which::LargestMagnitude,
            "LR" => Which::LargestReal,
            "SR" => Which::SmallestReal,
            "LI" => Which::LargestImag,
            "SI" => Which::SmallestImag,
            w => panic!("unknown selector {w}"),
        };
        let alg = Arnoldi::new()
            .krylovdim(num(c, "krylovdim"))
            .tol(num(c, "tol"))
            .eager(num::<u8>(c, "eager") == 1)
            .maxiter(100);
        let howmany = num(c, "howmany");
        let out = match (c["kind"].as_str(), complex) {
            ("schur", false) => run_schur::<f64>(&a, &x0, howmany, which, &alg),
            ("schur", true) => run_schur::<Complex64>(&a, &x0, howmany, which, &alg),
            ("eig", _) => run_eig(&a, &x0, howmany, which, &alg),
            ("real", _) => run_real(&a, &x0, howmany, which, &alg),
            (k, _) => panic!("unknown kind {k}"),
        };
        eprintln!(
            "{name}: krylov numiter {} numops {} converged {} | KrylovKit {} {} {}",
            out.numiter, out.numops, out.converged, c["numiter"], c["numops"], c["converged"]
        );
        let (re, im) = (floats(c, "values_re"), floats(c, "values_im"));
        assert_eq!(out.values.len(), re.len(), "{name}: number of values");
        let spec = real_spec
            .map(|d| d.iter().map(|&x| Complex64::new(x, 0.0)).collect())
            .unwrap_or_else(|| dense_spectrum(&a));
        let tol: f64 = num(c, "tol");
        for (i, v) in out.values.iter().enumerate() {
            let want = Complex64::new(re[i], im[i]);
            assert!(
                (v - want).norm() <= 1e-8 * want.norm().max(1.0),
                "{name}: {v} vs {want}"
            );
            if i < out.converged {
                assert!(out.normres[i] <= tol, "{name}: residual {i}");
                assert!(
                    nearest(&spec, *v) <= 1e-6,
                    "{name}: {v} not in the spectrum"
                );
            }
        }
        assert_eq!(
            out.converged,
            num::<usize>(c, "converged"),
            "{name}: converged"
        );
        assert_eq!(out.numiter, num::<usize>(c, "numiter"), "{name}: numiter");
        assert_eq!(out.numops, num::<usize>(c, "numops"), "{name}: numops");
    }
}

#[test]
fn single_precision_and_complex32() {
    let n = 40;
    let alg = Arnoldi::new().krylovdim(16).tol(1e-4);
    let mut g = SplitMix(41);
    let a = outliers(&mut g, n, false);
    let x0 = random_vec(&mut g, n, false);
    let spec = dense_spectrum(&a);
    let x = to_dvec::<f32>(&x0);
    let mut op = DenseOp::<f32>::new(&a);
    let mut ws = Workspace::with_capacity(&x, 16).unwrap();
    let (_, _, vals, info) =
        schursolve(&mut op, &x, 3, Which::LargestMagnitude, &alg, &mut ws).unwrap();
    assert!(info.converged >= 3);
    for v in &vals[..3] {
        let v = Complex64::new(v.re as f64, v.im as f64);
        assert!(nearest(&spec, v) < 1e-3, "{v}");
    }

    let mut g = SplitMix(42);
    let a = outliers(&mut g, n, true);
    let x0 = random_vec(&mut g, n, true);
    let spec = dense_spectrum(&a);
    let x = to_dvec::<Complex32>(&x0);
    let mut op = DenseOp::<Complex32>::new(&a);
    let mut ws = Workspace::with_capacity(&x, 16).unwrap();
    let (vals, vecs, info) =
        arnoldi_eigsolve(&mut op, &x, 2, Which::LargestImag, &alg, &mut ws).unwrap();
    assert!(info.converged >= 2);
    assert!(vals[0].im >= vals[1].im);
    for (v, r) in vals.iter().zip(&info.normres).take(2) {
        assert!(
            nearest(&spec, Complex64::new(v.re as f64, v.im as f64)) < 1e-3,
            "{v}"
        );
        assert!(*r < 1e-3);
    }
    assert_eq!(vecs.len(), vals.len());
}

#[test]
fn allocates_only_the_returned_vectors() {
    let n = 50;
    let mut g = SplitMix(43);
    let a = outliers(&mut g, n, true);
    let x0 = to_dvec::<Complex64>(&random_vec(&mut g, n, true));
    let alg = Arnoldi::new().krylovdim(12).tol(1e-10);
    let mut op = DenseOp::<Complex64>::new(&a);
    let mut ws = Workspace::with_capacity(&x0, 12).unwrap();
    for _ in 0..2 {
        let before = similar_calls();
        let (vals, _, info) =
            arnoldi_eigsolve(&mut op, &x0, 3, Which::LargestMagnitude, &alg, &mut ws).unwrap();
        assert!(info.numiter > 1, "must exercise restarts");
        assert_eq!(similar_calls() - before, 2 * vals.len());
        let before = similar_calls();
        let (_, q, _, _) =
            schursolve(&mut op, &x0, 3, Which::LargestMagnitude, &alg, &mut ws).unwrap();
        assert_eq!(similar_calls() - before, 2 * q.len());
    }
    // realeigsolve on a real map with a real spectrum.
    let mut g = SplitMix(46);
    let (a, _) = permuted_triangular(&mut g, n);
    let x0 = to_dvec::<f64>(&random_vec(&mut g, n, false));
    let mut op = DenseOp::<f64>::new(&a);
    let mut ws = Workspace::with_capacity(&x0, 12).unwrap();
    for _ in 0..2 {
        let before = similar_calls();
        let (vals, vecs, info) =
            realeigsolve(&mut op, &x0, 3, Which::LargestReal, &alg, &mut ws).unwrap();
        assert!(info.numiter > 1, "must exercise restarts");
        assert_eq!(vecs.len(), vals.len());
        assert_eq!(similar_calls() - before, 2 * vals.len());
    }
}

#[test]
fn typed_errors() {
    let n = 30;
    let mut g = SplitMix(44);
    let a = outliers(&mut g, n, false);
    let x0 = random_vec(&mut g, n, false);
    let xr = to_dvec::<f64>(&x0);
    let alg = Arnoldi::new().krylovdim(10).tol(1e-10);
    let mut op = DenseOp::<f64>::new(&a);
    let mut ws = Workspace::with_capacity(&xr, 10).unwrap();
    for which in [Which::LargestImag, Which::SmallestImag] {
        assert!(matches!(
            schursolve(&mut op, &xr, 1, which, &alg, &mut ws),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            realeigsolve(&mut op, &xr, 1, which, &alg, &mut ws),
            Err(Error::InvalidArgument(_))
        ));
        // The Hermitian solvers have real spectra.
        let h = hermitian(&mut g, n, false);
        let mut hop = DenseOp::<f64>::new(&h);
        assert!(matches!(
            eigsolve(
                &mut hop,
                &xr,
                1,
                which,
                &Lanczos::new().krylovdim(10),
                &mut ws
            ),
            Err(Error::InvalidArgument(_))
        ));
        let block = [to_dvec::<f64>(&x0)];
        let mut wsb = Workspace::with_capacity(&xr, 10).unwrap();
        assert!(matches!(
            block_eigsolve(
                &mut hop,
                &block,
                1,
                which,
                &BlockLanczos::new().krylovdim(10),
                &mut wsb
            ),
            Err(Error::InvalidArgument(_))
        ));
    }
    // The largest-magnitude eigenvalues of `outliers` are 3, -2.5, 2 and the
    // pair 1.5 ± 1.2i: asking realeigsolve for four is a complex eigenvalue.
    let alg = Arnoldi::new().krylovdim(20).tol(1e-10);
    let mut ws = Workspace::with_capacity(&xr, 20).unwrap();
    assert_eq!(
        realeigsolve(&mut op, &xr, 4, Which::LargestMagnitude, &alg, &mut ws).err(),
        Some(Error::ComplexEigenvalue { index: 3 })
    );
    let (vals, _, _) =
        realeigsolve(&mut op, &xr, 3, Which::LargestMagnitude, &alg, &mut ws).unwrap();
    assert!((vals[0] - 3.0).abs() < 0.5 && (vals[1] + 2.5).abs() < 0.5);
    assert_eq!(
        schursolve(&mut op, &xr, 21, Which::LargestMagnitude, &alg, &mut ws).err(),
        Some(Error::HowmanyTooLarge {
            howmany: 21,
            krylovdim: 20
        })
    );
    let zero = to_dvec::<f64>(&vec![Complex64::new(0.0, 0.0); n]);
    assert_eq!(
        schursolve(&mut op, &zero, 1, Which::LargestMagnitude, &alg, &mut ws).err(),
        Some(Error::ZeroStartVector)
    );
    // A map that returns NaN.
    let mut nan_op = |_: &DVec<f64>, y: &mut DVec<f64>| {
        y.0.fill(f64::NAN);
        Ok(())
    };
    assert_eq!(
        schursolve(&mut nan_op, &xr, 1, Which::LargestMagnitude, &alg, &mut ws).err(),
        Some(Error::NonFiniteIterate)
    );
    let mut small = Workspace::with_capacity(&xr, 5).unwrap();
    assert!(matches!(
        schursolve(&mut op, &xr, 1, Which::LargestMagnitude, &alg, &mut small),
        Err(Error::InsufficientCapacity { .. })
    ));
}

#[test]
fn real_pair_is_completed_and_exhausted_space_returns_what_exists() {
    // howmany = 4 ends inside the 2×2 block of the pair 1.5 ± 1.2i:
    // schursolve returns the whole pair.
    let n = 60;
    let mut g = SplitMix(45);
    let a = outliers(&mut g, n, false);
    let x0 = random_vec(&mut g, n, false);
    let alg = Arnoldi::new().krylovdim(20).tol(1e-10);
    let out = run_schur::<f64>(&a, &x0, 4, Which::LargestMagnitude, &alg);
    assert!(out.values.len() >= 5);
    assert!(out.values[3].im != 0.0 && out.values[3].conj() == out.values[4]);
    // A 6-dimensional invariant space: the iteration stops there.
    let d = DMatrix::from_fn(6, 6, |i, j| {
        Complex64::new(
            if i == j {
                i as f64 + 1.0
            } else if j == i + 1 {
                0.5
            } else {
                0.0
            },
            0.0,
        )
    });
    let x = vec![Complex64::new(1.0, 0.0); 6];
    let out = run_schur::<f64>(&d, &x, 3, Which::LargestReal, &Arnoldi::new().krylovdim(10));
    assert_eq!(out.numiter, 1);
    assert!(out.converged >= 3);
    for (i, v) in out.values.iter().take(3).enumerate() {
        assert!((v.re - (6 - i) as f64).abs() < 1e-10);
    }
}
