//! Krylov–Schur (`schursolve`, `arnoldi_eigsolve`, `realeigsolve`) against
//! dense oracles and KrylovKit counts.

mod common;

use common::*;
use krylov::{
    Arnoldi, BlockLanczos, Error, Lanczos, Orth, Which, Workspace, arnoldi_eigsolve,
    block_eigsolve, eigsolve, realeigsolve, schursolve,
};
use nalgebra::DMatrix;
use num_complex::{Complex32, Complex64};

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
    let mut ws = Workspace::with_capacity(&mut (), &x0, alg.get_krylovdim()).unwrap();
    let (t, q, vals, info) =
        schursolve(&mut (), &mut op, &x0, howmany, which, alg, &mut ws).unwrap();
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
    let mut ws = Workspace::with_capacity(&mut (), &x0, alg.get_krylovdim()).unwrap();
    let (vals, vecs, info) =
        arnoldi_eigsolve(&mut (), &mut op, &x0, howmany, which, alg, &mut ws).unwrap();
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
    let mut ws = Workspace::with_capacity(&mut (), &x0, alg.get_krylovdim()).unwrap();
    let (vals, vecs, info) =
        realeigsolve(&mut (), &mut op, &x0, howmany, which, alg, &mut ws).unwrap();
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

/// Sort key of `which` (largest first).
fn key(which: Which, l: Complex64) -> f64 {
    match which {
        Which::LargestMagnitude => l.norm(),
        Which::LargestReal => l.re,
        Which::SmallestReal => -l.re,
        Which::LargestImag => l.im,
        Which::SmallestImag => -l.im,
        _ => unreachable!(),
    }
}

/// KrylovKit.jl's Arnoldi cases (CGS2, `maxiter = 100`, `tol = 1e-10`) on
/// `outliers` and on `permuted_triangular` (exact real spectrum). The
/// converged values are the leading part of the dense spectrum in `which`
/// order (complex pairs complete). The counts follow KrylovKit exactly; they
/// are pinned on four representative cases, checked against KrylovKit.jl
/// v0.10.4: a real LM solve extended to keep a complex pair, a
/// restart-heavy SR solve, an eager SI `arnoldi_eigsolve` and a
/// `realeigsolve` that returns every converged value.
#[test]
fn krylov_schur_matches_dense_spectrum_and_krylovkit_counts() {
    use Which::*;
    #[derive(Clone, Copy, PartialEq)]
    enum Kind {
        Schur,
        Eig,
        Real,
    }
    use Kind::*;
    type Counts = (usize, usize, usize, usize);
    type Case = (
        &'static str,
        Kind,
        u64,
        usize,
        bool,
        usize,
        Which,
        usize,
        bool,
        Option<Counts>,
    );
    #[rustfmt::skip]
    let cases: [Case; 8] = [
        // name, kind, seed, n, complex, howmany, which, krylovdim, eager,
        // KrylovKit (nvals, converged, numiter, numops)
        ("schur_real_lm_pair", Schur, 31, 80, false, 4, LargestMagnitude, 20, false, Some((5, 5, 2, 28))),
        ("schur_real_lr_eager", Schur, 32, 60, false, 3, LargestReal, 16, true, None),
        ("schur_real_sr", Schur, 33, 60, false, 2, SmallestReal, 12, false, Some((2, 2, 33, 199))),
        ("schur_complex_li", Schur, 34, 60, true, 3, LargestImag, 20, false, None),
        ("eig_complex_lm", Eig, 35, 70, true, 4, LargestMagnitude, 20, false, None),
        ("eig_complex_si_eager", Eig, 36, 50, true, 2, SmallestImag, 14, true, Some((3, 3, 32, 196))),
        ("real_lr", Real, 37, 60, false, 3, LargestReal, 20, false, Some((6, 6, 7, 71))),
        ("real_sr_eager", Real, 38, 50, false, 2, SmallestReal, 16, true, None),
    ];
    let tol = 1e-10;
    for (name, kind, seed, n, complex, howmany, which, kd, eager, kk) in cases {
        let mut g = SplitMix(seed);
        let (a, spec) = if kind == Real {
            let (a, d) = permuted_triangular(&mut g, n);
            (a, d.iter().map(|&x| Complex64::new(x, 0.0)).collect())
        } else {
            let a = outliers(&mut g, n, complex);
            let spec = dense_spectrum(&a);
            (a, spec)
        };
        let x0 = random_vec(&mut g, n, complex);
        let alg = Arnoldi::new(kd, 100, tol).eager(eager);
        let out = match (kind, complex) {
            (Schur, false) => run_schur::<f64>(&a, &x0, howmany, which, &alg),
            (Schur, true) => run_schur::<Complex64>(&a, &x0, howmany, which, &alg),
            (Eig, _) => run_eig(&a, &x0, howmany, which, &alg),
            (Real, _) => run_real(&a, &x0, howmany, which, &alg),
        };
        assert!(
            out.converged >= howmany,
            "{name}: converged {}",
            out.converged
        );
        let mut want = spec.clone();
        want.sort_by(|x, y| key(which, *y).partial_cmp(&key(which, *x)).unwrap());
        let mut used = vec![false; out.converged];
        for (i, v) in out.values[..out.converged].iter().enumerate() {
            assert!(out.normres[i] <= tol, "{name}: residual {i}");
            let j = (0..out.converged)
                .filter(|&j| !used[j])
                .min_by(|&j, &k| {
                    (want[j] - v)
                        .norm()
                        .partial_cmp(&(want[k] - v).norm())
                        .unwrap()
                })
                .unwrap();
            used[j] = true;
            assert!(
                (want[j] - v).norm() <= 1e-8 * want[j].norm().max(1.0),
                "{name}: {v} vs {}",
                want[j]
            );
        }
        if let Some((nvals, converged, numiter, numops)) = kk {
            assert_eq!(out.values.len(), nvals, "{name}: number of values");
            assert_eq!(out.converged, converged, "{name}: converged");
            assert_eq!(out.numiter, numiter, "{name}: numiter");
            assert_eq!(out.numops, numops, "{name}: numops");
        }
    }
}

#[test]
fn single_precision_and_complex32() {
    let n = 40;
    let alg = Arnoldi::new(16, 100, 1e-4);
    let mut g = SplitMix(41);
    let a = outliers(&mut g, n, false);
    let x0 = random_vec(&mut g, n, false);
    let spec = dense_spectrum(&a);
    let x = to_dvec::<f32>(&x0);
    let mut op = DenseOp::<f32>::new(&a);
    let mut ws = Workspace::with_capacity(&mut (), &x, 16).unwrap();
    let (_, _, vals, info) = schursolve(
        &mut (),
        &mut op,
        &x,
        3,
        Which::LargestMagnitude,
        &alg,
        &mut ws,
    )
    .unwrap();
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
    let mut ws = Workspace::with_capacity(&mut (), &x, 16).unwrap();
    let (vals, vecs, info) =
        arnoldi_eigsolve(&mut (), &mut op, &x, 2, Which::LargestImag, &alg, &mut ws).unwrap();
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
    let alg = Arnoldi::new(12, 100, 1e-10);
    let mut op = DenseOp::<Complex64>::new(&a);
    let mut ws = Workspace::with_capacity(&mut (), &x0, 12).unwrap();
    for _ in 0..2 {
        let before = similar_calls();
        let (vals, _, info) = arnoldi_eigsolve(
            &mut (),
            &mut op,
            &x0,
            3,
            Which::LargestMagnitude,
            &alg,
            &mut ws,
        )
        .unwrap();
        assert!(info.numiter > 1, "must exercise restarts");
        assert_eq!(similar_calls() - before, 2 * vals.len());
        let before = similar_calls();
        let (_, q, _, _) = schursolve(
            &mut (),
            &mut op,
            &x0,
            3,
            Which::LargestMagnitude,
            &alg,
            &mut ws,
        )
        .unwrap();
        assert_eq!(similar_calls() - before, 2 * q.len());
    }
    // realeigsolve on a real map with a real spectrum.
    let mut g = SplitMix(46);
    let (a, _) = permuted_triangular(&mut g, n);
    let x0 = to_dvec::<f64>(&random_vec(&mut g, n, false));
    let mut op = DenseOp::<f64>::new(&a);
    let mut ws = Workspace::with_capacity(&mut (), &x0, 12).unwrap();
    for _ in 0..2 {
        let before = similar_calls();
        let (vals, vecs, info) =
            realeigsolve(&mut (), &mut op, &x0, 3, Which::LargestReal, &alg, &mut ws).unwrap();
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
    let alg = Arnoldi::new(10, 100, 1e-10);
    let mut op = DenseOp::<f64>::new(&a);
    let mut ws = Workspace::with_capacity(&mut (), &xr, 10).unwrap();
    for which in [Which::LargestImag, Which::SmallestImag] {
        assert!(matches!(
            schursolve(&mut (), &mut op, &xr, 1, which, &alg, &mut ws),
            Err(Error::InvalidArgument(_))
        ));
        assert!(matches!(
            realeigsolve(&mut (), &mut op, &xr, 1, which, &alg, &mut ws),
            Err(Error::InvalidArgument(_))
        ));
        // The Hermitian solvers have real spectra.
        let h = hermitian(&mut g, n, false);
        let mut hop = DenseOp::<f64>::new(&h);
        assert!(matches!(
            eigsolve(
                &mut (),
                &mut hop,
                &xr,
                1,
                which,
                &Lanczos::new(10, 100, 1e-12),
                &mut ws
            ),
            Err(Error::InvalidArgument(_))
        ));
        let block = [to_dvec::<f64>(&x0)];
        let mut wsb = Workspace::with_capacity(&mut (), &xr, 10).unwrap();
        assert!(matches!(
            block_eigsolve(
                &mut (),
                &mut hop,
                &block,
                1,
                which,
                &BlockLanczos::new(10, 100, 1e-12, 1e-12),
                &mut wsb
            ),
            Err(Error::InvalidArgument(_))
        ));
    }
    // The largest-magnitude eigenvalues of `outliers` are 3, -2.5, 2 and the
    // pair 1.5 ± 1.2i: asking realeigsolve for four is a complex eigenvalue.
    let alg = Arnoldi::new(20, 100, 1e-10);
    let mut ws = Workspace::with_capacity(&mut (), &xr, 20).unwrap();
    assert_eq!(
        realeigsolve(
            &mut (),
            &mut op,
            &xr,
            4,
            Which::LargestMagnitude,
            &alg,
            &mut ws
        )
        .err(),
        Some(Error::ComplexEigenvalue { index: 3 })
    );
    let (vals, _, _) = realeigsolve(
        &mut (),
        &mut op,
        &xr,
        3,
        Which::LargestMagnitude,
        &alg,
        &mut ws,
    )
    .unwrap();
    assert!((vals[0] - 3.0).abs() < 0.5 && (vals[1] + 2.5).abs() < 0.5);
    assert_eq!(
        schursolve(
            &mut (),
            &mut op,
            &xr,
            21,
            Which::LargestMagnitude,
            &alg,
            &mut ws
        )
        .err(),
        Some(Error::HowmanyTooLarge {
            howmany: 21,
            krylovdim: 20
        })
    );
    let zero = to_dvec::<f64>(&vec![Complex64::new(0.0, 0.0); n]);
    assert_eq!(
        schursolve(
            &mut (),
            &mut op,
            &zero,
            1,
            Which::LargestMagnitude,
            &alg,
            &mut ws
        )
        .err(),
        Some(Error::ZeroStartVector)
    );
    // A map that returns NaN.
    let mut nan_op = |_: &mut (), _: &DVec<f64>, y: &mut DVec<f64>| {
        y.0.fill(f64::NAN);
        Ok(())
    };
    assert_eq!(
        schursolve(
            &mut (),
            &mut nan_op,
            &xr,
            1,
            Which::LargestMagnitude,
            &alg,
            &mut ws
        )
        .err(),
        Some(Error::NonFiniteIterate)
    );
    let mut small = Workspace::with_capacity(&mut (), &xr, 5).unwrap();
    assert!(matches!(
        schursolve(
            &mut (),
            &mut op,
            &xr,
            1,
            Which::LargestMagnitude,
            &alg,
            &mut small
        ),
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
    let alg = Arnoldi::new(20, 100, 1e-10);
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
    let out = run_schur::<f64>(&d, &x, 3, Which::LargestReal, &Arnoldi::new(10, 100, 1e-12));
    assert_eq!(out.numiter, 1);
    assert!(out.converged >= 3);
    for (i, v) in out.values.iter().take(3).enumerate() {
        assert!((v.re - (6 - i) as f64).abs() < 1e-10);
    }
}

/// Vector work of the Arnoldi expansion (shared with GMRES), pinned per
/// orthogonalizer: `[inner, norm, axpby, copy_from]` and map applications
/// for one restarted `schursolve` (n = 40, K = 8, three Krylov subspaces).
#[test]
fn arnoldi_vector_operation_counts() {
    let mut g = SplitMix(84);
    let n = 40;
    let a = outliers(&mut g, n, false);
    let x0 = to_dvec::<f64>(&random_vec(&mut g, n, false));
    let mut got = Vec::new();
    for orth in [Orth::Cgs2, Orth::Mgs2] {
        let mut op = DenseOp::<f64>::new(&a);
        let mut ws = Workspace::with_capacity(&mut (), &x0, 8).unwrap();
        let alg = Arnoldi::new(8, 3, 0.0).orth(orth);
        let before = vector_ops();
        let info = schursolve(&mut (), &mut op, &x0, 2, Which::LargestReal, &alg, &mut ws)
            .unwrap()
            .3;
        assert_eq!(info.numops, op.applications);
        let [inner, norm, axpby, copy] = ops_since(before);
        got.push([inner, norm, axpby, copy, op.applications]);
    }
    assert_eq!(got, ARNOLDI_COUNTS);
}

const ARNOLDI_COUNTS: [[usize; 5]; 2] = [[176, 17, 332, 0, 16], [176, 17, 332, 0, 16]];
