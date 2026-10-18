//! `block_eigsolve` against nalgebra's dense eigensolver and KrylovKit
//! `BlockLanczos` fixtures.

mod common;

use std::collections::HashMap;

use common::*;
use krylov::{
    BlockLanczos, ConvergenceInfo, Error, Lanczos, Orth, Which, Workspace, block_eigsolve, eigsolve,
};
use nalgebra::DMatrix;
use num_complex::{Complex32, Complex64};

struct Out<S: TestScalar> {
    vals: Vec<f64>,
    vecs: Vec<Vec<Complex64>>,
    info: ConvergenceInfo<DVec<S>>,
}

fn run<S: TestScalar>(
    a: &DMatrix<Complex64>,
    x0: &[Vec<Complex64>],
    howmany: usize,
    which: Which,
    alg: BlockLanczos,
) -> Out<S> {
    let x0: Vec<DVec<S>> = x0.iter().map(|x| to_dvec(x)).collect();
    let mut op = DenseOp::<S>::new(a);
    let cap = alg.get_krylovdim() + x0.len() - 1;
    let mut ws = Workspace::with_capacity(&x0[0], cap).unwrap();
    let (vals, vecs, info) = block_eigsolve(&mut op, &x0, howmany, which, &alg, &mut ws).unwrap();
    assert_eq!(info.numops, op.applications);
    assert_eq!(vals.len(), vecs.len());
    assert_eq!(vals.len(), info.residual.len());
    Out {
        vals: vals.into_iter().map(to_f64).collect(),
        vecs: vecs.iter().map(to_c64).collect(),
        info,
    }
}

/// The reported residuals are `A·x - λ·x`, their norms are `normres`, and
/// the eigenvectors are orthonormal.
fn check_pairs<S: TestScalar>(a: &DMatrix<Complex64>, out: &Out<S>, eps: f64) {
    let scale = a.norm();
    for i in 0..out.vals.len() {
        let x = nalgebra::DVector::from_column_slice(&out.vecs[i]);
        let r = a * &x - &x * Complex64::new(out.vals[i], 0.0);
        let diff = diff_norm(r.as_slice(), &to_c64(&out.info.residual[i]));
        assert!(diff <= eps * scale, "pair {i}: residual mismatch {diff}");
        let nr = to_f64(out.info.normres[i]);
        assert!(
            (nr - r.norm()).abs() <= eps * scale,
            "pair {i}: {nr} vs {}",
            r.norm()
        );
    }
    let vs: Vec<DVec<Complex64>> = out.vecs.iter().map(|x| to_dvec(x)).collect();
    assert!(orthogonality_error(&vs) <= eps * 100.0);
}

fn block(g: &mut SplitMix, n: usize, bs: usize, complex: bool) -> Vec<Vec<Complex64>> {
    (0..bs).map(|_| random_vec(g, n, complex)).collect()
}

#[test]
fn degenerate_spectrum_finds_every_copy() {
    let mut g = SplitMix(51);
    // Lowest eigenvalue three-fold, then a double, then a generic tail.
    let mut eigs = vec![-3.0, -3.0, -3.0, -2.5, -2.5];
    eigs.extend((0..55).map(|i| -2.0 + 4.0 * i as f64 / 55.0));
    for complex in [false, true] {
        let a = with_spectrum(&mut g, &eigs, complex);
        let x0 = block(&mut g, 60, 3, complex);
        for orth in [Orth::Cgs2, Orth::Mgs2] {
            let alg = BlockLanczos::new(30, 100, 1e-10, 1e-12).orth(orth);
            let out = run::<Complex64>(&a, &x0, 5, Which::SmallestReal, alg);
            check_pairs(&a, &out, 1e-12);
            assert!(
                out.info.converged >= 5,
                "complex={complex}: {:?}",
                out.info.converged
            );
            for (v, e) in out.vals.iter().zip(&eigs) {
                assert!((v - e).abs() < 1e-9, "complex={complex}: {v} vs {e}");
            }
        }
    }
}

#[test]
fn matches_dense_spectrum_with_restarts() {
    let mut g = SplitMix(52);
    let a = hermitian(&mut g, 100, true);
    let exact = dense_eigenvalues(&a);
    let x0 = block(&mut g, 100, 2, true);
    for (which, want) in [
        (Which::SmallestReal, exact[..3].to_vec()),
        (
            Which::LargestReal,
            exact.iter().rev().take(3).copied().collect(),
        ),
        (Which::LargestMagnitude, {
            let mut m = exact.clone();
            m.sort_by(|x, y| y.abs().partial_cmp(&x.abs()).unwrap());
            m[..3].to_vec()
        }),
    ] {
        let alg = BlockLanczos::new(14, 100, 1e-10, 1e-12);
        let out = run::<Complex64>(&a, &x0, 3, which, alg);
        assert!(out.info.numiter > 1, "{which:?}: restarts expected");
        assert!(out.info.converged >= 3, "{which:?}");
        check_pairs(&a, &out, 1e-12);
        for (v, e) in out.vals.iter().zip(&want) {
            assert!((v - e).abs() < 1e-9, "{which:?}: {v} vs {e}");
        }
    }
    let a = hermitian(&mut g, 80, false);
    let exact = dense_eigenvalues(&a);
    let x0 = block(&mut g, 80, 3, false);
    let alg = BlockLanczos::new(18, 100, 1e-10, 1e-12).eager(true);
    let out = run::<f64>(&a, &x0, 2, Which::SmallestReal, alg);
    check_pairs(&a, &out, 1e-12);
    for (v, e) in out.vals.iter().zip(&exact) {
        assert!((v - e).abs() < 1e-9, "{v} vs {e}");
    }
}

#[test]
fn block_size_one_agrees_with_lanczos() {
    let mut g = SplitMix(53);
    let a = hermitian(&mut g, 60, false);
    let x = random_vec(&mut g, 60, false);
    let out = run::<f64>(
        &a,
        std::slice::from_ref(&x),
        3,
        Which::SmallestReal,
        BlockLanczos::new(20, 100, 1e-10, 1e-12),
    );
    let xv = to_dvec::<f64>(&x);
    let alg = Lanczos::new(20, 100, 1e-10);
    let mut ws = Workspace::with_capacity(&xv, 20).unwrap();
    let (vals, _, _) = eigsolve(
        &mut DenseOp::new(&a),
        &xv,
        3,
        Which::SmallestReal,
        &alg,
        &mut ws,
    )
    .unwrap();
    for (a, b) in out.vals.iter().zip(&vals) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
}

#[test]
fn invariant_subspace_and_dropped_blocks() {
    // Four distinct eigenvalues, each double: with a block of two the Krylov
    // space is exhausted at dimension 8.
    let mut g = SplitMix(54);
    let eigs: Vec<f64> = (0..40).map(|i| (i % 4) as f64).collect();
    let a = with_spectrum(&mut g, &eigs, false);
    let x0 = block(&mut g, 40, 2, false);
    let out = run::<f64>(
        &a,
        &x0,
        4,
        Which::SmallestReal,
        BlockLanczos::new(20, 100, 1e-12, 1e-12),
    );
    check_pairs(&a, &out, 1e-12);
    assert_eq!(out.info.numiter, 1);
    for (v, e) in out.vals.iter().zip([0.0, 0.0, 1.0, 1.0]) {
        assert!((v - e).abs() < 1e-10, "{v} vs {e}");
    }
    // A dependent start vector is dropped by the QR.
    let x1 = x0[0].iter().map(|z| z * 2.0).collect();
    let out = run::<f64>(
        &a,
        &[x0[0].clone(), x1],
        2,
        Which::SmallestReal,
        BlockLanczos::new(20, 100, 1e-12, 1e-12),
    );
    check_pairs(&a, &out, 1e-12);
    assert_eq!(out.info.numops, 4); // block of one, Krylov space of four
    // Restarts with a residual block smaller than the start block.
    let b = hermitian(&mut g, 60, true);
    let exact = dense_eigenvalues(&b);
    let y = block(&mut g, 60, 2, true);
    let dup = y[0].iter().map(|z| z * Complex64::new(0.0, 3.0)).collect();
    let alg = BlockLanczos::new(12, 100, 1e-10, 1e-12);
    let out = run::<Complex64>(
        &b,
        &[y[0].clone(), dup, y[1].clone()],
        3,
        Which::SmallestReal,
        alg,
    );
    assert!(out.info.numiter > 1 && out.info.converged >= 3);
    check_pairs(&b, &out, 1e-12);
    for (v, e) in out.vals.iter().zip(&exact) {
        assert!((v - e).abs() < 1e-9, "{v} vs {e}");
    }
    // A coarse qr_tol drops whole blocks early; the result stays consistent.
    let a = hermitian(&mut g, 50, true);
    let x0 = block(&mut g, 50, 2, true);
    let out = run::<Complex64>(
        &a,
        &x0,
        2,
        Which::SmallestReal,
        BlockLanczos::new(30, 100, 1e-12, 0.5),
    );
    check_pairs(&a, &out, 1e-12);
}

#[test]
fn single_precision() {
    let mut g = SplitMix(55);
    let mut eigs = vec![-2.0, -2.0];
    eigs.extend((0..38).map(|i| -1.0 + 2.0 * i as f64 / 38.0));
    let alg = BlockLanczos::new(20, 100, 1e-4, 1e-5);
    let a = with_spectrum(&mut g, &eigs, true);
    let out = run::<Complex32>(&a, &block(&mut g, 40, 2, true), 2, Which::SmallestReal, alg);
    assert!(out.info.converged >= 2);
    check_pairs(&a, &out, 1e-5);
    for v in &out.vals[..2] {
        assert!((v + 2.0).abs() < 1e-4, "{v}");
    }
    let a = with_spectrum(&mut g, &eigs, false);
    let out = run::<f32>(
        &a,
        &block(&mut g, 40, 2, false),
        2,
        Which::SmallestReal,
        alg,
    );
    assert!(out.info.converged >= 2);
    check_pairs(&a, &out, 1e-5);
    // A Complex32 operator of norm ~1e13 (where squares fit but cubes do not).
    let scale = 1e13;
    let a = with_spectrum(&mut g, &eigs, true) * Complex64::new(scale, 0.0);
    let alg = BlockLanczos::new(20, 100, 1e-4 * scale, 1e-5);
    let out = run::<Complex32>(&a, &block(&mut g, 40, 2, true), 2, Which::SmallestReal, alg);
    assert!(out.info.converged >= 2);
    for v in &out.vals[..2] {
        assert!((v / scale + 2.0).abs() < 1e-4, "{v}");
    }
}

#[test]
fn zero_qr_tol_drops_exactly_zero_vectors() {
    let mut g = SplitMix(57);
    let a = hermitian(&mut g, 30, false);
    let exact = dense_eigenvalues(&a);
    let x = random_vec(&mut g, 30, false);
    let zero = vec![Complex64::new(0.0, 0.0); 30];
    let alg = BlockLanczos::new(20, 100, 1e-10, 0.0);
    let out = run::<f64>(&a, &[x, zero], 1, Which::SmallestReal, alg);
    check_pairs(&a, &out, 1e-12);
    assert!((out.vals[0] - exact[0]).abs() < 1e-9);
}

#[test]
fn warm_call_allocates_two_blocks_and_the_results() {
    let mut g = SplitMix(56);
    let a = hermitian(&mut g, 60, true);
    let x0: Vec<DVec<Complex64>> = block(&mut g, 60, 3, true)
        .iter()
        .map(|x| to_dvec(x))
        .collect();
    let alg = BlockLanczos::new(15, 100, 1e-10, 1e-12);
    let mut op = DenseOp::new(&a);
    let mut ws = Workspace::with_capacity(&x0[0], 17).unwrap();
    for _ in 0..2 {
        let before = similar_calls();
        let (vals, _, info) =
            block_eigsolve(&mut op, &x0, 2, Which::SmallestReal, &alg, &mut ws).unwrap();
        assert!(info.numiter > 1);
        assert_eq!(similar_calls() - before, 2 * 3 + 2 * vals.len());
    }
}

#[test]
fn errors() {
    let a = DMatrix::<Complex64>::identity(4, 4);
    let x = DVec(vec![1.0f64, 2.0, 3.0, 4.0]);
    let mut op = DenseOp::<f64>::new(&a);
    let mut ws = Workspace::with_capacity(&x, 4).unwrap();
    let alg = BlockLanczos::new(3, 100, 1e-12, 1e-12);
    let sr = Which::SmallestReal;
    let e = |r: Result<_, Error<_>>| r.map(|_| ()).unwrap_err();
    assert!(matches!(
        e(block_eigsolve(&mut op, &[], 1, sr, &alg, &mut ws)),
        Error::InvalidArgument(_)
    ));
    let two = [DVec(vec![1.0; 4]), DVec(vec![0.0, 1.0, 0.0, 0.0])];
    assert_eq!(
        e(block_eigsolve(
            &mut op,
            &two,
            1,
            sr,
            &BlockLanczos::new(4, 100, 1e-12, 1e-12),
            &mut ws
        )),
        Error::InsufficientCapacity {
            needed: 5,
            capacity: 4
        }
    );
    assert!(matches!(
        e(block_eigsolve(
            &mut op,
            &two,
            1,
            sr,
            &BlockLanczos::new(2, 100, 1e-12, 1e-12),
            &mut ws
        )),
        Error::InvalidArgument(_)
    ));
    assert!(matches!(
        e(block_eigsolve(&mut op, &two, 4, sr, &alg, &mut ws)),
        Error::HowmanyTooLarge { .. }
    ));
    let zeros = [DVec(vec![0.0; 4]), DVec(vec![1e-14; 4])];
    assert_eq!(
        e(block_eigsolve(&mut op, &zeros, 1, sr, &alg, &mut ws)),
        Error::ZeroStartVector
    );
    let nan = [x, DVec(vec![f64::NAN; 4])];
    assert_eq!(
        e(block_eigsolve(&mut op, &nan, 1, sr, &alg, &mut ws)),
        Error::NonFiniteStartVector
    );
    assert!(matches!(
        e(block_eigsolve(
            &mut op,
            &two,
            1,
            sr,
            &BlockLanczos::new(3, 100, 1e-12, -1.0),
            &mut ws
        )),
        Error::InvalidArgument(_)
    ));
}

#[test]
fn matches_krylovkit_block_lanczos_fixtures() {
    // Format of fixtures/krylovkit/block.txt (see tests/krylovkit.rs).
    let mut cases: Vec<HashMap<String, String>> = Vec::new();
    let mut cur = HashMap::new();
    for line in include_str!("fixtures/krylovkit/block.txt").lines() {
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
    for c in &cases {
        let num = |k: &str| -> usize { c[k].parse().unwrap() };
        let complex = num("complex") == 1;
        let (m, copies) = (num("m"), num("copies"));
        let n = m * copies;
        let mut g = SplitMix(c["seed"].parse().unwrap());
        let h = hermitian(&mut g, m, complex);
        let mut a = DMatrix::<Complex64>::zeros(n, n);
        for b in 0..copies {
            a.view_mut((b * m, b * m), (m, m)).copy_from(&h);
        }
        let x0 = block(&mut g, n, num("bs"), complex);
        let which = match c["which"].as_str() {
            "SR" => Which::SmallestReal,
            "LR" => Which::LargestReal,
            "LM" => Which::LargestMagnitude,
            w => panic!("unknown selector {w}"),
        };
        let alg = BlockLanczos::new(num("krylovdim"), 100, c["tol"].parse().unwrap(), 1e-12)
            .eager(num("eager") == 1)
            .orth(Orth::Mgs2);
        let out = run::<Complex64>(&a, &x0, num("howmany"), which, alg);
        let expect: Vec<f64> = c["values"]
            .split_whitespace()
            .map(|x| x.parse().unwrap())
            .collect();
        let name = &c["case"];
        eprintln!(
            "{name}: krylov numiter {} numops {} converged {} | KrylovKit {} {} {}",
            out.info.numiter,
            out.info.numops,
            out.info.converged,
            c["numiter"],
            c["numops"],
            c["converged"]
        );
        assert_eq!(out.vals.len(), expect.len(), "{name}: number of values");
        assert_eq!(out.info.converged, num("converged"), "{name}");
        for (v, e) in out.vals.iter().zip(&expect) {
            assert!(
                (v - e).abs() <= 1e-9 * e.abs().max(1.0),
                "{name}: {v} vs {e}"
            );
        }
        check_pairs(&a, &out, 1e-12);
        let (ni, no) = (num("numiter"), num("numops"));
        assert!(
            out.info.numiter.abs_diff(ni) <= 1,
            "{name}: numiter {} vs {ni}",
            out.info.numiter
        );
        assert!(
            out.info.numops.abs_diff(no) * 10 <= no,
            "{name}: numops {} vs {no}",
            out.info.numops
        );
    }
}

/// Single precision at a tolerance below its reach restarts until the
/// projected matrices carry entries near 1e-24 (#20). The Householder
/// reflectors of the restart used to turn them into NaN.
#[test]
fn tight_tolerance_restarts_stay_finite_in_single_precision() {
    for (seed, kd, tol, maxiter, which) in [
        (10, 6, 1e-12, 200, Which::SmallestReal),
        (15, 5, 1e-10, 100, Which::LargestReal),
    ] {
        let mut g = SplitMix(1000 + seed);
        let n = 30;
        let a = if seed % 2 == 0 {
            hermitian(&mut g, n, true)
        } else {
            let eigs: Vec<f64> = (0..n)
                .map(|i| match i {
                    0 => -5.0,
                    i if i == n - 1 => 5.0,
                    i => -1.0 + 2.0 * i as f64 / n as f64,
                })
                .collect();
            with_spectrum(&mut g, &eigs, true)
        };
        let x0: Vec<DVec<Complex32>> = (0..2)
            .map(|_| to_dvec(&random_vec(&mut g, n, true)))
            .collect();
        let alg = BlockLanczos::new(kd, maxiter, tol, 1e-12);
        let mut ws = Workspace::with_capacity(&x0[0], kd + 2).unwrap();
        let (vals, _, _) = block_eigsolve(
            &mut DenseOp::<Complex32>::new(&a),
            &x0,
            2,
            which,
            &alg,
            &mut ws,
        )
        .unwrap();
        let exact = dense_eigenvalues(&a);
        let want = if which == Which::SmallestReal {
            exact[0]
        } else {
            exact[n - 1]
        };
        assert!(
            (f64::from(vals[0]) - want).abs() < 1e-4,
            "seed {seed}: {} vs {want}",
            vals[0]
        );
    }
}
