//! `block_eigsolve` against nalgebra's dense eigensolver and KrylovKit
//! `BlockLanczos` counts.

mod common;

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
    let mut ws = Workspace::with_capacity(&mut (), &x0[0], cap).unwrap();
    let (vals, vecs, info) =
        block_eigsolve(&mut (), &mut op, &x0, howmany, which, &alg, &mut ws).unwrap();
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
    let mut ws = Workspace::with_capacity(&mut (), &xv, 20).unwrap();
    let (vals, _, _) = eigsolve(
        &mut (),
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
    let mut ws = Workspace::with_capacity(&mut (), &x0[0], 17).unwrap();
    for _ in 0..2 {
        let before = similar_calls();
        let (vals, _, info) =
            block_eigsolve(&mut (), &mut op, &x0, 2, Which::SmallestReal, &alg, &mut ws).unwrap();
        assert!(info.numiter > 1);
        assert_eq!(similar_calls() - before, 2 * 3 + 2 * vals.len());
    }
}

#[test]
fn errors() {
    let a = DMatrix::<Complex64>::identity(4, 4);
    let x = DVec(vec![1.0f64, 2.0, 3.0, 4.0]);
    let mut op = DenseOp::<f64>::new(&a);
    let mut ws = Workspace::with_capacity(&mut (), &x, 4).unwrap();
    let alg = BlockLanczos::new(3, 100, 1e-12, 1e-12);
    let sr = Which::SmallestReal;
    let e = |r: Result<_, Error<_>>| r.map(|_| ()).unwrap_err();
    assert!(matches!(
        e(block_eigsolve(&mut (), &mut op, &[], 1, sr, &alg, &mut ws)),
        Error::InvalidArgument(_)
    ));
    let two = [DVec(vec![1.0; 4]), DVec(vec![0.0, 1.0, 0.0, 0.0])];
    assert_eq!(
        e(block_eigsolve(
            &mut (),
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
            &mut (),
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
        e(block_eigsolve(&mut (), &mut op, &two, 4, sr, &alg, &mut ws)),
        Error::HowmanyTooLarge { .. }
    ));
    let zeros = [DVec(vec![0.0; 4]), DVec(vec![1e-14; 4])];
    assert_eq!(
        e(block_eigsolve(
            &mut (),
            &mut op,
            &zeros,
            1,
            sr,
            &alg,
            &mut ws
        )),
        Error::ZeroStartVector
    );
    let nan = [x, DVec(vec![f64::NAN; 4])];
    assert_eq!(
        e(block_eigsolve(&mut (), &mut op, &nan, 1, sr, &alg, &mut ws)),
        Error::NonFiniteStartVector
    );
    assert!(matches!(
        e(block_eigsolve(
            &mut (),
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

/// KrylovKit.jl's `BlockLanczos` cases (MGS2, the only orthogonalizer it
/// supports here): `A = kron(I_copies, H)`, so every eigenvalue of `H` is
/// `copies`-fold and the expected values are the dense spectrum of `H`
/// repeated. `(nvals, converged, numiter, numops)` is pinned on two
/// representative cases, checked against KrylovKit.jl v0.10.4: numiter
/// equal, numops one lower (krylov skips KrylovKit's type-probe
/// application, `docs/krylovkit.md`).
#[test]
fn block_lanczos_matches_dense_spectrum_and_krylovkit_counts() {
    type Counts = (usize, usize, usize, usize);
    type Case = (
        &'static str,
        u64,
        usize,
        usize,
        bool,
        usize,
        usize,
        Which,
        usize,
        f64,
        bool,
        Option<Counts>,
    );
    #[rustfmt::skip]
    let cases: [Case; 4] = [
        // name, seed, m, copies, complex, bs, howmany, which, krylovdim, tol, eager, KrylovKit
        ("real_double_sr", 21, 30, 2, false, 2, 4, Which::SmallestReal, 20, 1e-10, false, None),
        ("complex_triple_lr", 22, 25, 3, true, 3, 3, Which::LargestReal, 30, 1e-10, false, Some((3, 3, 7, 103))),
        ("real_generic_sr_eager", 23, 60, 1, false, 2, 2, Which::SmallestReal, 24, 1e-10, true, None),
        // Restarts; KrylovKit returns every converged pair (6 > 4).
        ("complex_double_restart", 24, 40, 2, true, 2, 4, Which::SmallestReal, 16, 1e-9, false, Some((6, 6, 14, 117))),
    ];
    for (name, seed, m, copies, complex, bs, howmany, which, kd, tol, eager, kk) in cases {
        let n = m * copies;
        let mut g = SplitMix(seed);
        let h = hermitian(&mut g, m, complex);
        let mut a = DMatrix::<Complex64>::zeros(n, n);
        for b in 0..copies {
            a.view_mut((b * m, b * m), (m, m)).copy_from(&h);
        }
        let x0 = block(&mut g, n, bs, complex);
        let alg = BlockLanczos::new(kd, 100, tol, 1e-12)
            .eager(eager)
            .orth(Orth::Mgs2);
        let out = run::<Complex64>(&a, &x0, howmany, which, alg);
        let mut want = dense_eigenvalues(&h);
        if which == Which::LargestReal {
            want.reverse();
        }
        let want: Vec<f64> = want
            .iter()
            .flat_map(|&e| std::iter::repeat_n(e, copies))
            .collect();
        assert!(out.info.converged >= howmany, "{name}");
        assert!(out.vals.len() >= howmany, "{name}");
        for (v, e) in out.vals.iter().zip(&want) {
            assert!(
                (v - e).abs() <= 1e-9 * e.abs().max(1.0),
                "{name}: {v} vs {e}"
            );
        }
        check_pairs(&a, &out, 1e-12);
        if let Some((nvals, converged, ni, no)) = kk {
            assert_eq!(out.vals.len(), nvals, "{name}: number of values");
            assert_eq!(out.info.converged, converged, "{name}");
            assert_eq!(out.info.numiter, ni, "{name}: numiter");
            // No start vector is dropped here, so exactly KrylovKit's count
            // without its type-probe application.
            assert_eq!(out.info.numops, no - 1, "{name}: numops");
        }
    }
}

/// Single precision at a tolerance below its reach restarts until the
/// projected matrices carry entries near 1e-24 (#20). The Householder
/// reflectors of the restart must keep them finite.
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
        let mut ws = Workspace::with_capacity(&mut (), &x0[0], kd + 2).unwrap();
        let (vals, _, _) = block_eigsolve(
            &mut (),
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

/// Host vector with an overflow/underflow-safe (`hypot`) norm, so any range
/// failure in a residual comes from the solver's own arithmetic.
#[derive(Debug)]
struct Hv<T>(Vec<T>);

impl<T: krylov::RealScalar> krylov::Vector for Hv<T> {
    type Scalar = T;
    type Error = std::convert::Infallible;
    type Context = ();
    fn similar(&self, _: &mut ()) -> Result<Self, Self::Error> {
        Ok(Hv(vec![T::zero(); self.0.len()]))
    }
    fn copy_from(&mut self, _: &mut (), x: &Self) -> Result<(), Self::Error> {
        self.0.copy_from_slice(&x.0);
        Ok(())
    }
    fn inner(&self, _: &mut (), y: &Self) -> Result<T, Self::Error> {
        Ok(self
            .0
            .iter()
            .zip(&y.0)
            .fold(T::zero(), |s, (&a, &b)| s + a * b))
    }
    fn norm(&self, _: &mut ()) -> Result<T, Self::Error> {
        Ok(self.0.iter().fold(T::zero(), |s, &a| s.hypot(a)))
    }
    fn axpby(&mut self, _: &mut (), a: T, x: &Self, b: T) -> Result<(), Self::Error> {
        for (y, &x) in self.0.iter_mut().zip(&x.0) {
            *y = if b == T::zero() {
                a * x
            } else {
                a * x + b * *y
            };
        }
        Ok(())
    }
}

/// `block_eigsolve` of the symmetric `A` from the unit vectors
/// `e₀ … e_{p-1}`, `howmany = 1`, `LargestReal`. For every returned pair it
/// checks that `normres` is the norm of the returned residual (up to `1e-5`
/// relative and `tol` absolute, i.e. without changing the convergence
/// decision) and of `A·x - λ·x` (plus that evaluation's rounding), and that `converged` counts exactly
/// the pairs with `normres <= tol`.
fn residual_run<T: krylov::RealScalar>(
    a: &[&[f64]],
    p: usize,
    alg: BlockLanczos,
) -> Result<Vec<T>, Error<std::convert::Infallible>> {
    let n = a.len();
    let apply = |x: &[T]| -> Vec<T> {
        (0..n)
            .map(|i| {
                (0..n)
                    .filter(|&j| a[i][j] != 0.0)
                    .fold(T::zero(), |acc, j| acc + T::from(a[i][j]).unwrap() * x[j])
            })
            .collect()
    };
    let x0: Vec<Hv<T>> = (0..p)
        .map(|i| {
            Hv((0..n)
                .map(|j| if i == j { T::one() } else { T::zero() })
                .collect())
        })
        .collect();
    let mut op = |_: &mut (), x: &Hv<T>, y: &mut Hv<T>| {
        y.0 = apply(&x.0);
        Ok::<(), std::convert::Infallible>(())
    };
    let cap = alg.get_krylovdim() + p - 1;
    let mut ws = Workspace::with_capacity(&mut (), &x0[0], cap).unwrap();
    let (vals, vecs, info) =
        block_eigsolve(&mut (), &mut op, &x0, 1, Which::LargestReal, &alg, &mut ws)?;
    let tol = T::from(alg.get_tol()).unwrap();
    let rel = T::from(1e-5).unwrap();
    let amax = a
        .iter()
        .flat_map(|r| r.iter())
        .fold(T::zero(), |m, &x| m.max(T::from(x.abs()).unwrap()));
    let hypot = |v: &[T]| v.iter().fold(T::zero(), |acc, &x| acc.hypot(x));
    for (i, (&l, x)) in vals.iter().zip(&vecs).enumerate() {
        let direct: Vec<T> = apply(&x.0)
            .iter()
            .zip(&x.0)
            .map(|(&ax, &xi)| ax - l * xi)
            .collect();
        let nr = info.normres[i];
        // Evaluating A·x - λ·x directly adds rounding of order eps·‖A‖.
        let rounding = T::epsilon() * T::from(4 * n).unwrap() * amax;
        for (what, want, slack) in [
            ("returned", hypot(&info.residual[i].0), tol),
            ("direct", hypot(&direct), tol + rounding),
        ] {
            assert!(
                (nr - want).abs() <= rel * nr.max(want) + slack,
                "pair {i}: normres {:e} vs {what} {:e}",
                to_f64(nr),
                to_f64(want)
            );
        }
        assert_eq!(i < info.converged, nr <= tol, "pair {i}: {:e}", to_f64(nr));
    }
    Ok(vals)
}

#[test]
fn block_residuals_are_exact_at_extreme_scales() {
    // Squaring the residual norms or forming their Gram matrix unscaled
    // under- or overflows; the solver once reported the first Krylov space as
    // converged with zero residuals and zero eigenvalues.
    let pair = |s: f64| [vec![0.0, s], vec![s, 0.0]];
    let swap = |s: f64| {
        let mut m = vec![vec![0.0; 4]; 4];
        for i in 0..4 {
            m[i][(i + 2) % 4] = s;
        }
        m
    };
    let run = |m: &[Vec<f64>], p, alg, single: bool| {
        let rows: Vec<&[f64]> = m.iter().map(Vec::as_slice).collect();
        if single {
            residual_run::<f32>(&rows, p, alg).map(|v| v.into_iter().map(f64::from).collect())
        } else {
            residual_run::<f64>(&rows, p, alg)
        }
        .unwrap()
    };
    let rel = |v: f64, want: f64| ((v - want) / want).abs() < 1e-5;
    let v = run(
        &pair(1e-200),
        1,
        BlockLanczos::new(2, 1, 1e-210, 0.0),
        false,
    );
    assert!(rel(v[0], 1e-200), "{v:?}");
    let eager = |tol| BlockLanczos::new(3, 1, tol, 0.0).eager(true);
    let v = run(&swap(1e200), 2, eager(1e190), false);
    assert!(rel(v[0], 1e200) && rel(v[1], 1e200), "{v:?}");
    // Single precision: squares of 1e∓30 leave the f32 range.
    let v = run(&pair(1e-30), 1, BlockLanczos::new(2, 1, 1e-37, 0.0), true);
    assert!(rel(v[0], 1e-30), "{v:?}");
    let v = run(&swap(1e30), 2, eager(1e25), true);
    assert!(rel(v[0], 1e30) && rel(v[1], 1e30), "{v:?}");
    // Residual columns of very different size (a·e₂ and b·e₃): each column's
    // own scale must survive, or the small one vanishes and the pair with
    // residual a = ‖A·x - 2x‖ is reported converged.
    let mixed = |a: f64, b: f64| {
        vec![
            vec![2.0, 0.0, a, 0.0],
            vec![0.0, 1.0, 0.0, b],
            vec![a, 0.0, 0.0, 0.0],
            vec![0.0, b, 0.0, 0.0],
        ]
    };
    // The largest eigenvalue is 1/2 + √(1/4 + b²) ≈ b.
    let big = |v: &[f64], b: f64| rel(v[0], b);
    for (a, b, tol, single) in [
        (1e-100, 1e100, 1e-110, false),
        (1e-200, 1e200, 1e-210, false),
        (1e-12, 1e12, 1e-13, true),
    ] {
        let v = run(&mixed(a, b), 2, eager(tol), single);
        assert!(big(&v, b), "a = {a:e}: {v:?}");
    }
}

#[test]
fn block_residual_beyond_the_range_is_an_error() {
    // Every map output and its stable norm is finite, but the residual
    // norm of the first Ritz pair exceeds the largest finite number.
    let h = 1.5e308;
    let m: [&[f64]; 5] = [
        &[0.0, 0.0, 1.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0, 1.0, 0.0],
        &[1.0, 0.0, 0.0, 1.0, h],
        &[0.0, 1.0, 1.0, 0.0, h],
        &[0.0, 0.0, h, h, 0.0],
    ];
    let alg = BlockLanczos::new(3, 1, 1e-10, 0.0).eager(true);
    let out = residual_run::<f64>(&m, 2, alg);
    assert!(matches!(out, Err(Error::NonFiniteIterate)), "{out:?}");
}
