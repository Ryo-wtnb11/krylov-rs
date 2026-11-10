//! `davidson_eigsolve` against hand-chosen spectra and nalgebra
//! Rayleigh–Ritz oracles.

mod common;

use std::convert::Infallible;

use common::*;
use krylov::{
    Basis, ConvergenceInfo, Davidson, Error, MatRef, Orth, VecBasis, Which, Workspace,
    davidson_eigsolve,
};
use nalgebra::DMatrix;
use num_complex::{Complex32, Complex64};

type Ws<S> = Workspace<DVec<S>, VecBasis<DVec<S>>>;

/// block2's denominator guard of the diagonal preconditioner.
const GUARD: f64 = 1e-12;

struct Out<S: TestScalar> {
    vals: Vec<f64>,
    vecs: Vec<Vec<Complex64>>,
    info: ConvergenceInfo<DVec<S>>,
}

fn workspaces<S: TestScalar>(x: &DVec<S>, m: usize) -> (Ws<S>, Ws<S>) {
    (
        Workspace::with_capacity(&mut (), x, m).unwrap(),
        Workspace::with_capacity(&mut (), x, m).unwrap(),
    )
}

fn starts<S: TestScalar>(x0: &[Vec<Complex64>]) -> Vec<DVec<S>> {
    x0.iter().map(|x| to_dvec(x)).collect()
}

fn identity<S: TestScalar>(
    _: &mut (),
    _: S::Real,
    r: &DVec<S>,
    t: &mut DVec<S>,
) -> Result<(), Infallible> {
    t.0.copy_from_slice(&r.0);
    Ok(())
}

fn zero_precond<S: TestScalar>(
    _: &mut (),
    _: S::Real,
    _: &DVec<S>,
    t: &mut DVec<S>,
) -> Result<(), Infallible> {
    t.0.fill(S::zero());
    Ok(())
}

/// block2's guarded diagonal correction `tᵢ = rᵢ/(θ − dᵢ)`, `tᵢ = rᵢ` where
/// `|θ − dᵢ| <= GUARD`, with `d` the diagonal of `a`.
fn jacobi<S: TestScalar>(
    a: &DMatrix<Complex64>,
) -> impl FnMut(&mut (), S::Real, &DVec<S>, &mut DVec<S>) -> Result<(), Infallible> {
    let d: Vec<f64> = (0..a.nrows()).map(|i| a[(i, i)].re).collect();
    move |_, theta, r, t| {
        let theta = to_f64(theta);
        for ((ti, &ri), &di) in t.0.iter_mut().zip(&r.0).zip(&d) {
            let den = theta - di;
            *ti = if den.abs() > GUARD {
                ri * S::from_c64(Complex64::new(1.0 / den, 0.0))
            } else {
                ri
            };
        }
        Ok(())
    }
}

/// `Q·diag(eigs)·Qᴴ` with `Q = exp(angle·K)`, `K` a random anti-Hermitian
/// matrix: the spectrum is exactly `eigs` and the diagonal stays near it.
fn near_diagonal(g: &mut SplitMix, eigs: &[f64], complex: bool, angle: f64) -> DMatrix<Complex64> {
    let n = eigs.len();
    let gm = DMatrix::from_fn(n, n, |_, _| g.scalar(complex));
    let q = ((&gm - gm.adjoint()) * Complex64::new(0.5 * angle, 0.0)).exp();
    let d = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
        n,
        eigs.iter().map(|&e| Complex64::new(e, 0.0)),
    ));
    let a = &q * d * q.adjoint();
    (&a + a.adjoint()) * Complex64::new(0.5, 0.0)
}

#[allow(clippy::too_many_arguments)]
fn run<S: TestScalar>(
    a: &DMatrix<Complex64>,
    precond: &mut impl krylov::Preconditioner<DVec<S>>,
    x0: &[Vec<Complex64>],
    howmany: usize,
    which: Which,
    alg: Davidson,
    ws: &mut Ws<S>,
    ws_av: &mut Ws<S>,
) -> Out<S> {
    let x0 = starts::<S>(x0);
    let mut op = DenseOp::<S>::new(a);
    let (vals, vecs, info) = davidson_eigsolve(
        &mut (),
        &mut op,
        precond,
        &x0,
        howmany,
        which,
        &alg,
        ws,
        ws_av,
    )
    .unwrap();
    assert_eq!(info.numops, op.applications);
    assert_eq!(vals.len(), vecs.len());
    assert_eq!(vals.len(), info.residual.len());
    assert_eq!(vals.len(), info.normres.len());
    Out {
        vals: vals.into_iter().map(to_f64).collect(),
        vecs: vecs.iter().map(to_c64).collect(),
        info,
    }
}

fn cold<S: TestScalar>(
    a: &DMatrix<Complex64>,
    precond: &mut impl krylov::Preconditioner<DVec<S>>,
    x0: &[Vec<Complex64>],
    howmany: usize,
    which: Which,
    alg: Davidson,
) -> Out<S> {
    let (mut ws, mut ws_av) = workspaces(&to_dvec::<S>(&x0[0]), alg.get_krylovdim());
    run(a, precond, x0, howmany, which, alg, &mut ws, &mut ws_av)
}

/// The reported residuals are `A·x − θ·x` and their norms are `normres`, and
/// the eigenvectors are orthonormal, up to rounding in `S`. The residual
/// comes from the stored images `A·V`, which every restart rotates once
/// more, so its rounding grows with `numiter`; each returned vector combines
/// up to `krylovdim` unit basis vectors.
fn check_pairs<S: TestScalar>(a: &DMatrix<Complex64>, out: &Out<S>, krylovdim: usize) {
    let n = a.nrows();
    let relation = (out.info.numiter + 1) as f64 * rounding::<S>(n, op_norm(a));
    for i in 0..out.vals.len() {
        let x = nalgebra::DVector::from_column_slice(&out.vecs[i]);
        let r = a * &x - &x * Complex64::new(out.vals[i], 0.0);
        let diff = diff_norm(r.as_slice(), &to_c64(&out.info.residual[i]));
        within(diff, relation, &format!("pair {i}: residual vector"));
        let nr = to_f64(out.info.normres[i]);
        within(
            (nr - r.norm()).abs(),
            relation,
            &format!("pair {i}: normres"),
        );
    }
    let vs: Vec<DVec<Complex64>> = out.vecs.iter().map(|x| to_dvec(x)).collect();
    within(
        orthogonality_error(&vs),
        rounding::<S>(n, krylovdim as f64),
        "eigenvector orthonormality",
    );
}

/// Every returned pair converged: value within [`eig_bound`] of `want` and
/// direct dense residual within the same bound.
fn check_converged<S: TestScalar>(a: &DMatrix<Complex64>, out: &Out<S>, tol: f64, want: &[f64]) {
    let bound = eig_bound::<S>(tol, a.nrows(), op_norm(a));
    assert_eq!(out.info.converged, want.len(), "{:?}", out.info.normres);
    assert_eq!(out.vals.len(), want.len());
    for (i, (v, e)) in out.vals.iter().zip(want).enumerate() {
        within((v - e).abs(), bound, &format!("value {i}: {v} vs {e}"));
        within(
            residual(a, *v, &out.vecs[i]),
            bound,
            &format!("pair {i}: dense residual"),
        );
    }
}

#[test]
fn converges_to_known_spectrum_all_scalars() {
    fn go<S: TestScalar>(seed: u64) {
        let n = 40;
        // Mixed signs with distinct magnitudes at both ends.
        let mut eigs: Vec<f64> = (0..n)
            .map(|i| -3.0 + 6.0 * i as f64 / (n - 1) as f64)
            .collect();
        eigs[..3].copy_from_slice(&[-9.0, -7.0, -5.0]);
        eigs[n - 3..].copy_from_slice(&[4.0, 4.5, 8.0]);
        let mut g = SplitMix(seed);
        // A diagonal near the spectrum, but not so near that the plain
        // correction mostly reproduces the Ritz vector and stalls.
        let a = near_diagonal(&mut g, &eigs, <S as TestScalar>::COMPLEX, 0.5);
        let a_norm = op_norm(&a);
        // Ten times the rounding floor of one residual evaluation.
        let tol = 10.0 * rounding::<S>(n, a_norm);
        let cases: [(Which, [f64; 3]); 3] = [
            (Which::SmallestReal, [-9.0, -7.0, -5.0]),
            (Which::LargestReal, [8.0, 4.5, 4.0]),
            (Which::LargestMagnitude, [-9.0, 8.0, -7.0]),
        ];
        for b in [1, 3] {
            let x0: Vec<_> = (0..b)
                .map(|_| random_vec(&mut g, n, <S as TestScalar>::COMPLEX))
                .collect();
            // Seven vectors with four kept force restarts before three roots
            // converge.
            let alg = Davidson::new(7, 4, b, 200, tol, 1e-10);
            for (which, want) in cases {
                let out = cold::<S>(&a, &mut jacobi::<S>(&a), &x0, 3, which, alg);
                assert!(out.info.numiter >= 2, "{which:?} b={b}: no restart");
                check_converged(&a, &out, tol, &want);
                check_pairs(&a, &out, alg.get_krylovdim());
            }
        }
    }
    go::<f64>(1);
    go::<f32>(2);
    go::<Complex64>(3);
    go::<Complex32>(4);
}

#[test]
fn degenerate_root_found_with_every_copy_after_restarts() {
    let n = 30;
    let mut eigs: Vec<f64> = (0..n).map(|i| 3.0 + i as f64 * 0.25).collect();
    eigs[..6].copy_from_slice(&[1.0, 1.0, 1.0, 1.5, 1.6, 1.7]);
    let mut g = SplitMix(11);
    let a = with_spectrum(&mut g, &eigs, false);
    let tol = 1e-9;
    // Three random starts: their projections onto the eigenspace have rank 3.
    let x0: Vec<_> = (0..3).map(|_| random_vec(&mut g, n, false)).collect();
    let alg = Davidson::new(9, 3, 3, 200, tol, 1e-10);
    let out = cold::<f64>(&a, &mut identity::<f64>, &x0, 3, Which::SmallestReal, alg);
    assert!(
        out.info.numiter >= 3,
        "too few restarts: {}",
        out.info.numiter
    );
    check_converged(&a, &out, tol, &[1.0, 1.0, 1.0]);
    check_pairs(&a, &out, alg.get_krylovdim());
}

/// A `VecBasis` that counts appended vectors.
struct CountingBasis {
    inner: VecBasis<DVec<f64>>,
    pushes: usize,
}

impl Basis<DVec<f64>> for CountingBasis {
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    fn push(&mut self, cx: &mut (), v: &DVec<f64>, a: f64) -> Result<(), Infallible> {
        self.pushes += 1;
        self.inner.push(cx, v, a)
    }
    fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
    }
    fn project(&self, cx: &mut (), x: &DVec<f64>, out: &mut [f64]) -> Result<(), Infallible> {
        self.inner.project(cx, x, out)
    }
    fn combine_into(
        &self,
        cx: &mut (),
        c: &[f64],
        y: &mut DVec<f64>,
        a: f64,
        b: f64,
    ) -> Result<(), Infallible> {
        self.inner.combine_into(cx, c, y, a, b)
    }
    fn rotate(&mut self, cx: &mut (), q: MatRef<'_, f64>, keep: usize) -> Result<(), Infallible> {
        self.inner.rotate(cx, q, keep)
    }
    fn inner_at(&self, cx: &mut (), i: usize, x: &DVec<f64>) -> Result<f64, Infallible> {
        self.inner.inner_at(cx, i, x)
    }
    fn axpy_at(&self, cx: &mut (), i: usize, a: f64, y: &mut DVec<f64>) -> Result<(), Infallible> {
        self.inner.axpy_at(cx, i, a, y)
    }
}

#[test]
fn restarts_keep_cached_images_accurate() {
    let n = 60;
    let eigs: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.5).collect();
    let mut g = SplitMix(21);
    let a = with_spectrum(&mut g, &eigs, false);
    let tol = 1e-9;
    let x = to_dvec::<f64>(&random_vec(&mut g, n, false));
    let alg = Davidson::new(8, 4, 1, 500, tol, 1e-10);
    for orth in [Orth::Cgs2, Orth::Mgs2] {
        let alg = alg.orth(orth);
        let v = CountingBasis {
            inner: VecBasis::with_capacity(&mut (), &x, 8).unwrap(),
            pushes: 0,
        };
        let mut ws = Workspace::new(&mut (), v, &x).unwrap();
        let mut ws_av = Workspace::with_capacity(&mut (), &x, 8).unwrap();
        let mut op = DenseOp::<f64>::new(&a);
        let x0 = [DVec(x.0.clone())];
        let (vals, vecs, info) = davidson_eigsolve(
            &mut (),
            &mut op,
            &mut identity::<f64>,
            &x0,
            2,
            Which::SmallestReal,
            &alg,
            &mut ws,
            &mut ws_av,
        )
        .unwrap();
        assert!(info.numiter >= 2, "{orth:?}: no restart");
        assert_eq!(info.numops, op.applications);
        // One application per accepted basis vector: nothing is re-applied
        // at a restart.
        assert_eq!(ws.basis().pushes, info.numops, "{orth:?}");
        assert_eq!(ws.basis().pushes, op.applications, "{orth:?}");
        let out = Out {
            vals: vals.into_iter().collect(),
            vecs: vecs.iter().map(to_c64).collect(),
            info,
        };
        check_converged(&a, &out, tol, &[1.0, 1.5]);
        check_pairs(&a, &out, 8);
    }
}

#[test]
fn identity_preconditioner_spans_the_krylov_space() {
    let n = 20;
    let j = 5;
    let mut g = SplitMix(41);
    let a = hermitian(&mut g, n, true);
    let x = random_vec(&mut g, n, true);
    // Orthonormal basis of span[x, A·x, …, Aʲ·x] by Gram–Schmidt with
    // reorthogonalization in f64, then its Rayleigh–Ritz values.
    let mut q: Vec<nalgebra::DVector<Complex64>> = Vec::new();
    let mut v = nalgebra::DVector::from_column_slice(&x);
    for _ in 0..=j {
        for _ in 0..2 {
            for qi in &q {
                let c = qi.dotc(&v);
                v -= qi * c;
            }
        }
        let nv = v.norm();
        v /= Complex64::new(nv, 0.0);
        q.push(v.clone());
        v = &a * &v;
    }
    let qm = DMatrix::from_columns(&q);
    let want = dense_eigenvalues(&(qm.adjoint() * &a * &qm));

    let alg = Davidson::new(j + 1, 3, 1, 1, 0.0, 0.0);
    let out = cold::<Complex64>(
        &a,
        &mut identity::<Complex64>,
        &[x],
        3,
        Which::SmallestReal,
        alg,
    );
    assert_eq!(out.info.numops, j + 1);
    assert_eq!(out.info.numiter, 1);
    let lead = out.info.normres.iter().take_while(|&&r| r <= 0.0).count();
    assert_eq!(out.info.converged, lead);
    // The same subspace in both: each of the `j + 1` steps adds one
    // orthogonalization's rounding to its basis.
    let bound = (j + 1) as f64 * rounding::<Complex64>(n, op_norm(&a));
    for (i, (v, e)) in out.vals.iter().zip(&want).enumerate() {
        within((v - e).abs(), bound, &format!("Ritz value {i}"));
    }
}

#[test]
fn short_start_and_small_ritz_dimension() {
    let c = |x: f64| Complex64::new(x, 0.0);
    let e = |i: usize, n: usize| (0..n).map(|k| c(if k == i { 1.0 } else { 0.0 })).collect();

    // One invariant start, blocksize 3: only the one Ritz pair exists, it has
    // zero residual and nothing is corrected.
    let a = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![c(1.0), c(2.0), c(3.0)]));
    let alg = Davidson::new(5, 2, 3, 10, 1e-12, 1e-10);
    let out = cold::<f64>(
        &a,
        &mut identity::<f64>,
        &[e(0, 3)],
        2,
        Which::SmallestReal,
        alg,
    );
    assert_eq!(out.vals, [1.0]);
    assert_eq!(
        (out.info.converged, out.info.numops, out.info.numiter),
        (1, 1, 1)
    );

    // Dependent and zero starts are dropped: with `tol = ∞` the first pass
    // finishes, and `numops` counts only the two accepted starts.
    let mut g = SplitMix(51);
    let n = 10;
    let b = with_spectrum(
        &mut g,
        &(1..=n).map(|i| i as f64).collect::<Vec<_>>(),
        false,
    );
    let (u, w) = (random_vec(&mut g, n, false), random_vec(&mut g, n, false));
    let u2: Vec<_> = u.iter().map(|z| z * 2.0).collect();
    let x0 = [u, u2, vec![c(0.0); n], w];
    let alg = Davidson::new(6, 2, 1, 10, f64::INFINITY, 1e-10);
    let out = cold::<f64>(&b, &mut identity::<f64>, &x0, 2, Which::SmallestReal, alg);
    assert_eq!((out.info.numops, out.vals.len()), (2, 2));

    // A block where the lowest root's correction is always zero still
    // progresses through the other roots' corrections.
    let x0: Vec<_> = (0..3).map(|_| random_vec(&mut g, n, false)).collect();
    let mut dropped = 0;
    let mut partial = |_: &mut (), theta: f64, r: &DVec<f64>, t: &mut DVec<f64>| {
        if theta < 1.5 {
            dropped += 1;
            t.0.fill(0.0);
        } else {
            t.0.copy_from_slice(&r.0);
        }
        Ok::<(), Infallible>(())
    };
    let tol = 1e-9;
    let alg = Davidson::new(9, 3, 3, 100, tol, 1e-10);
    let out = cold::<f64>(&b, &mut partial, &x0, 1, Which::SmallestReal, alg);
    assert!(dropped > 0, "no correction was dropped");
    assert!(out.info.numops > 3, "no progress beyond the starts");
    check_converged(&b, &out, tol, &[1.0]);
}

#[test]
fn warm_call_matches_cold_and_allocates_only_results() {
    let n = 30;
    let mut g = SplitMix(61);
    let a1 = near_diagonal(
        &mut g,
        &(1..=n).map(|i| i as f64).collect::<Vec<_>>(),
        false,
        0.2,
    );
    let a2 = near_diagonal(
        &mut g,
        &(0..n).map(|i| -(i as f64)).collect::<Vec<_>>(),
        false,
        0.2,
    );
    let x1 = vec![random_vec(&mut g, n, false)];
    let x2 = vec![random_vec(&mut g, n, false), random_vec(&mut g, n, false)];
    let tol = 1e-9;
    let alg = Davidson::new(10, 3, 2, 100, tol, 1e-10);
    let (mut ws, mut ws_av) = workspaces(&to_dvec::<f64>(&x1[0]), 10);
    for (a, x0, want) in [
        (&a1, &x1, [1.0, 2.0]),
        (&a1, &x1, [1.0, 2.0]),
        (&a2, &x2, [-29.0, -28.0]),
    ] {
        let x0 = x0.as_slice();
        let s0 = similar_calls();
        let warm = run::<f64>(
            a,
            &mut jacobi::<f64>(a),
            x0,
            2,
            Which::SmallestReal,
            alg,
            &mut ws,
            &mut ws_av,
        );
        assert_eq!(similar_calls() - s0, 2 * warm.vals.len());
        let cold = cold::<f64>(a, &mut jacobi::<f64>(a), x0, 2, Which::SmallestReal, alg);
        // Same inputs, same arithmetic: a warm call keeps no state.
        assert_eq!(warm.vals, cold.vals);
        assert_eq!(warm.info.normres, cold.info.normres);
        assert_eq!(
            (warm.info.numops, warm.info.numiter),
            (cold.info.numops, cold.info.numiter)
        );
        check_converged(a, &warm, tol, &want);
    }

    // A short (immediately accepted) and a long run construct the same
    // number of vectors per returned pair.
    let short = Davidson::new(10, 3, 2, 100, f64::INFINITY, 1e-10);
    let s0 = similar_calls();
    let out = run::<f64>(
        &a1,
        &mut identity::<f64>,
        &x1,
        2,
        Which::SmallestReal,
        short,
        &mut ws,
        &mut ws_av,
    );
    assert_eq!(out.info.numops, 1);
    assert_eq!(similar_calls() - s0, 2 * out.vals.len());
}

#[test]
fn stagnation_and_budget_counts() {
    let c = |x: f64| Complex64::new(x, 0.0);
    let n = 10;
    let mut g = SplitMix(71);
    let a = with_spectrum(
        &mut g,
        &(1..=n).map(|i| i as f64).collect::<Vec<_>>(),
        false,
    );

    // Zero preconditioner: every correction is dropped in the first pass.
    let x0: Vec<_> = (0..3).map(|_| random_vec(&mut g, n, false)).collect();
    let alg = Davidson::new(6, 2, 1, 10, 1e-9, 1e-10);
    let out = cold::<f64>(
        &a,
        &mut zero_precond::<f64>,
        &x0,
        1,
        Which::SmallestReal,
        alg,
    );
    assert_eq!(
        (out.info.converged, out.info.numiter, out.info.numops),
        (0, 1, 3)
    );
    assert!(out.vals[0].is_finite() && out.info.normres[0].is_finite());

    // `[[0, I], [I, 0]]` from `p` coordinate starts: every Ritz value is 0
    // with unit residual. The zero-preconditioned run scans root 0 only and
    // adds one dropped attempt (one norm) to the work of a `tol = ∞` run
    // that scans root 0 and stops.
    let p = 4;
    let swap = DMatrix::from_fn(2 * p, 2 * p, |i, j| {
        c(if i.abs_diff(j) == p { 1.0 } else { 0.0 })
    });
    let ex: Vec<Vec<Complex64>> = (0..p)
        .map(|i| {
            (0..2 * p)
                .map(|k| c(if k == i { 1.0 } else { 0.0 }))
                .collect()
        })
        .collect();
    fn budget(
        a: &DMatrix<Complex64>,
        x0: &[Vec<Complex64>],
        tol: f64,
        pc: &mut impl krylov::Preconditioner<DVec<f64>>,
    ) -> ([usize; 4], ConvergenceInfo<DVec<f64>>) {
        let m = a.nrows();
        let alg = Davidson::new(m, 1, 1, 10, tol, 0.0);
        let (mut ws, mut ws_av) = workspaces(&to_dvec::<f64>(&x0[0]), m);
        let before = vector_ops();
        let out = run::<f64>(a, pc, x0, 1, Which::SmallestReal, alg, &mut ws, &mut ws_av);
        (ops_since(before), out.info)
    }
    let (stop, info) = budget(&swap, &ex, f64::INFINITY, &mut identity::<f64>);
    assert_eq!((info.converged, info.numops), (1, p));
    let (drop, info) = budget(&swap, &ex, 0.5, &mut zero_precond::<f64>);
    assert_eq!((info.converged, info.numiter, info.numops), (0, 1, p));
    let extra: [usize; 4] = core::array::from_fn(|i| drop[i] - stop[i]);
    assert_eq!(extra, [0, 1, 0, 0], "[inner, norm, axpby, copy]");

    // Plain Jacobi on diag(1, 3) from (1, 1)/√2: the correction is −x0,
    // which lies in the subspace and is dropped.
    let d = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![c(1.0), c(3.0)]));
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let alg = Davidson::new(2, 1, 1, 10, 1e-9, 1e-8);
    let out = cold::<f64>(
        &d,
        &mut jacobi::<f64>(&d),
        &[vec![c(s), c(s)]],
        1,
        Which::SmallestReal,
        alg,
    );
    assert_eq!(
        (out.info.converged, out.info.numiter, out.info.numops),
        (0, 1, 1)
    );

    // Immediate exact convergence.
    let d = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![c(1.0), c(2.0)]));
    let out = cold::<f64>(
        &d,
        &mut identity::<f64>,
        &[vec![c(1.0), c(0.0)]],
        1,
        Which::SmallestReal,
        alg,
    );
    assert_eq!(out.vals, [1.0]);
    assert_eq!(
        (out.info.converged, out.info.numiter, out.info.numops),
        (1, 1, 1)
    );
}

fn solve<S: TestScalar>(
    x0: &[DVec<S>],
    op: &mut impl krylov::LinearMap<DVec<S>>,
    precond: &mut impl krylov::Preconditioner<DVec<S>>,
    howmany: usize,
    which: Which,
    alg: Davidson,
    caps: (usize, usize),
) -> Option<Error<Infallible>> {
    let x = DVec(vec![S::zero(); 4]);
    let mut ws = Workspace::with_capacity(&mut (), &x, caps.0).unwrap();
    let mut ws_av = Workspace::with_capacity(&mut (), &x, caps.1).unwrap();
    davidson_eigsolve(
        &mut (),
        op,
        precond,
        x0,
        howmany,
        which,
        &alg,
        &mut ws,
        &mut ws_av,
    )
    .err()
}

#[test]
fn errors() {
    let c = |x: f64| Complex64::new(x, 0.0);
    let a = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![
        c(1.0),
        c(2.0),
        c(3.0),
        c(4.0),
    ]));
    let x0 = starts::<f64>(&[
        vec![c(1.0), c(1.0), c(0.0), c(0.0)],
        vec![c(0.0), c(1.0), c(1.0), c(1.0)],
    ]);
    let ok = Davidson::new(6, 2, 2, 10, 1e-8, 1e-10);
    let mut op = DenseOp::<f64>::new(&a);
    let id = &mut identity::<f64>;
    let sr = Which::SmallestReal;
    let invalid = |e: Option<Error<Infallible>>| matches!(e, Some(Error::InvalidArgument(_)));

    assert!(solve(&x0, &mut op, id, 1, sr, ok, (6, 6)).is_none());
    assert!(invalid(solve(&[], &mut op, id, 1, sr, ok, (6, 6))));
    for alg in [
        Davidson::new(6, 2, 0, 10, 1e-8, 1e-10),
        Davidson::new(6, 2, 2, 0, 1e-8, 1e-10),
        Davidson::new(6, 2, 2, 10, f64::NAN, 1e-10),
        Davidson::new(6, 2, 2, 10, -1.0, 1e-10),
        Davidson::new(6, 2, 2, 10, 1e-8, f64::NAN),
        Davidson::new(6, 2, 2, 10, 1e-8, -0.1),
        Davidson::new(6, 2, 2, 10, 1e-8, 1.0),
        // restartdim > krylovdim, without wrapping in krylovdim − restartdim.
        Davidson::new(2, usize::MAX, 1, 10, 1e-8, 1e-10),
        // blocksize > krylovdim − restartdim.
        Davidson::new(6, 2, 5, 10, 1e-8, 1e-10),
    ] {
        assert!(
            invalid(solve(&x0, &mut op, id, 1, sr, alg, (6, 6))),
            "{alg:?}"
        );
    }
    assert!(
        invalid(solve(&x0, &mut op, id, 0, sr, ok, (6, 6))),
        "howmany 0"
    );
    assert!(
        invalid(solve(&x0, &mut op, id, 3, sr, ok, (6, 6))),
        "restartdim < howmany"
    );
    let seven: Vec<_> = (0..7).map(|_| DVec(x0[0].0.clone())).collect();
    assert!(
        invalid(solve(&seven, &mut op, id, 1, sr, ok, (6, 6))),
        "x0 > krylovdim"
    );
    assert!(invalid(solve(
        &x0,
        &mut op,
        id,
        1,
        Which::LargestImag,
        ok,
        (6, 6)
    )));
    for caps in [(5, 6), (6, 5)] {
        assert!(matches!(
            solve(&x0, &mut op, id, 1, sr, ok, caps),
            Some(Error::InsufficientCapacity {
                needed: 6,
                capacity: 5
            })
        ));
    }

    // drop_tol is checked after conversion: 1 − 2⁻³⁰ rounds to 1 in f32.
    let near_one = 1.0 - 2f64.powi(-30);
    let alg = Davidson::new(6, 2, 2, 10, 1e-4, near_one);
    assert!(solve(&x0, &mut op, id, 1, sr, alg, (6, 6)).is_none());
    let x32 = starts::<f32>(&[vec![c(1.0), c(1.0), c(0.0), c(0.0)]]);
    assert!(invalid(solve(
        &x32,
        &mut DenseOp::<f32>::new(&a),
        &mut identity::<f32>,
        1,
        sr,
        alg,
        (6, 6)
    )));

    for bad in [f64::NAN, f64::INFINITY] {
        let x = [DVec(vec![1.0, bad, 0.0, 0.0])];
        assert!(matches!(
            solve(&x, &mut op, id, 1, sr, ok, (6, 6)),
            Some(Error::NonFiniteStartVector)
        ));
        let mut map = |_: &mut (), _: &DVec<f64>, y: &mut DVec<f64>| {
            y.0.fill(bad);
            Ok::<(), Infallible>(())
        };
        assert!(matches!(
            solve(&x0, &mut map, id, 1, sr, ok, (6, 6)),
            Some(Error::NonFiniteIterate)
        ));
        let mut pc = |_: &mut (), _: f64, _: &DVec<f64>, t: &mut DVec<f64>| {
            t.0.fill(bad);
            Ok::<(), Infallible>(())
        };
        assert!(matches!(
            solve(&x0, &mut op, &mut pc, 1, sr, ok, (6, 6)),
            Some(Error::NonFiniteIterate)
        ));
    }
    let zeros = [DVec(vec![0.0; 4]), DVec(vec![0.0; 4])];
    assert!(matches!(
        solve(&zeros, &mut op, id, 1, sr, ok, (6, 6)),
        Some(Error::ZeroStartVector)
    ));
}

/// A basis that reports a capacity but stores nothing: it is never touched
/// when the call is rejected up front.
struct LazyBasis(usize);

impl Basis<DVec<f64>> for LazyBasis {
    fn len(&self) -> usize {
        0
    }
    fn capacity(&self) -> usize {
        self.0
    }
    fn push(&mut self, _: &mut (), _: &DVec<f64>, _: f64) -> Result<(), Infallible> {
        unreachable!("push")
    }
    fn truncate(&mut self, _: usize) {
        unreachable!("truncate")
    }
    fn project(&self, _: &mut (), _: &DVec<f64>, _: &mut [f64]) -> Result<(), Infallible> {
        unreachable!("project")
    }
    fn combine_into(
        &self,
        _: &mut (),
        _: &[f64],
        _: &mut DVec<f64>,
        _: f64,
        _: f64,
    ) -> Result<(), Infallible> {
        unreachable!("combine_into")
    }
    fn rotate(&mut self, _: &mut (), _: MatRef<'_, f64>, _: usize) -> Result<(), Infallible> {
        unreachable!("rotate")
    }
    fn inner_at(&self, _: &mut (), _: usize, _: &DVec<f64>) -> Result<f64, Infallible> {
        unreachable!("inner_at")
    }
    fn axpy_at(&self, _: &mut (), _: usize, _: f64, _: &mut DVec<f64>) -> Result<(), Infallible> {
        unreachable!("axpy_at")
    }
}

#[test]
#[cfg(target_pointer_width = "64")]
fn oversized_projected_problem_is_rejected_before_any_work() {
    // krylovdim² overflows usize; 2³¹ squared fits but its f64 bytes do not.
    for m in [1usize << 32, 1usize << 31] {
        let x0 = [DVec(vec![1.0])];
        let mut ws = Workspace::new(&mut (), LazyBasis(m), &x0[0]).unwrap();
        let mut ws_av = Workspace::new(&mut (), LazyBasis(m), &x0[0]).unwrap();
        let mut op = |_: &mut (), x: &DVec<f64>, y: &mut DVec<f64>| {
            y.0.copy_from_slice(&x.0);
            Ok::<(), Infallible>(())
        };
        let alg = Davidson::new(m, 1, 1, 1, 0.0, 0.0);
        let e = davidson_eigsolve(
            &mut (),
            &mut op,
            &mut identity::<f64>,
            &x0,
            1,
            Which::SmallestReal,
            &alg,
            &mut ws,
            &mut ws_av,
        )
        .err();
        assert!(matches!(e, Some(Error::InvalidArgument(_))), "m = {m}");
    }
}
