//! `davidson_eigsolve` against hand-chosen spectra and nalgebra
//! Rayleigh–Ritz oracles.

mod common;

use std::convert::Infallible;

use common::*;
use krylov::{
    Basis, ConvergenceInfo, Correction, Davidson, Error, LinearMap, MatRef, Orth, VecBasis, Which,
    Workspace, davidson_eigsolve,
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

/// `n` eigenvalues of mixed signs with distinct magnitudes at both ends:
/// `-9, -7, -5`, an even grid on `[-3, 3]`, then `4, 4.5, 8`.
fn mixed_spectrum(n: usize) -> Vec<f64> {
    let mut eigs: Vec<f64> = (0..n)
        .map(|i| -3.0 + 6.0 * i as f64 / (n - 1) as f64)
        .collect();
    eigs[..3].copy_from_slice(&[-9.0, -7.0, -5.0]);
    eigs[n - 3..].copy_from_slice(&[4.0, 4.5, 8.0]);
    eigs
}

#[test]
fn converges_to_known_spectrum_all_scalars() {
    fn go<S: TestScalar>(seed: u64, correction: Correction) {
        let n = 40;
        let eigs = mixed_spectrum(n);
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
            let alg = Davidson::new(7, 4, b, 200, tol, 1e-10).correction(correction);
            for (which, want) in cases {
                let out = cold::<S>(&a, &mut jacobi::<S>(&a), &x0, 3, which, alg);
                assert!(
                    out.info.numiter >= 2,
                    "{correction:?} {which:?} b={b}: no restart"
                );
                check_converged(&a, &out, tol, &want);
                check_pairs(&a, &out, alg.get_krylovdim());
            }
        }
    }
    for correction in [Correction::Plain, Correction::Olsen] {
        go::<f64>(1, correction);
        go::<f32>(2, correction);
        go::<Complex64>(3, correction);
        go::<Complex32>(4, correction);
    }
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
    let alg = Davidson::new(9, 3, 3, 200, tol, 1e-10).correction(Correction::Plain);
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
    let alg = Davidson::new(8, 4, 1, 500, tol, 1e-10).correction(Correction::Plain);
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

    let alg = Davidson::new(j + 1, 3, 1, 1, 0.0, 0.0).correction(Correction::Plain);
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
    let plain = |alg: Davidson| alg.correction(Correction::Plain);
    let alg = plain(Davidson::new(5, 2, 3, 10, 1e-12, 1e-10));
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
    let alg = plain(Davidson::new(6, 2, 1, 10, f64::INFINITY, 1e-10));
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
    let alg = plain(Davidson::new(9, 3, 3, 100, tol, 1e-10));
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
    for correction in [Correction::Plain, Correction::Olsen] {
        warm_matches_cold(&a1, &a2, &x1, &x2, tol, correction);
    }
}

fn warm_matches_cold(
    a1: &DMatrix<Complex64>,
    a2: &DMatrix<Complex64>,
    x1: &[Vec<Complex64>],
    x2: &[Vec<Complex64>],
    tol: f64,
    correction: Correction,
) {
    let alg = Davidson::new(10, 3, 2, 100, tol, 1e-10).correction(correction);
    let (mut ws, mut ws_av) = workspaces(&to_dvec::<f64>(&x1[0]), 10);
    for (a, x0, want) in [
        (a1, x1, [1.0, 2.0]),
        (a1, x1, [1.0, 2.0]),
        (a2, x2, [-29.0, -28.0]),
    ] {
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
    let short = Davidson::new(10, 3, 2, 100, f64::INFINITY, 1e-10).correction(correction);
    let s0 = similar_calls();
    let out = run::<f64>(
        a1,
        &mut identity::<f64>,
        x1,
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
    let alg = Davidson::new(6, 2, 1, 10, 1e-9, 1e-10).correction(Correction::Plain);
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
    // adds one dropped attempt to the work of a `tol = ∞` run that scans
    // root 0 and stops: one norm for Plain; under Olsen also two inner
    // products and no update (`⟨x, M·r⟩ = ⟨x, M·x⟩ = 0` keeps `M·r = 0`).
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
        correction: Correction,
        pc: &mut impl krylov::Preconditioner<DVec<f64>>,
    ) -> ([usize; 4], ConvergenceInfo<DVec<f64>>) {
        let m = a.nrows();
        let alg = Davidson::new(m, 1, 1, 10, tol, 0.0).correction(correction);
        let (mut ws, mut ws_av) = workspaces(&to_dvec::<f64>(&x0[0]), m);
        let before = vector_ops();
        let out = run::<f64>(a, pc, x0, 1, Which::SmallestReal, alg, &mut ws, &mut ws_av);
        (ops_since(before), out.info)
    }
    for (correction, calls_per_attempt, want) in [
        (Correction::Plain, 1, [0, 1, 0, 0]),
        (Correction::Olsen, 2, [2, 1, 0, 0]),
    ] {
        let mut calls = 0;
        let mut counted = |cx: &mut (), th: f64, r: &DVec<f64>, t: &mut DVec<f64>| {
            calls += 1;
            identity::<f64>(cx, th, r, t)
        };
        let (stop, info) = budget(&swap, &ex, f64::INFINITY, correction, &mut counted);
        assert_eq!((info.converged, info.numops, calls), (1, p, 0));
        let mut calls = 0;
        let mut counted = |cx: &mut (), th: f64, r: &DVec<f64>, t: &mut DVec<f64>| {
            calls += 1;
            zero_precond::<f64>(cx, th, r, t)
        };
        let (drop, info) = budget(&swap, &ex, 0.5, correction, &mut counted);
        assert_eq!((info.converged, info.numiter, info.numops), (0, 1, p));
        assert_eq!(calls, calls_per_attempt, "{correction:?}: one attempt");
        let extra: [usize; 4] = core::array::from_fn(|i| drop[i] - stop[i]);
        assert_eq!(extra, want, "{correction:?}: [inner, norm, axpby, copy]");
    }

    // Exact Jacobi on a diagonal A from a nonconverged Ritz vector x: the
    // plain correction (θ − D)⁻¹·(D − θ)·x = −x lies in the subspace and is
    // dropped; Olsen's correction is not, and the iteration converges.
    let s = std::f64::consts::FRAC_1_SQRT_2;
    let t = 1.0 / 3f64.sqrt();
    for (diag, x0) in [
        (vec![1.0, 3.0], vec![c(s), c(s)]),
        (vec![1.0, 2.0, 4.0], vec![c(t), c(t), c(t)]),
    ] {
        let d = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(
            diag.iter().map(|&x| c(x)).collect(),
        ));
        let alg = Davidson::new(diag.len(), 1, 1, 10, 1e-9, 1e-8);
        let x0 = [x0];
        let out = cold::<f64>(
            &d,
            &mut jacobi::<f64>(&d),
            &x0,
            1,
            Which::SmallestReal,
            alg.correction(Correction::Plain),
        );
        assert_eq!(
            (out.info.converged, out.info.numiter, out.info.numops),
            (0, 1, 1)
        );
        let out = cold::<f64>(&d, &mut jacobi::<f64>(&d), &x0, 1, Which::SmallestReal, alg);
        check_converged(&d, &out, 1e-9, &[1.0]);
    }
    // The first Olsen direction on diag(1, 2, 4) from (1, 1, 1)/√3, by hand:
    // θ = 7/3, M = diag(3/4, 3, −3/5), M·r = −x, n = −1, d = ⟨x, M·x⟩ = 21/20,
    // so t = −x + (20/21)·M·x ∝ (−2, 13, −11).
    let d = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![c(1.0), c(2.0), c(4.0)]));
    let x = [vec![c(t), c(t), c(t)]];
    let want = span_eigenvalue(&d, &[x[0].clone(), vec![c(-2.0), c(13.0), c(-11.0)]]);
    let one = Davidson::new(2, 1, 1, 1, 0.0, 1e-10);
    let out = cold::<f64>(&d, &mut jacobi::<f64>(&d), &x, 1, Which::SmallestReal, one);
    assert_eq!(out.info.numops, 2);
    within(
        (out.vals[0] - want).abs(),
        eig_bound::<f64>(0.0, 3, op_norm(&d)),
        "first Olsen Ritz value",
    );

    // Immediate exact convergence.
    let d = DMatrix::from_diagonal(&nalgebra::DVector::from_vec(vec![c(1.0), c(2.0)]));
    let alg = Davidson::new(2, 1, 1, 10, 1e-9, 1e-8).correction(Correction::Plain);
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
        // Olsen's second call, on the Ritz vector, is checked as well.
        let mut calls = 0;
        let mut second = |_: &mut (), _: f64, r: &DVec<f64>, t: &mut DVec<f64>| {
            calls += 1;
            if calls == 2 {
                t.0.fill(bad);
            } else {
                t.0.copy_from_slice(&r.0);
            }
            Ok::<(), Infallible>(())
        };
        assert!(matches!(
            solve(&x0, &mut op, &mut second, 1, sr, ok, (6, 6)),
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

/// Lowest Rayleigh–Ritz value of `a` on the span of `vs`, from an f64
/// Gram–Schmidt basis with reorthogonalization.
fn span_eigenvalue(a: &DMatrix<Complex64>, vs: &[Vec<Complex64>]) -> f64 {
    let mut q: Vec<nalgebra::DVector<Complex64>> = Vec::new();
    for v in vs {
        let mut v = nalgebra::DVector::from_column_slice(v);
        for _ in 0..2 {
            for qi in &q {
                let c = qi.dotc(&v);
                v -= qi * c;
            }
        }
        let nv = v.norm();
        q.push(v / Complex64::new(nv, 0.0));
    }
    let qm = DMatrix::from_columns(&q);
    dense_eigenvalues(&(qm.adjoint() * a * &qm))[0]
}

/// `[[0, 1, 0], [1, 3, 0], [0, 0, c]]`: from `x0 = e1` the Ritz pair is
/// `x = e1`, `θ = 0` and the residual `r = e2`, all exact.
fn coupling(c: f64) -> DMatrix<Complex64> {
    let mut a = DMatrix::zeros(3, 3);
    a[(0, 1)] = Complex64::new(1.0, 0.0);
    a[(1, 0)] = Complex64::new(1.0, 0.0);
    a[(1, 1)] = Complex64::new(3.0, 0.0);
    a[(2, 2)] = Complex64::new(c, 0.0);
    a
}

/// Lowest eigenvalue of [`coupling`]`(c)` on span{e1, (0, α, β)}: the
/// projected matrix is `[[0, b], [b, s]]` with `b = α/√(α² + β²)` and
/// `s = (3α² + cβ²)/(α² + β²)`.
fn coupling_theta(c: f64, alpha: f64, beta: f64) -> f64 {
    let n2 = alpha * alpha + beta * beta;
    let b = alpha / n2.sqrt();
    let s = (3.0 * alpha * alpha + c * beta * beta) / n2;
    (s - (s * s + 4.0 * b * b).sqrt()) / 2.0
}

/// The fixed linear preconditioner `t ← M·v`, independent of `θ`.
fn fixed<S: TestScalar>(
    m: &DMatrix<Complex64>,
) -> impl FnMut(&mut (), S::Real, &DVec<S>, &mut DVec<S>) -> Result<(), Infallible> {
    let mut op = DenseOp::<S>::new(m);
    move |cx, _, v, t| op.apply_into(cx, v, t)
}

/// One Olsen correction of [`coupling`] from `x0 = e1` with the fixed `M`,
/// then a finishing pass.
fn first_olsen<S: TestScalar>(a: &DMatrix<Complex64>, m: &DMatrix<Complex64>) -> Out<S> {
    let c = |x: f64| Complex64::new(x, 0.0);
    let alg = Davidson::new(2, 1, 1, 1, 0.0, 1e-10);
    let e1 = vec![c(1.0), c(0.0), c(0.0)];
    cold::<S>(a, &mut fixed::<S>(m), &[e1], 1, Which::SmallestReal, alg)
}

#[test]
fn olsen_first_correction_matches_dense_formula() {
    fn go<S: TestScalar>() {
        let n = 6;
        let mut g = SplitMix(1);
        let a = hermitian(&mut g, n, <S as TestScalar>::COMPLEX);
        let x0 = random_vec(&mut g, n, <S as TestScalar>::COMPLEX);
        // Dense f64 reference: x, θ, r, the guarded Jacobi M and
        // t = M·r − ε·M·x with ε = ⟨x, M·r⟩/⟨x, M·x⟩ (x conjugated).
        let mut x = nalgebra::DVector::from_column_slice(&x0);
        x /= Complex64::new(x.norm(), 0.0);
        let theta = x.dotc(&(&a * &x)).re;
        let r = &a * &x - &x * Complex64::new(theta, 0.0);
        let jac = |v: &nalgebra::DVector<Complex64>| {
            nalgebra::DVector::from_fn(n, |i, _| {
                let den = theta - a[(i, i)].re;
                if den.abs() > GUARD { v[i] / den } else { v[i] }
            })
        };
        let (w, z) = (jac(&r), jac(&x));
        let eps = x.dotc(&w) / x.dotc(&z);
        let t = &w - &z * eps;
        let want = span_eigenvalue(&a, &[x0.clone(), t.as_slice().to_vec()]);

        let mut calls: Vec<(f64, Vec<Complex64>)> = Vec::new();
        let mut inner = jacobi::<S>(&a);
        let mut observed = |cx: &mut (), th: S::Real, v: &DVec<S>, t: &mut DVec<S>| {
            calls.push((to_f64(th), to_c64(v)));
            inner(cx, th, v, t)
        };
        let alg = Davidson::new(2, 1, 1, 1, 0.0, 1e-10);
        let out = cold::<S>(
            &a,
            &mut observed,
            std::slice::from_ref(&x0),
            1,
            Which::SmallestReal,
            alg,
        );
        let a_norm = op_norm(&a);
        assert_eq!(out.info.numops, 2, "the correction is accepted");
        within(
            (out.vals[0] - want).abs(),
            eig_bound::<S>(0.0, n, a_norm),
            "Ritz value on span{x0, t}",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].0, calls[1].0, "one shift for both calls");
        within(
            diff_norm(&calls[0].1, r.as_slice()),
            rounding::<S>(n, a_norm),
            "first call: the residual",
        );
        within(
            diff_norm(&calls[1].1, x.as_slice()),
            rounding::<S>(n, 1.0),
            "second call: the Ritz vector",
        );
    }
    go::<f64>();
    go::<Complex64>();
}

#[test]
fn olsen_zero_denominator_branches() {
    fn go<S: TestScalar>() {
        let c = |x: f64| Complex64::new(x, 0.0);
        let col = |v: [f64; 3]| nalgebra::DVector::from_iterator(3, v.map(c));
        let a = coupling(-1.0);
        let bound = eig_bound::<S>(0.0, 3, op_norm(&a));
        // (a) M·e1 = e3, M·e2 = e1 + e2: d = ⟨x, M·x⟩ = 0 and n = ⟨x, M·r⟩ = 1,
        // so t = −M·x = −e3 (Plain's span{e1, e2} gives (3 − √13)/2 and a
        // drop gives 0).
        // (b) M·e1 = e2, M·e2 = e3: n = d = 0, so t = M·r = e3.
        for (what, m) in [
            (
                "d = 0",
                [col([0.0, 0.0, 1.0]), col([1.0, 1.0, 0.0]), col([0.0; 3])],
            ),
            (
                "n = d = 0",
                [col([0.0, 1.0, 0.0]), col([0.0, 0.0, 1.0]), col([0.0; 3])],
            ),
        ] {
            let out = first_olsen::<S>(&a, &DMatrix::from_columns(&m));
            assert_eq!(out.info.numops, 2, "{what}: the correction is accepted");
            within(
                (out.vals[0] - coupling_theta(-1.0, 0.0, 1.0)).abs(),
                bound,
                &format!("{what}: Ritz value on span{{e1, e3}}"),
            );
        }
    }
    go::<f64>();
    go::<Complex64>();
}

#[test]
fn olsen_quotient_dynamic_range() {
    /// `M·e1 = z`, `M·e2 = w`, `M·e3 = e1`, so `n = w₀` and `d = z₀`; the
    /// first correction must span `(0, α, β)`. Complex cases also run with
    /// `z` and `w` times `i`, which leaves the span unchanged.
    fn case<S: TestScalar>(z: [Complex64; 3], w: [Complex64; 3], t: (f64, f64), what: &str) {
        let a = coupling(4.0);
        let want = coupling_theta(4.0, t.0, t.1);
        let bound = eig_bound::<S>(0.0, 3, op_norm(&a));
        let phases: &[Complex64] = if <S as TestScalar>::COMPLEX {
            &[Complex64::ONE, Complex64::I]
        } else {
            &[Complex64::ONE]
        };
        for &p in phases {
            let col = |v: [Complex64; 3]| nalgebra::DVector::from_iterator(3, v.map(|x| x * p));
            let e1 =
                nalgebra::DVector::from_vec(vec![Complex64::ONE, Complex64::ZERO, Complex64::ZERO]);
            let m = DMatrix::from_columns(&[col(z), col(w), e1]);
            let out = first_olsen::<S>(&a, &m);
            assert_eq!(out.info.numops, 2, "{what} (phase {p}): accepted");
            within(
                (out.vals[0] - want).abs(),
                bound,
                &format!("{what} (phase {p}): {} vs {want}", out.vals[0]),
            );
        }
    }
    let r = |x: f64| Complex64::new(x, 0.0);
    let (o, z0) = (r(1.0), r(0.0));
    // An unscaled `/` loses these: |w₀|² or |z₀|² leaves the range.
    case::<Complex64>(
        [o, z0, o],
        [r(1e200), r(1e200), z0],
        (1.0, -1.0),
        "(ii), q = 1e-200",
    );
    case::<Complex64>(
        [r(1e-200), z0, o],
        [r(2e-200), o, z0],
        (0.5, -1.0),
        "(ii), tiny",
    );
    case::<Complex64>(
        [r(2e-200), z0, o],
        [r(1e-200), o, z0],
        (1.0, -0.5),
        "(i), tiny",
    );
    case::<Complex64>(
        [r(2e200), z0, o],
        [r(1e200), o, z0],
        (1.0, -0.5),
        "(i), huge",
    );
    case::<Complex32>(
        [r(1e-30), z0, o],
        [r(2e-30), o, z0],
        (0.5, -1.0),
        "(ii), tiny",
    );

    /// Per precision: the smallest subnormal `u`, a value whose unscaled
    /// modulus overflows, and an inexact unit `b` (`b·(1/b) ≠ 1`).
    fn extremes<S: TestScalar>(u: f64, huge: f64, unit: f64) {
        let r = |x: f64| Complex64::new(x, 0.0);
        let (o, z0) = (r(1.0), r(0.0));
        let ci = |x: f64| Complex64::new(x, x);
        // A rounded modulus of a subnormal or an overflowing value is wrong.
        let b = ci(u);
        case::<S>([b, z0, o], [b, o, z0], (1.0, -1.0), "(i), subnormal");
        case::<S>([b, z0, o], [b * 2.0, o, z0], (0.5, -1.0), "(ii), subnormal");
        let b = ci(huge);
        case::<S>(
            [b, z0, o],
            [b, o, z0],
            (1.0, -1.0),
            "(i), overflowing modulus",
        );
        // A large tail turns a quotient off by one ulp into a different span.
        let (b, big) = (ci(unit), r(2f64.powi(60)));
        case::<S>([b, z0, big], [b, o, big], (1.0, 0.0), "(i), exact quotient");
        case::<S>(
            [b, z0, big],
            [b * 2.0, o, big * 2.0],
            (0.5, 0.0),
            "(ii), exact quotient",
        );
    }
    extremes::<Complex64>(f64::from_bits(1), 1.3e308, 1.2);
    extremes::<Complex32>(f64::from(f32::from_bits(1)), 3e38, 1.4);

    // A tiny correction δ·(0, 1, −1) survives only if `b/b` is exactly 1:
    // otherwise `b − q·b` leaves an `e1` component that dominates it, and
    // the orthogonalized candidate falls below `drop_tol`.
    let d = r(2f64.powi(-100));
    case::<f64>(
        [r(1.2), z0, d],
        [r(1.2), d, z0],
        (1.0, -1.0),
        "(i), tiny tail",
    );
    case::<f32>(
        [r(1.4), z0, d],
        [r(1.4), d, z0],
        (1.0, -1.0),
        "(i), tiny tail",
    );
    let ci = |x: f64| Complex64::new(x, x);
    case::<Complex64>(
        [ci(1.2), z0, d],
        [ci(1.2), d, z0],
        (1.0, -1.0),
        "(i), tiny tail",
    );
    case::<Complex32>(
        [ci(1.4), z0, d],
        [ci(1.4), d, z0],
        (1.0, -1.0),
        "(i), tiny tail",
    );
}

#[test]
fn olsen_needs_fewer_restarts_near_diagonal() {
    fn go<S: TestScalar>(seed: u64) {
        let n = 40;
        let mut g = SplitMix(seed);
        // A diagonal so close to the spectrum that the plain correction
        // mostly reproduces the Ritz vector.
        let a = near_diagonal(&mut g, &mixed_spectrum(n), <S as TestScalar>::COMPLEX, 0.05);
        // Ten times the rounding floor of one residual evaluation.
        let tol = 10.0 * rounding::<S>(n, op_norm(&a));
        let x0: Vec<_> = (0..3)
            .map(|_| random_vec(&mut g, n, <S as TestScalar>::COMPLEX))
            .collect();
        let alg = Davidson::new(7, 4, 3, 100, tol, 1e-10);
        let sr = Which::SmallestReal;
        let olsen = cold::<S>(&a, &mut jacobi::<S>(&a), &x0, 3, sr, alg);
        let plain = cold::<S>(
            &a,
            &mut jacobi::<S>(&a),
            &x0,
            3,
            sr,
            alg.correction(Correction::Plain),
        );
        check_converged(&a, &olsen, tol, &[-9.0, -7.0, -5.0]);
        check_pairs(&a, &olsen, alg.get_krylovdim());
        assert!(
            olsen.info.numiter < plain.info.numiter,
            "olsen {} vs plain {} restart cycles",
            olsen.info.numiter,
            plain.info.numiter
        );
    }
    go::<f64>(1);
    go::<Complex64>(3);
}
