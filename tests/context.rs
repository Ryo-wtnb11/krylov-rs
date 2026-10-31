//! The execution context reaches every vector, basis and map operation of
//! every solver through the one `&mut` borrow the caller passes.
//!
//! `Arena` owns all payloads and `H` is only an index into it, so no vector
//! operation can run without the context, and the map reads and writes the
//! same arena through `cx` (no `RefCell`, nothing captured mutably). A shared
//! operation counter injects one vector or map failure at every position of a
//! solve, which must come back as `Error::Vector` with no later operation.

mod common;

use std::marker::PhantomData;

use common::*;
use krylov::{
    AdjointMap, Arnoldi, BlockLanczos, Cg, Error, Gkl, Gmres, Lanczos, LinearMap, Orth,
    RestartResidual, Vector, Which, Workspace, arnoldi_eigsolve, block_eigsolve, eigsolve,
    expintegrator, exponentiate, linsolve, realeigsolve, schursolve, svdsolve,
};
use nalgebra::DMatrix;
use num_complex::Complex64;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fail {
    Vector,
    Map,
}

struct Arena<S> {
    data: Vec<Vec<S>>,
    vector_ops: usize,
    map_calls: usize,
    fail_vector_at: Option<usize>,
    fail_map_at: Option<usize>,
}

impl<S: TestScalar> Arena<S> {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            vector_ops: 0,
            map_calls: 0,
            fail_vector_at: None,
            fail_map_at: None,
        }
    }
    /// Stores an input vector without counting it as a solver operation.
    fn input(&mut self, x: &[Complex64]) -> H<S> {
        self.data.push(x.iter().map(|&z| S::from_c64(z)).collect());
        H(self.data.len() - 1, PhantomData)
    }
    fn vector_op(&mut self) -> Result<(), Fail> {
        let i = self.vector_ops;
        self.vector_ops += 1;
        if self.fail_vector_at == Some(i) {
            return Err(Fail::Vector);
        }
        Ok(())
    }
    /// `y ← A·x` (or `Aᴴ·x`) for the column-major `m × n` matrix `a`.
    fn apply(&mut self, a: &Mat<S>, adjoint: bool, x: &H<S>, y: &H<S>) -> Result<(), Fail> {
        let j = self.map_calls;
        self.map_calls += 1;
        if self.fail_map_at == Some(j) {
            return Err(Fail::Map);
        }
        let xs = self.data[x.0].clone();
        let ys = &mut self.data[y.0];
        assert_eq!(
            (xs.len(), ys.len()),
            if adjoint { (a.m, a.n) } else { (a.n, a.m) }
        );
        ys.fill(S::zero());
        for c in 0..a.n {
            for r in 0..a.m {
                let e = a.a[r + c * a.m];
                if adjoint {
                    ys[c] += e.cj() * xs[r];
                } else {
                    ys[r] += e * xs[c];
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct H<S>(usize, PhantomData<S>);

impl<S: TestScalar> Vector for H<S> {
    type Scalar = S;
    type Error = Fail;
    type Context = Arena<S>;
    fn similar(&self, cx: &mut Arena<S>) -> Result<Self, Fail> {
        cx.vector_op()?;
        let n = cx.data[self.0].len();
        cx.data.push(vec![S::zero(); n]);
        Ok(H(cx.data.len() - 1, PhantomData))
    }
    fn copy_from(&mut self, cx: &mut Arena<S>, x: &Self) -> Result<(), Fail> {
        cx.vector_op()?;
        let xs = cx.data[x.0].clone();
        cx.data[self.0].copy_from_slice(&xs);
        Ok(())
    }
    fn inner(&self, cx: &mut Arena<S>, y: &Self) -> Result<S, Fail> {
        cx.vector_op()?;
        let mut s = S::zero();
        for (a, b) in cx.data[self.0].iter().zip(&cx.data[y.0]) {
            s += a.cj() * *b;
        }
        Ok(s)
    }
    fn norm(&self, cx: &mut Arena<S>) -> Result<S::Real, Fail> {
        cx.vector_op()?;
        let s: f64 = cx.data[self.0].iter().map(|a| a.to_c64().norm_sqr()).sum();
        Ok(<S::Real as num_traits::NumCast>::from(s.sqrt()).unwrap())
    }
    fn axpby(&mut self, cx: &mut Arena<S>, a: S, x: &Self, b: S) -> Result<(), Fail> {
        cx.vector_op()?;
        let xs = cx.data[x.0].clone();
        for (s, x) in cx.data[self.0].iter_mut().zip(&xs) {
            *s = if b == S::zero() {
                a * *x
            } else {
                a * *x + b * *s
            };
        }
        Ok(())
    }
}

/// Column-major `m × n` matrix in the test scalar.
struct Mat<S> {
    a: Vec<S>,
    m: usize,
    n: usize,
}

impl<S: TestScalar> Mat<S> {
    fn new(a: &DMatrix<Complex64>) -> Self {
        Self {
            a: a.iter().map(|&z| S::from_c64(z)).collect(),
            m: a.nrows(),
            n: a.ncols(),
        }
    }
}

impl<S: TestScalar> LinearMap<H<S>> for Mat<S> {
    fn apply_into(&mut self, cx: &mut Arena<S>, x: &H<S>, y: &mut H<S>) -> Result<(), Fail> {
        cx.apply(self, false, x, y)
    }
}

impl<S: TestScalar> AdjointMap<H<S>> for Mat<S> {
    fn apply_adjoint_into(
        &mut self,
        cx: &mut Arena<S>,
        y: &H<S>,
        x: &mut H<S>,
    ) -> Result<(), Fail> {
        cx.apply(self, true, y, x)
    }
}

/// Runs `solve` (which returns the solver's `numops`) once cleanly, then once
/// per vector operation and per map application with a failure injected
/// there. Every failure must surface as `Error::Vector` immediately.
fn sweep<S: TestScalar>(name: &str, solve: impl Fn(&mut Arena<S>) -> Result<usize, Error<Fail>>) {
    let mut cx = Arena::new();
    let numops = solve(&mut cx).unwrap_or_else(|e| panic!("{name}: {e:?}"));
    // The map ran on the solver's own context, once per counted application.
    assert_eq!(cx.map_calls, numops, "{name}");
    let (ops, calls) = (cx.vector_ops, cx.map_calls);
    assert!(ops > 0 && calls > 0, "{name}");
    for k in 0..ops {
        let mut cx = Arena::new();
        cx.fail_vector_at = Some(k);
        assert_eq!(
            solve(&mut cx),
            Err(Error::Vector(Fail::Vector)),
            "{name} op {k}"
        );
        assert_eq!(cx.vector_ops, k + 1, "{name}: operation after failure {k}");
    }
    for j in 0..calls {
        let mut cx = Arena::new();
        cx.fail_map_at = Some(j);
        assert_eq!(
            solve(&mut cx),
            Err(Error::Vector(Fail::Map)),
            "{name} map {j}"
        );
        assert_eq!(cx.map_calls, j + 1, "{name}: application after failure {j}");
    }
}

fn hermitian_case() -> (DMatrix<Complex64>, Vec<Complex64>) {
    let mut g = SplitMix(31);
    let a = hermitian(&mut g, 14, false);
    let x = random_vec(&mut g, 14, false);
    (a, x)
}

#[test]
fn eigsolve_runs_on_the_context_and_matches_the_dense_spectrum() {
    let (a, x) = hermitian_case();
    let want = dense_eigenvalues(&a)[0];
    let mut cx = Arena::<f64>::new();
    let x0 = cx.input(&x);
    let mat = Mat::new(&a);
    // The closure map reaches the payloads only through the borrowed `cx`.
    let mut op = |cx: &mut Arena<f64>, x: &H<f64>, y: &mut H<f64>| cx.apply(&mat, false, x, y);
    let mut ws = Workspace::with_capacity(&mut cx, &x0, 14).unwrap();
    let alg = Lanczos::new(14, 1, 1e-10);
    let (vals, vecs, info) =
        eigsolve(&mut cx, &mut op, &x0, 1, Which::SmallestReal, &alg, &mut ws).unwrap();
    let bound = eig_bound::<f64>(alg.get_tol(), 14, op_norm(&a));
    within((vals[0] - want).abs(), bound, "lowest eigenvalue");
    assert_eq!(cx.map_calls, info.numops);
    let v: Vec<Complex64> = cx.data[vecs[0].0].iter().map(|s| s.to_c64()).collect();
    within(residual(&a, vals[0], &v), bound, "residual");
}

#[test]
fn hermitian_solvers_forward_the_context_and_propagate_failures() {
    let (a, x) = hermitian_case();
    let small = Lanczos::new(5, 3, 1e-14);
    for (name, orth) in [("eigsolve cgs2", Orth::Cgs2), ("eigsolve mgs2", Orth::Mgs2)] {
        let alg = small.orth(orth);
        let mat = Mat::new(&a);
        sweep::<f64>(name, |cx| {
            let x0 = cx.input(&x);
            let mut op =
                |cx: &mut Arena<f64>, x: &H<f64>, y: &mut H<f64>| cx.apply(&mat, false, x, y);
            let mut ws = Workspace::with_capacity(cx, &x0, 5).map_err(Error::Vector)?;
            let info = eigsolve(cx, &mut op, &x0, 2, Which::SmallestReal, &alg, &mut ws)?.2;
            Ok(info.numops)
        });
    }

    let mut g = SplitMix(32);
    let starts = [random_vec(&mut g, 14, false), random_vec(&mut g, 14, false)];
    let alg = BlockLanczos::new(6, 3, 1e-14, 1e-12);
    sweep::<f64>("block_eigsolve", |cx| {
        let x0 = [cx.input(&starts[0]), cx.input(&starts[1])];
        let mut ws = Workspace::with_capacity(cx, &x0[0], 7).map_err(Error::Vector)?;
        let mut op = Mat::new(&a);
        let (_, _, info) = block_eigsolve(cx, &mut op, &x0, 2, Which::LargestReal, &alg, &mut ws)?;
        Ok(info.numops)
    });

    let spd = with_spectrum(&mut g, &(1..=14).map(f64::from).collect::<Vec<_>>(), false);
    sweep::<f64>("linsolve cg", |cx| {
        let (b, x0) = (cx.input(&x), cx.input(&vec![Complex64::new(0.0, 0.0); 14]));
        let mut ws = Workspace::with_capacity(cx, &b, 0).map_err(Error::Vector)?;
        let (_, info) = linsolve(
            cx,
            &mut Mat::new(&spd),
            &b,
            &x0,
            &Cg::new(6, 1e-14),
            0.0,
            1.0,
            &mut ws,
        )?;
        Ok(info.numops)
    });

    for (name, alg) in [
        ("exponentiate lanczos", Lanczos::new(4, 3, 1e-12)),
        (
            "exponentiate lanczos eager",
            Lanczos::new(4, 3, 1e-12).eager(true),
        ),
    ] {
        sweep::<f64>(name, |cx| {
            let x0 = cx.input(&x);
            let mut ws = Workspace::with_capacity(cx, &x0, 4).map_err(Error::Vector)?;
            let (_, info) = exponentiate(cx, &mut Mat::new(&a), -0.5, &x0, &alg, &mut ws)?;
            Ok(info.numops)
        });
    }
}

#[test]
fn rectangular_svdsolve_shares_the_context_between_both_spaces() {
    let mut g = SplitMix(33);
    let (m, n) = (9, 13);
    let a = DMatrix::from_fn(m, n, |_, _| g.scalar(false));
    let x = random_vec(&mut g, m, false);
    let dom = vec![Complex64::new(0.0, 0.0); n];
    let alg = Gkl::new(5, 3, 1e-14);
    sweep::<f64>("svdsolve", |cx| {
        let x0 = cx.input(&x);
        let xd = cx.input(&dom);
        let mut ws_u = Workspace::with_capacity(cx, &x0, 5).map_err(Error::Vector)?;
        let mut ws_v = Workspace::with_capacity(cx, &xd, 5).map_err(Error::Vector)?;
        let mut op = Mat::new(&a);
        let (_, _, _, info) = svdsolve(
            cx,
            &mut op,
            &x0,
            2,
            Which::LargestReal,
            &alg,
            &mut ws_u,
            &mut ws_v,
        )?;
        Ok(info.numops)
    });
}

#[test]
fn general_solvers_forward_the_context_and_propagate_failures() {
    let mut g = SplitMix(34);
    let a = outliers(&mut g, 14, false);
    let x = random_vec(&mut g, 14, false);
    let alg = Arnoldi::new(6, 3, 1e-14);
    sweep::<f64>("schursolve", |cx| {
        let x0 = cx.input(&x);
        let mut ws = Workspace::with_capacity(cx, &x0, 6).map_err(Error::Vector)?;
        let (_, _, _, info) = schursolve(
            cx,
            &mut Mat::new(&a),
            &x0,
            2,
            Which::LargestMagnitude,
            &alg,
            &mut ws,
        )?;
        Ok(info.numops)
    });
    sweep::<f64>("realeigsolve", |cx| {
        let x0 = cx.input(&x);
        let mut ws = Workspace::with_capacity(cx, &x0, 6).map_err(Error::Vector)?;
        let (_, _, info) = realeigsolve(
            cx,
            &mut Mat::new(&a),
            &x0,
            1,
            Which::LargestReal,
            &alg,
            &mut ws,
        )?;
        Ok(info.numops)
    });
    for rr in [RestartResidual::Recomputed, RestartResidual::Recurrence] {
        let gm = Gmres::new(4, 3, 1e-14).restart_residual(rr);
        sweep::<f64>(&format!("linsolve gmres {rr:?}"), |cx| {
            let (b, x0) = (cx.input(&x), cx.input(&x));
            let mut ws = Workspace::with_capacity(cx, &b, 4).map_err(Error::Vector)?;
            let (_, info) = linsolve(cx, &mut Mat::new(&a), &b, &x0, &gm, 0.5, 1.0, &mut ws)?;
            Ok(info.numops)
        });
    }
    let u = [
        x.clone(),
        random_vec(&mut g, 14, false),
        random_vec(&mut g, 14, false),
    ];
    let alg = Arnoldi::new(4, 3, 1e-12);
    sweep::<f64>("expintegrator arnoldi", |cx| {
        let u: Vec<H<f64>> = u.iter().map(|ui| cx.input(ui)).collect();
        let mut ws = Workspace::with_capacity(cx, &u[0], 4).map_err(Error::Vector)?;
        let (_, info) = expintegrator(cx, &mut Mat::new(&a), 0.5, &u, &alg, &mut ws)?;
        Ok(info.numops)
    });

    let ac = outliers(&mut g, 12, true);
    let xc = random_vec(&mut g, 12, true);
    let alg = Arnoldi::new(6, 3, 1e-14);
    sweep::<Complex64>("arnoldi_eigsolve", |cx| {
        let x0 = cx.input(&xc);
        let mut ws = Workspace::with_capacity(cx, &x0, 6).map_err(Error::Vector)?;
        let (_, _, info) = arnoldi_eigsolve(
            cx,
            &mut Mat::new(&ac),
            &x0,
            2,
            Which::LargestImag,
            &alg,
            &mut ws,
        )?;
        Ok(info.numops)
    });
}

/// An erased engine as the context: `type Context = dyn Engine` is
/// `dyn Engine + 'static`, so a forwarding wrapper takes
/// `&mut (dyn Engine + 'static)`; a locally owned engine coerces to it.
trait Engine {
    fn record(&mut self);
}

struct Counting(usize);

impl Engine for Counting {
    fn record(&mut self) {
        self.0 += 1;
    }
}

struct EVec(DVec<f64>);

impl Vector for EVec {
    type Scalar = f64;
    type Error = std::convert::Infallible;
    type Context = dyn Engine;
    fn similar(&self, cx: &mut Self::Context) -> Result<Self, Self::Error> {
        cx.record();
        self.0.similar(&mut ()).map(EVec)
    }
    fn copy_from(&mut self, cx: &mut Self::Context, x: &Self) -> Result<(), Self::Error> {
        cx.record();
        self.0.copy_from(&mut (), &x.0)
    }
    fn inner(&self, cx: &mut Self::Context, y: &Self) -> Result<f64, Self::Error> {
        cx.record();
        self.0.inner(&mut (), &y.0)
    }
    fn norm(&self, cx: &mut Self::Context) -> Result<f64, Self::Error> {
        cx.record();
        self.0.norm(&mut ())
    }
    fn axpby(
        &mut self,
        cx: &mut Self::Context,
        a: f64,
        x: &Self,
        b: f64,
    ) -> Result<(), Self::Error> {
        cx.record();
        self.0.axpby(&mut (), a, &x.0, b)
    }
}

fn lowest(engine: &mut (dyn Engine + 'static), a: &DMatrix<Complex64>, x: &[Complex64]) -> f64 {
    let mut dense = DenseOp::<f64>::new(a);
    let mut op = |cx: &mut (dyn Engine + 'static), x: &EVec, y: &mut EVec| {
        cx.record();
        dense.apply_into(&mut (), &x.0, &mut y.0)
    };
    let x0 = EVec(to_dvec(x));
    let mut ws = Workspace::with_capacity(engine, &x0, 14).unwrap();
    let alg = Lanczos::new(14, 1, 1e-10);
    eigsolve(engine, &mut op, &x0, 1, Which::SmallestReal, &alg, &mut ws)
        .unwrap()
        .0[0]
}

#[test]
fn erased_static_engine_context_forwards_through_a_wrapper() {
    let (a, x) = hermitian_case();
    let mut engine = Counting(0);
    let got = lowest(&mut engine, &a, &x);
    // `lowest` runs Lanczos(14, 1, 1e-10) on the full 14-dimensional space.
    let bound = eig_bound::<f64>(1e-10, 14, op_norm(&a));
    within(
        (got - dense_eigenvalues(&a)[0]).abs(),
        bound,
        "lowest eigenvalue",
    );
    assert!(engine.0 > 0);
}
