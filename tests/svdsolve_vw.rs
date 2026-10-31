//! `svdsolve` with distinct domain and codomain vector types
//! (`AdjointMap<V, W>`, as KrylovKit's `GKLFactorization{TU, TV}`): the
//! domain is `DVec` of length `n`, the codomain a distinct newtype of length
//! `m`, so a swapped workspace or start vector does not compile.

mod common;

use common::*;
use krylov::{AdjointMap, Gkl, LinearMap, Vector, Which, Workspace, svdsolve};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;

/// Codomain vector: a distinct type delegating to `DVec` (whose `similar`
/// calls are counted).
#[derive(Debug, PartialEq)]
struct CoVec<S>(DVec<S>);

impl<S: TestScalar> Vector for CoVec<S> {
    type Scalar = S;
    type Error = std::convert::Infallible;
    type Context = ();
    fn similar(&self, cx: &mut ()) -> Result<Self, Self::Error> {
        Ok(CoVec(self.0.similar(cx)?))
    }
    fn copy_from(&mut self, cx: &mut (), x: &Self) -> Result<(), Self::Error> {
        self.0.copy_from(cx, &x.0)
    }
    fn inner(&self, cx: &mut (), y: &Self) -> Result<S, Self::Error> {
        self.0.inner(cx, &y.0)
    }
    fn norm(&self, cx: &mut ()) -> Result<S::Real, Self::Error> {
        self.0.norm(cx)
    }
    fn axpby(&mut self, cx: &mut (), a: S, x: &Self, b: S) -> Result<(), Self::Error> {
        self.0.axpby(cx, a, &x.0, b)
    }
}

fn apply<S: TestScalar>(m: &DMatrix<Complex64>, x: &DVec<S>, y: &mut DVec<S>) {
    assert_eq!((x.0.len(), y.0.len()), (m.ncols(), m.nrows()));
    let xv = DVector::from_iterator(x.0.len(), x.0.iter().map(|s| s.to_c64()));
    for (yi, v) in y.0.iter_mut().zip((m * xv).iter()) {
        *yi = S::from_c64(*v);
    }
}

/// The same dense `m × n` map, once as `DVec → CoVec` and once as
/// `DVec → DVec`.
struct Rect<S> {
    a: DMatrix<Complex64>,
    adj: DMatrix<Complex64>,
    _s: std::marker::PhantomData<S>,
}

impl<S: TestScalar> Rect<S> {
    fn new(a: &DMatrix<Complex64>) -> Self {
        Self {
            a: a.clone(),
            adj: a.adjoint(),
            _s: Default::default(),
        }
    }
}

impl<S: TestScalar> LinearMap<DVec<S>, CoVec<S>> for Rect<S> {
    fn apply_into(
        &mut self,
        _: &mut (),
        x: &DVec<S>,
        y: &mut CoVec<S>,
    ) -> Result<(), std::convert::Infallible> {
        apply(&self.a, x, &mut y.0);
        Ok(())
    }
}

impl<S: TestScalar> AdjointMap<DVec<S>, CoVec<S>> for Rect<S> {
    fn apply_adjoint_into(
        &mut self,
        _: &mut (),
        y: &CoVec<S>,
        x: &mut DVec<S>,
    ) -> Result<(), std::convert::Infallible> {
        apply(&self.adj, &y.0, x);
        Ok(())
    }
}

impl<S: TestScalar> LinearMap<DVec<S>> for Rect<S> {
    fn apply_into(
        &mut self,
        _: &mut (),
        x: &DVec<S>,
        y: &mut DVec<S>,
    ) -> Result<(), std::convert::Infallible> {
        apply(&self.a, x, y);
        Ok(())
    }
}

impl<S: TestScalar> AdjointMap<DVec<S>> for Rect<S> {
    fn apply_adjoint_into(
        &mut self,
        _: &mut (),
        y: &DVec<S>,
        x: &mut DVec<S>,
    ) -> Result<(), std::convert::Infallible> {
        apply(&self.adj, y, x);
        Ok(())
    }
}

fn rectangular(g: &mut SplitMix, m: usize, n: usize, complex: bool) -> DMatrix<Complex64> {
    DMatrix::from_fn(m, n, |_, _| g.scalar(complex))
}

/// Everything a run returns, flattened for comparison.
#[derive(Debug, PartialEq)]
struct Flat {
    values: Vec<f64>,
    normres: Vec<f64>,
    left: Vec<Vec<Complex64>>,
    right: Vec<Vec<Complex64>>,
    residual: Vec<Vec<Complex64>>,
    counts: (usize, usize, usize),
}

/// `svdsolve` on `DVec → CoVec`.
fn run_vw<S: TestScalar>(
    a: &DMatrix<Complex64>,
    x0: &[Complex64],
    which: Which,
    alg: &Gkl,
) -> Flat {
    let n = a.ncols();
    let x0 = CoVec(to_dvec::<S>(x0));
    let domain = to_dvec::<S>(&vec![Complex64::default(); n]);
    let kd = alg.get_krylovdim();
    let mut ws_u = Workspace::with_capacity(&mut (), &x0, kd).unwrap();
    let mut ws_v = Workspace::with_capacity(&mut (), &domain, kd).unwrap();
    let mut op = Rect::<S>::new(a);
    let (s, l, r, info) =
        svdsolve(&mut (), &mut op, &x0, 2, which, alg, &mut ws_u, &mut ws_v).unwrap();
    Flat {
        values: s.iter().map(|&x| to_f64(x)).collect(),
        normres: info.normres.iter().map(|&x| to_f64(x)).collect(),
        left: l.iter().map(|v| to_c64(&v.0)).collect(),
        right: r.iter().map(to_c64).collect(),
        residual: info.residual.iter().map(|v| to_c64(&v.0)).collect(),
        counts: (info.converged, info.numiter, info.numops),
    }
}

/// The same run with one vector type `DVec` for both spaces.
fn run_vv<S: TestScalar>(
    a: &DMatrix<Complex64>,
    x0: &[Complex64],
    which: Which,
    alg: &Gkl,
) -> Flat {
    let n = a.ncols();
    let x0 = to_dvec::<S>(x0);
    let domain = to_dvec::<S>(&vec![Complex64::default(); n]);
    let kd = alg.get_krylovdim();
    let mut ws_u = Workspace::with_capacity(&mut (), &x0, kd).unwrap();
    let mut ws_v = Workspace::with_capacity(&mut (), &domain, kd).unwrap();
    let mut op = Rect::<S>::new(a);
    let (s, l, r, info) =
        svdsolve(&mut (), &mut op, &x0, 2, which, alg, &mut ws_u, &mut ws_v).unwrap();
    Flat {
        values: s.iter().map(|&x| to_f64(x)).collect(),
        normres: info.normres.iter().map(|&x| to_f64(x)).collect(),
        left: l.iter().map(to_c64).collect(),
        right: r.iter().map(to_c64).collect(),
        residual: info.residual.iter().map(to_c64).collect(),
        counts: (info.converged, info.numiter, info.numops),
    }
}

fn dense_singular_values(a: &DMatrix<Complex64>) -> Vec<f64> {
    let mut s: Vec<f64> = a.clone().singular_values().iter().copied().collect();
    s.sort_by(|x, y| y.partial_cmp(x).unwrap());
    s
}

#[test]
fn distinct_codomain_type_matches_dense_and_the_one_type_run() {
    // LargestReal on a tall map, SmallestReal on a wide map (codomain not
    // larger than the domain, svdsolve's SmallestReal precondition).
    let cases = [
        (90, 70, false, Which::LargestReal),
        (90, 70, true, Which::LargestReal),
        (30, 45, false, Which::SmallestReal),
        (30, 45, true, Which::SmallestReal),
    ];
    for (m, n, complex, which) in cases {
        let mut g = SplitMix(401 + m as u64 + complex as u64);
        let a = rectangular(&mut g, m, n, complex);
        let x0 = random_vec(&mut g, m, complex);
        let alg = Gkl::new(24, 300, 1e-9);
        let (vw, vv) = if complex {
            (
                run_vw::<Complex64>(&a, &x0, which, &alg),
                run_vv::<Complex64>(&a, &x0, which, &alg),
            )
        } else {
            (
                run_vw::<f64>(&a, &x0, which, &alg),
                run_vv::<f64>(&a, &x0, which, &alg),
            )
        };
        // The type split changes no arithmetic: identical results.
        assert_eq!(vw, vv, "{m}×{n} {which:?}");
        assert!(
            vw.counts.0 >= 2,
            "{m}×{n} {which:?}: converged {}",
            vw.counts.0
        );
        let mut exact = dense_singular_values(&a);
        if which == Which::SmallestReal {
            exact.reverse();
        }
        let bound = if complex {
            eig_bound::<Complex64>(alg.get_tol(), m.max(n), op_norm(&a))
        } else {
            eig_bound::<f64>(alg.get_tol(), m.max(n), op_norm(&a))
        };
        for (v, e) in vw.values.iter().zip(&exact).take(vw.counts.0) {
            within(
                (v - e).abs(),
                bound,
                &format!("{m}×{n} {which:?}: {v} vs {e}"),
            );
        }
        assert!(vw.left.iter().all(|l| l.len() == m) && vw.right.iter().all(|r| r.len() == n));
    }
}

#[test]
fn warm_call_with_distinct_types_allocates_only_the_returned_triplets() {
    let mut g = SplitMix(441);
    let a = rectangular(&mut g, 30, 20, false);
    let x0 = CoVec(to_dvec::<f64>(&random_vec(&mut g, 30, false)));
    let domain = to_dvec::<f64>(&vec![Complex64::default(); 20]);
    let alg = Gkl::new(8, 100, 1e-10);
    let mut ws_u = Workspace::with_capacity(&mut (), &x0, 8).unwrap();
    let mut ws_v = Workspace::with_capacity(&mut (), &domain, 8).unwrap();
    let mut op = Rect::<f64>::new(&a);
    for _ in 0..2 {
        let before = similar_calls();
        let (vals, _, _, info) = svdsolve(
            &mut (),
            &mut op,
            &x0,
            2,
            Which::LargestReal,
            &alg,
            &mut ws_u,
            &mut ws_v,
        )
        .unwrap();
        assert!(info.numiter > 1);
        assert_eq!(similar_calls() - before, 3 * vals.len());
    }
}
