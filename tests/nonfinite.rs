//! Non-finite map output is `Error::NonFiniteIterate` in every solver,
//! returned at the poisoned application without spending more of the budget;
//! large finite values with a naive `√⟨x, x⟩` norm are not rejected.

mod common;

use std::convert::Infallible;

use common::*;
use krylov::{
    Arnoldi, BlockLanczos, Cg, Error, Gmres, Lanczos, LinearMap, Which, Workspace, block_eigsolve,
    eigsolve, exponentiate, linsolve, schursolve,
};
use nalgebra::DMatrix;
use num_complex::Complex64;

/// `DenseOp` whose output is `bad` everywhere from application `after + 1` on.
struct Poisoned {
    op: DenseOp<f64>,
    after: usize,
    bad: f64,
    calls: usize,
}

impl LinearMap<DVec<f64>> for Poisoned {
    fn apply_into(&mut self, x: &DVec<f64>, y: &mut DVec<f64>) -> Result<(), Infallible> {
        self.calls += 1;
        self.op.apply_into(x, y)?;
        if self.calls > self.after {
            y.0.fill(self.bad);
        }
        Ok(())
    }
}

/// Positive definite, so that every solver (CG included) runs normally.
fn spd(n: usize) -> DMatrix<Complex64> {
    let mut g = SplitMix(5);
    let eigs: Vec<f64> = (0..n).map(|i| 1.0 + i as f64).collect();
    with_spectrum(&mut g, &eigs, false)
}

/// Runs `f` with a map poisoned after `after` applications and checks the
/// error and that the poisoned application was the last one.
fn check(name: &str, f: impl Fn(&mut Poisoned) -> Option<Error<Infallible>>) {
    let a = spd(30);
    for (after, bad) in [(0, f64::NAN), (3, f64::NAN), (3, f64::INFINITY)] {
        let mut op = Poisoned {
            op: DenseOp::new(&a),
            after,
            bad,
            calls: 0,
        };
        assert_eq!(
            f(&mut op),
            Some(Error::NonFiniteIterate),
            "{name}: after {after}, {bad}"
        );
        assert_eq!(op.calls, after + 1, "{name}: after {after}, {bad}");
    }
}

#[test]
fn every_solver_reports_non_finite_map_output_at_once() {
    let mut g = SplitMix(6);
    let x = to_dvec::<f64>(&random_vec(&mut g, 30, false));
    let x2 = to_dvec::<f64>(&random_vec(&mut g, 30, false));
    let ws = || Workspace::with_capacity(&x, 12).unwrap();
    let sr = Which::SmallestReal;
    check("eigsolve", |op| {
        eigsolve(op, &x, 2, sr, &Lanczos::new(10, 100, 1e-12), &mut ws()).err()
    });
    check("block_eigsolve", |op| {
        let x0 = [DVec(x.0.clone()), DVec(x2.0.clone())];
        let alg = BlockLanczos::new(10, 100, 1e-12, 1e-12);
        block_eigsolve(op, &x0, 2, sr, &alg, &mut ws()).err()
    });
    check("schursolve", |op| {
        schursolve(op, &x, 2, sr, &Arnoldi::new(10, 100, 1e-12), &mut ws())
            .map(|_| ())
            .err()
    });
    check("cg", |op| {
        linsolve(op, &x, &x2, &Cg::new(100, 1e-12), 0.0, 1.0, &mut ws()).err()
    });
    check("gmres", |op| {
        let alg = Gmres::new(10, 100, 1e-12);
        linsolve(op, &x, &x2, &alg, 0.0, 1.0, &mut ws()).err()
    });
    check("exponentiate lanczos", |op| {
        exponentiate(op, -1.0, &x, &Lanczos::new(10, 100, 1e-12), &mut ws()).err()
    });
    check("exponentiate arnoldi", |op| {
        exponentiate(op, -1.0, &x, &Arnoldi::new(10, 100, 1e-12), &mut ws()).err()
    });
}

#[test]
fn non_finite_linsolve_inputs_stay_start_vector_errors() {
    let a = spd(5);
    let ok = DVec(vec![1.0; 5]);
    let bad = DVec(vec![1.0, f64::NAN, 1.0, 1.0, 1.0]);
    let mut ws = Workspace::with_capacity(&ok, 5).unwrap();
    for (b, x0) in [(&bad, &ok), (&ok, &bad)] {
        let e = linsolve(
            &mut DenseOp::new(&a),
            b,
            x0,
            &Cg::new(100, 1e-12),
            0.0,
            1.0,
            &mut ws,
        )
        .err();
        assert_eq!(e, Some(Error::NonFiniteStartVector));
    }
}

#[test]
fn overflowing_invariant_subspace_step_is_a_typed_error() {
    // A = 800·I₄: the Krylov space is invariant after one vector and
    // φ(800) overflows f64 (#9).
    let a = DMatrix::<Complex64>::identity(4, 4) * Complex64::new(800.0, 0.0);
    let x = DVec(vec![1.0, -0.5, 0.25, 2.0]);
    let mut ws = Workspace::with_capacity(&x, 30).unwrap();
    let e = exponentiate(
        &mut DenseOp::<f64>::new(&a),
        1.0,
        &x,
        &Lanczos::new(30, 100, 1e-12),
        &mut ws,
    )
    .err();
    assert_eq!(e, Some(Error::NonFiniteIterate));
}

#[test]
fn large_finite_values_with_a_naive_norm_are_accepted() {
    // Eigenvalues near 1e150 (f64) and 1e18 (f32): ‖A·v‖² (and CG's
    // ⟨p, A·p⟩ from x₀ = 0) stays finite in the vector's own precision, so
    // nothing here is non-finite.
    fn go<S: TestScalar>(scale: f64) {
        let n = 6;
        let a: Vec<S> = (0..n * n)
            .map(|k| {
                let (i, j) = (k % n, k / n);
                S::from_c64(Complex64::new(
                    if i == j { scale * (1 + i) as f64 } else { 0.0 },
                    0.0,
                ))
            })
            .collect();
        let mut op = |x: &Naive<S>, y: &mut Naive<S>| {
            for i in 0..n {
                y.0[i] = (0..n).fold(S::zero(), |s, j| s + a[i + j * n] * x.0[j]);
            }
            Ok::<(), Infallible>(())
        };
        let x = Naive(vec![S::from_c64(Complex64::new(1.0, 0.0)); n]);
        let mut ws = Workspace::with_capacity(&x, 6).unwrap();
        let (vals, _, _) = eigsolve(
            &mut op,
            &x,
            1,
            Which::LargestReal,
            &Lanczos::new(6, 100, scale * 1e-5),
            &mut ws,
        )
        .unwrap();
        let want = scale * n as f64;
        let got = num_traits::ToPrimitive::to_f64(&vals[0]).unwrap();
        assert!((got - want).abs() <= 1e-4 * want, "{got} vs {want}");
        let (y, _) = linsolve(
            &mut op,
            &x,
            &Naive(vec![S::zero(); n]),
            &Cg::new(6, 0.0),
            S::zero(),
            S::one(),
            &mut ws,
        )
        .unwrap();
        let y0 = y.0[0].to_c64().re;
        assert!((y0 * scale - 1.0).abs() <= 1e-3, "{y0}");
    }
    go::<f64>(1e150);
    go::<f32>(1e18);
}
