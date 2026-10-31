//! Hermitian Lanczos `eigsolve` and `exponentiate` on the inputs and options
//! of KrylovKit.jl's own runs (CGS2, SplitMix64 matrices and start vectors,
//! `maxiter = 100`).
//!
//! Results are checked against dense oracles on the same matrix. The
//! iteration and application counts are part of the KrylovKit
//! correspondence (`docs/krylovkit.md`); they are pinned on a few
//! representative cases, checked against KrylovKit.jl v0.10.4.

mod common;

use common::*;
use krylov::{Lanczos, Which, Workspace, eigsolve, exponentiate};
use num_complex::Complex64;

/// `(converged, numiter, numops)` of KrylovKit.jl v0.10.4 on the same case.
type Counts = (usize, usize, usize);

struct Case {
    name: &'static str,
    seed: u64,
    n: usize,
    complex: bool,
    krylovdim: usize,
    tol: f64,
    eager: bool,
}

impl Case {
    fn alg(&self) -> Lanczos {
        Lanczos::new(self.krylovdim, 100, self.tol).eager(self.eager)
    }

    fn inputs(&self) -> (nalgebra::DMatrix<Complex64>, Vec<Complex64>) {
        let mut g = SplitMix(self.seed);
        let a = hermitian(&mut g, self.n, self.complex);
        let x = random_vec(&mut g, self.n, self.complex);
        (a, x)
    }
}

/// Counts agree with KrylovKit up to one restart and 10 % of the
/// applications: the CGS2 step differs (see `docs/krylovkit.md`), so
/// convergence can be detected one cycle apart.
fn check_counts(name: &str, (converged, numiter, numops): Counts, got: Counts, iter_slack: usize) {
    assert_eq!(got.0, converged, "{name}: converged");
    assert!(
        got.1.abs_diff(numiter) <= iter_slack,
        "{name}: numiter {} vs {numiter}",
        got.1
    );
    assert!(
        got.2.abs_diff(numops) * 10 <= numops,
        "{name}: numops {} vs {numops}",
        got.2
    );
}

/// Returns the eigenvalues in `which` order, matched against the dense
/// spectrum, with converged pairs satisfying `‖A·x − λ·x‖ ≤ tol`.
fn eig_case<S: TestScalar>(c: &Case, howmany: usize, which: Which, kk: Option<(usize, Counts)>) {
    let (a, x) = c.inputs();
    let x0 = to_dvec::<S>(&x);
    let alg = c.alg();
    let mut ws = Workspace::with_capacity(&mut (), &x0, alg.get_krylovdim()).unwrap();
    let (vals, vecs, info) = eigsolve(
        &mut (),
        &mut DenseOp::new(&a),
        &x0,
        howmany,
        which,
        &alg,
        &mut ws,
    )
    .unwrap();
    let name = c.name;
    let mut want = dense_eigenvalues(&a);
    match which {
        Which::SmallestReal => {}
        Which::LargestReal => want.reverse(),
        Which::LargestMagnitude => want.sort_by(|x, y| y.abs().partial_cmp(&x.abs()).unwrap()),
        _ => unreachable!(),
    }
    assert!(
        info.converged >= howmany,
        "{name}: converged {}",
        info.converged
    );
    assert!(vals.len() >= howmany);
    let bound = eig_bound::<S>(c.tol, c.n, op_norm(&a));
    for (k, (v, x)) in vals.iter().zip(&vecs).enumerate().take(info.converged) {
        let v = to_f64(*v);
        within((v - want[k]).abs(), bound, &format!("{name}: value #{k}"));
        within(
            residual(&a, v, &to_c64(x)),
            bound,
            &format!("{name}: residual #{k}"),
        );
    }
    if let Some((nvals, counts)) = kk {
        assert_eq!(vals.len(), nvals, "{name}: number of returned values");
        check_counts(name, counts, (info.converged, info.numiter, info.numops), 1);
    }
}

/// `exp(t·A)·x` against nalgebra's dense exponential.
fn exp_case<S: TestScalar>(c: &Case, t: Complex64, kk: Option<Counts>) {
    let (a, x) = c.inputs();
    let x0 = to_dvec::<S>(&x);
    let alg = c.alg();
    let mut ws = Workspace::with_capacity(&mut (), &x0, alg.get_krylovdim()).unwrap();
    let (y, info) = exponentiate(
        &mut (),
        &mut DenseOp::new(&a),
        S::from_c64(t),
        &x0,
        &alg,
        &mut ws,
    )
    .unwrap();
    let name = c.name;
    let want = dense_expv(&a, t, &x);
    let bound = exp_bound::<S>(c.tol, t, info.numiter, c.n, norm(&want));
    within(
        diff_norm(&to_c64(&y), &want),
        bound,
        &format!("{name}: tol per unit time"),
    );
    assert_eq!(info.converged, 1, "{name}");
    if let Some(counts) = kk {
        // The step sequence follows KrylovKit's controller: steps within 10 %.
        let slack = counts.1.max(10) / 10;
        check_counts(
            name,
            counts,
            (info.converged, info.numiter, info.numops),
            slack,
        );
    }
}

const fn case(
    name: &'static str,
    seed: u64,
    n: usize,
    complex: bool,
    krylovdim: usize,
    tol: f64,
    eager: bool,
) -> Case {
    Case {
        name,
        seed,
        n,
        complex,
        krylovdim,
        tol,
        eager,
    }
}

#[test]
fn eigsolve_matches_dense_spectrum_and_krylovkit_counts() {
    use Which::*;
    eig_case::<f64>(
        &case("real_sr", 1, 120, false, 20, 1e-10, false),
        3,
        SmallestReal,
        None,
    );
    // KrylovKit returns every converged pair, here one more than requested.
    eig_case::<Complex64>(
        &case("complex_lm", 2, 100, true, 16, 1e-10, false),
        2,
        LargestMagnitude,
        Some((3, (3, 9, 70))),
    );
    eig_case::<f64>(
        &case("real_lr_eager", 3, 80, false, 30, 1e-11, true),
        2,
        LargestReal,
        None,
    );
    // A small Krylov dimension: many restarts.
    eig_case::<Complex64>(
        &case("complex_sr_small_k", 4, 60, true, 8, 1e-9, false),
        4,
        SmallestReal,
        Some((4, (4, 28, 99))),
    );
}

#[test]
fn exponentiate_matches_dense_and_krylovkit_steps() {
    // Imaginary time: `docs/krylovkit.md` cites this case's 21 steps.
    exp_case::<f64>(
        &case("real_imag_time", 5, 60, false, 12, 1e-12, false),
        Complex64::new(-2.0, 0.0),
        Some((1, 21, 273)),
    );
    exp_case::<Complex64>(
        &case("complex_real_time", 6, 60, true, 15, 1e-12, false),
        Complex64::new(0.0, -5.0),
        None,
    );
    exp_case::<Complex64>(
        &case("complex_imag_time_eager", 7, 50, true, 20, 1e-12, true),
        Complex64::new(-1.5, 0.0),
        None,
    );
}
