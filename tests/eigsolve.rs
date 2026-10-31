//! `eigsolve` against dense nalgebra eigenvalues.

mod common;

use common::*;
use krylov::{
    Arnoldi, Basis, Error, Lanczos, Orth, VecBasis, Which, Workspace, eigsolve, schursolve,
};
use nalgebra::DMatrix;
use num_complex::{Complex32, Complex64};

struct Case<'a> {
    a: &'a DMatrix<Complex64>,
    x0: &'a [Complex64],
    howmany: usize,
    which: Which,
    alg: Lanczos,
}

/// Runs eigsolve and checks converged values against the dense spectrum and
/// residuals against a recomputation, both within [`eig_bound`], and
/// eigenvector orthonormality up to rounding.
fn check<S: TestScalar>(c: &Case) -> krylov::ConvergenceInfo<DVec<S>> {
    let x0 = to_dvec::<S>(c.x0);
    let mut op = DenseOp::<S>::new(c.a);
    let mut ws = Workspace::with_capacity(&mut (), &x0, c.alg.get_krylovdim()).unwrap();
    let (vals, vecs, info) =
        eigsolve(&mut (), &mut op, &x0, c.howmany, c.which, &c.alg, &mut ws).unwrap();
    assert!(
        info.converged >= c.howmany,
        "converged {} of {}",
        info.converged,
        c.howmany
    );
    assert_eq!(vals.len(), vecs.len());
    assert_eq!(info.numops, op.applications);

    let mut exact = dense_eigenvalues(c.a);
    match c.which {
        Which::SmallestReal => {}
        Which::LargestReal => exact.reverse(),
        Which::LargestMagnitude => exact.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap()),
        w => panic!("Hermitian test with selector {w:?}"),
    }
    let n = c.a.nrows();
    let bound = eig_bound::<S>(c.alg.get_tol(), n, op_norm(c.a));
    for (i, (&v, x)) in vals.iter().zip(&vecs).enumerate() {
        let normres = to_f64(info.normres[i]);
        let rv = to_c64(&info.residual[i]);
        within(
            (norm(&rv) - normres).abs(),
            rounding::<S>(n, normres),
            &format!("normres {i}"),
        );
        if i >= info.converged {
            continue;
        }
        let v = to_f64(v);
        within(
            (v - exact[i]).abs(),
            bound,
            &format!("value {i} vs {}", exact[i]),
        );
        within(
            residual(c.a, v, &to_c64(x)),
            bound,
            &format!("residual {i}"),
        );
    }
    within(
        orthogonality_error(&vecs),
        rounding::<S>(n, 1.0),
        "eigenvector orthonormality",
    );
    info
}

#[test]
fn random_real_symmetric_all_selectors() {
    let mut g = SplitMix(11);
    let a = hermitian(&mut g, 150, false);
    let x0 = random_vec(&mut g, 150, false);
    for which in [
        Which::SmallestReal,
        Which::LargestReal,
        Which::LargestMagnitude,
    ] {
        let alg = Lanczos::new(24, 100, 1e-10);
        let info = check::<f64>(&Case {
            a: &a,
            x0: &x0,
            howmany: 4,
            which,
            alg,
        });
        assert!(info.numiter > 1, "test must exercise restarts");
        assert!(info.max_imag == 0.0);
    }
}

#[test]
fn random_complex_hermitian_cgs2_and_mgs2() {
    let mut g = SplitMix(12);
    let a = hermitian(&mut g, 120, true);
    let x0 = random_vec(&mut g, 120, true);
    for orth in [Orth::Cgs2, Orth::Mgs2] {
        let alg = Lanczos::new(20, 100, 1e-10).orth(orth);
        let info = check::<Complex64>(&Case {
            a: &a,
            x0: &x0,
            howmany: 3,
            which: Which::SmallestReal,
            alg,
        });
        assert!(info.numiter > 1);
        // ⟨v, A·v⟩ of a Hermitian A is real up to the rounding of A·v.
        within(
            info.max_imag,
            rounding::<Complex64>(120, op_norm(&a)),
            "max_imag",
        );
    }
}

/// Every converged value is within `bound` of an eigenvalue of the known
/// spectrum (whose distinct levels are more than `2·bound` apart), returned
/// no more often than its multiplicity, with orthonormal eigenvectors whose
/// residuals are within `bound`.
fn check_against_spectrum(
    a: &DMatrix<Complex64>,
    eigs: &[f64],
    vals: &[f64],
    vecs: &[DVec<Complex64>],
    converged: usize,
    bound: f64,
) {
    for &v in &vals[..converged] {
        let mult = eigs.iter().filter(|&&e| (e - v).abs() <= bound).count();
        let returned = vals[..converged]
            .iter()
            .filter(|&&w| (w - v).abs() <= 2.0 * bound)
            .count();
        assert!(mult >= 1, "{v} is not an eigenvalue");
        assert!(
            returned <= mult,
            "{v} returned {returned} times, multiplicity {mult}"
        );
    }
    for (&v, x) in vals[..converged].iter().zip(vecs) {
        within(residual(a, v, &to_c64(x)), bound, "residual");
    }
    within(
        orthogonality_error(&vecs[..converged]),
        rounding::<Complex64>(a.nrows(), 1.0),
        "eigenvector orthonormality",
    );
}

#[test]
fn degenerate_and_clustered_spectra() {
    let mut g = SplitMix(13);
    // Degenerate lowest level (multiplicity 3). In exact arithmetic a single
    // start vector sees each eigenspace once; in floating point, rounding
    // noise from orthogonalization can seed further copies (KrylovKit
    // documents the same). Only platform-independent properties are asserted.
    let mut eigs: Vec<f64> = (0..80).map(|i| i as f64 * 0.1).collect();
    eigs[1] = eigs[0];
    eigs[2] = eigs[0];
    let a = with_spectrum(&mut g, &eigs, true);
    let x0 = random_vec(&mut g, 80, true);
    let alg = Lanczos::new(20, 100, 1e-10);
    let x = to_dvec::<Complex64>(&x0);
    let mut ws = Workspace::with_capacity(&mut (), &x, 20).unwrap();
    let (vals, vecs, info) = eigsolve(
        &mut (),
        &mut DenseOp::new(&a),
        &x,
        2,
        Which::SmallestReal,
        &alg,
        &mut ws,
    )
    .unwrap();
    assert!(info.converged >= 2);
    let bound = eig_bound::<Complex64>(alg.get_tol(), 80, op_norm(&a));
    check_against_spectrum(&a, &eigs, &vals, &vecs, info.converged, bound);
    // The two lowest distinct levels are found, in order.
    let mut distinct: Vec<f64> = Vec::new();
    for &v in &vals[..info.converged] {
        if distinct.last().is_none_or(|&d| (v - d).abs() > 2.0 * bound) {
            distinct.push(v);
        }
    }
    assert!(distinct.len() >= 2, "{vals:?}");
    within(distinct[0].abs(), bound, "lowest level 0");
    within((distinct[1] - 0.3).abs(), bound, "second level 0.3");

    // Clustered at the bottom of the spectrum.
    let eigs: Vec<f64> = (0..100)
        .map(|i| {
            if i < 4 {
                i as f64 * 1e-4
            } else {
                1.0 + i as f64 * 0.05
            }
        })
        .collect();
    let a = with_spectrum(&mut g, &eigs, false);
    let x0 = random_vec(&mut g, 100, false);
    let alg = Lanczos::new(30, 100, 1e-11);
    check::<f64>(&Case {
        a: &a,
        x0: &x0,
        howmany: 4,
        which: Which::SmallestReal,
        alg,
    });
}

#[test]
fn invariant_subspace_stops_without_restart() {
    let mut g = SplitMix(14);
    let eigs: Vec<f64> = (0..50).map(|i| (i % 5) as f64 - 2.0).collect();
    let a = with_spectrum(&mut g, &eigs, false);
    let x0 = to_dvec::<f64>(&random_vec(&mut g, 50, false));
    let alg = Lanczos::new(20, 100, 1e-10);
    let mut ws = Workspace::with_capacity(&mut (), &x0, 20).unwrap();
    let (vals, vecs, info) = eigsolve(
        &mut (),
        &mut DenseOp::new(&a),
        &x0,
        8,
        Which::SmallestReal,
        &alg,
        &mut ws,
    )
    .unwrap();
    // Five distinct eigenvalues: the Krylov space is exhausted at dimension 5.
    assert_eq!(vals.len(), 5);
    assert_eq!(vecs.len(), 5);
    assert_eq!(info.numiter, 1);
    assert_eq!(info.numops, 5);
    assert_eq!(info.converged, 5);
    let bound = eig_bound::<f64>(alg.get_tol(), 50, op_norm(&a));
    for (v, e) in vals.iter().zip([-2.0, -1.0, 0.0, 1.0, 2.0]) {
        within((v - e).abs(), bound, &format!("eigenvalue {e}"));
    }
    // An exact eigenvector as start vector: one-dimensional invariant subspace.
    let mut ones = DMatrix::<Complex64>::zeros(4, 4);
    for i in 0..4 {
        ones[(i, i)] = Complex64::new(i as f64, 0.0);
    }
    let e2 = to_dvec::<f64>(&[0.0, 0.0, 1.0, 0.0].map(|x| Complex64::new(x, 0.0)));
    let mut ws = Workspace::with_capacity(&mut (), &e2, 3).unwrap();
    let (vals, _, info) = eigsolve(
        &mut (),
        &mut DenseOp::new(&ones),
        &e2,
        2,
        Which::LargestReal,
        &Lanczos::new(3, 100, 1e-12),
        &mut ws,
    )
    .unwrap();
    assert_eq!(vals, vec![2.0]);
    assert_eq!(info.numops, 1);
}

#[test]
fn tiny_krylov_dimensions() {
    let mut g = SplitMix(15);
    let a = hermitian(&mut g, 30, false);
    let x0 = random_vec(&mut g, 30, false);
    // krylovdim 2: keep = 1, one new vector per restart (steepest-descent like).
    let alg = Lanczos::new(2, 5000, 1e-8);
    check::<f64>(&Case {
        a: &a,
        x0: &x0,
        howmany: 1,
        which: Which::LargestReal,
        alg,
    });
    // krylovdim 3: keep is 1 or 2, i.e. at most two new vectors per restart.
    let alg = Lanczos::new(3, 5000, 1e-8);
    check::<f64>(&Case {
        a: &a,
        x0: &x0,
        howmany: 2,
        which: Which::LargestReal,
        alg,
    });
    // krylovdim 1 cannot restart: one Rayleigh quotient, reported unconverged.
    let x = to_dvec::<f64>(&x0);
    let mut ws = Workspace::with_capacity(&mut (), &x, 1).unwrap();
    let (vals, _, info) = eigsolve(
        &mut (),
        &mut DenseOp::new(&a),
        &x,
        1,
        Which::LargestReal,
        &Lanczos::new(1, 100, 1e-12),
        &mut ws,
    )
    .unwrap();
    assert_eq!(
        (vals.len(), info.converged, info.numops, info.numiter),
        (1, 0, 1, 1)
    );
}

#[test]
fn eager_returns_early() {
    let mut g = SplitMix(16);
    let eigs: Vec<f64> = (0..200)
        .map(|i| if i == 0 { -10.0 } else { i as f64 / 200.0 })
        .collect();
    let a = with_spectrum(&mut g, &eigs, false);
    let x0 = random_vec(&mut g, 200, false);
    let lazy = Lanczos::new(40, 100, 1e-10);
    let i_lazy = check::<f64>(&Case {
        a: &a,
        x0: &x0,
        howmany: 1,
        which: Which::SmallestReal,
        alg: lazy,
    });
    let eager = lazy.eager(true);
    let i_eager = check::<f64>(&Case {
        a: &a,
        x0: &x0,
        howmany: 1,
        which: Which::SmallestReal,
        alg: eager,
    });
    assert_eq!(i_lazy.numops, 40);
    assert!(i_eager.numops < 40, "eager numops {}", i_eager.numops);
}

#[test]
fn single_precision_real_and_complex() {
    let mut g = SplitMix(17);
    let a = hermitian(&mut g, 80, true);
    let x0 = random_vec(&mut g, 80, true);
    let alg = Lanczos::new(20, 100, 1e-4);
    check::<Complex32>(&Case {
        a: &a,
        x0: &x0,
        howmany: 2,
        which: Which::LargestReal,
        alg,
    });
    let a = hermitian(&mut g, 80, false);
    let x0 = random_vec(&mut g, 80, false);
    check::<f32>(&Case {
        a: &a,
        x0: &x0,
        howmany: 2,
        which: Which::SmallestReal,
        alg,
    });
}

#[test]
fn non_hermitian_imaginary_part_is_recorded() {
    let mut g = SplitMix(18);
    let n = 20;
    let a = DMatrix::from_fn(n, n, |_, _| g.scalar(true));
    let x0 = to_dvec::<Complex64>(&random_vec(&mut g, n, true));
    let mut ws = Workspace::with_capacity(&mut (), &x0, 10).unwrap();
    let alg = Lanczos::new(10, 1, 1e-12);
    let (_, _, info) = eigsolve(
        &mut (),
        &mut DenseOp::new(&a),
        &x0,
        1,
        Which::LargestMagnitude,
        &alg,
        &mut ws,
    )
    .unwrap();
    // Recorded beyond the rounding a Hermitian map would leave.
    assert!(info.max_imag > rounding::<Complex64>(n, op_norm(&a)));
}

#[test]
fn errors() {
    let a = DMatrix::<Complex64>::identity(4, 4);
    let x = DVec(vec![1.0f64, 2.0, 3.0, 4.0]);
    let mut op = DenseOp::<f64>::new(&a);
    let mut ws = Workspace::with_capacity(&mut (), &x, 3).unwrap();
    let run = |op: &mut DenseOp<f64>, x: &DVec<f64>, h, alg: Lanczos, ws: &mut Workspace<_, _>| {
        eigsolve(&mut (), op, x, h, Which::LargestReal, &alg, ws)
            .map(|_| ())
            .unwrap_err()
    };
    let alg = Lanczos::new(3, 100, 1e-12);
    assert_eq!(
        run(&mut op, &DVec(vec![0.0; 4]), 1, alg, &mut ws),
        Error::ZeroStartVector
    );
    assert_eq!(
        run(&mut op, &DVec(vec![f64::NAN; 4]), 1, alg, &mut ws),
        Error::NonFiniteStartVector
    );
    assert_eq!(
        run(&mut op, &x, 4, alg, &mut ws),
        Error::HowmanyTooLarge {
            howmany: 4,
            krylovdim: 3
        }
    );
    assert_eq!(
        run(&mut op, &x, 1, Lanczos::new(4, 100, 1e-12), &mut ws),
        Error::InsufficientCapacity {
            needed: 4,
            capacity: 3
        }
    );
    assert!(matches!(
        run(&mut op, &x, 1, Lanczos::new(3, 0, 1e-12), &mut ws),
        Error::InvalidArgument(_)
    ));
    assert!(matches!(
        run(&mut op, &x, 1, Lanczos::new(3, 100, f64::NAN), &mut ws),
        Error::InvalidArgument(_)
    ));
}

#[test]
fn warm_solve_allocates_only_returned_vectors() {
    let mut g = SplitMix(19);
    let a = hermitian(&mut g, 100, true);
    let x0 = to_dvec::<Complex64>(&random_vec(&mut g, 100, true));
    let alg = Lanczos::new(16, 100, 1e-10);
    let mut op = DenseOp::new(&a);
    let mut ws = Workspace::with_capacity(&mut (), &x0, 16).unwrap();
    for _ in 0..2 {
        let before = similar_calls();
        let (vals, vecs, info) =
            eigsolve(&mut (), &mut op, &x0, 3, Which::SmallestReal, &alg, &mut ws).unwrap();
        assert!(info.numiter > 1, "must exercise restarts");
        // Only the returned eigenvectors and residual vectors are new.
        assert_eq!(similar_calls() - before, 2 * vals.len());
        assert_eq!(vecs.len(), vals.len());
    }
}

#[test]
fn basis_stays_orthonormal_across_restarts() {
    // Wide spectrum with a tight target: plain Lanczos would lose
    // orthogonality long before convergence.
    let mut g = SplitMix(20);
    let eigs: Vec<f64> = (0..300)
        .map(|i| 10f64.powf(-4.0 + 8.0 * i as f64 / 299.0))
        .collect();
    let a = with_spectrum(&mut g, &eigs, false);
    let x0 = to_dvec::<f64>(&random_vec(&mut g, 300, false));
    for orth in [Orth::Cgs2, Orth::Mgs2] {
        let alg = Lanczos::new(40, 6, 1e-6).orth(orth);
        let mut ws = Workspace::with_capacity(&mut (), &x0, 40).unwrap();
        let _ = eigsolve(
            &mut (),
            &mut DenseOp::new(&a),
            &x0,
            2,
            Which::LargestReal,
            &alg,
            &mut ws,
        )
        .unwrap();
        let e = orthogonality_error(ws.basis().vectors());
        assert!(ws.basis().len() > 20);
        within(
            e,
            rounding::<f64>(300, 1.0),
            &format!("{orth:?}: orthogonality"),
        );
    }
}

#[test]
fn lanczos_step_uses_one_full_pass_after_the_recurrence() {
    // KrylovKit's `lanczosrecurrence` (CGS2 and MGS2): at basis size k, one
    // inner product for α (three-term recurrence) plus one full pass of k,
    // i.e. Σ_{k=1}^{K} (k + 1) = K(K + 3)/2 over one Krylov space of size K.
    // Orthogonalizing twice against the full basis would take K(K + 1).
    let mut g = SplitMix(21);
    let a = hermitian(&mut g, 60, true);
    let x0 = to_dvec::<Complex64>(&random_vec(&mut g, 60, true));
    let kd = 20;
    for orth in [Orth::Cgs2, Orth::Mgs2] {
        let alg = Lanczos::new(kd, 1, 0.0).orth(orth);
        let mut ws = Workspace::with_capacity(&mut (), &x0, kd).unwrap();
        let mut op = DenseOp::new(&a);
        let before = inner_calls();
        let (_, _, info) =
            eigsolve(&mut (), &mut op, &x0, 1, Which::SmallestReal, &alg, &mut ws).unwrap();
        assert_eq!(info.numops, kd);
        assert_eq!(inner_calls() - before, kd * (kd + 3) / 2, "{orth:?}");
        within(
            orthogonality_error(ws.basis().vectors()),
            rounding::<Complex64>(60, 1.0),
            &format!("{orth:?}: orthogonality"),
        );
    }
}

/// A caller-defined basis that forwards every method, including the
/// per-column `inner_at`/`axpy_at` that modified Gram–Schmidt requires.
struct Custom(VecBasis<DVec<f64>>);
impl Basis<DVec<f64>> for Custom {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn capacity(&self) -> usize {
        self.0.capacity()
    }
    fn push(&mut self, cx: &mut (), v: &DVec<f64>, a: f64) -> Result<(), std::convert::Infallible> {
        self.0.push(cx, v, a)
    }
    fn truncate(&mut self, len: usize) {
        self.0.truncate(len)
    }
    fn project(
        &self,
        cx: &mut (),
        x: &DVec<f64>,
        out: &mut [f64],
    ) -> Result<(), std::convert::Infallible> {
        self.0.project(cx, x, out)
    }
    fn combine_into(
        &self,
        cx: &mut (),
        c: &[f64],
        y: &mut DVec<f64>,
        a: f64,
        b: f64,
    ) -> Result<(), std::convert::Infallible> {
        self.0.combine_into(cx, c, y, a, b)
    }
    fn rotate(
        &mut self,
        cx: &mut (),
        q: krylov::MatRef<'_, f64>,
        keep: usize,
    ) -> Result<(), std::convert::Infallible> {
        self.0.rotate(cx, q, keep)
    }
    fn inner_at(
        &self,
        cx: &mut (),
        i: usize,
        x: &DVec<f64>,
    ) -> Result<f64, std::convert::Infallible> {
        self.0.inner_at(cx, i, x)
    }
    fn axpy_at(
        &self,
        cx: &mut (),
        i: usize,
        a: f64,
        y: &mut DVec<f64>,
    ) -> Result<(), std::convert::Infallible> {
        self.0.axpy_at(cx, i, a, y)
    }
}

#[test]
fn mgs2_on_a_custom_basis_takes_two_inner_products_per_basis_vector() {
    // KrylovKit `orthonormal.jl:reorthogonalize!!(::ModifiedGramSchmidt2)`:
    // two sequential passes over the k basis vectors at basis size k, so one
    // Arnoldi cycle of size K takes Σ_{k=1}^{K} 2k = K(K + 1) inner products
    // (a full projection per coefficient would take K(K + 1)(2K + 1)/3).
    let n = 64;
    let lap = DMatrix::from_fn(n, n, |i, j| {
        Complex64::new(
            match i.abs_diff(j) {
                0 => 2.0,
                1 => -1.0,
                _ => 0.0,
            },
            0.0,
        )
    });
    let mut e1 = vec![Complex64::new(0.0, 0.0); n];
    e1[0] = Complex64::new(1.0, 0.0);
    let x0 = to_dvec::<f64>(&e1);
    for kd in [8, 16, 32] {
        let basis = Custom(VecBasis::with_capacity(&mut (), &x0, kd).unwrap());
        let mut ws = Workspace::new(&mut (), basis, &x0).unwrap();
        let mut op = DenseOp::new(&lap);
        let alg = Arnoldi::new(kd, 1, 0.0).orth(Orth::Mgs2);
        let before = inner_calls();
        let info = schursolve(&mut (), &mut op, &x0, 1, Which::LargestReal, &alg, &mut ws)
            .unwrap()
            .3;
        assert_eq!(info.numops, kd);
        assert_eq!(inner_calls() - before, kd * (kd + 1), "K = {kd}");
    }
}

#[test]
fn vecbasis_rotate_matches_dense_product() {
    let mut g = SplitMix(22);
    let (n, m, keep) = (7, 5, 3);
    let x = DVec(vec![Complex64::new(0.0, 0.0); n]);
    let mut basis = VecBasis::with_capacity(&mut (), &x, m).unwrap();
    let cols: Vec<Vec<Complex64>> = (0..m).map(|_| random_vec(&mut g, n, true)).collect();
    for c in &cols {
        basis
            .push(&mut (), &DVec(c.clone()), Complex64::new(1.0, 0.0))
            .unwrap();
    }
    let q = unitary(&mut g, m, true);
    let qdata: Vec<Complex64> = q.iter().copied().collect();
    basis
        .rotate(&mut (), krylov::MatRef::new(&qdata, m, m), keep)
        .unwrap();
    assert_eq!(basis.len(), keep);
    for j in 0..keep {
        let expect: Vec<Complex64> = (0..n)
            .map(|r| (0..m).map(|i| cols[i][r] * q[(i, j)]).sum())
            .collect();
        // An m-term sum per entry: rounding of the inputs' Frobenius norm.
        let scale = cols.iter().map(|c| norm(c).powi(2)).sum::<f64>().sqrt();
        within(
            diff_norm(&basis.vectors()[j].0, &expect),
            rounding::<Complex64>(m, scale),
            "rotated column",
        );
    }
}
