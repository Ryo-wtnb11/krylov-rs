//! `exponentiate`, `expintegrator` and GMRES `linsolve` with vectors that
//! truncate their rank inside every rank-growing operation, the way a
//! tensor-network vector (MPS/TT) would. krylov needs no hook for this: every combination is a
//! user trait method (docs/guide.md, "Low-rank and tensor-network vectors").
//!
//! The vector is an `n × n` matrix `X` truncated by SVD to relative accuracy
//! `τ` (`‖X − X_r‖_F <= τ·‖X‖_F`, TT-Toolbox `round` semantics). The map is
//! `X ↦ L·X + X·Lᵀ` (Kronecker sum, the heat-equation structure of
//! Weigel–Riemer–Stoll arXiv:2607.11293). Its exact flow
//! `e^{tL}·X₀·e^{tLᵀ}` and the dense solve of the vectorized system are
//! oracles independent of krylov.

mod common;

use std::cell::Cell;

use common::*;
use krylov::{
    Arnoldi, Gmres, KrylovProcess, Lanczos, LinearMap, RestartResidual, Vector, Workspace,
    expintegrator, exponentiate, linsolve,
};
use nalgebra::DMatrix;
use num_complex::Complex64;

thread_local! {
    static MAX_RANK: Cell<usize> = const { Cell::new(0) };
    static DROPS: Cell<usize> = const { Cell::new(0) };
}

/// Low-rank matrix vector. Real scalar types keep the imaginary part zero.
struct LowRank<S> {
    x: DMatrix<Complex64>,
    tau: f64,
    _s: std::marker::PhantomData<S>,
}

impl<S: TestScalar> LowRank<S> {
    fn new(x: DMatrix<Complex64>, tau: f64) -> Self {
        let mut v = Self {
            x,
            tau,
            _s: std::marker::PhantomData,
        };
        v.truncate();
        v
    }

    /// Keeps the fewest singular triplets whose discarded weight is
    /// `<= τ·‖X‖_F`. Uses LAPACK `zgesvd`: nalgebra 0.35's `svd` returned
    /// singular vectors that do not reconstruct an exactly rank-2 16 × 16
    /// matrix (`‖U·Σ·Vᵀ − X‖ ≈ ‖X‖`).
    fn truncate(&mut self) {
        let (m, n) = (self.x.nrows(), self.x.ncols());
        let k = m.min(n);
        let mut a = self.x.as_slice().to_vec();
        let mut s = vec![0.0; k];
        let mut u = vec![Complex64::default(); m * k];
        let mut vt = vec![Complex64::default(); k * n];
        let mut rwork = vec![0.0; 5 * k];
        let mut info = 0;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let mut query = [Complex64::default()];
        // SAFETY: every buffer has the length zgesvd requires for jobu =
        // jobvt = 'S' with the leading dimensions passed.
        unsafe {
            lapack::zgesvd(
                b'S', b'S', mi, ni, &mut a, mi, &mut s, &mut u, mi, &mut vt, ki, &mut query, -1,
                &mut rwork, &mut info,
            );
        }
        let lwork = query[0].re as usize;
        let mut work = vec![Complex64::default(); lwork];
        // SAFETY: as above, with `work` of the queried length.
        unsafe {
            lapack::zgesvd(
                b'S',
                b'S',
                mi,
                ni,
                &mut a,
                mi,
                &mut s,
                &mut u,
                mi,
                &mut vt,
                ki,
                &mut work,
                lwork as i32,
                &mut rwork,
                &mut info,
            );
        }
        assert_eq!(info, 0, "zgesvd failed");
        // s is descending.
        let total: f64 = s.iter().map(|x| x * x).sum();
        let budget = self.tau * self.tau * total;
        let mut tail = 0.0;
        let mut r = k;
        while r > 0 && tail + s[r - 1] * s[r - 1] <= budget {
            tail += s[r - 1] * s[r - 1];
            r -= 1;
        }
        if r < s.iter().filter(|&&x| x > 0.0).count() {
            DROPS.with(|c| c.set(c.get() + 1));
        }
        MAX_RANK.with(|c| c.set(c.get().max(r)));
        let u = DMatrix::from_column_slice(m, k, &u);
        let vt = DMatrix::from_column_slice(k, n, &vt);
        let mut y = DMatrix::<Complex64>::zeros(m, n);
        for (i, &si) in s.iter().enumerate().take(r) {
            y += u.column(i) * Complex64::new(si, 0.0) * vt.row(i);
        }
        if !<S as TestScalar>::COMPLEX {
            y.apply(|z| *z = Complex64::new(z.re, 0.0));
        }
        self.x = y;
    }
}

impl<S: TestScalar> Vector for LowRank<S> {
    type Scalar = S;
    type Error = std::convert::Infallible;
    fn similar(&self) -> Result<Self, Self::Error> {
        Ok(Self {
            x: DMatrix::zeros(self.x.nrows(), self.x.ncols()),
            tau: self.tau,
            _s: std::marker::PhantomData,
        })
    }
    fn copy_from(&mut self, x: &Self) -> Result<(), Self::Error> {
        self.x.copy_from(&x.x);
        Ok(())
    }
    fn inner(&self, y: &Self) -> Result<S, Self::Error> {
        Ok(S::from_c64(self.x.dotc(&y.x)))
    }
    fn norm(&self) -> Result<S::Real, Self::Error> {
        Ok(<S::Real as num_traits::NumCast>::from(self.x.norm()).unwrap())
    }
    /// Rank grows only when both terms survive. Rescaling (`b = 0` or
    /// `a = 0`) never truncates again, which would let a repeatedly scaled
    /// vector drift by `τ` each time.
    fn axpby(&mut self, a: S, x: &Self, b: S) -> Result<(), Self::Error> {
        let (a, b) = (a.to_c64(), b.to_c64());
        if b == Complex64::new(0.0, 0.0) {
            self.x = &x.x * a;
        } else {
            self.x = &x.x * a + &self.x * b;
            if a != Complex64::new(0.0, 0.0) {
                self.truncate();
            }
        }
        Ok(())
    }
}

/// `X ↦ L·X + X·Lᵀ`, truncated like any product of a tensor-network map.
struct KronSum(DMatrix<Complex64>);

impl<S: TestScalar> LinearMap<LowRank<S>> for KronSum {
    fn apply_into(
        &mut self,
        x: &LowRank<S>,
        y: &mut LowRank<S>,
    ) -> Result<(), std::convert::Infallible> {
        y.x = &self.0 * &x.x + &x.x * self.0.transpose();
        y.truncate();
        Ok(())
    }
}

const N: usize = 32;
const KD: usize = 10;
const TOL: f64 = 1e-8;

/// Scaled 1D Laplacian `c·tridiag(1, −2, 1)` plus an optional advection
/// `d·(upper − lower)`, which makes `L` (and the map) non-normal.
fn laplacian(n: usize, c: f64, d: f64) -> DMatrix<Complex64> {
    DMatrix::from_fn(n, n, |i, j| {
        let v = if i == j {
            -2.0 * c
        } else if j == i + 1 {
            c + d
        } else if i == j + 1 {
            c - d
        } else {
            0.0
        };
        Complex64::new(v, 0.0)
    })
}

/// `Σ_i ρⁱ·c_i·d_iᵀ` over `terms` random rank-one terms: rank `terms`, with
/// singular values decaying roughly like `ρⁱ`.
fn start(g: &mut SplitMix, n: usize, complex: bool, terms: usize, rho: f64) -> DMatrix<Complex64> {
    let mut x = DMatrix::<Complex64>::zeros(n, n);
    let mut w = 1.0;
    for _ in 0..terms {
        let c = DMatrix::from_column_slice(n, 1, &random_vec(g, n, complex));
        let d = DMatrix::from_column_slice(n, 1, &random_vec(g, n, complex));
        x += &c * d.transpose() * Complex64::new(w, 0.0);
        w *= rho;
    }
    x
}

fn reset() {
    MAX_RANK.with(|c| c.set(0));
    DROPS.with(|c| c.set(0));
}

struct Run {
    /// `‖y − exact‖_F / ‖X₀‖_F`.
    err: f64,
    converged: usize,
    max_rank: usize,
    drops: usize,
}

fn run_exp<S: TestScalar, P: KrylovProcess>(
    l: &DMatrix<Complex64>,
    x0: &DMatrix<Complex64>,
    t: Complex64,
    tau: f64,
    alg: &P,
) -> Run {
    reset();
    let x = LowRank::<S>::new(x0.clone(), tau);
    let mut ws = Workspace::with_capacity(&x, KD).unwrap();
    let (y, info) =
        exponentiate(&mut KronSum(l.clone()), S::from_c64(t), &x, alg, &mut ws).unwrap();
    let e = (l * t).exp();
    let exact = &e * x0 * e.transpose();
    Run {
        err: (&y.x - &exact).norm() / x0.norm(),
        converged: info.converged,
        max_rank: MAX_RANK.with(Cell::get),
        drops: DROPS.with(Cell::get),
    }
}

/// Real `t` (diffusion, real vectors; Lanczos when the map is Hermitian)
/// and complex `t` (complex vectors, Arnoldi).
fn both(g: &mut SplitMix, l: &DMatrix<Complex64>, terms: usize, tau: f64) -> [Run; 2] {
    let hermitian = l == &l.adjoint();
    let lanczos = Lanczos::new(KD, 100, TOL);
    let arnoldi = Arnoldi::new(KD, 100, TOL);
    let x0 = start(g, N, false, terms, 0.1);
    let one = Complex64::new(1.0, 0.0);
    let real = if hermitian {
        run_exp::<f64, _>(l, &x0, one, tau, &lanczos)
    } else {
        run_exp::<f64, _>(l, &x0, one, tau, &arnoldi)
    };
    let x0 = start(g, N, true, terms, 0.1);
    let complex = run_exp::<Complex64, _>(l, &x0, Complex64::new(0.3, -0.5), tau, &arnoldi);
    [real, complex]
}

#[test]
fn truncation_bounds_the_rank_of_a_low_rank_flow() {
    // A rank-one start X₀ = c·dᵀ: the exact flow stays rank one, and a Krylov
    // vector built from Aʲ·X₀ with j ≤ k lies in span{Lⁱ·c}·span{(Lᵀ)ʲ·d}ᵀ,
    // so its rank is at most k + 1. Truncating rounding noise keeps every
    // vector well below full rank; without truncation the noise fills it.
    let l = laplacian(N, 0.3, 0.0);
    for tau in [TOL / 10.0, TOL] {
        let mut g = SplitMix(81);
        for r in both(&mut g, &l, 1, tau) {
            assert_eq!(r.converged, 1);
            assert!(r.err <= TOL, "τ = {tau}: error {}", r.err);
            assert!(r.max_rank <= N / 2, "τ = {tau}: rank {}", r.max_rank);
        }
    }
    let mut g = SplitMix(81);
    for r in both(&mut g, &l, 1, 0.0) {
        assert!(
            r.max_rank > N / 2,
            "without truncation rounding noise fills the rank"
        );
    }
}

#[test]
fn error_follows_the_truncation_tolerance() {
    // A start with decaying singular values (ρ = 0.1, full rank): truncation
    // discards real weight, and the error stays within a small multiple of
    // `tol + τ` (KIOPS-TT's default τ = tol/10 and the paper's τ = tol).
    for (l, what) in [
        (laplacian(N, 0.3, 0.0), "Hermitian"),
        (laplacian(N, 0.3, 0.1), "non-normal"),
    ] {
        for tau in [TOL / 10.0, TOL] {
            let mut g = SplitMix(82);
            for r in both(&mut g, &l, N, tau) {
                assert_eq!(r.converged, 1, "{what}");
                assert!(r.drops > 0, "{what}: truncation never dropped anything");
                assert!(
                    r.err <= 5.0 * (TOL + tau),
                    "{what}, τ = {tau}: error {}",
                    r.err
                );
            }
        }
    }
}

#[test]
fn coarse_truncation_exceeds_the_krylov_tolerance() {
    // τ = 1e3·tol: the step error estimate does not see truncation error, so
    // the run still reports convergence while missing the tolerance.
    let l = laplacian(N, 0.3, 0.0);
    let mut g = SplitMix(83);
    for r in both(&mut g, &l, N, 1e3 * TOL) {
        assert_eq!(r.converged, 1);
        assert!(
            r.err > 10.0 * TOL,
            "error {} should exceed the tolerance",
            r.err
        );
    }
}

#[test]
fn phi_combination_with_truncated_vectors() {
    // ẋ = A·x + u₁ with A the vectorized Kronecker sum (column-major vec:
    // vec(L·X + X·Lᵀ) = (I⊗L + L⊗I)·vec X), checked against the dense
    // augmented exponential.
    let n = 8;
    let id = DMatrix::<Complex64>::identity(n, n);
    let lanczos = Lanczos::new(KD, 100, TOL);
    let arnoldi = Arnoldi::new(KD, 100, TOL);
    for tau in [TOL / 10.0, TOL] {
        let mut g = SplitMix(84);
        for (adv, complex) in [(0.0, false), (0.05, false), (0.0, true), (0.05, true)] {
            let l = laplacian(n, 0.3, adv);
            let a = id.kronecker(&l) + l.kronecker(&id);
            let t = if complex {
                Complex64::new(0.3, -0.5)
            } else {
                Complex64::new(1.0, 0.0)
            };
            let u0 = start(&mut g, n, complex, n, 0.01);
            let u1 = start(&mut g, n, complex, n, 0.01);
            let exact = dense_phi_comb(&a, t, &[u0.as_slice().to_vec(), u1.as_slice().to_vec()]);
            // Lanczos for the symmetric map, Arnoldi for the advective one.
            let err = match (adv == 0.0, complex) {
                (true, false) => phi::<f64, _>(&l, &u0, &u1, t, tau, &lanczos),
                (true, true) => phi::<Complex64, _>(&l, &u0, &u1, t, tau, &lanczos),
                (false, false) => phi::<f64, _>(&l, &u0, &u1, t, tau, &arnoldi),
                (false, true) => phi::<Complex64, _>(&l, &u0, &u1, t, tau, &arnoldi),
            };
            let err = diff_norm(&err, &exact) / (u0.norm() + u1.norm());
            assert!(
                err <= 5.0 * (TOL + tau),
                "τ = {tau}, adv {adv}, complex {complex}: {err}"
            );
        }
    }
}

fn phi<S: TestScalar, P: KrylovProcess>(
    l: &DMatrix<Complex64>,
    u0: &DMatrix<Complex64>,
    u1: &DMatrix<Complex64>,
    t: Complex64,
    tau: f64,
    alg: &P,
) -> Vec<Complex64> {
    reset();
    let u = [
        LowRank::<S>::new(u0.clone(), tau),
        LowRank::<S>::new(u1.clone(), tau),
    ];
    let mut ws = Workspace::with_capacity(&u[0], KD).unwrap();
    let (y, info) =
        expintegrator(&mut KronSum(l.clone()), S::from_c64(t), &u, alg, &mut ws).unwrap();
    assert_eq!(info.converged, 1);
    assert!(
        DROPS.with(Cell::get) > 0,
        "truncation never dropped anything"
    );
    y.x.as_slice().to_vec()
}

struct Solve {
    /// `‖X − X*‖_F / ‖X*‖_F` against the dense solve.
    err: f64,
    converged: usize,
    numiter: usize,
    drops: usize,
}

/// `(3 + A)·X = B` with `A: X ↦ L·X + X·Lᵀ` (spectrum in `[0.6, 3]`) by
/// GMRES with `krylovdim = 3` (several restarts) and the absolute tolerance
/// `TOL·‖B‖`, checked against the dense LU solve of the vectorized system
/// `3 + I⊗L + L⊗I`, which shares no code with krylov.
fn run_gmres<S: TestScalar>(rhs: &DMatrix<Complex64>, tau: f64) -> Solve {
    reset();
    let n = rhs.nrows();
    let l = laplacian(n, 0.3, 0.1);
    let (a0, a1) = (Complex64::new(3.0, 0.0), Complex64::new(1.0, 0.0));
    let b = LowRank::<S>::new(rhs.clone(), tau);
    let x0 = LowRank::<S>::new(DMatrix::zeros(n, n), tau);
    let alg = Gmres::new(3, 50, TOL * rhs.norm());
    assert_eq!(alg.get_restart_residual(), RestartResidual::Recomputed);
    let mut ws = Workspace::with_capacity(&b, 3).unwrap();
    let (x, info) = linsolve(
        &mut KronSum(l.clone()),
        &b,
        &x0,
        &alg,
        S::from_c64(a0),
        S::from_c64(a1),
        &mut ws,
    )
    .unwrap();
    let id = DMatrix::<Complex64>::identity(n, n);
    let m = (id.kronecker(&l) + l.kronecker(&id)) * a1
        + DMatrix::<Complex64>::identity(n * n, n * n) * a0;
    let exact = m
        .lu()
        .solve(&nalgebra::DVector::from_column_slice(rhs.as_slice()))
        .unwrap();
    let exact = DMatrix::from_column_slice(n, n, exact.as_slice());
    Solve {
        err: (&x.x - &exact).norm() / exact.norm(),
        converged: info.converged,
        numiter: info.numiter,
        drops: DROPS.with(Cell::get),
    }
}

/// A real and a complex right-hand side with decaying singular values.
fn both_gmres(tau: f64) -> [Solve; 2] {
    let n = 16;
    let mut g = SplitMix(85);
    let b = start(&mut g, n, false, n, 0.1);
    let real = run_gmres::<f64>(&b, tau);
    let b = start(&mut g, n, true, n, 0.1);
    [real, run_gmres::<Complex64>(&b, tau)]
}

#[test]
fn gmres_with_truncated_vectors() {
    // The default `RestartResidual::Recomputed` restarts from `b − A·x`, so
    // each cycle sees the true residual of the truncated iterate, and the
    // error stays within a small multiple of `tol + τ`. (`Recurrence` also
    // converges here, at most one cycle later at τ ≈ tol: its restart
    // residual is off by about τ·‖b‖, but the final explicit check restarts
    // from the true residual, so the offset is not asserted.)
    for tau in [TOL / 10.0, TOL] {
        for r in both_gmres(tau) {
            assert_eq!(r.converged, 1, "τ = {tau}");
            assert!(r.numiter > 1, "restarts exercised");
            assert!(r.drops > 0, "truncation never dropped anything");
            assert!(r.err <= 5.0 * (TOL + tau), "τ = {tau}: error {}", r.err);
        }
    }
}

#[test]
fn gmres_coarse_truncation_sets_the_accuracy_floor() {
    // τ = 1e3·tol: the recomputed residual cannot drop below the truncation
    // floor, so the run reports non-convergence instead of a false success,
    // and the error sits at the floor, far above the tolerance.
    for r in both_gmres(1e3 * TOL) {
        assert_eq!(r.converged, 0);
        assert!(r.err > 10.0 * TOL, "error {} should exceed tol", r.err);
    }
}
