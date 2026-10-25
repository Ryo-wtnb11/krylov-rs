//! Portable public-API comparison with the LAPACK baseline (copy this file
//! and its [[bench]] entry there). No backend types or test helpers.
//! The baseline binary still needs its provider linked externally, e.g.
//! RUSTFLAGS="-l framework=Accelerate" on macOS (or "-l openblas" on Linux).
//! Run with RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
//! cargo bench --bench backend [-- --smoke] [-- --sparse --n=N]. One warmup, five timed repeats;
//! smoke uses small matrices and two repeats. Matrix/workspace setup is untimed.
//! Dense operators use identical plain Rust loops on both backends. Timings
//! include Krylov/vector work; consult META counts when comparing backends.

use krylov::{
    AdjointMap, Arnoldi, Gkl, Gmres, Lanczos, LinearMap, Scalar, Vector, Which, Workspace,
    arnoldi_eigsolve, eigsolve, exponentiate, linsolve, schursolve, svdsolve,
};
use num_complex::Complex64;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::{convert::Infallible, hint::black_box, time::Instant};

trait BenchScalar: Scalar<Real = f64> {
    const NAME: &'static str;
    fn make(re: f64, im: f64) -> Self;
    fn cj(self) -> Self;
    fn abs2(self) -> f64;
}
impl BenchScalar for f64 {
    const NAME: &'static str = "f64";
    fn make(re: f64, _: f64) -> Self {
        re
    }
    fn cj(self) -> Self {
        self
    }
    fn abs2(self) -> f64 {
        self * self
    }
}
impl BenchScalar for Complex64 {
    const NAME: &'static str = "c64";
    fn make(re: f64, im: f64) -> Self {
        Self::new(re, im)
    }
    fn cj(self) -> Self {
        self.conj()
    }
    fn abs2(self) -> f64 {
        self.norm_sqr()
    }
}
struct V<S>(Vec<S>);
impl<S: BenchScalar> Vector for V<S> {
    type Scalar = S;
    type Error = Infallible;
    fn similar(&self) -> Result<Self, Infallible> {
        Ok(Self(vec![S::zero(); self.0.len()]))
    }
    fn copy_from(&mut self, x: &Self) -> Result<(), Infallible> {
        self.0.copy_from_slice(&x.0);
        Ok(())
    }
    fn inner(&self, y: &Self) -> Result<S, Infallible> {
        Ok(self
            .0
            .iter()
            .zip(&y.0)
            .fold(S::zero(), |s, (&a, &b)| s + a.cj() * b))
    }
    fn norm(&self) -> Result<f64, Infallible> {
        Ok(self.0.iter().map(|&v| v.abs2()).sum::<f64>().sqrt())
    }
    fn axpby(&mut self, a: S, x: &Self, b: S) -> Result<(), Infallible> {
        for (y, &x) in self.0.iter_mut().zip(&x.0) {
            *y = if b == S::zero() {
                a * x
            } else {
                a * x + b * *y
            };
        }
        Ok(())
    }
}
struct Dense<S> {
    a: Vec<S>,
    m: usize,
    n: usize,
    // `--sparse`: keep only the tridiagonal band as (row, col, value), so a
    // matvec is O(n) and the backend's projected problem is a larger share.
    band: Option<Vec<(usize, usize, S)>>,
}
impl<S: BenchScalar> Dense<S> {
    fn finish(mut self) -> Self {
        if SPARSE.load(Relaxed) {
            let n = self.n;
            self.band = Some(
                (0..self.m)
                    .flat_map(|i| (i.saturating_sub(1)..(i + 2).min(n)).map(move |j| (i, j)))
                    .map(|(i, j)| (i, j, self.a[i * n + j]))
                    .collect(),
            );
        }
        self
    }
}
static SPARSE: AtomicBool = AtomicBool::new(false);
impl<S: BenchScalar> LinearMap<V<S>> for Dense<S> {
    fn apply_into(&mut self, x: &V<S>, y: &mut V<S>) -> Result<(), Infallible> {
        if let Some(band) = &self.band {
            y.0.fill(S::zero());
            for &(i, j, a) in band {
                y.0[i] += a * x.0[j];
            }
            return Ok(());
        }
        for (row, yi) in self.a.chunks_exact(self.n).zip(&mut y.0) {
            *yi = row
                .iter()
                .zip(&x.0)
                .fold(S::zero(), |s, (&a, &x)| s + a * x);
        }
        Ok(())
    }
}
impl<S: BenchScalar> AdjointMap<V<S>> for Dense<S> {
    fn apply_adjoint_into(&mut self, y: &V<S>, x: &mut V<S>) -> Result<(), Infallible> {
        x.0.fill(S::zero());
        if let Some(band) = &self.band {
            for &(i, j, a) in band {
                x.0[j] += a.cj() * y.0[i];
            }
            return Ok(());
        }
        for (row, &yi) in self.a.chunks_exact(self.n).zip(&y.0) {
            for (xi, &a) in x.0.iter_mut().zip(row) {
                *xi += a.cj() * yi;
            }
        }
        Ok(())
    }
}
fn random(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5
}
fn dense<S: BenchScalar>(m: usize, n: usize, hermitian: bool) -> Dense<S> {
    let mut seed = 42;
    let scale = 1.0 / (n as f64).sqrt();
    let mut a: Vec<S> = (0..m * n)
        .map(|_| S::make(random(&mut seed) * scale, random(&mut seed) * scale))
        .collect();
    if hermitian {
        for i in 0..n {
            for j in 0..i {
                a[i * n + j] = a[j * n + i].cj();
            }
        }
    }
    for i in 0..m.min(n) {
        // Separated outliers make all requested eigen/singular triplets
        // reachable without an unbounded benchmark convergence problem.
        a[i * n + i] = S::make(if i < 16 { 4.0 + 0.4 * i as f64 } else { 1.0 }, 0.0);
    }
    Dense { a, m, n, band: None }.finish()
}
fn generator<S: BenchScalar>(n: usize, hermitian: bool) -> Dense<S> {
    let mut a = Dense { band: None, ..dense::<S>(n, n, hermitian) };
    for x in &mut a.a {
        *x *= S::make(0.1, 0.0);
    }
    // An isolated zero mode prevents decay to zero. The other modes span
    // [-101,-1], so t=10 needs multiple spaces even at krylovdim=80.
    for i in 0..n {
        a.a[i] = S::zero();
        a.a[i * n] = S::zero();
        if i > 0 {
            a.a[i * n + i] = S::make(-1.0 - 100.0 * (i - 1) as f64 / (n - 1) as f64, 0.0);
        }
    }
    a.finish()
}
fn measure(name: &str, repeats: usize, mut run: impl FnMut() -> (usize, usize)) {
    black_box(run());
    let mut times = Vec::with_capacity(repeats);
    let mut counts = (0, 0);
    for _ in 0..repeats {
        let start = Instant::now();
        counts = black_box(run());
        times.push(start.elapsed().as_nanos());
    }
    times.sort_unstable();
    let median = if repeats % 2 == 0 {
        (times[repeats / 2 - 1] + times[repeats / 2]) / 2
    } else {
        times[repeats / 2]
    };
    let name = format!("{}{name}", if SPARSE.load(Relaxed) { "sparse_" } else { "" });
    println!("BENCH {name} median_ns={median} min_ns={}", times[0]);
    println!("META {name} numiter={} numops={}", counts.0, counts.1);
}
fn run<S: BenchScalar>(smoke: bool, size: Option<usize>) {
    let (n, m, repeats) = match (smoke, size) {
        (true, _) => (48, 36, 2),
        (false, Some(n)) => (n, n * 3 / 4, 5),
        (false, None) => (2000, 1500, 5),
    };
    let dims = if smoke { [12, 24] } else { [30, 80] };
    let counts = if smoke { [2, 4] } else { [4, 16] };
    let mut h = dense::<S>(n, n, true);
    let mut a = dense::<S>(n, n, false);
    let x = V((0..n)
        .map(|i| S::make(1.0 + (i as f64).sin(), 0.0))
        .collect());
    for kd in dims {
        let mut ws = Workspace::with_capacity(&x, kd).unwrap();
        for howmany in counts {
            let name = format!("eig_lanczos_{}_n{n}_h{howmany}_k{kd}", S::NAME);
            measure(&name, repeats, || {
                let out = eigsolve(
                    &mut h,
                    &x,
                    howmany,
                    Which::LargestReal,
                    &Lanczos::new(kd, 100, 1e-9),
                    &mut ws,
                )
                .unwrap();
                assert!(out.2.converged >= howmany);
                let counts = (out.2.numiter, out.2.numops);
                black_box(out);
                counts
            });
            let name = format!("schur_arnoldi_{}_n{n}_h{howmany}_k{kd}", S::NAME);
            measure(&name, repeats, || {
                let out = schursolve(
                    &mut a,
                    &x,
                    howmany,
                    Which::LargestReal,
                    &Arnoldi::new(kd, 100, 1e-9),
                    &mut ws,
                )
                .unwrap();
                assert!(out.3.converged >= howmany);
                let counts = (out.3.numiter, out.3.numops);
                black_box(out);
                counts
            });
        }
        let t = S::make(10.0, 0.0);
        for lanczos in [true, false] {
            let mut generator = generator::<S>(n, lanczos);
            let name = format!(
                "exp_{}_{}_n{n}_k{kd}",
                if lanczos { "lanczos" } else { "arnoldi" },
                S::NAME
            );
            measure(&name, repeats, || {
                let out = if lanczos {
                    exponentiate(
                        &mut generator,
                        t,
                        &x,
                        &Lanczos::new(kd, 1000, 1e-9),
                        &mut ws,
                    )
                } else {
                    exponentiate(
                        &mut generator,
                        t,
                        &x,
                        &Arnoldi::new(kd, 1000, 1e-9),
                        &mut ws,
                    )
                }
                .unwrap();
                assert!(out.1.converged > 0);
                assert!(out.1.numiter > 1, "exponential benchmark must restart");
                let counts = (out.1.numiter, out.1.numops);
                black_box(out);
                counts
            });
        }
        let zero = V(vec![S::zero(); n]);
        let name = format!("linsolve_gmres_{}_n{n}_k{kd}", S::NAME);
        measure(&name, repeats, || {
            let out = linsolve(
                &mut a,
                &x,
                &zero,
                &Gmres::new(kd, 100, 1e-9),
                S::make(1.0, 0.0),
                S::make(1.0, 0.0),
                &mut ws,
            )
            .unwrap();
            assert!(out.1.converged > 0);
            let counts = (out.1.numiter, out.1.numops);
            black_box(out);
            counts
        });
        let mut rectangular = dense::<S>(n, m, false);
        let domain = V(vec![S::zero(); rectangular.n]);
        assert_eq!(rectangular.m, x.0.len());
        let mut ws_v = Workspace::with_capacity(&domain, kd).unwrap();
        for howmany in counts {
            let name = format!("svd_gkl_{}_m{n}_n{m}_h{howmany}_k{kd}", S::NAME);
            measure(&name, repeats, || {
                let out = svdsolve(
                    &mut rectangular,
                    &x,
                    howmany,
                    Which::LargestReal,
                    &Gkl::new(kd, 100, 1e-9),
                    &mut ws,
                    &mut ws_v,
                )
                .unwrap();
                assert!(out.3.converged >= howmany);
                let counts = (out.3.numiter, out.3.numops);
                black_box(out);
                counts
            });
        }
    }
}
fn main() {
    let smoke = std::env::args().any(|a| a == "--smoke");
    SPARSE.store(std::env::args().any(|a| a == "--sparse"), Relaxed);
    // `--n=N` overrides the full-run dimension (rectangular: N x 3N/4).
    let size = std::env::args().find_map(|a| a.strip_prefix("--n=")?.parse().ok());
    run::<f64>(smoke, size);
    run::<Complex64>(smoke, size);
    let n = if smoke { 48 } else { size.unwrap_or(2000) };
    let mut a = dense::<Complex64>(n, n, false);
    let x = V(vec![Complex64::new(1.0, 0.5); n]);
    for kd in if smoke { [12, 24] } else { [30, 80] } {
        let mut ws = Workspace::with_capacity(&x, kd).unwrap();
        let name = format!("eig_arnoldi_c64_n{n}_h4_k{kd}");
        measure(&name, if smoke { 2 } else { 5 }, || {
            let out = arnoldi_eigsolve(
                &mut a,
                &x,
                4,
                Which::LargestReal,
                &Arnoldi::new(kd, 100, 1e-9),
                &mut ws,
            )
            .unwrap();
            assert!(out.2.converged >= 4);
            let counts = (out.2.numiter, out.2.numops);
            black_box(out);
            counts
        });
    }
}
