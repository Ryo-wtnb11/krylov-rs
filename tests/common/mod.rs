//! Shared test helpers: an instrumented host vector, a reproducible RNG and
//! dense nalgebra oracles.
#![allow(dead_code)]

// krylov declares LAPACK; the test binary links the dev-dependency provider.
extern crate lapack_src as _;

use std::cell::Cell;

use krylov::{RealScalar, Scalar, Vector};
use nalgebra::DMatrix;
use num_complex::{Complex32, Complex64};

thread_local! {
    static SIMILAR_CALLS: Cell<usize> = const { Cell::new(0) };
}

/// Number of `similar` calls on this thread so far.
pub fn similar_calls() -> usize {
    SIMILAR_CALLS.with(Cell::get)
}

/// Test scalar: a krylov scalar convertible to and from `Complex64`.
pub trait TestScalar: Scalar {
    fn from_c64(z: Complex64) -> Self;
    fn to_c64(self) -> Complex64;
    const COMPLEX: bool;
    fn cj(self) -> Self {
        Self::from_c64(self.to_c64().conj())
    }
}
impl TestScalar for f64 {
    fn from_c64(z: Complex64) -> Self {
        z.re
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self, 0.0)
    }
    const COMPLEX: bool = false;
}
impl TestScalar for f32 {
    fn from_c64(z: Complex64) -> Self {
        z.re as f32
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self as f64, 0.0)
    }
    const COMPLEX: bool = false;
}
impl TestScalar for Complex64 {
    fn from_c64(z: Complex64) -> Self {
        z
    }
    fn to_c64(self) -> Complex64 {
        self
    }
    const COMPLEX: bool = true;
}
impl TestScalar for Complex32 {
    fn from_c64(z: Complex64) -> Self {
        Complex32::new(z.re as f32, z.im as f32)
    }
    fn to_c64(self) -> Complex64 {
        Complex64::new(self.re as f64, self.im as f64)
    }
    const COMPLEX: bool = true;
}

pub fn to_f64<R: RealScalar>(r: R) -> f64 {
    num_traits::ToPrimitive::to_f64(&r).unwrap()
}

/// Host vector whose `similar` calls are counted.
#[derive(Debug, PartialEq)]
pub struct DVec<S>(pub Vec<S>);

impl<S: TestScalar> Vector for DVec<S> {
    type Scalar = S;
    type Error = std::convert::Infallible;
    fn similar(&self) -> Result<Self, Self::Error> {
        SIMILAR_CALLS.with(|c| c.set(c.get() + 1));
        Ok(DVec(vec![S::zero(); self.0.len()]))
    }
    fn copy_from(&mut self, x: &Self) -> Result<(), Self::Error> {
        self.0.copy_from_slice(&x.0);
        Ok(())
    }
    fn inner(&self, y: &Self) -> Result<S, Self::Error> {
        let mut s = S::zero();
        for (a, b) in self.0.iter().zip(&y.0) {
            s += a.cj() * *b;
        }
        Ok(s)
    }
    fn norm(&self) -> Result<S::Real, Self::Error> {
        let s: f64 = self.0.iter().map(|a| a.to_c64().norm_sqr()).sum();
        Ok(<S::Real as num_traits::NumCast>::from(s.sqrt()).unwrap())
    }
    fn axpby(&mut self, a: S, x: &Self, b: S) -> Result<(), Self::Error> {
        if b == S::zero() {
            for (s, x) in self.0.iter_mut().zip(&x.0) {
                *s = a * *x;
            }
        } else {
            for (s, x) in self.0.iter_mut().zip(&x.0) {
                *s = a * *x + b * *s;
            }
        }
        Ok(())
    }
}

/// SplitMix64 mapped exactly to `k·2⁻⁵² - 1 ∈ [-1, 1)`; identical to the
/// generator in `fixtures/krylovkit/generate.jl`.
pub struct SplitMix(pub u64);
impl SplitMix {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn unif(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * 2f64.powi(-52) - 1.0
    }
    pub fn scalar(&mut self, complex: bool) -> Complex64 {
        let re = self.unif();
        let im = if complex { self.unif() } else { 0.0 };
        Complex64::new(re, im)
    }
}

/// Random Hermitian matrix (real symmetric when `!complex`), filled in the
/// same order as `generate.jl`.
pub fn hermitian(g: &mut SplitMix, n: usize, complex: bool) -> DMatrix<Complex64> {
    let mut a = DMatrix::zeros(n, n);
    for j in 0..n {
        for i in 0..=j {
            if i == j || !complex {
                let x = Complex64::new(g.unif(), 0.0);
                a[(i, j)] = x;
                a[(j, i)] = x;
            } else {
                let x = g.scalar(true);
                a[(i, j)] = x;
                a[(j, i)] = x.conj();
            }
        }
    }
    a
}

/// KrylovKit's `outliers` (see `fixtures/krylovkit/arnoldi.jl`).
pub fn outliers(g: &mut SplitMix, n: usize, complex: bool) -> DMatrix<Complex64> {
    let mut a = DMatrix::zeros(n, n);
    let s = (n as f64).sqrt();
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = g.scalar(complex) / s;
        }
    }
    let c = Complex64::new;
    if complex {
        a[(0, 0)] += c(0.0, 3.0);
        a[(1, 1)] += c(-2.5, 0.0);
        a[(2, 2)] += c(2.0, 1.0);
        a[(3, 3)] += c(-1.5, -1.5);
    } else {
        a[(0, 0)] += c(3.0, 0.0);
        a[(1, 1)] += c(-2.5, 0.0);
        a[(2, 2)] += c(2.0, 0.0);
        a[(3, 3)] += c(1.5, 0.0);
        a[(4, 4)] += c(1.5, 0.0);
        a[(3, 4)] += c(1.2, 0.0);
        a[(4, 3)] -= c(1.2, 0.0);
    }
    a
}

pub fn random_vec(g: &mut SplitMix, n: usize, complex: bool) -> Vec<Complex64> {
    (0..n).map(|_| g.scalar(complex)).collect()
}

/// Random unitary (orthogonal when `!complex`) from the QR of a random matrix.
pub fn unitary(g: &mut SplitMix, n: usize, complex: bool) -> DMatrix<Complex64> {
    let m = DMatrix::from_fn(n, n, |_, _| g.scalar(complex));
    m.qr().q()
}

/// `Q·diag(eigs)·Qᴴ`.
pub fn with_spectrum(g: &mut SplitMix, eigs: &[f64], complex: bool) -> DMatrix<Complex64> {
    let n = eigs.len();
    let q = unitary(g, n, complex);
    let d = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
        n,
        eigs.iter().map(|&e| Complex64::new(e, 0.0)),
    ));
    let a = &q * d * q.adjoint();
    (&a + a.adjoint()) * Complex64::new(0.5, 0.0)
}

pub fn to_dvec<S: TestScalar>(x: &[Complex64]) -> DVec<S> {
    DVec(x.iter().map(|&z| S::from_c64(z)).collect())
}

pub fn to_c64<S: TestScalar>(x: &DVec<S>) -> Vec<Complex64> {
    x.0.iter().map(|&s| s.to_c64()).collect()
}

/// Dense operator in the test scalar's arithmetic, with an application count.
pub struct DenseOp<S> {
    n: usize,
    a: Vec<S>, // column-major
    pub applications: usize,
}

impl<S: TestScalar> DenseOp<S> {
    pub fn new(a: &DMatrix<Complex64>) -> Self {
        Self {
            n: a.nrows(),
            a: a.iter().map(|&z| S::from_c64(z)).collect(),
            applications: 0,
        }
    }
}

impl<S: TestScalar> krylov::LinearMap<DVec<S>> for DenseOp<S> {
    fn apply_into(&mut self, x: &DVec<S>, y: &mut DVec<S>) -> Result<(), std::convert::Infallible> {
        self.applications += 1;
        y.0.fill(S::zero());
        for (j, &xj) in x.0.iter().enumerate() {
            for (yi, &aij) in y.0.iter_mut().zip(&self.a[j * self.n..(j + 1) * self.n]) {
                *yi += aij * xj;
            }
        }
        Ok(())
    }
}

/// Ascending eigenvalues of a Hermitian matrix.
pub fn dense_eigenvalues(a: &DMatrix<Complex64>) -> Vec<f64> {
    let mut v: Vec<f64> = a
        .clone()
        .symmetric_eigen()
        .eigenvalues
        .iter()
        .copied()
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

/// `exp(t·A)·x` by nalgebra's dense matrix exponential.
pub fn dense_expv(a: &DMatrix<Complex64>, t: Complex64, x: &[Complex64]) -> Vec<Complex64> {
    let e = (a * t).exp();
    let y = e * nalgebra::DVector::from_column_slice(x);
    y.iter().copied().collect()
}

pub fn norm(x: &[Complex64]) -> f64 {
    x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

pub fn diff_norm(x: &[Complex64], y: &[Complex64]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(a, b)| (a - b).norm_sqr())
        .sum::<f64>()
        .sqrt()
}

/// `‖A·x - λ·x‖` in f64.
pub fn residual(a: &DMatrix<Complex64>, lambda: f64, x: &[Complex64]) -> f64 {
    let xv = nalgebra::DVector::from_column_slice(x);
    let r = a * &xv - xv * Complex64::new(lambda, 0.0);
    r.norm()
}

/// `max |⟨vᵢ, vⱼ⟩ - δᵢⱼ|`.
pub fn orthogonality_error<S: TestScalar>(vs: &[DVec<S>]) -> f64 {
    let mut e: f64 = 0.0;
    for (i, a) in vs.iter().enumerate() {
        for (j, b) in vs.iter().enumerate() {
            let ip: Complex64 =
                a.0.iter()
                    .zip(&b.0)
                    .map(|(x, y)| x.to_c64().conj() * y.to_c64())
                    .sum();
            let target = if i == j { 1.0 } else { 0.0 };
            e = e.max((ip - target).norm());
        }
    }
    e
}
