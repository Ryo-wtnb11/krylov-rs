//! Scalar types and the small column-major matrix reference.

use core::fmt::Debug;
use core::ops::Neg;

use num_complex::Complex;
use num_traits::{Float, NumAssign};

/// A field scalar: `f32`, `f64`, `Complex32` or `Complex64`.
///
/// The trait is sealed: krylov implements it for exactly these four types,
/// and the arithmetic helpers it needs internally are not part of the public
/// API. Absolute values inside krylov use only `sqrt` (no platform `hypot`).
///
/// ```compile_fail
/// #[derive(Clone, Copy, Debug, PartialEq)]
/// struct Mine;
/// impl krylov::Scalar for Mine { type Real = f64; } // sealed: rejected
/// ```
pub trait Scalar:
    sealed::Ops<<Self as Scalar>::Real>
    + Copy
    + PartialEq
    + Debug
    + NumAssign
    + Neg<Output = Self>
    + 'static
{
    /// The underlying real type (`f32` or `f64`).
    type Real: RealScalar;
}

/// A real scalar: `f32` or `f64`. Sealed like [`Scalar`].
pub trait RealScalar: Scalar<Real = Self> + Float + sealed::RealOps {}

pub(crate) mod sealed {
    use num_complex::Complex;
    use num_traits::Float;

    use crate::dense::Work;

    /// Internal scalar helpers; unnameable outside the crate, which also
    /// seals [`Scalar`](super::Scalar).
    pub trait Ops<R>: Sized {
        fn from_real(re: R) -> Self;
        fn re(self) -> R;
        fn im(self) -> R;
        fn conj(self) -> Self;
        fn scalar_exp(self) -> Self;
        /// `|self|`, using only `sqrt`.
        fn modulus(self) -> R;
        fn all_finite(self) -> bool;
        /// `self·r` component-wise, so that `±∞ + 0i` stays `±∞ + 0i` (a full
        /// complex product with `r + 0i` would make `0·∞ = NaN`).
        fn mul_real(self, r: R) -> Self;
        /// Eigendecomposition of the Hermitian `n × n` matrix `a` (upper
        /// triangle read, destroyed): ascending `eig`, eigenvectors in `z`.
        /// LAPACK `xSYEVR` for real, `xHEEVR` for complex scalars.
        fn heig(
            a: &mut [Self],
            n: usize,
            eig: &mut [R],
            z: &mut [Self],
            w: &mut Work<Self, R>,
        ) -> Result<(), ()>;
        /// `xGEBAL('B')` on the `n × n` matrix `a` (balanced in place):
        /// returns the 0-based `(ilo, ihi)`; `scale` (`n`) holds the scaling
        /// factors inside `ilo..=ihi` and the 1-based permutation indices
        /// outside it, as LAPACK stores them.
        fn gebal(a: &mut [Self], n: usize, scale: &mut [R]) -> Result<(usize, usize), ()>;
        /// `xGESV`: `b ← A⁻¹·b` for the `n × n` `a` (overwritten by its LU
        /// factors) and the `nrhs` columns of `b`; `ipiv` holds `n` entries.
        /// `Err(())` also for an exactly singular `a` (`info > 0`).
        fn gesv(
            a: &mut [Self],
            b: &mut [Self],
            n: usize,
            nrhs: usize,
            ipiv: &mut [i32],
        ) -> Result<(), ()>;
    }

    /// Internal real helpers; seals [`RealScalar`](super::RealScalar).
    pub trait RealOps: Float {
        /// `sqrt(a² + b²)` without undue overflow, using only `sqrt`.
        fn hypot2(self, b: Self) -> Self {
            let (a, b) = (self.abs(), b.abs());
            let (big, small) = if a >= b { (a, b) } else { (b, a) };
            if big == Self::zero() || !big.is_finite() {
                return big + small;
            }
            let r = small / big;
            big * (Self::one() + r * r).sqrt()
        }
        fn from_f64(x: f64) -> Self {
            <Self as num_traits::NumCast>::from(x).unwrap_or_else(Self::nan)
        }
        fn from_usize(n: usize) -> Self {
            <Self as num_traits::NumCast>::from(n).unwrap_or_else(Self::nan)
        }

        // LAPACK hooks, implemented in `dense/lapack.rs`. `Err(())` is a
        // nonzero LAPACK `info`.

        /// `xSTEGR`: eigenvalues of the symmetric tridiagonal matrix with
        /// diagonal `d` and off-diagonal `e[..n-1]` (`e.len() >= n`, destroyed)
        /// into `d` (ascending), eigenvectors into `z` (`n × n`).
        fn stegr(
            d: &mut [Self],
            e: &mut [Self],
            z: &mut [Self],
            w: &mut Work<Self, Self>,
        ) -> Result<(), ()>;
        /// `xBDSQR` on the `n × n` lower (`lower`) or upper bidiagonal matrix
        /// with diagonal `d`, off-diagonal `e[..n-1]`: `B = U·diag(d)·VT` with
        /// `d` descending, `U ← U·Q` and `VT ← Pᵀ·VT` (`n × n` each).
        fn bdsqr(
            lower: bool,
            d: &mut [Self],
            e: &mut [Self],
            vt: &mut [Self],
            u: &mut [Self],
            w: &mut Work<Self, Self>,
        ) -> Result<(), ()>;
        /// `xSYEVR` (see [`Ops::heig`]).
        fn syevr(
            a: &mut [Self],
            n: usize,
            eig: &mut [Self],
            z: &mut [Self],
            w: &mut Work<Self, Self>,
        ) -> Result<(), ()>;
        /// `xHEEVR` (see [`Ops::heig`]).
        fn heevr(
            a: &mut [Complex<Self>],
            n: usize,
            eig: &mut [Self],
            z: &mut [Complex<Self>],
            w: &mut Work<Complex<Self>, Self>,
        ) -> Result<(), ()>;
        /// `xGEBAL('B')` for real matrices (see [`Ops::gebal`]).
        fn gebal_r(a: &mut [Self], n: usize, scale: &mut [Self]) -> Result<(usize, usize), ()>;
        /// `xGEBAL('B')` for complex matrices (see [`Ops::gebal`]).
        fn gebal_c(
            a: &mut [Complex<Self>],
            n: usize,
            scale: &mut [Self],
        ) -> Result<(usize, usize), ()>;
        /// `xGESV` for real matrices (see [`Ops::gesv`]).
        fn gesv_r(
            a: &mut [Self],
            b: &mut [Self],
            n: usize,
            nrhs: usize,
            ipiv: &mut [i32],
        ) -> Result<(), ()>;
        /// `xGESV` for complex matrices (see [`Ops::gesv`]).
        fn gesv_c(
            a: &mut [Complex<Self>],
            b: &mut [Complex<Self>],
            n: usize,
            nrhs: usize,
            ipiv: &mut [i32],
        ) -> Result<(), ()>;
    }
}

macro_rules! real_scalar {
    ($t:ty) => {
        impl sealed::Ops<$t> for $t {
            fn from_real(re: $t) -> $t {
                re
            }
            fn re(self) -> $t {
                self
            }
            fn im(self) -> $t {
                0.0
            }
            fn conj(self) -> $t {
                self
            }
            fn scalar_exp(self) -> $t {
                Float::exp(self)
            }
            fn modulus(self) -> $t {
                Float::abs(self)
            }
            fn all_finite(self) -> bool {
                Float::is_finite(self)
            }
            fn mul_real(self, r: $t) -> $t {
                self * r
            }
            fn heig(
                a: &mut [Self],
                n: usize,
                eig: &mut [$t],
                z: &mut [Self],
                w: &mut crate::dense::Work<Self, $t>,
            ) -> Result<(), ()> {
                <$t as sealed::RealOps>::syevr(a, n, eig, z, w)
            }
            fn gebal(a: &mut [Self], n: usize, scale: &mut [$t]) -> Result<(usize, usize), ()> {
                <$t as sealed::RealOps>::gebal_r(a, n, scale)
            }
            fn gesv(
                a: &mut [Self],
                b: &mut [Self],
                n: usize,
                nrhs: usize,
                ipiv: &mut [i32],
            ) -> Result<(), ()> {
                <$t as sealed::RealOps>::gesv_r(a, b, n, nrhs, ipiv)
            }
        }
        impl Scalar for $t {
            type Real = $t;
        }
        impl RealScalar for $t {}

        impl sealed::Ops<$t> for Complex<$t> {
            fn from_real(re: $t) -> Self {
                Complex::new(re, 0.0)
            }
            fn re(self) -> $t {
                self.re
            }
            fn im(self) -> $t {
                self.im
            }
            fn conj(self) -> Self {
                Complex::conj(&self)
            }
            fn scalar_exp(self) -> Self {
                Complex::exp(self)
            }
            fn modulus(self) -> $t {
                sealed::RealOps::hypot2(self.re, self.im)
            }
            fn all_finite(self) -> bool {
                self.re.is_finite() && self.im.is_finite()
            }
            fn mul_real(self, r: $t) -> Self {
                self * r // num-complex: Complex<T> * T scales each component
            }
            fn heig(
                a: &mut [Self],
                n: usize,
                eig: &mut [$t],
                z: &mut [Self],
                w: &mut crate::dense::Work<Self, $t>,
            ) -> Result<(), ()> {
                <$t as sealed::RealOps>::heevr(a, n, eig, z, w)
            }
            fn gebal(a: &mut [Self], n: usize, scale: &mut [$t]) -> Result<(usize, usize), ()> {
                <$t as sealed::RealOps>::gebal_c(a, n, scale)
            }
            fn gesv(
                a: &mut [Self],
                b: &mut [Self],
                n: usize,
                nrhs: usize,
                ipiv: &mut [i32],
            ) -> Result<(), ()> {
                <$t as sealed::RealOps>::gesv_c(a, b, n, nrhs, ipiv)
            }
        }
        impl Scalar for Complex<$t> {
            type Real = $t;
        }
    };
}
real_scalar!(f32);
real_scalar!(f64);

/// Real type of a vector's scalar.
pub type Real<V> = <<V as crate::Vector>::Scalar as Scalar>::Real;

/// A borrowed column-major matrix of shape `nrows × ncols`, leading dimension
/// `nrows`.
#[derive(Clone, Copy, Debug)]
pub struct MatRef<'a, S> {
    data: &'a [S],
    nrows: usize,
    ncols: usize,
}

impl<'a, S: Copy> MatRef<'a, S> {
    /// Wraps `data`, which holds the columns one after another.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != nrows * ncols`.
    pub fn new(data: &'a [S], nrows: usize, ncols: usize) -> Self {
        assert_eq!(data.len(), nrows * ncols, "MatRef: data length mismatch");
        Self { data, nrows, ncols }
    }
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    /// Entry `(i, j)`.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn get(&self, i: usize, j: usize) -> S {
        assert!(
            i < self.nrows && j < self.ncols,
            "MatRef: index out of bounds"
        );
        self.data[i + j * self.nrows]
    }
    /// Column `j` as a slice.
    ///
    /// # Panics
    ///
    /// Panics if `j >= ncols`.
    pub fn col(&self, j: usize) -> &'a [S] {
        assert!(j < self.ncols, "MatRef: column out of bounds");
        &self.data[j * self.nrows..(j + 1) * self.nrows]
    }
}
