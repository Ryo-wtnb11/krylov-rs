//! Scalar types and the small column-major matrix reference.

use core::fmt::Debug;
use core::ops::Neg;

use num_complex::Complex;
use num_traits::{Float, NumAssign};

/// A field scalar: `f32`, `f64`, `Complex32` or `Complex64`.
///
/// The trait is implemented by krylov for exactly these four types. Absolute
/// values are computed with [`RealScalar::hypot2`] (square root only), so
/// branches that depend on them do not depend on a platform `hypot`.
pub trait Scalar: Copy + PartialEq + Debug + NumAssign + Neg<Output = Self> + 'static {
    /// The underlying real type (`f32` or `f64`).
    type Real: RealScalar;

    /// Embeds a real number.
    fn from_real(re: Self::Real) -> Self;
    /// Real part.
    fn re(self) -> Self::Real;
    /// Imaginary part (zero for real scalars).
    fn im(self) -> Self::Real;
    /// Complex conjugate (identity for real scalars).
    fn conj(self) -> Self;
    /// Exponential.
    fn scalar_exp(self) -> Self;

    /// Modulus `|self|`.
    fn modulus(self) -> Self::Real {
        self.re().hypot2(self.im())
    }
    /// `true` when both parts are finite.
    fn all_finite(self) -> bool {
        self.re().is_finite() && self.im().is_finite()
    }
}

/// A real scalar: `f32` or `f64`.
pub trait RealScalar: Scalar<Real = Self> + Float {
    /// `sqrt(a² + b²)` without undue overflow, using only `sqrt`.
    fn hypot2(self, b: Self) -> Self {
        let (a, b) = (Float::abs(self), Float::abs(b));
        let (big, small) = if a >= b { (a, b) } else { (b, a) };
        if big == Self::zero() || !big.is_finite() {
            return big + small;
        }
        let r = small / big;
        big * (Self::one() + r * r).sqrt()
    }

    /// Converts an `f64` constant.
    fn from_f64(x: f64) -> Self {
        <Self as num_traits::NumCast>::from(x).unwrap_or_else(Self::nan)
    }

    /// Converts a count.
    fn from_usize(n: usize) -> Self {
        <Self as num_traits::NumCast>::from(n).unwrap_or_else(Self::nan)
    }
}

macro_rules! real_scalar {
    ($t:ty) => {
        impl Scalar for $t {
            type Real = $t;
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
        }
        impl RealScalar for $t {}

        impl Scalar for Complex<$t> {
            type Real = $t;
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
