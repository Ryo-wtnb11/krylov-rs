//! Givens rotations and the upper-triangular solve of GMRES.

use num_traits::{One, Zero};

use crate::scalar::Scalar;
use crate::scalar::sealed::RealOps as _;

/// Rotation `G` acting on entries `(i1, i2)`:
/// `G = [c s; -conj(s) c]` with real `c ≥ 0` (LinearAlgebra's `Givens`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Givens<S: Scalar> {
    i1: usize,
    i2: usize,
    c: S::Real,
    s: S,
}

impl<S: Scalar> Givens<S> {
    /// The identity on `(0, 1)`, a placeholder for unused slots.
    pub(crate) fn identity() -> Self {
        Self {
            i1: 0,
            i2: 1,
            c: S::Real::one(),
            s: S::zero(),
        }
    }

    /// `x ← G·x`.
    pub(crate) fn lmul(&self, x: &mut [S]) {
        let (a, b) = (x[self.i1], x[self.i2]);
        x[self.i1] = S::from_real(self.c) * a + self.s * b;
        x[self.i2] = -(self.s.conj() * a) + S::from_real(self.c) * b;
    }

    /// `x ← Gᴴ·x`.
    pub(crate) fn lmul_adjoint(&self, x: &mut [S]) {
        let (a, b) = (x[self.i1], x[self.i2]);
        x[self.i1] = S::from_real(self.c) * a - self.s * b;
        x[self.i2] = self.s.conj() * a + S::from_real(self.c) * b;
    }
}

/// The rotation on `(i1, i2)` with `G·[f; g] = [r; 0]`, and `r`.
///
/// Same convention as LinearAlgebra's `givens(f, g, i1, i2)` (`c ≥ 0`), but
/// with moduli from `sqrt` only, so the result is libm-independent.
pub(crate) fn givens<S: Scalar>(f: S, g: S, i1: usize, i2: usize) -> (Givens<S>, S) {
    let (af, ag) = (f.modulus(), g.modulus());
    let zero = S::Real::zero();
    // Divide by a real through its reciprocal: dividing by a complex-typed
    // real squares it and overflows near the range limits.
    let inv = |x: S::Real| S::from_real(S::Real::one() / x);
    let (c, s, r) = if ag == zero {
        (S::Real::one(), S::zero(), f)
    } else if af == zero {
        (zero, g.conj() * inv(ag), S::from_real(ag))
    } else {
        let h = af.hypot2(ag);
        let phase = f * inv(af);
        (af / h, phase * (g.conj() * inv(h)), phase * S::from_real(h))
    };
    (Givens { i1, i2, c, s }, r)
}

/// Solves `R[..n, ..n]·y = y` in place for upper-triangular `R` (column-major,
/// leading dimension `ld`); `Err` on an exactly zero pivot.
///
/// Port of KrylovKit v0.10.4 `src/dense/linalg.jl:ldiv!(::UpperTriangular)`.
pub(crate) fn upper_solve<S: Scalar>(r: &[S], ld: usize, y: &mut [S], n: usize) -> Result<(), ()> {
    for j in (0..n).rev() {
        let d = r[j + j * ld];
        if d == S::zero() {
            return Err(());
        }
        let yj = y[j] / d;
        y[j] = yj;
        for i in 0..j {
            y[i] -= r[i + j * ld] * yj;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    #[test]
    fn givens_annihilates_and_is_unitary() {
        let cases = [
            (Complex64::new(0.3, -1.2), Complex64::new(-2.0, 0.7)),
            (Complex64::new(0.0, 0.0), Complex64::new(1.5, -0.5)),
            (Complex64::new(-4.0, 0.0), Complex64::new(0.0, 0.0)),
            (Complex64::new(1e-200, 0.0), Complex64::new(1e200, 1e200)),
        ];
        for (f, g) in cases {
            let (rot, r) = givens(f, g, 0, 1);
            let mut x = [f, g];
            rot.lmul(&mut x);
            let scale = f.norm().max(g.norm());
            assert!((x[0] - r).norm() <= 1e-15 * scale, "{f} {g}");
            assert!(x[1].norm() <= 1e-15 * scale, "{f} {g}");
            assert!((r.norm() - f.norm().hypot(g.norm())).abs() <= 1e-15 * scale);
            rot.lmul_adjoint(&mut x);
            assert!((x[0] - f).norm() <= 1e-15 * scale && (x[1] - g).norm() <= 1e-15 * scale);
        }
        let (rot, r) = givens(-3.0f64, 4.0, 2, 0);
        let mut x = [4.0, 9.0, -3.0];
        rot.lmul(&mut x);
        assert!((x[2] - r).abs() < 1e-15 && x[0].abs() < 1e-15 && x[1] == 9.0);
    }

    #[test]
    fn upper_solve_inverts_and_rejects_zero_pivot() {
        // R = [2 1 -1; 0 3 4; 0 0 -5] column-major, ld = 4 (padded).
        let r = [2.0, 0.0, 0.0, 9.9, 1.0, 3.0, 0.0, 9.9, -1.0, 4.0, -5.0, 9.9];
        let x = [1.0f64, -2.0, 0.5];
        let mut y: [f64; 4] = [2.0 - 2.0 - 0.5, -6.0 + 2.0, -2.5, 7.0];
        upper_solve(&r, 4, &mut y, 3).unwrap();
        for (a, b) in y.iter().zip(&x) {
            assert!((a - b).abs() < 1e-15);
        }
        assert_eq!(y[3], 7.0);
        let mut z = [1.0f64, 1.0];
        assert!(upper_solve(&[1.0, 0.0, 1.0, 0.0], 2, &mut z, 2).is_err());
    }
}
