//! Schur form of the Arnoldi Rayleigh quotient: LAPACK `xHSEQR`, reordering
//! (`xTREXC`) and eigenvectors (`xTREVC`), driven as KrylovKit v0.10.4
//! `src/dense/linalg.jl` drives them.

use num_complex::Complex;
use num_traits::{Float, One, Zero};

use super::Work;
use crate::scalar::Scalar;
use crate::scalar::sealed::RealOps as _;

type R<S> = <S as Scalar>::Real;

/// Schur form `h ← T = Qᴴ·H·Q`, `z ← z·Q` of the `n × n` upper Hessenberg
/// `h` (column-major) by LAPACK `xHSEQR(job='S', compz='V')`, eigenvalues in
/// `ev` (real: 2×2 blocks in standard form, the `+im` eigenvalue first).
///
/// `Err(())` on non-finite input or a LAPACK failure.
///
/// Unlike KrylovKit's bare `hseqr!` call, `h` is first scaled by a power of
/// four into `[sqrt(SAFMIN)/ULP, its inverse]`, as `xGEEV`/`xGEES` do: LAHQR
/// takes subdiagonal entries `≤ SAFMIN·n/ULP` (about 1e-291 in f64) for
/// zero, so a matrix with entries near 1e-300 would deflate completely and
/// return its diagonal. Powers of four are exact, inputs inside the range are
/// not touched, and `xGEES`'s post-unscale repair keeps `ev` consistent with
/// T when unscaling underflows a 2×2 block.
pub(crate) fn hschur<S: Scalar>(
    h: &mut [S],
    z: &mut [S],
    n: usize,
    ev: &mut [Complex<R<S>>],
    w: &mut Work<S, R<S>>,
) -> Result<(), ()> {
    if !h[..n * n].iter().all(|x| x.all_finite()) {
        return Err(());
    }
    let s = range_scale::<R<S>>(h[..n * n].iter().map(|x| x.re().abs().max(x.im().abs())));
    let one = R::<S>::one();
    if s != one {
        let sc = S::from_real(s);
        h[..n * n].iter_mut().for_each(|x| *x *= sc);
    }
    S::hseqr_raw(h, z, n, ev, w)?;
    if s != one {
        let inv = one / s;
        let ic = S::from_real(inv);
        h[..n * n].iter_mut().for_each(|x| *x *= ic);
        if S::COMPLEX {
            for (i, e) in ev[..n].iter_mut().enumerate() {
                *e = Complex::new(h[i + i * n].re(), h[i + i * n].im());
            }
        } else {
            if s > one {
                repair_underflowed_blocks(h, z, n, ev);
            }
            for (i, e) in ev[..n].iter_mut().enumerate() {
                *e = Complex::new(h[i + i * n].re(), e.im * inv);
            }
        }
    }
    Ok(())
}

/// Power of four that brings `max |h_ij|` into `[SMLNUM, BIGNUM]` with
/// `SMLNUM = sqrt(SAFMIN)/ULP` (`xGEEV`); 1 when already inside.
fn range_scale<T: crate::scalar::RealScalar>(abs: impl Iterator<Item = T>) -> T {
    let one = T::one();
    let four = (one + one) * (one + one);
    let anrm = abs.fold(T::zero(), T::max);
    let smlnum = T::min_positive_value().sqrt() / T::epsilon();
    let bignum = one / smlnum;
    let mut s = one;
    if anrm > T::zero() {
        while anrm * s < smlnum {
            s *= four;
        }
        while anrm * s > bignum {
            s /= four;
        }
    }
    s
}

/// `xGEES` post-unscale repair (real only): a 2×2 block whose off-diagonal
/// entry underflowed holds two real eigenvalues. It is made triangular
/// (swapping the basis pair when only the upper entry vanished) and its
/// eigenvalues are marked real.
fn repair_underflowed_blocks<S: Scalar>(
    h: &mut [S],
    z: &mut [S],
    n: usize,
    ev: &mut [Complex<R<S>>],
) {
    let zero = R::<S>::zero();
    let mut i = 0;
    while i + 1 < n {
        if ev[i].im == zero {
            i += 1;
            continue;
        }
        let (lower, upper) = (i + 1 + i * n, i + (i + 1) * n);
        if h[lower] == S::zero() {
            ev[i].im = zero;
            ev[i + 1].im = zero;
        } else if h[upper] == S::zero() {
            ev[i].im = zero;
            ev[i + 1].im = zero;
            for r in 0..i {
                h.swap(r + i * n, r + (i + 1) * n);
            }
            for col in i + 2..n {
                h.swap(i + col * n, i + 1 + col * n);
            }
            for r in 0..n {
                z.swap(r + i * n, r + (i + 1) * n);
            }
            h[upper] = h[lower];
            h[lower] = S::zero();
        }
        i += 2;
    }
}

/// Reorders the Schur form so that the eigenvalues at the 0-based positions
/// `order[0], order[1], …` come first, moving one block at a time with
/// `xTREXC` (KrylovKit `permuteschur!`). A real 2×2 block moves as a whole;
/// `order` must list its two positions consecutively.
///
/// `Err(())` when LAPACK rejects a swap or `order` would split a 2×2 block.
pub(crate) fn permute_schur<S: Scalar>(
    t: &mut [S],
    q: &mut [S],
    n: usize,
    order: &[usize],
    p: &mut Vec<usize>,
    w: &mut Work<S, R<S>>,
) -> Result<(), ()> {
    p.clear();
    p.extend_from_slice(order);
    let mut i = 0;
    while i < p.len() {
        let ifirst = p[i];
        let single = S::COMPLEX || ifirst + 1 == n || t[ifirst + 1 + ifirst * n] == S::zero();
        if !single && p.get(i + 1) != Some(&(ifirst + 1)) {
            return Err(());
        }
        S::trexc(t, q, n, ifirst, i, w)?;
        let (width, from) = if single { (1, i + 1) } else { (2, i + 2) };
        let pi = p[i];
        for pk in &mut p[from..] {
            if *pk < pi {
                *pk += width;
            }
        }
        i += width;
    }
    Ok(())
}

/// Eigenvalues of the leading `n × n` block of the Schur form `t` (leading
/// dimension `ld`): the diagonal, and for a real 2×2 block the half-trace
/// formula of KrylovKit `schur2eigvals`.
pub(crate) fn schur_eigvals<S: Scalar>(t: &[S], ld: usize, n: usize, out: &mut Vec<Complex<R<S>>>) {
    out.clear();
    let at = |i: usize, j: usize| t[i + j * ld].re();
    let two = R::<S>::one() + R::<S>::one();
    for i in 0..n {
        let v = if S::COMPLEX {
            Complex::new(t[i + i * ld].re(), t[i + i * ld].im())
        } else if i + 1 < n && at(i + 1, i) != R::<S>::zero() {
            let halftr = (at(i, i) + at(i + 1, i + 1)) / two;
            let diff = (at(i, i) - at(i + 1, i + 1)) / two;
            let d = diff * diff + at(i, i + 1) * at(i + 1, i);
            Complex::new(halftr, (-d).sqrt())
        } else if i > 0 && at(i, i - 1) != R::<S>::zero() {
            let halftr = (at(i, i) + at(i - 1, i - 1)) / two;
            let diff = -(at(i, i) - at(i - 1, i - 1)) / two;
            let d = diff * diff + at(i, i - 1) * at(i - 1, i);
            Complex::new(halftr, -(-d).sqrt())
        } else {
            Complex::new(at(i, i), R::<S>::zero())
        };
        out.push(v);
    }
}

/// Right eigenvectors of the `n × n` upper triangular Schur factor `t`
/// (complex, or real with no 2×2 blocks) into the columns of `vr`, each
/// normalized to unit 2-norm (KrylovKit `schur2eigvecs` /
/// `schur2realeigvecs`).
pub(crate) fn schur_eigvecs<S: Scalar>(
    t: &mut [S],
    n: usize,
    vr: &mut [S],
    w: &mut Work<S, R<S>>,
) -> Result<(), ()> {
    S::trevc(t, n, vr, w)?;
    for col in vr[..n * n].chunks_exact_mut(n.max(1)) {
        let nrm = col
            .iter()
            .fold(R::<S>::zero(), |acc, x| acc.hypot2(x.modulus()));
        if nrm > R::<S>::zero() {
            let inv = S::from_real(R::<S>::one() / nrm);
            col.iter_mut().for_each(|x| *x *= inv);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::sealed::Ops;
    use nalgebra::DMatrix;
    use num_complex::{Complex32, Complex64};

    fn real_schur(h: &mut [f64], z: &mut [f64], n: usize, w: &mut [Complex64]) -> Result<(), ()> {
        hschur(h, z, n, w, &mut Work::default())
    }

    fn complex_schur(
        h: &mut [Complex64],
        z: &mut [Complex64],
        n: usize,
        w: &mut [Complex64],
    ) -> Result<(), ()> {
        hschur(h, z, n, w, &mut Work::default())
    }

    fn lcg(s: &mut u64) -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    fn c(re: f64, im: f64) -> Complex64 {
        Complex64::new(re, im)
    }

    /// Runs the dispatched Schur kernel on the Hessenberg form of `a`, with
    /// `Z` started at the Hessenberg `Q`, so that `a = Z·T·Zᴴ` on success.
    type SchurOut<S> = (DMatrix<S>, DMatrix<S>, Vec<Complex<<S as Scalar>::Real>>);

    fn schur_of<S>(a: &DMatrix<S>) -> Result<SchurOut<S>, ()>
    where
        S: Scalar + nalgebra::ComplexField,
    {
        let n = a.nrows();
        let (q, hm) = a.clone().hessenberg().unpack();
        let mut h: Vec<S> = hm.as_slice().to_vec();
        let mut z: Vec<S> = q.as_slice().to_vec();
        let zero = <S::Real as num_traits::Zero>::zero();
        let mut w = vec![Complex::new(zero, zero); n];
        hschur(&mut h, &mut z, n, &mut w, &mut Work::default())?;
        Ok((
            DMatrix::from_column_slice(n, n, &h),
            DMatrix::from_column_slice(n, n, &z),
            w,
        ))
    }

    /// Structure and backward-error checks shared by every case; `scale` is
    /// the norm the residual is measured against.
    fn check_real(a: &DMatrix<f64>, t: &DMatrix<f64>, z: &DMatrix<f64>, w: &[Complex64], tol: f64) {
        check_real_residual(a, t, z, tol);
        check_real_structure(t, w);
    }

    /// Largest entry modulus; dividing by it first keeps nalgebra's
    /// Frobenius norm (which squares entries) finite for 1e300 inputs.
    fn amax<S: nalgebra::ComplexField<RealField = f64>>(m: &DMatrix<S>) -> f64 {
        m.iter()
            .map(|x| x.clone().modulus())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE)
    }

    fn check_real_residual(a: &DMatrix<f64>, t: &DMatrix<f64>, z: &DMatrix<f64>, tol: f64) {
        let n = a.nrows();
        let m = amax(a);
        let (a, t) = (a / m, t / m);
        let scale = a.norm();
        let back = (z * t * z.transpose() - &a).norm();
        assert!(back <= tol * scale, "backward error {back:e} vs {scale:e}");
        assert!((z.transpose() * z - DMatrix::identity(n, n)).norm() <= tol);
    }

    fn check_real_structure(t: &DMatrix<f64>, w: &[Complex64]) {
        let n = t.nrows();
        let mut i = 0;
        while i < n {
            for r in i + 2..n {
                assert_eq!(t[(r, i)], 0.0, "below subdiagonal");
            }
            if i + 1 < n && t[(i + 1, i)] != 0.0 {
                // Standardized 2×2 block holding a complex pair.
                assert!(i + 2 >= n || t[(i + 2, i + 1)] == 0.0, "adjacent blocks");
                assert_eq!(t[(i, i)], t[(i + 1, i + 1)]);
                assert_ne!(
                    t[(i, i + 1)].is_sign_negative(),
                    t[(i + 1, i)].is_sign_negative()
                );
                // The imaginary part is rounded differently by `dlanv2` and
                // `schur_eigvals`, and across LAPACK builds; the real part and
                // the conjugate pairing are exact in both producers.
                let im = t[(i, i + 1)].abs().sqrt() * t[(i + 1, i)].abs().sqrt();
                assert_eq!(w[i].re, t[(i, i)]);
                assert!(
                    (w[i].im - im).abs() <= 4.0 * f64::EPSILON * im,
                    "{} vs {im}",
                    w[i].im
                );
                assert_eq!(w[i + 1], w[i].conj());
                i += 2;
            } else {
                assert_eq!(w[i], c(t[(i, i)], 0.0));
                i += 1;
            }
        }
    }

    fn check_complex(
        a: &DMatrix<Complex64>,
        t: &DMatrix<Complex64>,
        z: &DMatrix<Complex64>,
        w: &[Complex64],
        tol: f64,
    ) {
        check_complex_residual(a, t, z, tol);
        check_complex_structure(t, w);
    }

    fn check_complex_residual(
        a: &DMatrix<Complex64>,
        t: &DMatrix<Complex64>,
        z: &DMatrix<Complex64>,
        tol: f64,
    ) {
        let n = a.nrows();
        // Divide by the real amax: nalgebra's complex `/` forms |m|².
        let m = amax(a);
        let (a, t) = (a.map(|x| x / m), t.map(|x| x / m));
        let scale = a.norm();
        let back = (z * t * z.adjoint() - &a).norm();
        assert!(back <= tol * scale, "backward error {back:e} vs {scale:e}");
        assert!((z.adjoint() * z - DMatrix::identity(n, n)).norm() <= tol);
    }

    fn check_complex_structure(t: &DMatrix<Complex64>, w: &[Complex64]) {
        let n = t.nrows();
        for j in 0..n {
            for r in j + 1..n {
                assert_eq!(t[(r, j)], c(0.0, 0.0), "not triangular");
            }
            assert_eq!(w[j], t[(j, j)]);
        }
    }

    /// Greedy nearest matching of two eigenvalue multisets.
    fn max_mismatch(got: &[Complex64], want: &[Complex64]) -> f64 {
        let mut used = vec![false; want.len()];
        let mut worst: f64 = 0.0;
        for g in got {
            let (j, d) = want
                .iter()
                .enumerate()
                .filter(|(j, _)| !used[*j])
                .map(|(j, x)| (j, (g - x).norm()))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            used[j] = true;
            worst = worst.max(d);
        }
        worst
    }

    fn random(n: usize, seed: u64) -> DMatrix<f64> {
        let mut s = seed;
        DMatrix::from_fn(n, n, |_, _| lcg(&mut s))
    }

    fn random_complex(n: usize, seed: u64) -> DMatrix<Complex64> {
        let mut s = seed;
        DMatrix::from_fn(n, n, |_, _| c(lcg(&mut s), lcg(&mut s)))
    }

    fn orthogonal(n: usize, seed: u64) -> DMatrix<f64> {
        random(n, seed).qr().q()
    }

    const TOL: f64 = 1e-13;

    #[test]
    fn real_random_matches_dense_eigenvalues() {
        for n in [1, 2, 3, 4, 7, 16, 40, 75] {
            for seed in [1, 2] {
                let a = random(n, seed * 100 + n as u64);
                let (t, z, w) = schur_of(&a).unwrap();
                check_real(&a, &t, &z, &w, TOL * n as f64);
                let want: Vec<Complex64> = a.complex_eigenvalues().iter().copied().collect();
                assert!(max_mismatch(&w, &want) <= 1e-10, "n={n}");
            }
        }
    }

    #[test]
    fn complex_random_matches_dense_eigenvalues() {
        for n in [1, 2, 3, 5, 16, 40] {
            let a = random_complex(n, 7 + n as u64);
            let (t, z, w) = schur_of(&a).unwrap();
            check_complex(&a, &t, &z, &w, TOL * n as f64);
            let want: Vec<Complex64> = a.schur().eigenvalues().unwrap().iter().copied().collect();
            assert!(max_mismatch(&w, &want) <= 1e-10, "n={n}");
        }
    }

    /// LAPACK `dchkhs`-style structured inputs, real arithmetic.
    #[test]
    fn real_adversarial_matrices() {
        let n = 8;
        let q = orthogonal(n, 5);
        let mut cases: Vec<(&str, DMatrix<f64>, Vec<Complex64>)> = Vec::new();
        cases.push(("zero", DMatrix::zeros(n, n), vec![c(0.0, 0.0); n]));
        cases.push(("identity", DMatrix::identity(n, n), vec![c(1.0, 0.0); n]));
        // Jordan block: eigenvalue 2 is only determined to ε^(1/n).
        let jordan = DMatrix::from_fn(n, n, |i, j| match j as isize - i as isize {
            0 => 2.0,
            1 => 1.0,
            _ => 0.0,
        });
        cases.push(("jordan", jordan.clone(), vec![c(2.0, 0.0); n]));
        cases.push((
            "jordan-rotated",
            &q * &jordan * q.transpose(),
            vec![c(2.0, 0.0); n],
        ));
        // Nearly defective: Jordan block with a tiny corner coupling.
        let mut near = jordan.clone();
        near[(n - 1, 0)] = 1e-12;
        // (λ-2)⁸ = 1e-12: the eigenvalues lie on the circle |λ-2| = 1e-1.5.
        let r = 1e-12f64.powf(1.0 / n as f64);
        let want: Vec<Complex64> = (0..n)
            .map(|k| {
                c(2.0, 0.0)
                    + Complex64::from_polar(r, 2.0 * std::f64::consts::PI * k as f64 / n as f64)
            })
            .collect();
        cases.push(("nearly-defective", &q * &near * q.transpose(), want));
        // Clustered complex-conjugate pairs: the same rotation block four times.
        let mut blocks = DMatrix::zeros(n, n);
        for b in 0..n / 2 {
            let (i, d) = (2 * b, 1e-9 * b as f64);
            blocks[(i, i)] = 0.5 + d;
            blocks[(i + 1, i + 1)] = 0.5 + d;
            blocks[(i, i + 1)] = 2.0;
            blocks[(i + 1, i)] = -2.0;
        }
        let want: Vec<Complex64> = (0..n / 2)
            .flat_map(|b| {
                let d = 1e-9 * b as f64;
                [c(0.5 + d, 2.0), c(0.5 + d, -2.0)]
            })
            .collect();
        cases.push(("conjugate-clusters", &q * &blocks * q.transpose(), want));
        // Geometrically spaced eigenvalues 1 … 1e-15 (dchkhs type 6-like).
        let geo = DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                10f64.powf(-15.0 * i as f64 / (n - 1) as f64)
            } else {
                0.0
            }
        });
        let want: Vec<Complex64> = (0..n).map(|i| c(geo[(i, i)], 0.0)).collect();
        cases.push(("geometric", &q * &geo * q.transpose(), want));
        // Cyclic permutation: unit-modulus spectrum, standard shifts stagnate.
        let cyc = DMatrix::from_fn(n, n, |i, j| f64::from(u8::from((i + 1) % n == j)));
        let want: Vec<Complex64> = (0..n)
            .map(|k| Complex64::from_polar(1.0, 2.0 * std::f64::consts::PI * k as f64 / n as f64))
            .collect();
        cases.push(("cyclic", cyc, want));
        for (name, a, want) in &cases {
            let (t, z, w) = schur_of(a).unwrap_or_else(|()| panic!("{name}: no convergence"));
            check_real(a, &t, &z, &w, 1e-12);
            let tol = match *name {
                "jordan" | "jordan-rotated" => 1e-1,
                "nearly-defective" => 1e-3,
                _ => 1e-7,
            };
            assert!(max_mismatch(&w, want) <= tol, "{name}: {w:?}");
        }
    }

    /// Extreme magnitudes: uniformly scaled by 1e±300 and graded across
    /// 1e-300…1e290, plus subnormal subdiagonals.
    #[test]
    fn real_extreme_magnitudes() {
        let n = 10;
        // Hessenberg input built directly: a dense reduction of 1e300 entries
        // would overflow in the oracle, not in the kernel under test.
        let mut a = random(n, 11);
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    a[(i, j)] = 0.0;
                }
            }
        }
        let want: Vec<Complex64> = a.complex_eigenvalues().iter().copied().collect();
        for s in [1e300, 1e-300] {
            let a = &a * s;
            let mut h: Vec<f64> = a.as_slice().to_vec();
            let mut z: Vec<f64> = DMatrix::<f64>::identity(n, n).as_slice().to_vec();
            let mut w = vec![c(0.0, 0.0); n];
            real_schur(&mut h, &mut z, n, &mut w).unwrap();
            let (t, zm) = (
                DMatrix::from_column_slice(n, n, &h),
                DMatrix::from_column_slice(n, n, &z),
            );
            // Residual in units of s, so 1e300·1e300 never forms.
            check_real_residual(&a, &t, &zm, 1e-12);
            check_real_structure(&t, &w);
            let scaled: Vec<Complex64> = want.iter().map(|x| x * s).collect();
            assert!(max_mismatch(&w, &scaled) <= 1e-10 * s, "scale {s}");
        }
        let _w = vec![c(0.0, 0.0); n];
        // Graded upper Hessenberg with entries from 1e-300 (top left) to
        // 1e290 (bottom right): backward error and normwise eigenvalue
        // accuracy must hold.
        let mut s = 3;
        let g = DMatrix::from_fn(n, n, |i, j| {
            if i > j + 1 {
                0.0
            } else {
                (1.0 + lcg(&mut s)) * 10f64.powf(-300.0 + 590.0 * (i + j) as f64 / 18.0)
            }
        });
        let mut h: Vec<f64> = g.as_slice().to_vec();
        let mut z: Vec<f64> = DMatrix::<f64>::identity(n, n).as_slice().to_vec();
        let mut w = vec![c(0.0, 0.0); n];
        real_schur(&mut h, &mut z, n, &mut w).unwrap();
        let (t, zm) = (
            DMatrix::from_column_slice(n, n, &h),
            DMatrix::from_column_slice(n, n, &z),
        );
        check_real(&g, &t, &zm, &w, 1e-12);
        let want: Vec<Complex64> = g.complex_eigenvalues().iter().copied().collect();
        assert!(g.norm().is_infinite(), "the scaled norm below is needed");
        assert!(max_mismatch(&w, &want) <= 1e-13 * amax(&g) * (&g / amax(&g)).norm());
        // Subnormal subdiagonal entries deflate without producing NaN.
        let mut sub = random(n, 13);
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    sub[(i, j)] = 0.0;
                }
            }
        }
        sub[(3, 2)] = 1e-310;
        sub[(7, 6)] = -4e-320;
        let mut h: Vec<f64> = sub.as_slice().to_vec();
        let mut z: Vec<f64> = DMatrix::<f64>::identity(n, n).as_slice().to_vec();
        real_schur(&mut h, &mut z, n, &mut w).unwrap();
        let (t, zm) = (
            DMatrix::from_column_slice(n, n, &h),
            DMatrix::from_column_slice(n, n, &z),
        );
        check_real(&sub, &t, &zm, &w, 1e-12);
        let want: Vec<Complex64> = sub.complex_eigenvalues().iter().copied().collect();
        assert!(max_mismatch(&w, &want) <= 1e-10);
    }

    #[test]
    fn complex_adversarial_matrices() {
        let n = 8;
        let (q, _) = random_complex(n, 21).qr().unpack();
        let jordan = DMatrix::from_fn(n, n, |i, j| match j as isize - i as isize {
            0 => c(1.0, -1.0),
            1 => c(0.0, 1.0),
            _ => c(0.0, 0.0),
        });
        let diag = DMatrix::from_fn(n, n, |i, j| {
            if i == j {
                c(1.0, 1e-8 * i as f64)
            } else {
                c(0.0, 0.0)
            }
        });
        let cases: Vec<(&str, DMatrix<Complex64>, Vec<Complex64>, f64)> = vec![
            ("zero", DMatrix::zeros(n, n), vec![c(0.0, 0.0); n], 0.0),
            (
                "jordan",
                &q * &jordan * q.adjoint(),
                vec![c(1.0, -1.0); n],
                1e-1,
            ),
            (
                "cluster",
                &q * &diag * q.adjoint(),
                (0..n).map(|i| diag[(i, i)]).collect(),
                1e-6,
            ),
        ];
        for (name, a, want, tol) in &cases {
            let (t, z, w) = schur_of(a).unwrap_or_else(|()| panic!("{name}: no convergence"));
            check_complex(a, &t, &z, &w, 1e-12);
            assert!(max_mismatch(&w, want) <= *tol, "{name}: {w:?}");
        }
        // 1e±300 entries, on a directly built Hessenberg matrix (a dense
        // reduction in the oracle would overflow first).
        let mut b = random_complex(n, 22);
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    b[(i, j)] = c(0.0, 0.0);
                }
            }
        }
        let want: Vec<Complex64> = b
            .clone()
            .schur()
            .eigenvalues()
            .unwrap()
            .iter()
            .copied()
            .collect();
        for s in [1e300, 1e-300] {
            let a = &b * c(s, 0.0);
            let mut h: Vec<Complex64> = a.as_slice().to_vec();
            let mut z: Vec<Complex64> = DMatrix::<Complex64>::identity(n, n).as_slice().to_vec();
            let mut w = vec![c(0.0, 0.0); n];
            complex_schur(&mut h, &mut z, n, &mut w).unwrap();
            let t = DMatrix::from_column_slice(n, n, &h);
            let zm = DMatrix::from_column_slice(n, n, &z);
            check_complex_residual(&a, &t, &zm, 1e-12);
            check_complex_structure(&t, &w);
            let scaled: Vec<Complex64> = want.iter().map(|x| x * s).collect();
            assert!(max_mismatch(&w, &scaled) <= 1e-10 * s, "scale {s}");
        }
    }

    #[test]
    fn single_precision() {
        let n = 12;
        let a = random(n, 31);
        let a32 = a.map(|x| x as f32);
        let (t, z, w) = schur_of(&a32).unwrap();
        let back = (&z * &t * z.transpose() - &a32).norm();
        assert!(back <= 1e-5 * a32.norm());
        let want: Vec<Complex64> = a.complex_eigenvalues().iter().copied().collect();
        let got: Vec<Complex64> = w
            .iter()
            .map(|x| c(f64::from(x.re), f64::from(x.im)))
            .collect();
        assert!(max_mismatch(&got, &want) <= 1e-4);

        let b = random_complex(n, 32);
        let b32 = b.map(|x| Complex32::new(x.re as f32, x.im as f32));
        let (t, z, w) = schur_of(&b32).unwrap();
        assert!((&z * &t * z.adjoint() - &b32).norm() <= 1e-5 * b32.norm());
        let want: Vec<Complex64> = b.schur().eigenvalues().unwrap().iter().copied().collect();
        let got: Vec<Complex64> = w
            .iter()
            .map(|x| c(f64::from(x.re), f64::from(x.im)))
            .collect();
        assert!(max_mismatch(&got, &want) <= 1e-4);
    }

    #[test]
    fn non_finite_input_is_an_error() {
        let n = 5;
        let id = |n: usize| DMatrix::<f64>::identity(n, n).as_slice().to_vec();
        let a = random(n, 41);
        let (_, hm) = a.hessenberg().unpack();
        let mut w = vec![c(0.0, 0.0); n];
        for bad in [f64::NAN, f64::INFINITY] {
            let mut h: Vec<f64> = hm.as_slice().to_vec();
            h[n + 1] = bad;
            assert_eq!(real_schur(&mut h, &mut id(n), n, &mut w), Err(()));
            let mut hc: Vec<Complex64> = h.iter().map(|&x| c(x, 0.0)).collect();
            let mut zc: Vec<Complex64> = id(n).iter().map(|&x| c(x, 0.0)).collect();
            assert_eq!(complex_schur(&mut hc, &mut zc, n, &mut w), Err(()));
        }
    }

    /// Why the range scaling exists: LAPACK `hseqr` itself, called unscaled
    /// as KrylovKit calls it, treats every subdiagonal entry of a matrix near
    /// 1e-300 as negligible and returns its diagonal.
    #[test]
    fn unscaled_hseqr_deflates_tiny_matrices_wrongly() {
        let n = 10;
        let mut a = random(n, 11);
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    a[(i, j)] = 0.0;
                }
            }
        }
        let a = a * 1e-300;
        let id: Vec<f64> = DMatrix::<f64>::identity(n, n).as_slice().to_vec();
        let mut w = vec![c(0.0, 0.0); n];
        let (mut h, mut z) = (a.as_slice().to_vec(), id.clone());
        f64::hseqr_raw(&mut h, &mut z, n, &mut w, &mut Work::default()).unwrap();
        let t = DMatrix::from_column_slice(n, n, &h);
        let zm = DMatrix::from_column_slice(n, n, &z);
        let back = (&zm * (&t / amax(&a)) * zm.transpose() - &a / amax(&a)).norm();
        assert!(back > 1e-3, "unscaled LAPACK unexpectedly fine: {back:e}");
        let (mut h, mut z) = (a.as_slice().to_vec(), id);
        real_schur(&mut h, &mut z, n, &mut w).unwrap();
        let t = DMatrix::from_column_slice(n, n, &h);
        let zm = DMatrix::from_column_slice(n, n, &z);
        check_real(&a, &t, &zm, &w, 1e-12);
    }

    #[test]
    fn unscaled_underflow_keeps_eigenvalues_consistent_with_t() {
        // Subnormal-scale input: the 2×2 block's lower entry underflows to
        // zero after unscaling, so T is triangular and w must be real.
        let a0 = [
            -2.7797856262260443e-308,
            -3.0533251061780536e-308,
            2.720006882486198e-307,
            1.5446640576704915e-307,
        ];
        let mut h = a0.to_vec();
        let mut z = vec![1.0, 0.0, 0.0, 1.0];
        let mut w = vec![c(0.0, 0.0); 2];
        real_schur(&mut h, &mut z, 2, &mut w).unwrap();
        let (t, zm) = (
            DMatrix::from_column_slice(2, 2, &h),
            DMatrix::from_column_slice(2, 2, &z),
        );
        check_real_structure(&t, &w);
        check_real_residual(&DMatrix::from_column_slice(2, 2, &a0), &t, &zm, 1e-12);
        // The swap branch: a lower-triangular block (upper entry zero) is
        // turned upper triangular, also in Z.
        let mut h = vec![1.0, 3.0, 0.0, 1.0];
        let mut z = vec![1.0, 0.0, 0.0, 1.0];
        let mut w = vec![c(1.0, 2.0), c(1.0, -2.0)];
        repair_underflowed_blocks(&mut h, &mut z, 2, &mut w);
        assert_eq!(h, [1.0, 0.0, 3.0, 1.0]);
        assert_eq!(z, [0.0, 1.0, 1.0, 0.0]);
        assert_eq!(w, [c(1.0, 0.0), c(1.0, 0.0)]);
    }

    /// The range scaling must not touch inputs already inside the safe range:
    /// the result is LAPACK's unscaled one, bit for bit.
    #[test]
    fn range_scaling_is_the_identity_in_range() {
        let n = 20;
        let mut a = random(n, 51);
        let mut b = random_complex(n, 52);
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    a[(i, j)] = 0.0;
                    b[(i, j)] = c(0.0, 0.0);
                }
            }
        }
        let id: Vec<f64> = DMatrix::<f64>::identity(n, n).as_slice().to_vec();
        let (mut h1, mut z1, mut w1) = (a.as_slice().to_vec(), id.clone(), vec![c(0.0, 0.0); n]);
        let (mut h2, mut z2, mut w2) = (h1.clone(), id.clone(), w1.clone());
        real_schur(&mut h1, &mut z1, n, &mut w1).unwrap();
        f64::hseqr_raw(&mut h2, &mut z2, n, &mut w2, &mut Work::default()).unwrap();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let cbits = |v: &[Complex64]| {
            v.iter()
                .flat_map(|x| [x.re.to_bits(), x.im.to_bits()])
                .collect::<Vec<_>>()
        };
        assert_eq!(
            (bits(&h1), bits(&z1), cbits(&w1)),
            (bits(&h2), bits(&z2), cbits(&w2))
        );
        let idc: Vec<Complex64> = id.iter().map(|&x| c(x, 0.0)).collect();
        let (mut h1, mut z1, mut w1) = (b.as_slice().to_vec(), idc.clone(), vec![c(0.0, 0.0); n]);
        let (mut h2, mut z2, mut w2) = (h1.clone(), idc, w1.clone());
        complex_schur(&mut h1, &mut z1, n, &mut w1).unwrap();
        Complex64::hseqr_raw(&mut h2, &mut z2, n, &mut w2, &mut Work::default()).unwrap();
        assert_eq!(
            (cbits(&h1), cbits(&z1), cbits(&w1)),
            (cbits(&h2), cbits(&z2), cbits(&w2))
        );
    }

    /// Eigenvalues, in order, against Julia 1.11 `LAPACK.hseqr!` (LAPACK
    /// 3.12.0) on the fixed matrices of `tests/fixtures/lapack/hseqr.jl`.
    #[test]
    fn matches_lapack_hseqr_fixture() {
        let re = |i: usize, j: usize| ((7 * i + 3 * j * j) % 11) as f64 / 4.0 - 1.25;
        let im = |i: usize, j: usize| ((5 * i + 2 * j) % 7) as f64 / 2.0 - 1.5;
        let fixture = include_str!("../../tests/fixtures/lapack/hseqr.txt");
        let hex = |s: &str| f64::from_bits(u64::from_str_radix(s, 16).unwrap());
        let mut cases = 0;
        for line in fixture.lines().filter(|l| !l.starts_with('#')) {
            let mut it = line.split_whitespace();
            let kind = it.next().unwrap();
            let n: usize = it.next().unwrap().parse().unwrap();
            let want: Vec<Complex64> = it
                .map(|p| {
                    let (a, b) = p.split_once(':').unwrap();
                    c(hex(a), hex(b))
                })
                .collect();
            assert_eq!(want.len(), n);
            // 1-based Julia indices; entries are exact multiples of 1/4.
            let mut w = vec![c(0.0, 0.0); n];
            if kind == "real" {
                let mut h: Vec<f64> = (0..n * n)
                    .map(|k| {
                        let (i, j) = (k % n + 1, k / n + 1);
                        if i <= j + 1 { re(i, j) } else { 0.0 }
                    })
                    .collect();
                let mut z = vec![0.0; n * n];
                real_schur(&mut h, &mut z, n, &mut w).unwrap();
            } else {
                let mut h: Vec<Complex64> = (0..n * n)
                    .map(|k| {
                        let (i, j) = (k % n + 1, k / n + 1);
                        if i <= j + 1 {
                            c(re(i, j), im(i, j))
                        } else {
                            c(0.0, 0.0)
                        }
                    })
                    .collect();
                let mut z = vec![c(0.0, 0.0); n * n];
                complex_schur(&mut h, &mut z, n, &mut w).unwrap();
            }
            for (k, (g, x)) in w.iter().zip(&want).enumerate() {
                assert!(
                    (g - x).norm() <= 1e-12 * (1.0 + x.norm()),
                    "{kind} {n} #{k}: {g} vs {x}"
                );
            }
            cases += 1;
        }
        assert_eq!(cases, 4);
    }

    #[test]
    fn reordering_and_eigenvectors_of_the_schur_form() {
        let n = 12;
        // Real: sort by descending modulus, keeping 2×2 pairs together.
        let mut a = random(n, 61);
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    a[(i, j)] = 0.0;
                }
            }
        }
        let mut h = a.as_slice().to_vec();
        let mut z = DMatrix::<f64>::identity(n, n).as_slice().to_vec();
        let mut w = vec![c(0.0, 0.0); n];
        let mut work = Work::default();
        hschur(&mut h, &mut z, n, &mut w, &mut work).unwrap();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&x, &y| w[y].norm().partial_cmp(&w[x].norm()).unwrap());
        let mut p = Vec::new();
        permute_schur(&mut h, &mut z, n, &order, &mut p, &mut work).unwrap();
        let (t, zm) = (
            DMatrix::from_column_slice(n, n, &h),
            DMatrix::from_column_slice(n, n, &z),
        );
        check_real_residual(&a, &t, &zm, 1e-12);
        let mut ev = Vec::new();
        schur_eigvals(&h, n, n, &mut ev);
        check_real_structure(&t, &ev);
        let want: Vec<Complex64> = order.iter().map(|&k| w[k]).collect();
        for (g, x) in ev.iter().zip(&want) {
            assert!((g - x).norm() < 1e-12, "{g} vs {x}");
        }
        // An order that separates the two halves of a complex pair is refused.
        let pair = (0..n - 1)
            .find(|&i| t[(i + 1, i)] != 0.0)
            .expect("random matrix has a pair");
        let mut bad: Vec<usize> = (0..n).collect();
        bad.swap(pair, pair + 1);
        bad.swap(0, pair + 1);
        let (mut h2, mut z2) = (h.clone(), z.clone());
        assert!(permute_schur(&mut h2, &mut z2, n, &bad, &mut p, &mut work).is_err());

        // Complex: arbitrary order, then eigenvectors of T.
        let mut b = random_complex(n, 62);
        for i in 0..n {
            for j in 0..n {
                if i > j + 1 {
                    b[(i, j)] = c(0.0, 0.0);
                }
            }
        }
        let mut h = b.as_slice().to_vec();
        let mut z = DMatrix::<Complex64>::identity(n, n).as_slice().to_vec();
        let mut wc = Work::default();
        hschur(&mut h, &mut z, n, &mut w, &mut wc).unwrap();
        let order: Vec<usize> = (0..n).rev().collect();
        permute_schur(&mut h, &mut z, n, &order, &mut p, &mut wc).unwrap();
        let (t, zm) = (
            DMatrix::from_column_slice(n, n, &h),
            DMatrix::from_column_slice(n, n, &z),
        );
        check_complex_residual(&b, &t, &zm, 1e-12);
        for (j, &k) in order.iter().enumerate() {
            assert!((t[(j, j)] - w[k]).norm() < 1e-12);
        }
        let mut vr = vec![c(0.0, 0.0); n * n];
        let mut tt = h.clone();
        schur_eigvecs(&mut tt, n, &mut vr, &mut wc).unwrap();
        assert_eq!(tt, h, "trevc restores T");
        let x = DMatrix::from_column_slice(n, n, &vr);
        for j in 0..n {
            let col = x.column(j);
            assert!((col.norm() - 1.0).abs() < 1e-14);
            assert!(
                (&t * col - col * t[(j, j)]).norm() < 1e-12,
                "eigenvector {j}"
            );
        }
    }
}
