//! Schur decomposition of a small upper Hessenberg matrix.
//!
//! Ports of LAPACK 3.12.0 `DLAHQR` (real, Francis double shift), `ZLAHQR`
//! (complex, single shift) and `DLANV2`, with `WANTT = WANTZ = true`,
//! `ILO = ILOZ = 1`, `IHI = IHIZ = n`. This is the routine KrylovKit v0.10.4
//! `hschur!` reaches through `xHSEQR` for `n ≤ 75`. Indices below are the
//! 1-based LAPACK ones; `ix` maps them to column-major storage.

use num_complex::Complex;

use super::cplx::{cabs, cabs1, cdiv, csqrt, phase};
use crate::scalar::RealScalar;

const KEXSH: usize = 10;

/// LAPACK `SIGN(A, B)`.
fn sign<R: RealScalar>(a: R, b: R) -> R {
    a.abs().copysign(b)
}

/// Rejects non-finite input, which would otherwise spin until `ITMAX`.
fn all_finite<R: RealScalar>(h: &[R]) -> bool {
    h.iter().all(|x| x.is_finite())
}

/// Real Schur form `H ← Zᴴ·H·Z` of the upper Hessenberg `n × n` matrix `h`
/// (column-major), accumulated into `z` (`Z ← Z·Q`), with eigenvalues in
/// `w`. 2×2 diagonal blocks are in standard form (equal diagonal, off-diagonal
/// entries of opposite sign); a complex pair is stored `+im` first.
///
/// `Err(())` when the input is not finite or the QR iteration exceeds the
/// LAPACK budget `30·max(10, n)` per eigenvalue.
pub(crate) fn real_schur<R: RealScalar>(
    h: &mut [R],
    z: &mut [R],
    n: usize,
    w: &mut [Complex<R>],
) -> Result<(), ()> {
    if !all_finite(h) {
        return Err(());
    }
    let s = range_scale(h.iter().map(|x| x.abs()));
    h.iter_mut().for_each(|x| *x *= s);
    real_schur_with(h, z, n, w, 30 * n.max(10))?;
    let inv = R::one() / s;
    h.iter_mut().for_each(|x| *x *= inv);
    w.iter_mut().for_each(|x| *x *= inv);
    Ok(())
}

/// Power of four that brings `max |h_ij|` into `[SMLNUM, BIGNUM]` with
/// `SMLNUM = sqrt(SAFMIN)/ULP`, as `DGEEV`/`ZGEEV` scale before `xHSEQR`.
///
/// LAHQR treats subdiagonal entries `≤ SAFMIN·n/ULP` as zero, so without
/// this a matrix with entries near 1e-300 deflates completely and returns
/// its diagonal. KrylovKit calls `hseqr!` unscaled; scaling by a power of
/// four is exact, also for the `sqrt(|b|)·sqrt(|c|)` of a 2×2 block, and only
/// removes that failure.
fn range_scale<R: RealScalar>(abs: impl Iterator<Item = R>) -> R {
    let one = R::one();
    let four = (one + one) * (one + one);
    let anrm = abs.fold(R::zero(), R::max);
    let smlnum = R::min_positive_value().sqrt() / R::epsilon();
    let bignum = one / smlnum;
    let mut s = one;
    if anrm > R::zero() {
        while anrm * s < smlnum {
            s *= four;
        }
        while anrm * s > bignum {
            s /= four;
        }
    }
    s
}

fn real_schur_with<R: RealScalar>(
    h: &mut [R],
    z: &mut [R],
    n: usize,
    w: &mut [Complex<R>],
    itmax: usize,
) -> Result<(), ()> {
    debug_assert!(h.len() == n * n && z.len() == n * n && w.len() == n);
    if !all_finite(h) {
        return Err(());
    }
    let zero = R::zero();
    let one = R::one();
    let two = one + one;
    let ix = |i: usize, j: usize| (i - 1) + (j - 1) * n;
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        w[0] = Complex::new(h[0], zero);
        return Ok(());
    }
    for j in 1..=n.saturating_sub(3) {
        h[ix(j + 2, j)] = zero;
        h[ix(j + 3, j)] = zero;
    }
    if n >= 3 {
        h[ix(n, n - 2)] = zero;
    }
    let dat1 = R::from_f64(0.75);
    let dat2 = R::from_f64(-0.4375);
    let safmin = R::min_positive_value();
    let ulp = R::epsilon();
    let smlnum = safmin * (R::from_usize(n) / ulp);
    let (i1, i2) = (1, n);
    let mut kdefl = 0;
    let mut v = [zero; 3];

    let mut i = n;
    while i >= 1 {
        let mut l = 1;
        let mut done = false;
        for _its in 0..=itmax {
            // Look for a single small subdiagonal element.
            let mut k = i;
            while k > l {
                let hkk1 = h[ix(k, k - 1)].abs();
                if hkk1 <= smlnum {
                    break;
                }
                let mut tst = h[ix(k - 1, k - 1)].abs() + h[ix(k, k)].abs();
                if tst == zero {
                    if k >= 3 {
                        tst += h[ix(k - 1, k - 2)].abs();
                    }
                    if k < n {
                        tst += h[ix(k + 1, k)].abs();
                    }
                }
                // Ahues & Tisseur conservative deflation criterion.
                if hkk1 <= ulp * tst {
                    let hk1k = h[ix(k - 1, k)].abs();
                    let ab = hkk1.max(hk1k);
                    let ba = hkk1.min(hk1k);
                    let d = (h[ix(k - 1, k - 1)] - h[ix(k, k)]).abs();
                    let aa = h[ix(k, k)].abs().max(d);
                    let bb = h[ix(k, k)].abs().min(d);
                    let s = aa + ab;
                    if ba * (ab / s) <= smlnum.max(ulp * (bb * (aa / s))) {
                        break;
                    }
                }
                k -= 1;
            }
            l = k;
            if l > 1 {
                h[ix(l, l - 1)] = zero;
            }
            if l + 1 >= i {
                done = true;
                break;
            }
            kdefl += 1;

            // Shifts.
            let (mut h11, mut h12, mut h21, mut h22);
            if kdefl % (2 * KEXSH) == 0 {
                let s = h[ix(i, i - 1)].abs() + h[ix(i - 1, i - 2)].abs();
                h11 = dat1 * s + h[ix(i, i)];
                h12 = dat2 * s;
                h21 = s;
                h22 = h11;
            } else if kdefl % KEXSH == 0 {
                let s = h[ix(l + 1, l)].abs() + h[ix(l + 2, l + 1)].abs();
                h11 = dat1 * s + h[ix(l, l)];
                h12 = dat2 * s;
                h21 = s;
                h22 = h11;
            } else {
                h11 = h[ix(i - 1, i - 1)];
                h21 = h[ix(i, i - 1)];
                h12 = h[ix(i - 1, i)];
                h22 = h[ix(i, i)];
            }
            let s = h11.abs() + h12.abs() + h21.abs() + h22.abs();
            let (rt1r, rt1i, rt2r, rt2i);
            if s == zero {
                (rt1r, rt1i, rt2r, rt2i) = (zero, zero, zero, zero);
            } else {
                h11 /= s;
                h21 /= s;
                h12 /= s;
                h22 /= s;
                let tr = (h11 + h22) / two;
                let det = (h11 - tr) * (h22 - tr) - h12 * h21;
                let rtdisc = det.abs().sqrt();
                if det >= zero {
                    rt1r = tr * s;
                    rt2r = rt1r;
                    rt1i = rtdisc * s;
                    rt2i = -rt1i;
                } else {
                    // Real shifts: use only the one closer to H(i,i), twice.
                    let a = tr + rtdisc;
                    let b = tr - rtdisc;
                    let r = if (a - h22).abs() <= (b - h22).abs() {
                        a * s
                    } else {
                        b * s
                    };
                    rt1r = r;
                    rt2r = r;
                    rt1i = zero;
                    rt2i = zero;
                }
            }

            // Look for two consecutive small subdiagonal elements.
            let mut m = i - 2;
            loop {
                let mut h21s = h[ix(m + 1, m)];
                let s = (h[ix(m, m)] - rt2r).abs() + rt2i.abs() + h21s.abs();
                h21s = h[ix(m + 1, m)] / s;
                v[0] = h21s * h[ix(m, m + 1)] + (h[ix(m, m)] - rt1r) * ((h[ix(m, m)] - rt2r) / s)
                    - rt1i * (rt2i / s);
                v[1] = h21s * (h[ix(m, m)] + h[ix(m + 1, m + 1)] - rt1r - rt2r);
                v[2] = h21s * h[ix(m + 2, m + 1)];
                let s = v[0].abs() + v[1].abs() + v[2].abs();
                v[0] /= s;
                v[1] /= s;
                v[2] /= s;
                if m == l {
                    break;
                }
                let h00 = h[ix(m - 1, m - 1)].abs();
                if h[ix(m, m - 1)].abs() * (v[1].abs() + v[2].abs())
                    <= ulp * v[0].abs() * (h00 + h[ix(m, m)].abs() + h[ix(m + 1, m + 1)].abs())
                {
                    break;
                }
                m -= 1;
            }

            // Double-shift QR sweep.
            for k in m..i {
                let nr = 3.min(i - k + 1);
                if k > m {
                    for (r, vr) in v.iter_mut().enumerate().take(nr) {
                        *vr = h[ix(k + r, k - 1)];
                    }
                }
                let (alpha, x) = v.split_first_mut().expect("v has 3 entries");
                let t1 = larfg(alpha, &mut x[..nr - 1]);
                if k > m {
                    h[ix(k, k - 1)] = v[0];
                    h[ix(k + 1, k - 1)] = zero;
                    if k < i - 1 {
                        h[ix(k + 2, k - 1)] = zero;
                    }
                } else if m > l {
                    // Use the following instead of H(k,k-1) = -H(k,k-1) to
                    // avoid a bug when v(2) and v(3) underflow (LAPACK note).
                    h[ix(k, k - 1)] = h[ix(k, k - 1)] * (one - t1);
                }
                let v2 = v[1];
                let t2 = t1 * v2;
                if nr == 3 {
                    let v3 = v[2];
                    let t3 = t1 * v3;
                    for j in k..=i2 {
                        let sum = h[ix(k, j)] + v2 * h[ix(k + 1, j)] + v3 * h[ix(k + 2, j)];
                        h[ix(k, j)] = h[ix(k, j)] - sum * t1;
                        h[ix(k + 1, j)] = h[ix(k + 1, j)] - sum * t2;
                        h[ix(k + 2, j)] = h[ix(k + 2, j)] - sum * t3;
                    }
                    for j in i1..=(k + 3).min(i) {
                        let sum = h[ix(j, k)] + v2 * h[ix(j, k + 1)] + v3 * h[ix(j, k + 2)];
                        h[ix(j, k)] = h[ix(j, k)] - sum * t1;
                        h[ix(j, k + 1)] = h[ix(j, k + 1)] - sum * t2;
                        h[ix(j, k + 2)] = h[ix(j, k + 2)] - sum * t3;
                    }
                    for j in 1..=n {
                        let sum = z[ix(j, k)] + v2 * z[ix(j, k + 1)] + v3 * z[ix(j, k + 2)];
                        z[ix(j, k)] = z[ix(j, k)] - sum * t1;
                        z[ix(j, k + 1)] = z[ix(j, k + 1)] - sum * t2;
                        z[ix(j, k + 2)] = z[ix(j, k + 2)] - sum * t3;
                    }
                } else if nr == 2 {
                    for j in k..=i2 {
                        let sum = h[ix(k, j)] + v2 * h[ix(k + 1, j)];
                        h[ix(k, j)] = h[ix(k, j)] - sum * t1;
                        h[ix(k + 1, j)] = h[ix(k + 1, j)] - sum * t2;
                    }
                    for j in i1..=i {
                        let sum = h[ix(j, k)] + v2 * h[ix(j, k + 1)];
                        h[ix(j, k)] = h[ix(j, k)] - sum * t1;
                        h[ix(j, k + 1)] = h[ix(j, k + 1)] - sum * t2;
                    }
                    for j in 1..=n {
                        let sum = z[ix(j, k)] + v2 * z[ix(j, k + 1)];
                        z[ix(j, k)] = z[ix(j, k)] - sum * t1;
                        z[ix(j, k + 1)] = z[ix(j, k + 1)] - sum * t2;
                    }
                }
            }
        }
        if !done {
            return Err(());
        }

        if l == i {
            w[i - 1] = Complex::new(h[ix(i, i)], zero);
        } else {
            // l == i - 1: a 2×2 block; standardize it and apply the rotation.
            let (mut a, mut b, mut c, mut d) = (
                h[ix(i - 1, i - 1)],
                h[ix(i - 1, i)],
                h[ix(i, i - 1)],
                h[ix(i, i)],
            );
            let (rt1, rt2, cs, sn) = lanv2(&mut a, &mut b, &mut c, &mut d);
            h[ix(i - 1, i - 1)] = a;
            h[ix(i - 1, i)] = b;
            h[ix(i, i - 1)] = c;
            h[ix(i, i)] = d;
            w[i - 2] = rt1;
            w[i - 1] = rt2;
            for j in i + 1..=i2 {
                rot(h, ix(i - 1, j), ix(i, j), cs, sn);
            }
            for j in i1..=i - 2 {
                rot(h, ix(j, i - 1), ix(j, i), cs, sn);
            }
            for j in 1..=n {
                rot(z, ix(j, i - 1), ix(j, i), cs, sn);
            }
        }
        kdefl = 0;
        i = l - 1;
    }
    Ok(())
}

/// BLAS `DROT` on one pair: `(x, y) ← (c·x + s·y, c·y − s·x)`.
fn rot<R: RealScalar>(a: &mut [R], p: usize, q: usize, c: R, s: R) {
    let (x, y) = (a[p], a[q]);
    a[p] = c * x + s * y;
    a[q] = c * y - s * x;
}

/// LAPACK `DLARFG`: overwrites `alpha` with `β` and `x` with `v[1..]`, and
/// returns `τ`, so that `(I − τ·v·vᵀ)·[alpha; x] = [β; 0]` with `v[0] = 1`.
fn larfg<R: RealScalar>(alpha: &mut R, x: &mut [R]) -> R {
    let zero = R::zero();
    let one = R::one();
    let nrm = |x: &[R]| x.iter().fold(zero, |acc, &xi| acc.hypot2(xi));
    let mut xnorm = nrm(x);
    if xnorm == zero {
        return zero;
    }
    let mut beta = -sign(alpha.hypot2(xnorm), *alpha);
    let safmin = R::min_positive_value() / (R::epsilon() / (one + one));
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = one / safmin;
        loop {
            knt += 1;
            x.iter_mut().for_each(|xi| *xi *= rsafmn);
            beta *= rsafmn;
            *alpha *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnorm = nrm(x);
        beta = -sign(alpha.hypot2(xnorm), *alpha);
    }
    let tau = (beta - *alpha) / beta;
    let scal = one / (*alpha - beta);
    x.iter_mut().for_each(|xi| *xi *= scal);
    for _ in 0..knt {
        beta *= safmin;
    }
    *alpha = beta;
    tau
}

/// LAPACK 3.12 `DLANV2`: Schur factorization of a real 2×2 matrix in
/// standard form. Updates `(a, b, c, d)` in place and returns the two
/// eigenvalues and the rotation `(cs, sn)`.
pub(crate) fn lanv2<R: RealScalar>(
    a: &mut R,
    b: &mut R,
    c: &mut R,
    d: &mut R,
) -> (Complex<R>, Complex<R>, R, R) {
    let zero = R::zero();
    let one = R::one();
    let two = one + one;
    let half = one / two;
    let multpl = two * two;
    let safmin = R::min_positive_value();
    let eps = R::epsilon();
    // DLAMCH('B')**INT(LOG(SAFMIN/EPS)/LOG(2)/2): the exponents are exact
    // integers, so take them from the representation instead of libm `log`.
    let safmn2 = two.powi((log2_exact(safmin) - log2_exact(eps)) / 2);
    let safmx2 = one / safmn2;
    let (cs, sn);
    if *c == zero {
        cs = one;
        sn = zero;
    } else if *b == zero {
        cs = zero;
        sn = one;
        core::mem::swap(a, d);
        *b = -*c;
        *c = zero;
    } else if *a - *d == zero && sign(one, *b) != sign(one, *c) {
        cs = one;
        sn = zero;
    } else {
        let mut temp = *a - *d;
        let mut p = half * temp;
        let bcmax = b.abs().max(c.abs());
        let bcmis = b.abs().min(c.abs()) * sign(one, *b) * sign(one, *c);
        let mut scale = p.abs().max(bcmax);
        let mut zz = (p / scale) * p + (bcmax / scale) * bcmis;
        if zz >= multpl * eps {
            // Real eigenvalues.
            zz = p + sign(scale.sqrt() * zz.sqrt(), p);
            *a = *d + zz;
            *d -= (bcmax / zz) * bcmis;
            let tau = c.hypot2(zz);
            cs = zz / tau;
            sn = *c / tau;
            *b -= *c;
            *c = zero;
        } else {
            // Complex or nearly equal real eigenvalues: equalize the diagonal.
            let mut count = 0;
            let mut sigma = *b + *c;
            loop {
                count += 1;
                scale = temp.abs().max(sigma.abs());
                if scale >= safmx2 {
                    sigma *= safmn2;
                    temp *= safmn2;
                    if count <= 20 {
                        continue;
                    }
                }
                if scale <= safmn2 {
                    sigma *= safmx2;
                    temp *= safmx2;
                    if count <= 20 {
                        continue;
                    }
                }
                break;
            }
            p = half * temp;
            let tau = sigma.hypot2(temp);
            let mut cs1 = (half * (one + sigma.abs() / tau)).sqrt();
            let mut sn1 = -(p / (tau * cs1)) * sign(one, sigma);
            // [aa bb; cc dd] = [a b; c d]·[cs −sn; sn cs]
            let aa = *a * cs1 + *b * sn1;
            let bb = -*a * sn1 + *b * cs1;
            let cc = *c * cs1 + *d * sn1;
            let dd = -*c * sn1 + *d * cs1;
            // [a b; c d] = [cs sn; −sn cs]·[aa bb; cc dd]
            *a = aa * cs1 + cc * sn1;
            *b = bb * cs1 + dd * sn1;
            *c = -aa * sn1 + cc * cs1;
            *d = -bb * sn1 + dd * cs1;
            let temp = half * (*a + *d);
            *a = temp;
            *d = temp;
            if *c != zero {
                if *b != zero {
                    if sign(one, *b) == sign(one, *c) {
                        // Real eigenvalues: reduce to upper triangular form.
                        let sab = b.abs().sqrt();
                        let sac = c.abs().sqrt();
                        let p = sign(sab * sac, *c);
                        let tau = one / (*b + *c).abs().sqrt();
                        *a = temp + p;
                        *d = temp - p;
                        *b -= *c;
                        *c = zero;
                        let cs2 = sab * tau;
                        let sn2 = sac * tau;
                        let t = cs1 * cs2 - sn1 * sn2;
                        sn1 = cs1 * sn2 + sn1 * cs2;
                        cs1 = t;
                    }
                } else {
                    *b = -*c;
                    *c = zero;
                    let t = cs1;
                    cs1 = -sn1;
                    sn1 = t;
                }
            }
            cs = cs1;
            sn = sn1;
        }
    }
    let (rt1, rt2) = if *c == zero {
        (Complex::new(*a, zero), Complex::new(*d, zero))
    } else {
        let im = b.abs().sqrt() * c.abs().sqrt();
        (Complex::new(*a, im), Complex::new(*d, -im))
    };
    (rt1, rt2, cs, sn)
}

/// `log₂ x` for a positive power of two `x`.
fn log2_exact<R: RealScalar>(x: R) -> i32 {
    let (mantissa, exponent, _) = x.integer_decode();
    i32::from(exponent) + (63 - mantissa.leading_zeros() as i32)
}

/// Complex Schur form of the upper Hessenberg `n × n` matrix `h`
/// (column-major), accumulated into `z`, with eigenvalues in `w`
/// (`w[i] = H[i,i]`). Same failure contract as [`real_schur`].
pub(crate) fn complex_schur<R: RealScalar>(
    h: &mut [Complex<R>],
    z: &mut [Complex<R>],
    n: usize,
    w: &mut [Complex<R>],
) -> Result<(), ()> {
    if !h.iter().all(|x| x.re.is_finite() && x.im.is_finite()) {
        return Err(());
    }
    let s = range_scale(h.iter().map(|x| x.re.abs().max(x.im.abs())));
    h.iter_mut().for_each(|x| *x *= s);
    complex_schur_with(h, z, n, w, 30 * n.max(10))?;
    let inv = R::one() / s;
    h.iter_mut().for_each(|x| *x *= inv);
    w.iter_mut().for_each(|x| *x *= inv);
    Ok(())
}

fn complex_schur_with<R: RealScalar>(
    h: &mut [Complex<R>],
    z: &mut [Complex<R>],
    n: usize,
    w: &mut [Complex<R>],
    itmax: usize,
) -> Result<(), ()> {
    debug_assert!(h.len() == n * n && z.len() == n * n && w.len() == n);
    if !h.iter().all(|x| x.re.is_finite() && x.im.is_finite()) {
        return Err(());
    }
    let rzero = R::zero();
    let rone = R::one();
    let czero = Complex::new(rzero, rzero);
    let cone = Complex::new(rone, rzero);
    let half = rone / (rone + rone);
    let ix = |i: usize, j: usize| (i - 1) + (j - 1) * n;
    if n == 0 {
        return Ok(());
    }
    if n == 1 {
        w[0] = h[0];
        return Ok(());
    }
    for j in 1..=n.saturating_sub(3) {
        h[ix(j + 2, j)] = czero;
        h[ix(j + 3, j)] = czero;
    }
    if n >= 3 {
        h[ix(n, n - 2)] = czero;
    }
    let (jlo, jhi) = (1, n);
    // Make the subdiagonal real.
    for i in 2..=n {
        let hi = h[ix(i, i - 1)];
        if hi.im != rzero {
            let sc = hi / cabs1(hi);
            let sc = sc.conj() / cabs(sc);
            h[ix(i, i - 1)] = Complex::new(cabs(hi), rzero);
            for j in i..=jhi {
                h[ix(i, j)] = sc * h[ix(i, j)];
            }
            for j in jlo..=jhi.min(i + 1) {
                h[ix(j, i)] = sc.conj() * h[ix(j, i)];
            }
            for j in 1..=n {
                z[ix(j, i)] = sc.conj() * z[ix(j, i)];
            }
        }
    }
    let dat1 = R::from_f64(0.75);
    let safmin = R::min_positive_value();
    let ulp = R::epsilon();
    let smlnum = safmin * (R::from_usize(n) / ulp);
    let (i1, i2) = (1, n);
    let mut kdefl = 0;

    let mut i = n;
    while i >= 1 {
        let mut l = 1;
        let mut done = false;
        for _its in 0..=itmax {
            let mut k = i;
            while k > l {
                let hkk1 = h[ix(k, k - 1)];
                if cabs1(hkk1) <= smlnum {
                    break;
                }
                let mut tst = cabs1(h[ix(k - 1, k - 1)]) + cabs1(h[ix(k, k)]);
                if tst == rzero {
                    if k >= 3 {
                        tst += h[ix(k - 1, k - 2)].re.abs();
                    }
                    if k < n {
                        tst += h[ix(k + 1, k)].re.abs();
                    }
                }
                if hkk1.re.abs() <= ulp * tst {
                    let ab = cabs1(hkk1).max(cabs1(h[ix(k - 1, k)]));
                    let ba = cabs1(hkk1).min(cabs1(h[ix(k - 1, k)]));
                    let dif = cabs1(h[ix(k - 1, k - 1)] - h[ix(k, k)]);
                    let aa = cabs1(h[ix(k, k)]).max(dif);
                    let bb = cabs1(h[ix(k, k)]).min(dif);
                    let s = aa + ab;
                    if ba * (ab / s) <= smlnum.max(ulp * (bb * (aa / s))) {
                        break;
                    }
                }
                k -= 1;
            }
            l = k;
            if l > 1 {
                h[ix(l, l - 1)] = czero;
            }
            if l >= i {
                done = true;
                break;
            }
            kdefl += 1;

            // Single shift.
            let t;
            if kdefl % (2 * KEXSH) == 0 {
                let s = dat1 * h[ix(i, i - 1)].re.abs();
                t = h[ix(i, i)] + s;
            } else if kdefl % KEXSH == 0 {
                let s = dat1 * h[ix(l + 1, l)].re.abs();
                t = h[ix(l, l)] + s;
            } else {
                let mut tt = h[ix(i, i)];
                let u = csqrt(h[ix(i - 1, i)]) * csqrt(h[ix(i, i - 1)]);
                let mut s = cabs1(u);
                if s != rzero {
                    let x = (h[ix(i - 1, i - 1)] - tt) * half;
                    let sx = cabs1(x);
                    s = s.max(sx);
                    let xs = x / s;
                    let us = u / s;
                    let mut y = csqrt(xs * xs + us * us) * s;
                    if sx > rzero {
                        let xn = x / sx;
                        if xn.re * y.re + xn.im * y.im < rzero {
                            y = -y;
                        }
                    }
                    tt -= u * cdiv(u, x + y);
                }
                t = tt;
            }

            // Look for two consecutive small subdiagonal elements.
            let mut v = [czero; 2];
            let mut m = i - 1;
            let mut found = false;
            while m > l {
                let h11 = h[ix(m, m)];
                let h22 = h[ix(m + 1, m + 1)];
                let mut h11s = h11 - t;
                let mut h21 = h[ix(m + 1, m)].re;
                let s = cabs1(h11s) + h21.abs();
                h11s /= s;
                h21 /= s;
                v = [h11s, Complex::new(h21, rzero)];
                let h10 = h[ix(m, m - 1)].re;
                if h10.abs() * h21.abs() <= ulp * (cabs1(h11s) * (cabs1(h11) + cabs1(h22))) {
                    found = true;
                    break;
                }
                m -= 1;
            }
            if !found {
                m = l;
                let h11 = h[ix(l, l)];
                let mut h11s = h11 - t;
                let mut h21 = h[ix(l + 1, l)].re;
                let s = cabs1(h11s) + h21.abs();
                h11s /= s;
                h21 /= s;
                v = [h11s, Complex::new(h21, rzero)];
            }

            // Single-shift QR sweep.
            for k in m..i {
                if k > m {
                    v = [h[ix(k, k - 1)], h[ix(k + 1, k - 1)]];
                }
                let (alpha, x) = v.split_first_mut().expect("v has 2 entries");
                let t1 = zlarfg(alpha, &mut x[0]);
                if k > m {
                    h[ix(k, k - 1)] = v[0];
                    h[ix(k + 1, k - 1)] = czero;
                }
                let v2 = v[1];
                let t2 = (t1 * v2).re;
                for j in k..=i2 {
                    let sum = t1.conj() * h[ix(k, j)] + h[ix(k + 1, j)] * t2;
                    h[ix(k, j)] -= sum;
                    h[ix(k + 1, j)] -= sum * v2;
                }
                for j in i1..=(k + 2).min(i) {
                    let sum = t1 * h[ix(j, k)] + h[ix(j, k + 1)] * t2;
                    h[ix(j, k)] -= sum;
                    h[ix(j, k + 1)] -= sum * v2.conj();
                }
                for j in 1..=n {
                    let sum = t1 * z[ix(j, k)] + z[ix(j, k + 1)] * t2;
                    z[ix(j, k)] -= sum;
                    z[ix(j, k + 1)] -= sum * v2.conj();
                }
                if k == m && m > l {
                    // The first reflector is not a pure rotation when the
                    // shift was taken at row m > l; restore real subdiagonals.
                    let temp = phase(cone - t1);
                    h[ix(m + 1, m)] *= temp.conj();
                    if m + 2 <= i {
                        h[ix(m + 2, m + 1)] *= temp;
                    }
                    for j in m..=i {
                        if j != m + 1 {
                            for jj in j + 1..=i2 {
                                h[ix(j, jj)] *= temp;
                            }
                            for jj in i1..j {
                                h[ix(jj, j)] *= temp.conj();
                            }
                            for jj in 1..=n {
                                z[ix(jj, j)] *= temp.conj();
                            }
                        }
                    }
                }
            }

            // Ensure H(i, i-1) is real.
            let temp = h[ix(i, i - 1)];
            if temp.im != rzero {
                let rtemp = cabs(temp);
                h[ix(i, i - 1)] = Complex::new(rtemp, rzero);
                let temp = temp / rtemp;
                for jj in i + 1..=i2 {
                    h[ix(i, jj)] *= temp.conj();
                }
                for jj in i1..i {
                    h[ix(jj, i)] *= temp;
                }
                for jj in 1..=n {
                    z[ix(jj, i)] *= temp;
                }
            }
        }
        if !done {
            return Err(());
        }
        w[i - 1] = h[ix(i, i)];
        kdefl = 0;
        i = l - 1;
    }
    Ok(())
}

/// LAPACK `ZLARFG` for `n = 2`: overwrites `alpha` with the real `β` and `x`
/// with `v[1]`, and returns `τ`, so that `Hᴴ·[alpha; x] = [β; 0]` with
/// `H = I − τ·v·vᴴ`.
fn zlarfg<R: RealScalar>(alpha: &mut Complex<R>, x: &mut Complex<R>) -> Complex<R> {
    let zero = R::zero();
    let one = R::one();
    let mut xnorm = cabs(*x);
    let (mut alphr, mut alphi) = (alpha.re, alpha.im);
    if xnorm == zero && alphi == zero {
        return Complex::new(zero, zero);
    }
    let lapy3 = |a: R, b: R, c: R| a.hypot2(b).hypot2(c);
    let mut beta = -sign(lapy3(alphr, alphi, xnorm), alphr);
    let safmin = R::min_positive_value() / (R::epsilon() / (one + one));
    let rsafmn = one / safmin;
    let mut knt = 0;
    if beta.abs() < safmin {
        loop {
            knt += 1;
            *x *= rsafmn;
            beta *= rsafmn;
            alphi *= rsafmn;
            alphr *= rsafmn;
            if !(beta.abs() < safmin && knt < 20) {
                break;
            }
        }
        xnorm = cabs(*x);
        *alpha = Complex::new(alphr, alphi);
        beta = -sign(lapy3(alphr, alphi, xnorm), alphr);
    }
    let tau = Complex::new((beta - alphr) / beta, -alphi / beta);
    *x *= cdiv(Complex::new(one, zero), *alpha - beta);
    for _ in 0..knt {
        beta *= safmin;
    }
    *alpha = Complex::new(beta, zero);
    tau
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::Scalar;
    use crate::scalar::sealed::Ops;
    use nalgebra::DMatrix;
    use num_complex::{Complex32, Complex64};

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
        <S as Ops<S::Real>>::hschur(&mut h, &mut z, n, &mut w)?;
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

    fn check_real_residual(a: &DMatrix<f64>, t: &DMatrix<f64>, z: &DMatrix<f64>, tol: f64) {
        let n = a.nrows();
        let scale = a.norm().max(f64::MIN_POSITIVE);
        let back = (z * t * z.transpose() - a).norm();
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
                let im = t[(i, i + 1)].abs().sqrt() * t[(i + 1, i)].abs().sqrt();
                assert_eq!(w[i], c(t[(i, i)], im));
                assert_eq!(w[i + 1], c(t[(i, i)], -im));
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
        let scale = a.norm().max(f64::MIN_POSITIVE);
        assert!(
            (z * t * z.adjoint() - a).norm() <= tol * scale,
            "backward error"
        );
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
            check_real_residual(&(&a / s), &(&t / s), &zm, 1e-12);
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
        assert!(max_mismatch(&w, &want) <= 1e-13 * g.norm());
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
            check_complex_residual(&b, &(&t * c(1.0 / s, 0.0)), &zm, 1e-12);
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
    fn non_finite_input_and_exhausted_budget_are_errors() {
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
        // With no iterations allowed, an unreduced Hessenberg matrix cannot
        // deflate: the budget path reports failure instead of looping.
        let mut h: Vec<f64> = hm.as_slice().to_vec();
        assert_eq!(real_schur_with(&mut h, &mut id(n), n, &mut w, 0), Err(()));
        let mut hc: Vec<Complex64> = hm.as_slice().iter().map(|&x| c(x, 0.0)).collect();
        let mut zc: Vec<Complex64> = id(n).iter().map(|&x| c(x, 0.0)).collect();
        assert_eq!(complex_schur_with(&mut hc, &mut zc, n, &mut w, 0), Err(()));
    }

    #[test]
    fn lanv2_standardizes_every_branch() {
        // (a, b, c, d) covering c = 0, b = 0, equal diagonal with opposite
        // signs, real split, complex pair, and tiny/huge entries that need
        // the SAFMN2 rescaling.
        let cases = [
            (1.0, 2.0, 0.0, 3.0),
            (1.0, 0.0, 2.0, 3.0),
            (1.0, 2.0, -3.0, 1.0),
            (4.0, 1.0, 2.0, -1.0),
            (1.0, -5.0, 2.0, 1.5),
            (1e-300, 3e-300, -2e-300, 1.5e-300),
            (1e300, 3e300, -2e300, 1.5e300),
            (0.0, 1.0, -1e-320, 0.0),
        ];
        let cases: [(f64, f64, f64, f64); 8] = cases;
        for (a0, b0, c0, d0) in cases {
            let (mut a, mut b, mut cc, mut d) = (a0, b0, c0, d0);
            let (r1, r2, cs, sn) = lanv2(&mut a, &mut b, &mut cc, &mut d);
            assert!((cs * cs + sn * sn - 1.0).abs() < 1e-15);
            // [a b; c d] = Rᵀ·[a0 b0; c0 d0]·R with R = [cs −sn; sn cs].
            let m0 = nalgebra::Matrix2::new(a0, b0, c0, d0);
            let r = nalgebra::Matrix2::new(cs, -sn, sn, cs);
            let m = nalgebra::Matrix2::new(a, b, cc, d);
            let scale = m0.norm();
            assert!(
                (r.transpose() * m0 * r - m).norm() <= 1e-15 * scale,
                "{a0} {b0} {c0} {d0}"
            );
            if cc == 0.0 {
                assert_eq!((r1, r2), (c(a, 0.0), c(d, 0.0)));
            } else {
                assert_eq!(a, d);
                assert_ne!(b.is_sign_negative(), cc.is_sign_negative());
                assert_eq!(r1.conj(), r2);
                assert!(r1.im > 0.0);
            }
        }
    }
}
