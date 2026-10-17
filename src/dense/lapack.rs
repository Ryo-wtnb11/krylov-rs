//! LAPACK calls for the small projected problems, made as KrylovKit v0.10.4
//! makes them.
//!
//! krylov depends only on the `lapack` declarations. The final binary links
//! a provider (OpenBLAS, Accelerate, MKL, …), e.g. through `lapack-src`. The
//! calls below are monomorphized per scalar type through the sealed
//! `RealOps`/`Ops` hooks:
//!
//! - `xSTEGR(jobz='V', range='A')`: symmetric tridiagonal eigenproblem
//!   (KrylovKit `src/dense/linalg.jl:stegr!`, used by `tridiageigh!`).
//! - `xBDSQR`: bidiagonal SVD (`linalg.jl:bidiagsvd!`).
//! - `xSYEVR`/`xHEEVR(jobz='V', range='A', uplo='U', abstol=-1)`: dense
//!   Hermitian eigenproblem (Julia 1.11 `eigen(::Hermitian)`, which
//!   KrylovKit's block Lanczos calls).
//!
//! Workspace sizes are queried (`lwork = -1`) on every call, as Julia does.
//! The buffers live in a caller-owned [`Work`] that only grows.

use num_complex::Complex;

use crate::scalar::sealed::RealOps;

/// Reusable LAPACK workspace: scalar `work`, real `rwork`/eigenvalues and
/// integer `iwork`/`isuppz`.
#[derive(Debug)]
pub struct Work<S, R> {
    s: Vec<S>,
    r: Vec<R>,
    i: Vec<i32>,
}

impl<S, R> Default for Work<S, R> {
    fn default() -> Self {
        Self {
            s: Vec::new(),
            r: Vec::new(),
            i: Vec::new(),
        }
    }
}

/// Non-finite input is rejected before the call: depending on the provider,
/// LAPACK may iterate to its budget, return garbage with `info = 0`, or hang.
fn finite<T: num_traits::Float>(x: &[T]) -> bool {
    x.iter().all(|v| v.is_finite())
}

fn int(n: usize) -> Result<i32, ()> {
    i32::try_from(n).map_err(|_| ())
}

/// LAPACK returns workspace sizes as floating-point values.
fn size_of_query(x: f64) -> usize {
    // ponytail: f64 holds every workspace size LAPACK can report for n < 2³¹.
    x.max(1.0) as usize
}

macro_rules! lapack_real {
    ($t:ty, $stegr:ident, $bdsqr:ident, $syevr:ident, $heevr:ident) => {
        impl RealOps for $t {
            fn stegr(
                d: &mut [$t],
                e: &mut [$t],
                z: &mut [$t],
                w: &mut Work<$t, $t>,
            ) -> Result<(), ()> {
                let n = d.len();
                assert!(e.len() >= n && z.len() >= n * n, "stegr: buffer too small");
                if n == 0 {
                    return Ok(());
                }
                if !finite(d) || !finite(&e[..n - 1]) {
                    return Err(());
                }
                let ni = int(n)?;
                w.r.resize(n, 0.0);
                w.i.resize(2 * n + 1, 0);
                let (mut m, mut info) = (0, 0);
                let (mut lw, mut liw) = ([0.0 as $t], [0i32]);
                let (isuppz, _) = w.i.split_at_mut(2 * n);
                // SAFETY: d, e have n entries, z n·n, w.r n, isuppz 2n; the
                // query (lwork = liwork = -1) writes only lw[0] and liw[0].
                unsafe {
                    lapack::$stegr(
                        b'V', b'A', ni, d, e, 0.0, 0.0, 1, ni, 0.0, &mut m, &mut w.r, z, ni,
                        isuppz, &mut lw, -1, &mut liw, -1, &mut info,
                    );
                }
                if info != 0 {
                    return Err(());
                }
                let (lwork, liwork) = (size_of_query(f64::from(lw[0])), liw[0].max(1) as usize);
                w.s.resize(lwork, 0.0);
                w.i.resize(2 * n + liwork, 0);
                let (isuppz, iwork) = w.i.split_at_mut(2 * n);
                // SAFETY: as above, with work/iwork of the queried sizes.
                unsafe {
                    lapack::$stegr(
                        b'V',
                        b'A',
                        ni,
                        d,
                        e,
                        0.0,
                        0.0,
                        1,
                        ni,
                        0.0,
                        &mut m,
                        &mut w.r,
                        z,
                        ni,
                        isuppz,
                        &mut w.s,
                        int(lwork)?,
                        iwork,
                        int(liwork)?,
                        &mut info,
                    );
                }
                if info != 0 || m != ni {
                    return Err(());
                }
                d.copy_from_slice(&w.r[..n]);
                Ok(())
            }

            fn bdsqr(
                lower: bool,
                d: &mut [$t],
                e: &mut [$t],
                vt: &mut [$t],
                u: &mut [$t],
                w: &mut Work<$t, $t>,
            ) -> Result<(), ()> {
                let n = d.len();
                assert!(
                    e.len() + 1 >= n && vt.len() >= n * n && u.len() >= n * n,
                    "bdsqr: buffer too small"
                );
                if n == 0 {
                    return Ok(());
                }
                if !finite(d) || !finite(&e[..n - 1]) {
                    return Err(());
                }
                let ni = int(n)?;
                w.s.resize(4 * n, 0.0);
                let mut info = 0;
                let mut c: [$t; 0] = [];
                // SAFETY: d has n entries, e n-1, vt and u n·n (ld n), work
                // 4n; ncc = 0 so c is never read.
                unsafe {
                    lapack::$bdsqr(
                        if lower { b'L' } else { b'U' },
                        ni,
                        &[ni],
                        &[ni],
                        &[0],
                        d,
                        e,
                        vt,
                        ni,
                        u,
                        ni,
                        &mut c,
                        1,
                        &mut w.s,
                        &mut info,
                    );
                }
                if info != 0 { Err(()) } else { Ok(()) }
            }

            fn syevr(
                a: &mut [$t],
                n: usize,
                eig: &mut [$t],
                z: &mut [$t],
                w: &mut Work<$t, $t>,
            ) -> Result<(), ()> {
                assert!(
                    a.len() >= n * n && eig.len() >= n && z.len() >= n * n,
                    "syevr: buffer too small"
                );
                if n == 0 {
                    return Ok(());
                }
                if !finite(&a[..n * n]) {
                    return Err(());
                }
                let ni = int(n)?;
                w.i.resize(2 * n + 1, 0);
                let (mut m, mut info) = (0, 0);
                let (mut lw, mut liw) = ([0.0 as $t], [0i32]);
                let (isuppz, _) = w.i.split_at_mut(2 * n);
                // SAFETY: a is n·n (ld n), eig n, z n·n, isuppz 2n; the query
                // writes only lw[0] and liw[0].
                unsafe {
                    lapack::$syevr(
                        b'V', b'A', b'U', ni, a, ni, 0.0, 0.0, 0, 0, -1.0, &mut m, eig, z, ni,
                        isuppz, &mut lw, -1, &mut liw, -1, &mut info,
                    );
                }
                if info != 0 {
                    return Err(());
                }
                let (lwork, liwork) = (size_of_query(f64::from(lw[0])), liw[0].max(1) as usize);
                w.s.resize(lwork, 0.0);
                w.i.resize(2 * n + liwork, 0);
                let (isuppz, iwork) = w.i.split_at_mut(2 * n);
                // SAFETY: as above, with work/iwork of the queried sizes.
                unsafe {
                    lapack::$syevr(
                        b'V',
                        b'A',
                        b'U',
                        ni,
                        a,
                        ni,
                        0.0,
                        0.0,
                        0,
                        0,
                        -1.0,
                        &mut m,
                        eig,
                        z,
                        ni,
                        isuppz,
                        &mut w.s,
                        int(lwork)?,
                        iwork,
                        int(liwork)?,
                        &mut info,
                    );
                }
                if info != 0 || m != ni {
                    Err(())
                } else {
                    Ok(())
                }
            }

            fn heevr(
                a: &mut [Complex<$t>],
                n: usize,
                eig: &mut [$t],
                z: &mut [Complex<$t>],
                w: &mut Work<Complex<$t>, $t>,
            ) -> Result<(), ()> {
                assert!(
                    a.len() >= n * n && eig.len() >= n && z.len() >= n * n,
                    "heevr: buffer too small"
                );
                if n == 0 {
                    return Ok(());
                }
                if !a[..n * n]
                    .iter()
                    .all(|x| x.re.is_finite() && x.im.is_finite())
                {
                    return Err(());
                }
                let ni = int(n)?;
                w.i.resize(2 * n + 1, 0);
                let (mut m, mut info) = (0, 0);
                let mut lw = [Complex::new(0.0 as $t, 0.0)];
                let (mut lrw, mut liw) = ([0.0 as $t], [0i32]);
                let (isuppz, _) = w.i.split_at_mut(2 * n);
                // SAFETY: a is n·n (ld n), eig n, z n·n, isuppz 2n; the query
                // writes only lw[0], lrw[0] and liw[0].
                unsafe {
                    lapack::$heevr(
                        b'V', b'A', b'U', ni, a, ni, 0.0, 0.0, 0, 0, -1.0, &mut m, eig, z, ni,
                        isuppz, &mut lw, -1, &mut lrw, -1, &mut liw, -1, &mut info,
                    );
                }
                if info != 0 {
                    return Err(());
                }
                let lwork = size_of_query(f64::from(lw[0].re));
                let lrwork = size_of_query(f64::from(lrw[0]));
                let liwork = liw[0].max(1) as usize;
                w.s.resize(lwork, Complex::new(0.0, 0.0));
                w.r.resize(lrwork, 0.0);
                w.i.resize(2 * n + liwork, 0);
                let (isuppz, iwork) = w.i.split_at_mut(2 * n);
                // SAFETY: as above, with work/rwork/iwork of the queried sizes.
                unsafe {
                    lapack::$heevr(
                        b'V',
                        b'A',
                        b'U',
                        ni,
                        a,
                        ni,
                        0.0,
                        0.0,
                        0,
                        0,
                        -1.0,
                        &mut m,
                        eig,
                        z,
                        ni,
                        isuppz,
                        &mut w.s,
                        int(lwork)?,
                        &mut w.r,
                        int(lrwork)?,
                        iwork,
                        int(liwork)?,
                        &mut info,
                    );
                }
                if info != 0 || m != ni {
                    Err(())
                } else {
                    Ok(())
                }
            }
        }
    };
}

lapack_real!(f32, sstegr, sbdsqr, ssyevr, cheevr);
lapack_real!(f64, dstegr, dbdsqr, dsyevr, zheevr);
