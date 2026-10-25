//! Borrowed, sequential faer adapters for the small projected problems.
//! Scratch belongs to one solver invocation and is reused across iterations.

use crate::scalar::{RealScalar, Scalar, sealed::RealOps};
use core::marker::PhantomData;
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::{
    evd,
    lu::partial_pivoting::{factor, solve},
    svd,
};
use faer::{
    ColMut, MatMut, MatRef, Par,
    diag::{DiagMut, DiagRef},
    prelude::{Reborrow, ReborrowMut},
};
use num_complex::Complex;

/// Private dense-workspace families (the containing module is private).
#[derive(Clone, Copy)]
pub enum Problem {
    /// Tridiagonal eigendecomposition.
    Tridiag,
    /// Hermitian eigendecomposition.
    Hermitian,
    /// Hessenberg Schur, reordering and eigenvectors.
    Schur,
    /// Bidiagonal SVD.
    Bidiag,
    /// LU factorization and solve.
    Lu,
}

/// Aligned arena, including adapter temporaries. Default is useful for one-off
/// dense calls; solver owners pre-size with `new` before entering iteration.
pub struct Work<S, R> {
    mem: MemBuffer,
    req: StackReq,
    marker: PhantomData<(S, R)>,
}

impl<S, R> Default for Work<S, R> {
    fn default() -> Self {
        Self {
            mem: MemBuffer::new(StackReq::EMPTY),
            req: StackReq::EMPTY,
            marker: PhantomData,
        }
    }
}

impl<S, R> Work<S, R> {
    fn stack(&mut self, req: StackReq) -> &mut MemStack {
        let combined = StackReq::any_of(&[self.req, req]);
        if combined.size_bytes() > self.req.size_bytes()
            || combined.align_bytes() > self.req.align_bytes()
        {
            self.mem = MemBuffer::new(combined);
            self.req = combined;
        }
        MemStack::new(&mut self.mem)
    }
}

impl<S: Scalar<Real = R>, R: RealScalar> Work<S, R> {
    pub(crate) fn new(nmax: usize, problem: Problem) -> Self {
        // Query every supported size: backend recursion/block thresholds can
        // make scratch nonmonotonic. No allocations during these queries.
        let req = (0..=nmax).fold(StackReq::EMPTY, |req, n| {
            StackReq::any_of(&[req, R::work_req(n, problem, S::COMPLEX)])
        });
        Self {
            mem: MemBuffer::new(req),
            req,
            marker: PhantomData,
        }
    }
}

fn schur_req<T: faer::traits::ComplexField>(n: usize) -> StackReq {
    StackReq::any_of(&[
        StackReq::all_of(&[
            StackReq::new::<T::Real>(2 * n),
            evd::hessenberg_schur_scratch::<T>(n, true, Par::Seq, Default::default()),
        ]),
        evd::reorder_schur_scratch::<T>(n),
        evd::eigenvectors_from_schur_scratch::<T>(n),
    ])
}

fn tri_req<T: faer::traits::RealField>(n: usize) -> StackReq {
    StackReq::all_of(&[
        StackReq::new::<T>(n),
        evd::tridiagonal_self_adjoint_evd_scratch::<T>(
            n,
            evd::ComputeEigenvectors::Yes,
            Par::Seq,
            Default::default(),
        ),
    ])
}

fn heig_req<T: faer::traits::ComplexField>(n: usize) -> StackReq {
    StackReq::all_of(&[
        StackReq::new::<T>(n),
        evd::self_adjoint_evd_scratch::<T>(
            n,
            evd::ComputeEigenvectors::Yes,
            Par::Seq,
            Default::default(),
        ),
    ])
}

fn lu_req<T: faer::traits::ComplexField>(n: usize, nrhs: usize) -> StackReq {
    StackReq::all_of(&[
        StackReq::new::<usize>(n).array(2),
        StackReq::any_of(&[
            factor::lu_in_place_scratch::<usize, T>(n, n, Par::Seq, Default::default()),
            solve::solve_in_place_scratch::<usize, T>(n, nrhs, Par::Seq),
        ]),
    ])
}

fn balance<T: faer::traits::ComplexField>(
    a: &mut [T],
    n: usize,
    scale: &mut [T::Real],
    perm: &mut [usize],
) -> Result<(usize, usize), ()> {
    let mut empty = [];
    let active = evd::balance_in_place(
        MatMut::from_column_major_slice_mut(&mut a[..n * n], n, n),
        perm,
        scale,
        Par::Seq,
        MemStack::new(&mut empty),
    )
    .map_err(|_| ())?;
    Ok((active.start, active.end))
}

fn gesv<T: faer::traits::ComplexField, R>(
    a: &mut [T],
    b: &mut [T],
    n: usize,
    nrhs: usize,
    w: &mut Work<T, R>,
) -> Result<(), ()> {
    if n == 0 || nrhs == 0 {
        return Ok(());
    }
    let a = MatMut::from_column_major_slice_mut(&mut a[..n * n], n, n);
    let b = MatMut::from_column_major_slice_mut(&mut b[..n * nrhs], n, nrhs);
    if !a.rb().is_all_finite() || !b.rb().is_all_finite() {
        return Err(());
    }
    let (mut perm, stack) = w.stack(lu_req::<T>(n, nrhs)).make_with(n, |_| 0usize);
    let (mut inv, stack) = stack.make_with(n, |_| 0usize);
    let mut a = a;
    let (_, p) = factor::lu_in_place(
        a.rb_mut(),
        &mut perm,
        &mut inv,
        Par::Seq,
        stack,
        Default::default(),
    );
    if (0..n).any(|i| a[(i, i)] == faer::traits::math_utils::zero::<T>()) {
        return Err(());
    }
    solve::solve_in_place(a.rb(), a.rb(), p, b, Par::Seq, stack);
    Ok(())
}

macro_rules! backend_real {
    ($t:ty) => {
        impl RealOps for $t {
            fn work_req(n: usize, problem: Problem, complex: bool) -> StackReq {
                match problem {
                    Problem::Tridiag => tri_req::<$t>(n),
                    Problem::Hermitian => {
                        if complex {
                            heig_req::<Complex<$t>>(n)
                        } else {
                            heig_req::<$t>(n)
                        }
                    }
                    Problem::Schur => {
                        if complex {
                            schur_req::<Complex<$t>>(n)
                        } else {
                            schur_req::<$t>(n)
                        }
                    }
                    Problem::Bidiag => {
                        svd::bidiagonal_svd_scratch::<$t>(n, Par::Seq, Default::default())
                    }
                    Problem::Lu => {
                        if complex {
                            lu_req::<Complex<$t>>(n, n)
                        } else {
                            lu_req::<$t>(n, n)
                        }
                    }
                }
            }
            fn hseqr(
                h: &mut [$t],
                z: &mut [$t],
                n: usize,
                ev: &mut [Complex<$t>],
                w: &mut Work<$t, $t>,
            ) -> Result<(), ()> {
                let (mut wr, stack) = w.stack(schur_req::<$t>(n)).make_with(n, |_| 0.0);
                let (mut wi, stack) = stack.make_with(n, |_| 0.0);
                evd::hessenberg_schur_real(
                    MatMut::from_column_major_slice_mut(&mut h[..n * n], n, n),
                    Some(MatMut::from_column_major_slice_mut(&mut z[..n * n], n, n)),
                    ColMut::from_slice_mut(&mut wr),
                    ColMut::from_slice_mut(&mut wi),
                    Par::Seq,
                    stack,
                    Default::default(),
                )
                .map_err(|_| ())?;
                // The QR kernel can round wr differently from the final T diagonal
                // after deflation updates. T is authoritative for Schur ordering.
                for i in 0..n {
                    ev[i] = Complex::new(h[i + i * n], wi[i]);
                }
                Ok(())
            }
            fn chseqr(
                h: &mut [Complex<$t>],
                z: &mut [Complex<$t>],
                n: usize,
                ev: &mut [Complex<$t>],
                w: &mut Work<Complex<$t>, $t>,
            ) -> Result<(), ()> {
                evd::hessenberg_schur_cplx(
                    MatMut::from_column_major_slice_mut(&mut h[..n * n], n, n),
                    Some(MatMut::from_column_major_slice_mut(&mut z[..n * n], n, n)),
                    ColMut::from_slice_mut(&mut ev[..n]),
                    Par::Seq,
                    w.stack(schur_req::<Complex<$t>>(n)),
                    Default::default(),
                )
                .map_err(|_| ())
            }
            fn trexc(
                t: &mut [$t],
                q: &mut [$t],
                n: usize,
                ifst: usize,
                ilst: usize,
                w: &mut Work<$t, $t>,
            ) -> Result<(), ()> {
                evd::reorder_schur_real(
                    MatMut::from_column_major_slice_mut(&mut t[..n * n], n, n),
                    Some(MatMut::from_column_major_slice_mut(&mut q[..n * n], n, n)),
                    ifst,
                    ilst,
                    Par::Seq,
                    w.stack(evd::reorder_schur_scratch::<$t>(n)),
                )
                .map(|_| ())
                .map_err(|_| ())
            }
            fn ctrexc(
                t: &mut [Complex<$t>],
                q: &mut [Complex<$t>],
                n: usize,
                ifst: usize,
                ilst: usize,
            ) -> Result<(), ()> {
                // The complex reorder operation needs no scratch.
                let mut empty = [];
                evd::reorder_schur_cplx(
                    MatMut::from_column_major_slice_mut(&mut t[..n * n], n, n),
                    Some(MatMut::from_column_major_slice_mut(&mut q[..n * n], n, n)),
                    ifst,
                    ilst,
                    Par::Seq,
                    MemStack::new(&mut empty),
                );
                Ok(())
            }
            fn trevc(t: &[$t], n: usize, vr: &mut [$t], w: &mut Work<$t, $t>) -> Result<(), ()> {
                evd::eigenvectors_from_schur_real(
                    MatRef::from_column_major_slice(&t[..n * n], n, n),
                    None,
                    MatMut::from_column_major_slice_mut(&mut vr[..n * n], n, n),
                    Par::Seq,
                    w.stack(evd::eigenvectors_from_schur_scratch::<$t>(n)),
                    Default::default(),
                )
                .map_err(|_| ())
            }
            fn ctrevc(
                t: &mut [Complex<$t>],
                n: usize,
                vr: &mut [Complex<$t>],
                w: &mut Work<Complex<$t>, $t>,
            ) -> Result<(), ()> {
                evd::eigenvectors_from_schur_cplx(
                    MatRef::from_column_major_slice(&t[..n * n], n, n),
                    None,
                    MatMut::from_column_major_slice_mut(&mut vr[..n * n], n, n),
                    Par::Seq,
                    w.stack(evd::eigenvectors_from_schur_scratch::<Complex<$t>>(n)),
                    Default::default(),
                )
                .map_err(|_| ())
            }
            fn gebal_r(
                a: &mut [$t],
                n: usize,
                scale: &mut [$t],
                perm: &mut [usize],
            ) -> Result<(usize, usize), ()> {
                balance(a, n, scale, perm)
            }
            fn gebal_c(
                a: &mut [Complex<$t>],
                n: usize,
                scale: &mut [$t],
                perm: &mut [usize],
            ) -> Result<(usize, usize), ()> {
                balance(a, n, scale, perm)
            }
            fn gesv_r(
                a: &mut [$t],
                b: &mut [$t],
                n: usize,
                nrhs: usize,
                w: &mut Work<$t, $t>,
            ) -> Result<(), ()> {
                gesv(a, b, n, nrhs, w)
            }
            fn gesv_c(
                a: &mut [Complex<$t>],
                b: &mut [Complex<$t>],
                n: usize,
                nrhs: usize,
                w: &mut Work<Complex<$t>, $t>,
            ) -> Result<(), ()> {
                gesv(a, b, n, nrhs, w)
            }
            fn stegr(
                d: &mut [$t],
                e: &mut [$t],
                z: &mut [$t],
                w: &mut Work<$t, $t>,
            ) -> Result<(), ()> {
                let n = d.len();
                if n == 0 {
                    return Ok(());
                }
                if !d.iter().chain(&e[..n - 1]).all(|x| x.is_finite()) {
                    return Err(());
                }
                let (mut eig, stack) = w.stack(tri_req::<$t>(n)).make_with(n, |_| 0.0);
                evd::tridiagonal_self_adjoint_evd(
                    DiagRef::from_slice(d),
                    DiagRef::from_slice(&e[..n - 1]),
                    DiagMut::from_slice_mut(&mut eig),
                    Some(MatMut::from_column_major_slice_mut(&mut z[..n * n], n, n)),
                    Par::Seq,
                    stack,
                    Default::default(),
                )
                .map_err(|_| ())?;
                d.copy_from_slice(&eig);
                Ok(())
            }
            fn bdsqr(
                lower: bool,
                d: &mut [$t],
                e: &mut [$t],
                v: &mut [$t],
                u: &mut [$t],
                w: &mut Work<$t, $t>,
            ) -> Result<(), ()> {
                let n = d.len();
                svd::bidiagonal_svd(
                    d,
                    &e[..n.saturating_sub(1)],
                    !lower,
                    MatMut::from_column_major_slice_mut(&mut u[..n * n], n, n),
                    MatMut::from_column_major_slice_mut(&mut v[..n * n], n, n),
                    Par::Seq,
                    w.stack(svd::bidiagonal_svd_scratch::<$t>(
                        n,
                        Par::Seq,
                        Default::default(),
                    )),
                    Default::default(),
                )
                .map_err(|_| ())
            }
            fn syevr(
                a: &mut [$t],
                n: usize,
                eig: &mut [$t],
                z: &mut [$t],
                w: &mut Work<$t, $t>,
            ) -> Result<(), ()> {
                if !a[..n * n].iter().all(|x| x.is_finite()) {
                    return Err(());
                }
                for j in 0..n {
                    for i in j + 1..n {
                        a[i + j * n] = a[j + i * n];
                    }
                }
                evd::self_adjoint_evd(
                    MatRef::from_column_major_slice(&a[..n * n], n, n),
                    DiagMut::from_slice_mut(&mut eig[..n]),
                    Some(MatMut::from_column_major_slice_mut(&mut z[..n * n], n, n)),
                    Par::Seq,
                    w.stack(heig_req::<$t>(n)),
                    Default::default(),
                )
                .map_err(|_| ())
            }
            fn heevr(
                a: &mut [Complex<$t>],
                n: usize,
                eig: &mut [$t],
                z: &mut [Complex<$t>],
                w: &mut Work<Complex<$t>, $t>,
            ) -> Result<(), ()> {
                if !a[..n * n]
                    .iter()
                    .all(|x| x.re.is_finite() && x.im.is_finite())
                {
                    return Err(());
                }
                for j in 0..n {
                    for i in j + 1..n {
                        a[i + j * n] = a[j + i * n].conj();
                    }
                }
                let (mut values, stack) = w
                    .stack(heig_req::<Complex<$t>>(n))
                    .make_with(n, |_| Complex::new(0.0, 0.0));
                evd::self_adjoint_evd(
                    MatRef::from_column_major_slice(&a[..n * n], n, n),
                    DiagMut::from_slice_mut(&mut values),
                    Some(MatMut::from_column_major_slice_mut(&mut z[..n * n], n, n)),
                    Par::Seq,
                    stack,
                    Default::default(),
                )
                .map_err(|_| ())?;
                for i in 0..n {
                    eig[i] = values[i].re;
                }
                Ok(())
            }
        }
    };
}
backend_real!(f32);
backend_real!(f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::sealed::Ops;
    use nalgebra::{DMatrix, DVector};
    use num_complex::Complex64 as C;

    use std::{
        alloc::{GlobalAlloc, Layout, System},
        cell::Cell,
    };
    struct Counting;
    thread_local! { static ALLOCS: Cell<usize> = const { Cell::new(0) }; }
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
            // SAFETY: the allocation request is forwarded unchanged.
            unsafe { System.alloc(layout) }
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: this allocation was obtained from System above.
            unsafe { System.dealloc(ptr, layout) }
        }
    }
    #[global_allocator]
    static GLOBAL: Counting = Counting;
    fn allocations() -> usize {
        ALLOCS.with(Cell::get)
    }

    #[test]
    fn tridiagonal_closed_form_and_negative_controls() {
        let a = DMatrix::from_row_slice(3, 3, &[2., -1., 0., -1., 2., -1., 0., -1., 2.]);
        let mut d = [2.; 3];
        let mut z = [f64::NAN; 9];
        f64::stegr(&mut d, &mut [-1., -1., 0.], &mut z, &mut Work::default()).unwrap();
        for (got, want) in d.iter().zip([2. - 2f64.sqrt(), 2., 2. + 2f64.sqrt()]) {
            assert!((got - want).abs() < 2e-15);
        }
        let z = DMatrix::from_column_slice(3, 3, &z);
        let lam = DMatrix::from_diagonal(&DVector::from_column_slice(&d));
        assert!((&a * &z - &z * &lam).norm() < 3e-15);
        assert!((z.transpose() * &z - DMatrix::identity(3, 3)).norm() < 2e-15);
        let mut wrong = z.clone();
        wrong.swap_columns(0, 2);
        assert!((&a * &wrong - &wrong * &lam).norm() > 1.0);
        assert!(
            f64::stegr(
                &mut [2., f64::NAN],
                &mut [1., 0.],
                &mut [0.; 4],
                &mut Work::default()
            )
            .is_err()
        );
    }

    #[test]
    #[ignore = "upstream: t4a-faer tridiagonal range safety"]
    fn tridiagonal_range_overflow() {
        let mut d = [-1e308, 1e308];
        let result = f64::stegr(&mut d, &mut [0.; 2], &mut [0.; 4], &mut Work::default());
        assert_eq!(result, Ok(()));
        assert_eq!(d, [-1e308, 1e308], "finite exact eigenvalues; got {d:?}");
    }

    #[test]
    #[ignore = "upstream: t4a-faer tridiagonal range safety"]
    fn tridiagonal_range_subnormal() {
        let s = 1e-310;
        let mut d = [2. * s; 3];
        let result = f64::stegr(
            &mut d,
            &mut [-s, -s, 0.],
            &mut [0.; 9],
            &mut Work::default(),
        );
        assert_eq!(result, Ok(()));
        for (got, want) in d.iter().zip([2. - 2f64.sqrt(), 2., 2. + 2f64.sqrt()]) {
            assert!(
                (got / s - want).abs() < 1e-12,
                "representable eigenvalue {want}*s; got {d:?}"
            );
        }
    }

    #[test]
    fn hermitian_reads_upper_with_conjugation() {
        let a = DMatrix::from_row_slice(
            2,
            2,
            &[
                C::new(2., 0.),
                C::new(1., 2.),
                C::new(1., -2.),
                C::new(5., 0.),
            ],
        );
        let mut input = a.as_slice().to_vec();
        input[1] = C::new(90., 70.);
        let mut d = [0.; 2];
        let mut z = [C::new(f64::NAN, 0.); 4];
        C::heig(&mut input, 2, &mut d, &mut z, &mut Work::default()).unwrap();
        for (got, want) in d
            .iter()
            .zip([(7. - 29f64.sqrt()) / 2., (7. + 29f64.sqrt()) / 2.])
        {
            assert!((got - want).abs() < 2e-15);
        }
        let z = DMatrix::from_column_slice(2, 2, &z);
        let lam = DMatrix::from_diagonal(&DVector::from_iterator(2, d.map(|x| C::new(x, 0.))));
        assert!((&a * &z - &z * &lam).norm() < 4e-15);
        assert!((z.adjoint() * &z - DMatrix::identity(2, 2)).norm() < 2e-15);
        let mut wrong = a.clone();
        wrong[(0, 1)] = C::new(90., -70.);
        wrong[(1, 0)] = C::new(90., 70.);
        assert!((&wrong * &z - &z * &lam).norm() > 1.0); // lower-triangle authority
        assert!((a.transpose() * &z - &z * &lam).norm() > 1.0); // missing conjugation
    }

    #[test]
    fn hessenberg_accumulates_initial_z_and_orders_pairs() {
        let h = DMatrix::from_row_slice(2, 2, &[1., 2., -2., 1.]);
        let z0 = DMatrix::from_row_slice(2, 2, &[0., 1., 1., 0.]);
        let mut t = h.as_slice().to_vec();
        let mut z = z0.as_slice().to_vec();
        let mut ev = [C::default(); 2];
        f64::hseqr_raw(&mut t, &mut z, 2, &mut ev, &mut Work::default()).unwrap();
        let t = DMatrix::from_column_slice(2, 2, &t);
        let z = DMatrix::from_column_slice(2, 2, &z);
        let original = &z0 * &h * z0.transpose();
        assert!((&z * &t * z.transpose() - &original).norm() < 1e-14);
        assert!((ev[0] - C::new(1., 2.)).norm() < 1e-14);
        assert_eq!(ev[1], ev[0].conj());
        assert!((&t - &original).norm() > 1.0); // ignoring Z0
        let mut reversed = ev;
        reversed.reverse();
        assert_ne!(reversed, ev);
        let hc = h.map(|x| C::new(x, 0.));
        let zc = z0.map(|x| C::new(x, 0.));
        let mut t = hc.as_slice().to_vec();
        let mut z = zc.as_slice().to_vec();
        C::hseqr_raw(&mut t, &mut z, 2, &mut ev, &mut Work::default()).unwrap();
        let t = DMatrix::from_column_slice(2, 2, &t);
        let z = DMatrix::from_column_slice(2, 2, &z);
        assert!((&z * &t * z.adjoint() - &zc * &hc * zc.adjoint()).norm() < 1e-14);
        let q = zc.adjoint() * z;
        assert!((&q * &t * q.adjoint() - &zc * &hc * zc.adjoint()).norm() > 1.0); // ignoring Z0
    }

    #[test]
    fn schur_moves_mixed_blocks_both_directions_and_rejects_split_pairs() {
        use crate::dense::{permute_schur, schur_eigvals};
        let a = DMatrix::from_row_slice(
            4,
            4,
            &[
                5., 0.2, 0.3, 0.4, 0., 1., 2., 0.5, 0., -2., 1., 0.6, 0., 0., 0., 9.,
            ],
        );
        let mut t = a.as_slice().to_vec();
        let mut q = DMatrix::<f64>::identity(4, 4).as_slice().to_vec();
        let mut w = Work::new(4, Problem::Schur);
        <f64 as Ops<f64>>::trexc(&mut t, &mut q, 4, 1, 3, &mut w).unwrap();
        let mut ev = Vec::new();
        schur_eigvals(&t, 4, 4, &mut ev);
        for (got, want) in ev.iter().zip([5., 9., 1., 1.]) {
            assert!((got.re - want).abs() < 1e-14);
        }
        <f64 as Ops<f64>>::trexc(&mut t, &mut q, 4, 3, 0, &mut w).unwrap(); // second row denotes pair start
        schur_eigvals(&t, 4, 4, &mut ev);
        for (got, want) in ev.iter().zip([1., 1., 5., 9.]) {
            assert!((got.re - want).abs() < 1e-14);
        }
        let tm = DMatrix::from_column_slice(4, 4, &t);
        let qm = DMatrix::from_column_slice(4, 4, &q);
        assert!((&qm * tm * qm.transpose() - a).norm() < 1e-13);
        assert!(permute_schur(&mut t, &mut q, 4, &[0, 2, 1, 3], &mut Vec::new(), &mut w).is_err());
        // Complex moves in both directions preserve a genuinely complex similarity.
        let a = DMatrix::from_row_slice(
            3,
            3,
            &[
                C::new(1., 2.),
                C::new(2., 1.),
                C::new(-1., 1.),
                C::default(),
                C::new(3., -1.),
                C::new(4., 2.),
                C::default(),
                C::default(),
                C::new(5., 3.),
            ],
        );
        let mut t = a.as_slice().to_vec();
        let mut q = DMatrix::<C>::identity(3, 3).as_slice().to_vec();
        let mut w = Work::default();
        C::trexc(&mut t, &mut q, 3, 0, 2, &mut w).unwrap();
        assert_eq!(t[8], a[(0, 0)]);
        C::trexc(&mut t, &mut q, 3, 2, 0, &mut w).unwrap();
        assert_eq!(t[0], a[(0, 0)]);
        let tm = DMatrix::from_column_slice(3, 3, &t);
        let qm = DMatrix::from_column_slice(3, 3, &q);
        assert!((&qm * tm * qm.adjoint() - a).norm() < 1e-13);
    }

    #[test]
    fn rejected_schur_swap_keeps_earlier_successful_swaps() {
        // Inline upstream public_schur_reorder.rs example (t4a-faer 0.24.9).
        let (s, m, l) = (2f64.powi(-100), 2f64.powi(-50), 2f64.powi(100));
        let pairs = DMatrix::from_row_slice(
            4,
            4,
            &[
                0.,
                s,
                l,
                l * 0.3,
                -1. / s,
                0.,
                -l * 0.7,
                l * 0.4,
                0.,
                0.,
                0.,
                m,
                0.,
                0.,
                -1. / m,
                0.,
            ],
        );
        let a = DMatrix::from_fn(5, 5, |i, j| {
            if i == 2 || j == 2 {
                if i == j { 5. } else { 0. }
            } else {
                pairs[(if i > 2 { i - 1 } else { i }, if j > 2 { j - 1 } else { j })]
            }
        });
        let mut t = a.as_slice().to_vec();
        let mut q = DMatrix::<f64>::identity(5, 5).as_slice().to_vec();
        let (mut expected, mut eq) = (t.clone(), q.clone());
        let mut w = Work::default();
        <f64 as Ops<f64>>::trexc(&mut expected, &mut eq, 5, 0, 2, &mut w).unwrap();
        assert_ne!(expected, t); // negative control: reverting all swaps is wrong
        assert!(<f64 as Ops<f64>>::trexc(&mut t, &mut q, 5, 0, 3, &mut w).is_err());
        assert_eq!(t, expected);
        assert_eq!(q, eq);
        let tm = DMatrix::from_column_slice(5, 5, &t);
        let qm = DMatrix::from_column_slice(5, 5, &q);
        assert!((&qm * (tm / l) * qm.transpose() - a / l).norm() < 1e-13);
    }

    #[test]
    fn schur_right_eigenvectors_overwrite_poison_and_preserve_t() {
        let a = DMatrix::from_row_slice(2, 2, &[1., 2., 0., 3.]);
        let mut t = a.as_slice().to_vec();
        let mut v = [f64::NAN; 4];
        <f64 as Ops<f64>>::trevc(&mut t, 2, &mut v, &mut Work::default()).unwrap();
        assert_eq!(t, a.as_slice());
        let v = DMatrix::from_column_slice(2, 2, &v);
        let d = DMatrix::from_diagonal(&DVector::from_vec(vec![1., 3.]));
        assert!((&a * &v - &v * &d).norm() < 1e-14);
        assert!((a.transpose() * &v - &v * &d).norm() > 1.0); // left vectors
        for col in v.column_iter() {
            assert!((col.norm() - 1.).abs() < 1e-14);
        }
        let a = DMatrix::from_row_slice(
            2,
            2,
            &[
                C::new(1., 1.),
                C::new(2., 3.),
                C::default(),
                C::new(4., -2.),
            ],
        );
        let mut t = a.as_slice().to_vec();
        let mut v = [C::new(f64::NAN, f64::NAN); 4];
        C::trevc(&mut t, 2, &mut v, &mut Work::default()).unwrap();
        assert_eq!(t, a.as_slice());
        let v = DMatrix::from_column_slice(2, 2, &v);
        let d = DMatrix::from_diagonal(&a.diagonal());
        assert!((&a * &v - &v * &d).norm() < 1e-14);
        assert!((&a * v.conjugate() - v.conjugate() * &d).norm() > 1.0);
        assert!((a.adjoint() * &v - &v * &d).norm() > 1.0);
        for col in v.column_iter() {
            assert!((col.norm() - 1.).abs() < 1e-14);
        }
        assert!(
            C::trevc(
                &mut [C::new(f64::NAN, 0.)],
                1,
                &mut [C::default()],
                &mut Work::default()
            )
            .is_err()
        );
        assert!(
            <f64 as Ops<f64>>::trevc(&mut [f64::INFINITY], 1, &mut [0.], &mut Work::default())
                .is_err()
        );
    }

    #[test]
    fn lu_pivoting_multiple_rhs_and_negative_controls() {
        let a = DMatrix::from_row_slice(
            3,
            3,
            &[
                C::default(),
                C::new(1., 2.),
                C::new(2., 0.),
                C::new(1., 0.),
                C::new(3., -1.),
                C::new(1., 0.),
                C::new(4., 1.),
                C::new(1., 0.),
                C::new(2., -1.),
            ],
        );
        let exact = DMatrix::from_row_slice(
            3,
            2,
            &[
                C::new(1., 2.),
                C::new(-1., 0.),
                C::new(2., -1.),
                C::new(3., 1.),
                C::new(1., 1.),
                C::new(2., -2.),
            ],
        );
        let b = &a * &exact;
        let mut lu = a.as_slice().to_vec();
        let mut x = b.as_slice().to_vec();
        C::gesv(&mut lu, &mut x, 3, 2, &mut Work::default()).unwrap();
        let x = DMatrix::from_column_slice(3, 2, &x);
        assert!((&x - &exact).norm() < 1e-13);
        assert!((&a * &x - &b).norm() < 1e-13);
        assert!((a.transpose() * &x - &b).norm() > 1.0);
        let mut wrong = x;
        wrong.swap_rows(0, 2);
        assert!((&a * wrong - &b).norm() > 1.0);
        assert!(
            f64::gesv(
                &mut [1., 2., 2., 4.],
                &mut [1., 1.],
                2,
                1,
                &mut Work::default()
            )
            .is_err()
        );
        assert!(
            C::gesv(
                &mut [C::new(f64::INFINITY, 0.)],
                &mut [C::new(1., 0.)],
                1,
                1,
                &mut Work::default()
            )
            .is_err()
        );
    }

    #[test]
    fn bidiagonal_orientation_signs_and_v_columns() {
        for lower in [false, true] {
            let a = DMatrix::from_row_slice(
                2,
                2,
                if lower {
                    &[-3., 0., 4., 0.]
                } else {
                    &[-3., 4., 0., 0.]
                },
            );
            let mut d = [-3., 0.];
            let mut u = [f64::NAN; 4];
            let mut v = [f64::NAN; 4];
            f64::bdsqr(
                lower,
                &mut d,
                &mut [4.],
                &mut v,
                &mut u,
                &mut Work::default(),
            )
            .unwrap();
            assert!((d[0] - 5.).abs() < 1e-14);
            assert_eq!(d[1], 0.);
            let u = DMatrix::from_column_slice(2, 2, &u);
            let v = DMatrix::from_column_slice(2, 2, &v);
            let s = DMatrix::from_diagonal(&DVector::from_column_slice(&d));
            let back = &u * &s * v.transpose();
            assert!((&back - &a).norm() < 1e-14);
            assert!((&back - a.transpose()).norm() > 1.0);
            assert!((u.transpose() * &u - DMatrix::identity(2, 2)).norm() < 1e-14);
            assert!((v.transpose() * &v - DMatrix::identity(2, 2)).norm() < 1e-14);
            let mut wrong = u;
            wrong.column_mut(0).scale_mut(-1.);
            assert!((wrong * &s * v.transpose() - &a).norm() > 1.0);
        }
        // For the rank-one 2x2 example one null column can make V symmetric;
        // a nonsymmetric full-rank 3x3 factor detects interpreting V as Vᵀ.
        let a = DMatrix::from_row_slice(3, 3, &[-3., 4., 0., 0., 2., 1., 0., 0., 1.]);
        let mut d = [-3., 2., 1.];
        let mut u = [0.; 9];
        let mut v = [0.; 9];
        f64::bdsqr(
            false,
            &mut d,
            &mut [4., 1.],
            &mut v,
            &mut u,
            &mut Work::default(),
        )
        .unwrap();
        let u = DMatrix::from_column_slice(3, 3, &u);
        let v = DMatrix::from_column_slice(3, 3, &v);
        let s = DMatrix::from_diagonal(&DVector::from_column_slice(&d));
        assert!((&u * &s * v.transpose() - &a).norm() < 1e-13);
        assert!((&u * &s * v - &a).norm() > 1.0);
    }

    #[test]
    fn scratch_reuse_across_recursion_thresholds_and_empty_singleton() {
        let mut tri = Work::<f64, f64>::new(129, Problem::Tridiag);
        let mut heig = Work::<C, f64>::new(129, Problem::Hermitian);
        let mut svd = Work::<f64, f64>::new(129, Problem::Bidiag);
        let pointers = (tri.mem.as_ptr(), heig.mem.as_ptr(), svd.mem.as_ptr());
        let mut observed = Vec::new();
        for n in [0, 1, 127, 128, 129, 128, 1, 127] {
            let mut d = vec![2.; n];
            let mut e = vec![0.; n];
            let mut z = vec![0.; n * n];
            f64::stegr(&mut d, &mut e, &mut z, &mut tri).unwrap(); // warm
            let before = allocations();
            f64::stegr(&mut d, &mut e, &mut z, &mut tri).unwrap();
            let tri_allocs = allocations() - before;
            assert!(d.iter().all(|x| *x == 2.));
            let mut u = vec![0.; n * n];
            f64::bdsqr(false, &mut d, &mut e, &mut z, &mut u, &mut svd).unwrap(); // warm
            let before = allocations();
            f64::bdsqr(false, &mut d, &mut e, &mut z, &mut u, &mut svd).unwrap();
            let svd_allocs = allocations() - before;
            assert!(d.iter().all(|x| *x == 2.));
            let mut a = vec![C::default(); n * n];
            let mut zc = a.clone();
            for i in 0..n {
                a[i + i * n] = C::new(2., 0.);
            }
            C::heig(&mut a, n, &mut d, &mut zc, &mut heig).unwrap(); // warm backend dispatch/cache
            let before = allocations();
            C::heig(&mut a, n, &mut d, &mut zc, &mut heig).unwrap();
            let heig_allocs = allocations() - before;
            observed.push((n, tri_allocs, svd_allocs, heig_allocs));
            assert!(d.iter().all(|x| *x == 2.));
            assert_eq!(
                pointers,
                (tri.mem.as_ptr(), heig.mem.as_ptr(), svd.mem.as_ptr())
            );
        }
        // Negative control: a fresh simultaneously live arena has new storage.
        let before = allocations();
        let fresh = Work::<f64, f64>::new(129, Problem::Tridiag);
        assert!(allocations() > before);
        assert_ne!(fresh.mem.as_ptr(), tri.mem.as_ptr());
        let mut ev = [];
        f64::hseqr_raw(&mut [], &mut [], 0, &mut ev, &mut Work::default()).unwrap();
        let mut ev = [C::default()];
        C::hseqr_raw(
            &mut [C::new(3., 2.)],
            &mut [C::new(1., 0.)],
            1,
            &mut ev,
            &mut Work::default(),
        )
        .unwrap();
        assert_eq!(ev[0], C::new(3., 2.));
        assert!(
            observed
                .iter()
                .all(|&(_, t, s, h)| t == 0 && s == 0 && h == 0),
            "warm allocations (n, tridiagonal, bidiagonal, Hermitian): {observed:?}"
        );
    }
}
