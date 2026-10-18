//! Allocation contract of the Arnoldi exponential integrator: the dense
//! exponential's scratch is allocated once per call, not per step-size
//! trial (#16). A counting global allocator sees every heap allocation.

mod common;

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use common::*;
use krylov::{Arnoldi, Workspace, expintegrator};
use num_complex::Complex64;

struct Counting;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCS.try_with(|c| c.set(c.get() + 1));
        // SAFETY: forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` above, i.e. from `System`.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

fn allocations() -> usize {
    ALLOCS.with(Cell::get)
}

#[test]
fn arnoldi_allocations_do_not_grow_with_the_step_count() {
    let mut g = SplitMix(71);
    let n = 40;
    let a = outliers(&mut g, n, true);
    let u: Vec<DVec<Complex64>> = (0..3)
        .map(|_| to_dvec(&random_vec(&mut g, n, true)))
        .collect();
    let alg = Arnoldi::new(8, 100, 1e-10);
    let mut ws = Workspace::with_capacity(&u[0], 8).unwrap();
    let mut run = |t: f64| {
        let mut op = DenseOp::<Complex64>::new(&a);
        let before = allocations();
        let (_, info) = expintegrator(&mut op, Complex64::new(0.0, -t), &u, &alg, &mut ws).unwrap();
        (allocations() - before, info.numiter)
    };
    run(0.5); // warm
    let (short, it_short) = run(0.5);
    let (long, it_long) = run(12.0);
    // Many more Krylov spaces, each with at least one dense exponential.
    assert!(it_long >= 5 * it_short, "{it_short} vs {it_long}");
    assert_eq!(short, long, "allocations: {short} vs {long}");
}
