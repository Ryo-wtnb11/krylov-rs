# krylov

Krylov subspace algorithms for Rust over **user-defined vector types**.

`krylov` owns only the procedure: iteration control, orthogonalization, the
small projected (dense, k×k) problem, convergence tests and restarts. Large
vector arithmetic and the linear map stay with the caller, behind three traits
(`Vector`, `Basis`, `LinearMap`). The same algorithms therefore run on plain
`Vec<f64>`, on symmetric tensors, or on device/distributed storage, with static
dispatch and no knowledge of where the data lives.

## Credit

The API, the algorithms and their result semantics are modelled on
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) by Jutho Haegeman and
contributors (reference revision: v0.10.4), and the vector interface on
[VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl). Entry points
keep KrylovKit's names (`eigsolve`, `exponentiate`, …) and its conventions:
`tol` is an absolute residual norm (per unit time for `exponentiate`),
converged eigenvalues are counted from the front, more or fewer than
`howmany` values may be returned, and non-convergence is reported through the
convergence information rather than as an error.

## Example

```rust
use krylov::{eigsolve, exponentiate, Error, Lanczos, LinearMap, Vector, Which, Workspace};

fn evolve_ground_state<V: Vector<Scalar = f64>>(
    h: &mut impl LinearMap<V>,
    x0: &V,
) -> Result<V, Error<V::Error>> {
    let alg = Lanczos::new().krylovdim(30).tol(1e-12);
    // The only basis allocations: done once, reused while the space is fixed.
    let mut ws = Workspace::with_capacity(x0, alg.get_krylovdim()).map_err(Error::Vector)?;
    let (_energies, mut states, info) = eigsolve(h, x0, 1, Which::SmallestReal, &alg, &mut ws)?;
    assert!(info.converged >= 1);
    // exp(-0.1·H)·ψ with the same workspace.
    let (y, _) = exponentiate(h, -0.1, &states.swap_remove(0), &alg, &mut ws)?;
    Ok(y)
}
```

A runnable version with a `Vec<f64>` vector type is in the crate
documentation (`src/lib.rs`).

## Stage 1 contents

- Traits `Vector`, `Basis`, `LinearMap` (also implemented for
  `FnMut(&V, &mut V) -> Result<(), V::Error>`), the `Workspace`, and the default
  `VecBasis` (in-place Householder `rotate` with one scratch vector).
- `eigsolve` with `Lanczos`: thick restart, `eager`, `Which::{LargestMagnitude,
  LargestReal, SmallestReal}`, absolute residual `tol`, `ConvergenceInfo`
  (including the largest discarded `Im α`).
- `exponentiate` with `Lanczos`: KrylovKit's `expintegrator` formulation and
  time-step controller (`tol` per unit time, `maxiter` Krylov subspaces).

## Deviations from KrylovKit

- **Orthogonalization default is CGS2** (classical Gram–Schmidt with one
  reorthogonalization), not MGS2: three reductions per iteration (two
  projections and one norm) instead of a number proportional to the Krylov
  dimension, which matters on devices and across MPI ranks. MGS2 is available.
- **Preallocated workspace.** The caller creates a `Workspace` once; the
  iteration and its restarts allocate no vectors. KrylovKit allocates new
  vectors per expansion and per restart.
- **No internal threading and no global state.** Parallelism belongs to the
  vector and map implementations.
- **Fallible vector operations.** Every vector operation returns
  `Result<_, V::Error>`, and errors are propagated as `krylov::Error::Vector`.
- **No random restart.** Reaching an invariant subspace stops the iteration.
  The caller always supplies the start vector.
- **Small dense problems are computed in the crate.** The tridiagonal
  eigenproblem uses implicit QL instead of LAPACK `stegr`; `exp`/`φ` of the
  tridiagonal Rayleigh quotient use its eigendecomposition instead of a Padé
  exponential of the augmented matrix. The step controller's error estimate
  `e_kᵀ φ₂(sΔτ·T) e₁` uses the Taylor series in `T` while `|sΔτ|·‖T‖∞ < k/2`,
  because for small steps that entry is far below `ε·max|φ₂|`, which is all an
  eigendecomposition resolves; this keeps the step sequence equal to KrylovKit's.
- **CGS2 Lanczos step** orthogonalizes against the full basis twice; KrylovKit's
  CGS2 recurrence uses the three-term recurrence for its first pass.
- Restricted inputs, reported as typed errors: `krylovdim >= 1`,
  `maxiter >= 1`, `tol >= 0` (`> 0` for `exponentiate`), finite `t`. With
  `krylovdim == 1`, `eigsolve` cannot restart and returns the first Rayleigh
  quotient (KrylovKit fails there). `exponentiate` with `t == 0` returns `x`
  without applying the map.

## Status

Pre-release. Not published on crates.io; the API may change.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
