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

## Status

Pre-release. Not published on crates.io; the API may change.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
