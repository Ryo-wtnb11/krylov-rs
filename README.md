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

## Stage 2 contents

- `expintegrator(op, t, u, &Lanczos, ws)`: `Σ_j tʲ·φ_j(t·A)·u_j` for
  `u = [u₀, u₁, …]`, the solution at `t` of `ẋ = A·x + Σ_j tʲ/j!·u_{j+1}`,
  `x(0) = u₀`. `exponentiate` is its single-vector case; a call allocates the
  result plus `len(u) - 2` scratch vectors (none for one or two vectors).
- `linsolve(op, b, x0, &alg, a0, a1, ws)`: `(a₀ + a₁·A)·x = b` by `Cg`
  (Hermitian positive definite, real `a₀`, `a₁`) or restarted `Gmres`
  (basis capacity `>= krylovdim`), selected through the sealed `LinearSolver`
  trait. `tol` is absolute and met when `normres < tol`. A call allocates the
  solution and its residual.
- `block_eigsolve(op, x0: &[V], howmany, which, &BlockLanczos, ws)`:
  thick-restart block Lanczos for Hermitian maps with repeated eigenvalues;
  a start block of size `b` resolves multiplicities up to `b`. The basis needs
  capacity `krylovdim + b - 1`; a call allocates `2b` scratch vectors plus the
  results.
- `svdsolve(op, x0, howmany, which, &Gkl, ws_u, ws_v)`: largest
  (`LargestReal`) or smallest (`SmallestReal`) singular triplets of an
  `AdjointMap` by thick-restart Golub–Kahan–Lanczos. `x0` and `ws_u` live in
  the codomain, `ws_v` in the domain (one Rust vector type, possibly different
  spaces). A call allocates the `3·n` returned vectors.

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
  `e_kᵀ φ_{p+1}(sΔτ·T) e₁` uses the Taylor series in `T` while
  `|sΔτ|·‖T‖∞ < k/2`, because for small steps that entry is far below
  `ε·max|φ_{p+1}|`, which is all an eigendecomposition resolves; this keeps the
  step sequence equal to KrylovKit's.
- **CGS2 Lanczos step** orthogonalizes against the full basis twice; KrylovKit's
  CGS2 recurrence uses the three-term recurrence for its first pass.
- Restricted inputs, reported as typed errors: `krylovdim >= 1`,
  `maxiter >= 1`, `tol >= 0` (`> 0` for `exponentiate`), finite `t`. With
  `krylovdim == 1`, `eigsolve` cannot restart and returns the first Rayleigh
  quotient (KrylovKit fails there). `exponentiate`/`expintegrator` with
  `t == 0` return `u₀` without applying the map; `t = ±∞` is rejected
  (KrylovKit allows it for `p = 1`); an empty `u` is an `InvalidArgument`.
- **`expintegrator` with `w_{p+1} = 0` and `p ≥ 2`** (the vector the Krylov
  space would start from) takes the exact polynomial step
  `y + Σ_{j<p} (sΔτ)ʲ/j!·w_{j+1}` to `t` without a Krylov space; KrylovKit
  divides by the zero norm there.
- **`linsolve`**: GMRES forms the restart residual from the dense Givens
  product (no basis rotation, capacity `krylovdim` instead of `+ 1`).
  `numops` counts every application, including CG's residual recompute and
  the apply that starts each GMRES cycle, which KrylovKit omits. CG checks
  `maxiter` after every iteration (KrylovKit always runs two) and stops on an
  exactly zero residual. Non-positive curvature in CG is
  `Error::NotPositiveDefinite` and an exactly singular GMRES pivot is
  `Error::SingularKrylovSpace` (KrylovKit: NaN and `SingularException`).
- **`block_eigsolve`** orthogonalizes every new vector against the full basis
  (CGS2 by default, or MGS2) instead of KrylovKit's three-term block step plus
  one MGS pass (KrylovKit accepts only MGS2), and skips KrylovKit's
  type-probe application, so `numops` is one lower (lower by
  `1 + dropped` when the QR drops start vectors, since KrylovKit counts the
  input block size). Diagonal blocks are
  symmetrized, `(M + Mᴴ)/2`, where KrylovKit reads the upper triangle; the
  discarded part is reported in `max_imag`. The small Hermitian eigenproblem
  uses Householder tridiagonalization and the QL solver instead of LAPACK.
  When every vector of a new block falls below `qr_tol` the iteration stops
  as an invariant subspace (KrylovKit throws `ArgumentError("blocklength
  must be >(0)")` there). After a
  restart the residual norm is that of the transformed residual block
  (KrylovKit keeps the pre-restart norm). Exactly zero vectors are dropped
  even with `qr_tol == 0`. Typed errors: an empty block or
  `krylovdim <= block size` (every restart would be a no-op) is an
  `InvalidArgument`, a block entirely below `qr_tol` is `ZeroStartVector`.
- **Complex Householder pivots** are formed scaled by `1/ν`, so complex
  operators reach the same range limits as real ones (entries whose squares
  stay finite).
- **`svdsolve`** orthogonalizes each new vector twice against the full basis
  on both sides (six reductions per step; KrylovKit's CGS2 GKL keeps the
  right side local and makes one pass on the left). When `(I - VVᴴ)Aᴴu`
  drops to `tol` the iteration stops with the exact triplets of the
  invariant pair (`(k+1) × k` bidiagonal); KrylovKit divides by that norm.
  An invariant subspace smaller than `howmany` returns its triplets
  (KrylovKit fails indexing). The bidiagonal SVD is an in-crate implicit QR
  (Golub–Kahan with Wilkinson shifts) instead of LAPACK `bdsqr`.
  `SmallestReal` on a tall map is not supported in substance: the codomain
  process targets the zero eigenvalues of `AAᴴ`, which are reported as an
  unconverged `σ ≈ 0`; use the adjoint map. (KrylovKit reports such a zero
  as converged although its true residual is `O(σ_min)`.)

## Vector contract and distributed use

- `Vector::axpby(a, x, b)` with `b == 0` must overwrite `self` without reading
  it (BLAS `beta = 0`); krylov reuses workspace vectors with stale contents.
- A `Workspace::with_capacity(x, k)` holds `k + 3` vectors: `k` basis vectors,
  one scratch vector for the in-place restart rotation, and the operator input
  and output.
- Distributed vectors: the results of `inner`, `norm` and `Basis::project` must
  be bitwise identical on every rank, all ranks must run the same binary with
  the same libm, and errors must be returned on every rank. Under this contract
  every branch is deterministic. `eigsolve` and the tridiagonal eigensolver
  branch only on arithmetic and `sqrt`; `exponentiate`'s step acceptance also
  branches on libm `exp`, `ln` and `pow`.

## Status

Pre-release. Not published on crates.io; the API may change.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
