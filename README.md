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
    let alg = Lanczos::new(30, 100, 1e-12);
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

## Parameters

The problem-dependent numbers are required constructor arguments, with no
defaults: `Lanczos::new(krylovdim, maxiter, tol)`, likewise `Arnoldi`, `Gkl`
and `Gmres`, `BlockLanczos::new(krylovdim, maxiter, tol, qr_tol)` and
`Cg::new(maxiter, tol)`.

- `krylovdim` fixes the memory (for tensor-network vectors, whole MPS): a
  `Workspace` of capacity `krylovdim` holds `krylovdim + 3` vectors, namely
  the basis, one restart scratch vector, and the operator input and output
  (the residual lives in the operator output). `BlockLanczos` with a start
  block of size `b` needs capacity `krylovdim + b - 1`, and `svdsolve` uses
  two workspaces of capacity `krylovdim` (codomain and domain).
- `tol` is the accuracy contract and depends on the scalar type (a tolerance
  reachable in `f64` is not in `f32`).
- `maxiter` bounds the work.

Algorithm-variant options keep builder defaults: `.orth(Orth)` (default
`Orth::Cgs2`) and `.eager(bool)` (default `false`). The arguments are
validated when a solver is called and violations are returned as
`Error::InvalidArgument` (see the restricted inputs below). KrylovKit gives
these parameters keyword defaults (`krylovdim = 30`, `maxiter = 100`,
`tol = 1e-12`, `100` for `BlockLanczos`'s `krylovdim`); krylov does not, so
cost and accuracy are always visible at the call site.

## LAPACK provider

The small projected problems call LAPACK exactly where KrylovKit does
(`xSTEGR`, `xBDSQR`, `xSYEVR`/`xHEEVR`, `xHSEQR`, `xTREXC`, `xTREVC`). krylov
depends only on the
[`lapack`](https://crates.io/crates/lapack) declarations and never selects an
implementation; the final binary must link one, for example:

```toml
[dependencies]
lapack-src = { version = "0.13", features = ["openblas"] } # or "accelerate", "intel-mkl-dynamic-parallel", "netlib"
```

```rust
extern crate lapack_src as _; // once, in the binary (or a crate it links)
```

Any other provider of the Fortran LAPACK symbols works the same way (a
system library passed to the linker, or `lapack-inject`). On Linux the system
OpenBLAS (`apt install libopenblas-dev`) is picked up with
`openblas-src = { version = "0.10", default-features = false, features = ["system"] }`.
krylov's own tests use Accelerate on macOS and the system OpenBLAS elsewhere.

## Stage 1 contents

- Traits `Vector`, `Basis`, `LinearMap<V, W = V>` (also implemented for
  `FnMut(&V, &mut W) -> Result<(), V::Error>`), the `Workspace`, and the default
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
  `AdjointMap<V, W>` (`V → W`, `W = V` by default) by thick-restart
  Golub–Kahan–Lanczos. `x0`, `ws_u`, the left vectors and the residuals live in
  the codomain `W`, `ws_v` and the right vectors in the domain `V`. The two
  may be one Rust vector type for different spaces or distinct types with the
  same scalar and error types (as KrylovKit's `GKLFactorization{TU, TV}`). A call allocates the `3·n` returned vectors. `SmallestReal`
  requires codomain dimension ≤ domain dimension; for a tall map, call it on
  the adjoint and swap left and right vectors.

## Stage 3 contents

- Non-Hermitian Krylov–Schur (Arnoldi with thick restarts) with `Arnoldi`
  settings, selected by `Which` including `LargestImag`/`SmallestImag`
  (complex vector types only):
  - `schursolve(op, x0, howmany, which, &Arnoldi, ws)`: partial Schur form
    `(T, Q, λ, info)`; `T` is quasi-triangular with standardized 2×2 blocks
    for real vectors, and a complex-conjugate pair is never split.
  - `arnoldi_eigsolve`: eigenpairs, for complex vector types (complex
    eigenvectors need complex storage).
  - `realeigsolve`: real eigenpairs of a real map, for real vector types.
  A call allocates the `2·n` returned vectors.
- `exponentiate`/`expintegrator` with `&Arnoldi` for general maps (KrylovKit
  `Union{Lanczos,Arnoldi}`; the process is a type parameter through the
  sealed `KrylovProcess` trait). The step controller is shared; Arnoldi takes
  the dense exponential of KrylovKit's `(k+p+1)²` augmented Hessenberg
  matrix (a port of Julia 1.11 `exp!` on LAPACK `gebal`/`gesv`) per
  step-size trial. A call allocates the same `p` vectors as with `Lanczos`,
  and the dense scratch once, for the largest `(k+p+1)²` (KrylovKit allocates
  it per trial).

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
- **`exp`/`φ` of the tridiagonal Lanczos Rayleigh quotient** use its eigendecomposition
  (LAPACK `stegr`, as KrylovKit's `eigsolve`) instead of a Padé exponential of
  the augmented matrix. The step controller's error estimate
  `e_kᵀ φ_{p+1}(sΔτ·T) e₁` uses the Taylor series in `T` while
  `|sΔτ|·‖T‖∞ < k/2`, because for small steps that entry is far below
  `ε·max|φ_{p+1}|`, which is all an eigendecomposition resolves; this keeps the
  step sequence equal to KrylovKit's.
- **CGS2 Lanczos step** orthogonalizes against the full basis twice; KrylovKit's
  CGS2 recurrence uses the three-term recurrence for its first pass.
- Restricted inputs, reported as typed errors: `krylovdim >= 1`
  (`>= 2` for `Gkl`, `>` the start block size for `BlockLanczos`),
  `maxiter >= 1`, `tol >= 0` (`> 0` for `exponentiate`), finite `t`. With
  `krylovdim == 1`, `eigsolve` cannot restart and returns the first Rayleigh
  quotient (KrylovKit fails there). `exponentiate`/`expintegrator` with
  `t == 0` return `u₀` without applying the map; `t = ±∞` is rejected
  (KrylovKit allows it for `p = 1`); an empty `u` is an `InvalidArgument`.
- **`expintegrator` with `w_{p+1} = 0` and `p ≥ 2`** (the vector the Krylov
  space would start from) takes the exact polynomial step
  `y + Σ_{j<p} (sΔτ)ʲ/j!·w_{j+1}` to `t` without a Krylov space; KrylovKit
  divides by the zero norm there.
- **`expintegrator` non-finite steps.** A NaN step error estimate rejects
  the step like an overflowed one (KrylovKit's `ω >= 1` test accepts NaN).
  An estimate still non-finite at the smallest allowed step (or the forced
  last step), a non-finite Krylov start norm `β`, or non-finite step update
  coefficients return `Error::NonFiniteIterate` instead of a NaN/∞ result.
  For complex vectors with real `t`, the Lanczos `φ` functions are evaluated
  in real arithmetic, so an overflow stays `∞ + 0i`.
- **`linsolve`**: GMRES restarts by default from the explicitly computed
  residual `b − a₀x − a₁A·x` (`RestartResidual::Recomputed`, one extra
  application per restart); KrylovKit uses the basis recurrence, available
  as `Gmres::restart_residual(RestartResidual::Recurrence)`, which krylov
  forms from the dense Givens product (no basis rotation, capacity
  `krylovdim` instead of `+ 1`). `numops` counts every application,
  including CG's residual recompute, the recomputed restart residuals and
  the apply that starts each GMRES cycle, which KrylovKit omits. CG checks
  `maxiter` after every iteration (KrylovKit always runs two) and stops on an
  exactly zero residual. Non-positive curvature in CG is
  `Error::NotPositiveDefinite` and an exactly singular GMRES pivot is
  `Error::SingularKrylovSpace` (KrylovKit: NaN and `SingularException`).
- **Non-finite map output** in every solver is `Error::NonFiniteIterate`,
  returned as soon as a recurrence scalar (a projection coefficient, a
  residual norm, CG's `⟨p, A·p⟩`) is NaN or infinite, with no further map
  applications. KrylovKit propagates the NaN into its results or fails
  later in a dense routine. A non-finite start vector, including `b` and
  `x0` of `linsolve`, stays `Error::NonFiniteStartVector`.
- **`block_eigsolve`** orthogonalizes every new vector against the full basis
  (CGS2 by default, or MGS2) instead of KrylovKit's three-term block step plus
  one MGS pass (KrylovKit accepts only MGS2), and skips KrylovKit's
  type-probe application, so `numops` is one lower (lower by
  `1 + dropped` when the QR drops start vectors, since KrylovKit counts the
  input block size). Diagonal blocks are
  symmetrized, `(M + Mᴴ)/2`, where KrylovKit reads the upper triangle; the
  discarded part is reported in `max_imag`. The small Hermitian eigenproblem
  is LAPACK `syevr`/`heevr`, as Julia's `eigen(Hermitian)`.
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
  (KrylovKit fails indexing). The bidiagonal SVD is LAPACK `bdsqr` on the
  lower bidiagonal matrix, as KrylovKit; the `(k+1) × k` invariant case is
  first reduced to upper bidiagonal form by Givens rotations.
  `SmallestReal` requires codomain dimension ≤ domain dimension (ARPACK's
  rule in `EXAMPLES/SVD/dsvd.f`: reverse the roles of `A` and `Aᴴ` when
  `m > n`); on a tall map call it on the adjoint. Otherwise the codomain
  process targets the zero eigenvalues of `AAᴴ`. krylov reports such a
  `σ ≈ 0` with its true residual (at least `σ_min − σ`) and never marks it
  converged. KrylovKit, which documents no precondition, reports it as
  converged although its true residual is `O(σ_min)`. A different Ritz
  scheme does not lift the precondition. The `(k+1) × k` projection
  (PROPACK) gives the Ritz values of `AᴴA`, which is this method on the
  adjoint from `Aᴴ·x0`, so the spurious zeros move to wide maps. Harmonic or
  refined Ritz values target the genuine zeros of `AAᴴ` as well. SciPy,
  IRLBA and PRIMME choose the side from the matrix shape, which a
  matrix-free map does not provide.
- **Krylov–Schur** (`schursolve`, `arnoldi_eigsolve`, `realeigsolve`):
  - `realeigsolve` returns `Error::ComplexEigenvalue { index }` when one of
    the first `howmany` eigenvalues sits in a 2×2 Schur block coupled by more
    than `tol`; KrylovKit warns and silently drops the imaginary part.
    Couplings `≤ tol` are dropped as in KrylovKit, and the largest imaginary
    part dropped is `info.max_imag` (0 for the other two entry points).
  - The vector type decides what is available at compile time: complex
    eigenvectors only through `arnoldi_eigsolve` (complex vectors), real
    eigenpairs only through `realeigsolve` (real vectors; KrylovKit's
    `RealVec` for real-linear maps on complex vectors is not supported).
    `LargestImag`/`SmallestImag` with real vectors are an `InvalidArgument`;
    KrylovKit's `schursolve` fails later trying to split a 2×2 block.
  - Before `hseqr` the Rayleigh quotient is scaled by a power of four into
    `[sqrt(SAFMIN)/ULP, its inverse]` (as LAPACK's `xGEEV`/`xGEES` drivers
    do, including `xGEES`'s repair after unscaling); KrylovKit calls `hseqr`
    unscaled, which deflates a quotient with entries near `1e-300` to its
    diagonal. In-range quotients are passed to LAPACK unchanged.
  - A reordering swap rejected by LAPACK `trexc` is
    `Error::SchurReorderFailed` (KrylovKit throws `LAPACKException`). A map
    producing NaN or infinity is `Error::NonFiniteIterate`. With
    `krylovdim == 1` the first subspace is final (KrylovKit indexes out of
    bounds), and `realeigsolve` on an invariant subspace smaller than
    `howmany` returns what it has.
- **Arnoldi `expintegrator`** follows KrylovKit step for step. Julia's
  `exp!` scaling exponent `⌈log₂(‖A‖₁/5.4)⌉` is found by exact halving
  (identical unless libm's `log2` rounds across an integer). A map producing
  NaN or infinity is `Error::NonFiniteIterate`, and a non-finite augmented matrix or
  an exactly singular Padé denominator is `Error::DenseNoConvergence` (Julia:
  `SingularException`). `info.max_imag` is 0.
- **`Which` is `#[non_exhaustive]`**; the Hermitian solvers (`eigsolve`,
  `block_eigsolve`, `svdsolve`) reject `LargestImag`/`SmallestImag` with
  `InvalidArgument`, as KrylovKit does.

## Vector contract and distributed use

- `Vector::axpby(a, x, b)` with `b == 0` must overwrite `self` without reading
  it (BLAS `beta = 0`); krylov reuses workspace vectors with stale contents.
- A `Workspace::with_capacity(x, k)` holds `k + 3` vectors: `k` basis vectors,
  one scratch vector for the in-place restart rotation, and the operator input
  and output.
- Distributed vectors: the results of `inner`, `norm` and `Basis::project` must
  be bitwise identical on every rank, all ranks must run the same binary with
  the same libm and the same LAPACK build (provider, version and thread
  count), and errors must be returned on every rank. Under this contract
  every branch is deterministic. Outside LAPACK, `eigsolve` branches only on
  arithmetic and `sqrt`; `exponentiate`'s step acceptance also branches on
  libm `exp`, `ln` and `pow`.

## Low-rank and tensor-network vectors

Vectors whose representation grows under linear combination (MPS/TT, low-rank
matrices) work with `exponentiate`, `expintegrator` and GMRES `linsolve`
without any krylov hook. Every combination krylov performs is a trait method
that the user implements, so truncation (e.g. SVD rounding to a relative accuracy `τ`)
goes inside those methods:

- `LinearMap::apply_into`: truncate the product, as with any tensor-network
  map.
- `Vector::axpby(a, x, b)` with `a ≠ 0` and `b ≠ 0`, and
  `Basis::combine_into`: truncate the combination. (`exponentiate`,
  `expintegrator` and GMRES never call `Basis::rotate`; only the
  thick-restart solvers below do, and those are unsupported under truncation.) A real
  tensor-network `Basis` should implement `combine_into` as one truncated
  sum (for an MPS, bond `kχ` before truncation); the default `VecBasis` loops
  over `axpby` and so truncates once per term.
- Do not truncate pure rescalings (`Basis::push`, `axpby` with `b == 0` or
  `a == 0`). Repeatedly truncating an already truncated vector drifts by `τ`
  each time.

This places the rounding where KIOPS-TT does it (Weigel–Riemer–Stoll,
arXiv:2607.11293, and its `kiops_tt.m`): after the operator product, after
orthogonalization and before the norm, and in the solution update. Choose
`τ` about `tol/10` (KIOPS-TT's default); the paper's experiments use
`τ = tol` and degrade for larger `τ`. The step error estimate and
`info.converged` do not see truncation error, so a coarse `τ` reports
convergence while missing `tol`. The exponential needs only the Arnoldi
relation, not an orthonormal basis (KIOPS, Gaudreault–Rainwater–Tokman
2018), so the `O(τ)` loss of orthogonality is harmless there.
`tests/truncated.rs` checks this with SVD-truncated matrix vectors on
`X ↦ L·X + X·Lᵀ` against the exact flow.

`linsolve` with `Gmres` works under the same contract with the default
`RestartResidual::Recomputed`: every restart cycle starts from `b − A·x`
and so minimizes the true residual, instead of relying only on the final
explicit check. The attainable residual is bounded below by the truncation
floor, about `τ·‖b‖`; with `tol` (absolute) below that floor the run reports
non-convergence rather than a false success. `RestartResidual::Recurrence`
is not recommended for truncating vectors: its restart residual differs from
`b − A·x` by about `τ·‖b‖`, so each cycle minimizes a perturbed residual.
`tests/truncated.rs` solves
`(a₀ + A)·X = B` for the same map against a dense solve.

Not supported under truncation: `Cg`, whose short recurrence relies on exact
orthogonality, and the thick-restart solvers (`eigsolve`, `block_eigsolve`,
`svdsolve`, `schursolve` and its eigen variants), whose restart relation the
truncated `rotate` breaks.

## Status

Pre-1.0: the API may change between minor versions.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
