# krylov user guide

This guide collects the usage contracts that do not fit on a single item's
API documentation. The entry points and their result semantics are
documented on [docs.rs](https://docs.rs/krylov); the correspondence with
KrylovKit.jl is in [krylovkit.md](krylovkit.md).

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
`Error::InvalidArgument` (see the restricted inputs in
[krylovkit.md](krylovkit.md#restricted-inputs)).

## LAPACK provider

The small projected problems call LAPACK exactly where KrylovKit does
(`xSTEGR`, `xBDSQR`, `xSYEVR`/`xHEEVR`, `xHSEQR`, `xTREXC`, `xTREVC`). krylov
depends only on the [`lapack`](https://crates.io/crates/lapack) declarations
and never selects an implementation; the final binary must link one, for
example:

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
that the user implements, so truncation (e.g. SVD rounding to a relative
accuracy `τ`) goes inside those methods:

- `LinearMap::apply_into`: truncate the product, as with any tensor-network
  map.
- `Vector::axpby(a, x, b)` with `a ≠ 0` and `b ≠ 0`, and
  `Basis::combine_into`: truncate the combination. (`exponentiate`,
  `expintegrator` and GMRES never call `Basis::rotate`; only the
  thick-restart solvers below do, and those are unsupported under
  truncation.) A real tensor-network `Basis` should implement `combine_into`
  as one truncated sum (for an MPS, bond `kχ` before truncation); the default
  `VecBasis` loops over `axpby` and so truncates once per term.
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
`tests/truncated.rs` solves `(a₀ + A)·X = B` for the same map against a dense
solve.

Not supported under truncation: `Cg`, whose short recurrence relies on exact
orthogonality, and the thick-restart solvers (`eigsolve`, `block_eigsolve`,
`svdsolve`, `schursolve` and its eigen variants), whose restart relation the
truncated `rotate` breaks.
