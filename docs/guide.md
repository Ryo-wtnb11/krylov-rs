# krylov user guide

The [API documentation](https://docs.rs/krylov) gives signatures and examples;
[krylovkit.md](krylovkit.md) records intentional differences from KrylovKit.
For algorithm theory, see [KrylovKit](https://github.com/Jutho/KrylovKit.jl/tree/v0.10.4).

## Ownership and conventions

krylov owns the algorithms and small dense problems. The caller implements
the [storage boundary](../src/traits.rs):

- `Vector` owns arithmetic; `similar` creates independent writable storage,
  even if cloning the type would share storage. Scalars are `f32`, `f64`,
  `Complex32` or `Complex64`; `inner(x, y)` is conjugate-linear in `x`.
- `Basis` owns block operations on ordered columns `V = [v₀, …, vₖ₋₁]`:
  projection is `Vᴴx`, combination is `a·V·c + b·y`, and rotation keeps the
  first `keep` columns of `V·Q`. `MatRef` stores columns consecutively.
  The per-column `inner_at` (`⟨vᵢ, x⟩`) and `axpy_at` (`y ← a·vᵢ + y`) are
  required too: MGS2 and the Lanczos recurrence call them once per
  coefficient, so each must cost one vector operation, not a block one.
- `LinearMap<V, W = V>` writes `A·x` into caller-supplied storage;
  `AdjointMap<V, W>` also writes `Aᴴ·y`. Domain and codomain share scalar,
  error and context types. `FnMut(&mut V::Context, &V, &mut W) -> Result<(), V::Error>`
  closures also implement `LinearMap`.
- `Vector::Context` is the caller's execution context (a device engine, a
  thread pool, a communicator; `()` for host vectors). Every solver,
  workspace constructor and vector/basis/map operation takes `cx: &mut
  V::Context`; krylov forwards that one borrow sequentially and never stores
  it, so a map reaches the engine through `cx` rather than capturing it.
  Domain and codomain share the context type. A trait-object context
  `type Context = dyn Engine` means `dyn Engine + 'static`: wrappers take
  `&mut (dyn Engine + 'static)`, which a locally owned engine coerces to.

There are no backend adapters, internal threads, caches or global state.
Vector/map implementations own placement and parallelism.

## Solver selection

| Entry point | Settings and map requirement |
| --- | --- |
| `eigsolve` | `Lanczos`; Hermitian map |
| `block_eigsolve` | `BlockLanczos`; Hermitian map, start block bounds resolvable multiplicity |
| `schursolve` | `Arnoldi`; general map, partial Schur form |
| `arnoldi_eigsolve` / `realeigsolve` | `Arnoldi`; complex / real vector scalar respectively |
| `svdsolve` | `Gkl`; adjoint map, start vector in the codomain |
| `linsolve` | `Cg` for Hermitian positive definite `a₀ + a₁A` with real coefficients; `Gmres` otherwise |
| `exponentiate` / `expintegrator` | `Lanczos` for Hermitian maps, `Arnoldi` otherwise |

`KrylovProcess` (`Lanczos`/`Arnoldi`) and `LinearSolver` (`Cg`/`Gmres`) are
sealed: callers cannot add implementations.

`exponentiate` returns `exp(tA)x`; `expintegrator` returns `Σⱼ tʲφⱼ(tA)uⱼ`
(see its [input convention](https://docs.rs/krylov/latest/krylov/fn.expintegrator.html)).
Complex time requires a complex vector scalar (e.g. `t = -iτ` for real-time evolution).

## Parameters and workspace

Cost and accuracy parameters are required: `new(krylovdim, maxiter, tol)` for
`Lanczos`, `Arnoldi`, `Gkl` and `Gmres`; `BlockLanczos` also takes `qr_tol`,
and `Cg::new(maxiter, tol)` needs no Krylov dimension. Where available,
`.orth(Orth)` defaults to `Cgs2` (classical Gram–Schmidt with reorthogonalization)
and `.eager(bool)` to `false`. Parameters are checked at the solver call;
see [restricted inputs](krylovkit.md#restricted-inputs).

`maxiter` limits Krylov subspaces (GMRES restart cycles, exponential time
steps), except for CG where it limits iterations. Tolerances must suit the
scalar precision.

`Workspace::with_capacity(cx, x, k)` preallocates `k + 3` vectors: `k` basis
slots, one rotation scratch, and operator input/output. Reuse it for the same
vector space. A custom `Basis` can be supplied through `Workspace::new`.

| Solver | Required basis capacity |
| --- | --- |
| `Cg` | 0 (basis unused) |
| `BlockLanczos`, start block size `b` | `krylovdim + b - 1` |
| `Gkl` | `krylovdim` in each of two workspaces (codomain and domain) |
| Others | `krylovdim` |

Iterations and restarts construct no new vectors in crate code; results and
some per-call scratch do allocate. Each entry point documents these counts.
This does not constrain allocation inside caller-provided trait methods.

## Results and errors

`tol` is absolute: residual norms for eigen/SVD/linear solves, an error
estimate per unit time for exponential integration. Eigen/Schur convergence
uses `<= tol`; SVD and linear solves use `< tol`. The exponential controller
compares its accumulated error estimate with `tol·|t|`; its fixed-point
shortcut instead accepts a forcing norm `‖Ay + u₁‖ < tol` (`u₁ = 0` if absent).
Non-convergence is returned in `info.converged`, not as an error.

Eigen/SVD values follow `Which` order; `converged` counts the leading entries.
Results can exceed `howmany` when more entries converge, or contain fewer
when the subspace is exhausted. Real Schur results do not split conjugate
pairs. `numops` counts map applications; `numiter` follows `maxiter`'s units.
Lanczos `max_imag` records the largest discarded `|Im α|`.
Typed failures, including propagated vector errors, are defined by
[`Error`](../src/error.rs); they are separate from non-convergence.

## LAPACK provider

The final binary links a provider as shown in the [README](../README.md#installation).
System linking is also possible. The crate's tests use `lapack-src` with
Accelerate on macOS; elsewhere they use its OpenBLAS feature together with
`openblas-src = { version = "0.10", default-features = false, features = ["system"] }`.
See [lapack-src](https://github.com/blas-lapack-rs/lapack-src) for providers.

## Vector contract and distributed use

`Vector::axpby(cx, a, x, b)` with `b == 0` must overwrite `self` without reading
it: workspace contents can be stale or non-finite. Map destinations likewise
hold arbitrary previous contents.

For distributed vectors, `inner`, `norm` and `Basis::project` results must be
bitwise identical on every rank. All ranks must use the same binary, libm and
LAPACK build (provider, version and thread count), and return errors on every
rank. These are caller obligations for identical scalar decisions and to
avoid blocking other ranks in reductions; krylov supplies no MPI adapter.

## Low-rank and tensor-network vectors

For `exponentiate`, `expintegrator` and GMRES, truncation can live in
`apply_into`, rank-growing `axpby` and `Basis::combine_into`; these solvers
never call `Basis::rotate`. Preserve pure copies/rescalings. A custom basis
can truncate a sum once; `VecBasis` combines terms through repeated `axpby`.
See [KIOPS-TT](https://arxiv.org/abs/2607.11293) for the method and
[tests/truncated.rs](../tests/truncated.rs) for small SVD-truncated matrix examples.

Exponential convergence estimates exclude truncation error: `converged == 1`
does not certify total error below `tol`. GMRES defaults to
`RestartResidual::Recomputed`, starting each cycle from `b − a₀x − a₁A·x`;
the recurrence option can accumulate truncation error in that residual.
The attainable accuracy depends on the vector implementation and problem.
Truncating arithmetic is unsupported for CG and thick-restart eigen/SVD/Schur
solvers, whose orthogonality or rotated recurrence relations it can break.
