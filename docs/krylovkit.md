# krylov and KrylovKit.jl

krylov is modelled on [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl)
(reference revision v0.10.4; the tests pin a few KrylovKit v0.10.4
iteration counts and values inline, on the same inputs and options). This document lists
what is implemented and every intentional difference from KrylovKit, each
with its reason.

## Shared conventions

Entry points keep KrylovKit's names (`eigsolve`, `exponentiate`, …) and its
conventions: `tol` is an absolute residual norm (per unit time for
`exponentiate`), converged eigenvalues are counted from the front, more or
fewer than `howmany` values may be returned, and non-convergence is reported
through the convergence information rather than as an error.

## What is implemented

- Traits `Vector`, `Basis`, `LinearMap<V, W = V>` (also implemented for
  `FnMut(&V, &mut W) -> Result<(), V::Error>`), `AdjointMap`, the
  `Workspace`, and the default `VecBasis` (in-place Householder `rotate` with
  one scratch vector).
- `eigsolve` with `Lanczos`: thick restart, `eager`,
  `Which::{LargestMagnitude, LargestReal, SmallestReal}`, `ConvergenceInfo`
  (including the largest discarded `Im α`).
- `exponentiate(op, t, x, &alg, ws)` and
  `expintegrator(op, t, u, &alg, ws)`: `Σ_j tʲ·φ_j(t·A)·u_j` for
  `u = [u₀, u₁, …]`, the solution at `t` of `ẋ = A·x + Σ_j tʲ/j!·u_{j+1}`,
  `x(0) = u₀`, with KrylovKit's `expintegrator` formulation and time-step
  controller. `alg` is `&Lanczos` (Hermitian maps) or `&Arnoldi` (general
  maps), selected through the sealed `KrylovProcess` trait (KrylovKit
  `Union{Lanczos,Arnoldi}`). Arnoldi takes the dense exponential of
  KrylovKit's `(k+p+1)²` augmented Hessenberg matrix (Julia 1.11 `exp!` on
  LAPACK `gebal`/`gesv`).
- `linsolve(op, b, x0, &alg, a0, a1, ws)`: `(a₀ + a₁·A)·x = b` by `Cg`
  (Hermitian positive definite, real `a₀`, `a₁`) or restarted `Gmres`,
  selected through the sealed `LinearSolver` trait.
- `block_eigsolve(op, x0: &[V], howmany, which, &BlockLanczos, ws)`:
  thick-restart block Lanczos for Hermitian maps with repeated eigenvalues;
  a start block of size `b` resolves multiplicities up to `b`.
- `svdsolve(op, x0, howmany, which, &Gkl, ws_u, ws_v)`: largest or smallest
  singular triplets of an `AdjointMap<V, W>` by thick-restart
  Golub–Kahan–Lanczos. Domain and codomain may be distinct vector types with
  the same scalar and error types (as KrylovKit's `GKLFactorization{TU, TV}`).
- Non-Hermitian Krylov–Schur (Arnoldi with thick restarts) with `Arnoldi`
  settings, selected by `Which` including `LargestImag`/`SmallestImag`
  (complex vector types only): `schursolve` (partial Schur form; a real
  complex-conjugate pair is never split), `arnoldi_eigsolve` (complex vector
  types) and `realeigsolve` (real vector types).

Workspace sizes and per-call allocations are stated on each entry point's
API documentation (the three Krylov–Schur entry points allocate only the
`2·n` returned vectors).

## Differences

### API and execution model

- **Explicit parameters.** `krylovdim`, `maxiter` and `tol` (and
  `BlockLanczos`'s `qr_tol`) are required constructor arguments. KrylovKit
  gives them keyword defaults (`krylovdim = 30`, `maxiter = 100`,
  `tol = 1e-12`, `100` for `BlockLanczos`'s `krylovdim`).
  *Reason:* cost and accuracy are always visible at the call site, and a
  tolerance depends on the scalar type (#28).
- **Orthogonalization default is CGS2** (classical Gram–Schmidt with one
  reorthogonalization), not MGS2. MGS2 is available.
  *Reason:* three reductions per iteration (two projections and one norm)
  instead of a number proportional to the Krylov dimension, which matters on
  devices and across MPI ranks.
- **Preallocated workspace.** The caller creates a `Workspace` once; the
  iteration and its restarts allocate no vectors. KrylovKit allocates new
  vectors per expansion and per restart.
  *Reason:* the caller owns allocation and placement of vectors (host,
  device, distributed), and `krylovdim` fixes the memory, which for
  tensor-network vectors is whole MPS.
- **No internal threading and no global state.**
  *Reason:* parallelism belongs to the vector and map implementations.
- **Fallible vector operations.** Every vector operation returns
  `Result<_, V::Error>`, and errors are propagated as `krylov::Error::Vector`.
  *Reason:* not recorded.
- **No random restart.** Reaching an invariant subspace stops the iteration.
  The caller always supplies the start vector.
  *Reason:* krylov has no random number generator (design: no RNG, no
  threads, no caches); every start vector comes from the caller.
- **`Which` is `#[non_exhaustive]`**; the Hermitian solvers (`eigsolve`,
  `block_eigsolve`, `svdsolve`) reject `LargestImag`/`SmallestImag` with
  `InvalidArgument`, as KrylovKit does.
  *Reason:* not recorded.
- **Non-finite map output** in every solver is `Error::NonFiniteIterate`,
  returned as soon as a recurrence scalar (a projection coefficient, a
  residual norm, CG's `⟨p, A·p⟩`) is NaN or infinite, with no further map
  applications. KrylovKit propagates the NaN into its results or fails
  later in a dense routine. A non-finite start vector, including `b` and
  `x0` of `linsolve`, stays `Error::NonFiniteStartVector`.
  *Reason:* one typed error at the first non-finite output instead of
  NaN results or a later dense failure (#9).

### Dense kernels (all solvers)

- **Complex Householder pivots** (Lanczos, block Lanczos and GKL restarts,
  `VecBasis::rotate`) are formed scaled by `1/ν`, so complex
  operators reach the same range limits as real ones (entries whose squares
  stay finite).
  *Reason:* the unscaled form underflows or overflows to NaN on valid
  input (#20).

### Restricted inputs

Reported as typed errors: `krylovdim >= 1` (`>= 2` for `Gkl`, `>` the start
block size for `BlockLanczos`), `maxiter >= 1`, `tol >= 0` (`> 0` for
`exponentiate`), finite `t`. With `krylovdim == 1`, `eigsolve` cannot restart
and returns the first Rayleigh quotient (KrylovKit fails there).
`exponentiate`/`expintegrator` with `t == 0` return `u₀` without applying the
map; `t = ±∞` is rejected (KrylovKit allows it for `p = 1`); an empty `u` is
an `InvalidArgument`.
*Reason:* for `krylovdim == 1`, KrylovKit fails there; for the other
restrictions and the `t == 0` shortcut, not recorded.

### Lanczos and the exponential integrator

- **CGS2 Lanczos step** orthogonalizes against the full basis twice;
  KrylovKit's CGS2 recurrence uses the three-term recurrence for its first
  pass.
  *Reason:* two full projections give classical CGS2 orthogonality
  (`O(ε)`) of the basis across restarts; the extra vector passes (about `4k`
  against `2k + 4`) are small next to a map application.
- **`exp`/`φ` of the tridiagonal Lanczos Rayleigh quotient** use its
  eigendecomposition (LAPACK `stegr`, as KrylovKit's `eigsolve`) instead of a
  Padé exponential of the augmented matrix. The step controller's error
  estimate `e_kᵀ φ_{p+1}(sΔτ·T) e₁` uses the Taylor series in `T` while
  `|sΔτ|·‖T‖∞ < k/2`.
  *Reason:* for the eigendecomposition itself, not recorded. The Taylor
  series is needed because for small steps that entry is far below
  `ε·max|φ_{p+1}|`, which is all an eigendecomposition resolves; it keeps the
  step sequence equal to KrylovKit's (without it, 100 steps instead of
  KrylovKit's 21 on the imaginary-time case in `tests/krylovkit.rs`).
- **`expintegrator` with `w_{p+1} = 0` and `p ≥ 2`** (the vector the Krylov
  space would start from) takes the exact polynomial step
  `y + Σ_{j<p} (sΔτ)ʲ/j!·w_{j+1}` to `t` without a Krylov space.
  *Reason:* KrylovKit divides by the zero norm there; the `φ_p` term
  vanishes identically, so there is nothing to approximate.
- **`expintegrator` non-finite steps.** A NaN step error estimate rejects
  the step like an overflowed one (KrylovKit's `ω >= 1` test accepts NaN).
  An estimate still non-finite at the smallest allowed step (or the forced
  last step), a non-finite Krylov start norm `β`, or non-finite step update
  coefficients return `Error::NonFiniteIterate` instead of a NaN/∞ result.
  For complex vectors with real `t`, the Lanczos `φ` functions are evaluated
  in real arithmetic, so an overflow stays `∞ + 0i`.
  *Reason:* a NaN estimate must not pass the step controller and return a
  NaN result (#13).
- **Arnoldi `expintegrator`** follows KrylovKit step for step. Julia's
  `exp!` scaling exponent `⌈log₂(‖A‖₁/5.4)⌉` is found by exact halving
  (identical unless libm's `log2` rounds across an integer). A map producing
  NaN or infinity is `Error::NonFiniteIterate`, and a non-finite augmented
  matrix or an exactly singular Padé denominator is
  `Error::DenseNoConvergence` (Julia: `SingularException`). `info.max_imag`
  is 0. The dense scratch is allocated once per call, for the largest
  `(k+p+1)²` (KrylovKit allocates it per step-size trial).
  *Reason:* Julia's `2^si` is an `Int` and overflows for
  `‖A‖₁ > 2.5e19` (NaN result), while exact halving stays finite (#14); typed errors as for the other
  solvers (#9); scratch allocated once to remove per-trial allocation (#16).

### linsolve

- **GMRES restart residual.** GMRES restarts by default from the explicitly
  computed residual `b − a₀x − a₁A·x` (`RestartResidual::Recomputed`, one
  extra application per restart); KrylovKit uses the basis recurrence,
  available as `Gmres::restart_residual(RestartResidual::Recurrence)`, which
  krylov forms from the dense Givens product (no basis rotation, capacity
  `krylovdim` instead of `+ 1`).
  *Reason:* every cycle then minimizes the true residual, also for vectors
  whose arithmetic truncates (#29); the recurrence form uses the Givens
  product to avoid mutating the basis and to save one vector.
- **`numops`** counts every application, including CG's residual recompute,
  the recomputed restart residuals and the apply that starts each GMRES
  cycle, which KrylovKit omits.
  *Reason:* not recorded.
- **CG stopping.** CG checks `maxiter` after every iteration (KrylovKit
  always runs two) and stops on an exactly zero residual.
  *Reason:* krylov honors `maxiter` literally and avoids a `0/0`.
- **Typed breakdowns.** Non-positive curvature in CG is
  `Error::NotPositiveDefinite` and an exactly singular GMRES pivot is
  `Error::SingularKrylovSpace` (KrylovKit: NaN and `SingularException`).
  *Reason:* not recorded.

### block_eigsolve

- **Orthogonalization.** Every new vector is orthogonalized against the full
  basis (CGS2 by default, or MGS2) instead of KrylovKit's three-term block
  step plus one MGS pass (KrylovKit accepts only MGS2).
  *Reason:* keeps one orthogonalization path, shared with Lanczos, and needs
  no access to the previous block.
- **`numops`** is one lower: krylov skips KrylovKit's type-probe
  application (lower by `1 + dropped` when the QR drops start vectors, since
  KrylovKit counts the input block size).
  *Reason:* krylov's count is the true number of applications.
- **Diagonal blocks** are symmetrized, `(M + Mᴴ)/2`, where KrylovKit reads
  the upper triangle; the discarded part is reported in `max_imag`. The small
  Hermitian eigenproblem is LAPACK `syevr`/`heevr`, as Julia's
  `eigen(Hermitian)`.
  *Reason:* the difference is of the order of the non-Hermiticity, which is
  what `max_imag` reports.
- **Exhausted block.** When every vector of a new block falls below `qr_tol`
  the iteration stops as an invariant subspace (KrylovKit throws
  `ArgumentError("blocklength must be >(0)")` there). After a restart the
  residual norm is that of the transformed residual block (KrylovKit keeps
  the pre-restart norm). Exactly zero vectors are dropped even with
  `qr_tol == 0`.
  *Reason:* not recorded.
- **Typed errors.** An empty block or `krylovdim <= block size` (every
  restart would be a no-op) is an `InvalidArgument`, a block entirely below
  `qr_tol` is `ZeroStartVector`.
  *Reason:* with `krylovdim <= block size` every restart would be a no-op.

### svdsolve

- **Orthogonalization.** Each new vector is orthogonalized twice against the
  full basis on both sides (six reductions per step; KrylovKit's CGS2 GKL
  keeps the right side local and makes one pass on the left).
  *Reason:* gives `O(ε)` orthogonality of both bases across restarts; the
  three extra reductions and about `4k` extra vector passes per step are
  small next to two map applications.
- **Incompatible adjoint.** The start check `⟨u, A·v⟩ = ‖Aᴴu‖` failing is
  the typed `Error::IncompatibleAdjoint` (KrylovKit: `ArgumentError`).
  *Reason:* not recorded.
- **Invariant subspace.** When `(I - VVᴴ)Aᴴu` drops to `tol` the iteration
  stops with the exact triplets of the invariant pair (`(k+1) × k`
  bidiagonal); KrylovKit divides by that norm. An invariant subspace smaller
  than `howmany` returns its triplets (KrylovKit fails indexing). The
  bidiagonal SVD is LAPACK `bdsqr` on the lower bidiagonal matrix, as
  KrylovKit; the `(k+1) × k` invariant case is first reduced to upper
  bidiagonal form by Givens rotations.
  *Reason:* avoids KrylovKit's division by a vanishing norm and its
  indexing failure; the invariant pair is exact there.
- **`SmallestReal` requires codomain dimension ≤ domain dimension**
  (ARPACK's rule in `EXAMPLES/SVD/dsvd.f`: reverse the roles of `A` and `Aᴴ`
  when `m > n`); on a tall map call it on the adjoint. Otherwise the codomain
  process targets the zero eigenvalues of `AAᴴ`. krylov reports such a
  `σ ≈ 0` with its true residual (at least `σ_min − σ`) and never marks it
  converged. KrylovKit, which documents no precondition, reports it as
  converged although its true residual is `O(σ_min)`.
  *Reason:* a different Ritz scheme does not lift the precondition (#7). The
  `(k+1) × k` projection (PROPACK) gives the Ritz values of `AᴴA`, which is
  this method on the adjoint from `Aᴴ·x0`, so the spurious zeros move to
  wide maps. Harmonic or refined Ritz values target the genuine zeros of
  `AAᴴ` as well. SciPy, IRLBA and PRIMME choose the side from the matrix
  shape, which a matrix-free map does not provide.

### Krylov–Schur (`schursolve`, `arnoldi_eigsolve`, `realeigsolve`)

- **Complex eigenvalues in `realeigsolve`** return
  `Error::ComplexEigenvalue { index }` when one of the first `howmany`
  eigenvalues sits in a 2×2 Schur block coupled by more than `tol`;
  KrylovKit warns and silently drops the imaginary part. Couplings `≤ tol`
  are dropped as in KrylovKit, and the largest imaginary part dropped is
  `info.max_imag` (0 for the other two entry points).
  *Reason:* KrylovKit then silently returns wrong real values, although its
  docstring promises an `ArgumentError`.
- **The vector type decides what is available at compile time:** complex
  eigenvectors only through `arnoldi_eigsolve` (complex vectors), real
  eigenpairs only through `realeigsolve` (real vectors; KrylovKit's `RealVec`
  for real-linear maps on complex vectors is not supported).
  `LargestImag`/`SmallestImag` with real vectors are an `InvalidArgument`;
  KrylovKit's `schursolve` fails later trying to split a 2×2 block.
  *Reason:* complex eigenvectors need complex storage; `RealVec`, not
  recorded.
- **Range scaling before `hseqr`.** The Rayleigh quotient is scaled by a
  power of four into `[sqrt(SAFMIN)/ULP, its inverse]` (as LAPACK's
  `xGEEV`/`xGEES` drivers do, including `xGEES`'s repair after unscaling).
  In-range quotients are passed to LAPACK unchanged.
  *Reason:* KrylovKit calls `hseqr` unscaled, which deflates a quotient with
  entries near `1e-300` to its diagonal.
- **Failures and edge cases.** A reordering swap rejected by LAPACK `trexc`
  is `Error::SchurReorderFailed` (KrylovKit throws `LAPACKException`). A map
  producing NaN or infinity is `Error::NonFiniteIterate`. With
  `krylovdim == 1` the first subspace is final (KrylovKit indexes out of
  bounds), and `realeigsolve` on an invariant subspace smaller than
  `howmany` returns what it has.
  *Reason:* typed errors instead of exceptions or out-of-bounds indexing
  (#9 for non-finite output).
