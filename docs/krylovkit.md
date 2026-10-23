# krylov and KrylovKit.jl

The semantic reference is [KrylovKit.jl v0.10.4](https://github.com/Jutho/KrylovKit.jl/tree/v0.10.4).
The [guide](guide.md) covers solver selection and shared result conventions;
this file records intentional differences and their reasons. Representative
values and iteration/application counts are recorded inline in the tests.

## API and execution model

- **Explicit parameters:** cost and accuracy parameters are constructor
  arguments rather than KrylovKit keyword defaults, so callers choose them
  for the problem and scalar precision.
- **CGS2 default:** MGS2 remains available. Block projections allow batched
  reductions on devices or distributed storage instead of sequential inner
  products; Lanczos keeps its three-term recurrence plus one full pass.
- **Caller-owned workspace:** iteration and restart code constructs no new
  vectors. Per-call scratch and results are documented on each entry point;
  caller-provided trait methods control their own allocations and placement.
- **No internal threads, caches, global state or RNG:** these stay outside
  the algorithm layer. Callers supply starts; invariant subspaces stop
  eigen/SVD iterations without random restarts.
- **Fallible operations and non-exhaustive enums:** storage failures propagate
  as `Error::Vector`; `Error` and `Which` permit additional variants.
  Failures are returned as typed errors rather than Julia exceptions.
- **Non-finite iterates:** a non-finite recurrence scalar or norm returns
  `Error::NonFiniteIterate` without further map applications, rather than
  propagating NaNs or failing later in a dense routine. Non-finite start
  norms use `NonFiniteStartVector`. See [tests/nonfinite.rs](../tests/nonfinite.rs).

## Restricted inputs

Solvers require `maxiter >= 1`, non-NaN `tol >= 0`, and `krylovdim >= 1`
(`>= 2` for GKL, strictly greater than the nonempty start block size for
block Lanczos). Block `qr_tol` is non-NaN and nonnegative. Violations return
`InvalidArgument`, preventing empty or ineffective iterations.
Insufficient workspace capacity and `howmany > krylovdim` have dedicated errors.

Exponential integration requires `tol > 0`, finite `t` and nonempty `u`;
`t == 0` returns `u₀` without applying the map. Infinite `t` is rejected
(KrylovKit permits it for `p = 1`) to keep the step controller finite.
With `krylovdim == 1`, Lanczos eigsolve and Krylov–Schur return the first
subspace rather than attempting a restart with nothing to retain.

Hermitian solvers reject imaginary-part selectors; SVD accepts only
`LargestReal` and `SmallestReal`. Arnoldi imaginary-part selectors require
complex vectors, avoiding a later attempt to split real 2×2 Schur blocks.

## Dense kernels

Small projected problems use LAPACK; krylov never selects its provider.
Givens rotations, φ functions, restart restoration and the matrix-exponential
driver live in crate code (see [src/dense](../src/dense)).

Householder restarts and `VecBasis::rotate` keep the sign convention
`H·x = +‖x‖e₀`. Input is scaled by powers of two and complex pivots are
safeguarded against underflow/overflow, extending KrylovKit's reflector to
extreme scales. LAPACK's reflector cannot replace it directly because its
sign/phase convention differs. See [householder.rs](../src/dense/householder.rs).

## Lanczos and the exponential integrator

- **Lanczos projected functions:** `exp`/φ use the tridiagonal
  eigendecomposition (`stegr`) rather than an augmented Padé exponential.
  This reduces evaluation to scalar φ functions. For small steps the
  error-estimate entry uses a Taylor series to preserve relative accuracy below the
  eigendecomposition's resolution. [tests/krylovkit.rs](../tests/krylovkit.rs)
  compares the controller to KrylovKit, including its 21-step imaginary-time
  case (counts are checked with slack).
- **Vanishing Krylov start:** for `p >= 2`, a zero `wₚ₊₁` takes the remaining
  polynomial step without a Krylov space, avoiding division by zero; the
  φ term vanishes. Here `p = max(u.len() - 1, 1)`. Coefficients are preflighted
  in scalar arithmetic; only on overflow, Horner evaluation reuses the last
  `w` buffer with exactly `p - 1` vector updates and no new allocation.
  Ordinary evaluation keeps its arithmetic order. KrylovKit v0.10.4 has no
  `p >= 2` zero-start shortcut.
- **Non-finite steps:** NaN estimates are rejected like overflowed ones. An
  estimate still non-finite at the minimum or forced final step, a non-finite
  Krylov start norm, or a non-finite update coefficient returns
  `NonFiniteIterate`. Krylov polynomial coefficients are checked before their
  vector updates. A final non-finite result norm also returns `NonFiniteIterate`,
  catching vector overflow even with finite coefficients. This costs one extra
  norm of the result, except at `t = 0` where the input check suffices. Backends
  should use overflow-safe norms; a naive norm can reject finite vectors.
  KrylovKit v0.10.4 checks neither polynomial coefficients nor the result norm.
  Real arguments use real φ arithmetic even for complex vectors, preserving
  `∞ + 0i` so overflow rejection works.
  Non-finite complex `t / |t|` falls back to reciprocal scaling to avoid
  squaring `|t|` at extreme scales; finite normalization is unchanged.
- **Arnoldi projected exponential:** follows Julia 1.11's `exp!` sequence,
  but finds the scaling exponent by exact halving to avoid integer overflow
  in `2^si` (only libm rounding at an integer boundary can change the exponent).
  Non-finite augmented matrices or singular Padé denominators return
  `DenseNoConvergence`. Dense scratch is reused across step-size trials to
  avoid per-trial allocation; `info.max_imag` is zero.

## linsolve

- **GMRES restart residual:** defaults to the recomputed `b − a₀x − a₁A·x`,
  costing one extra map application per restart so cycles start from the
  current residual even with truncating arithmetic. KrylovKit's recurrence
  option is available as `RestartResidual::Recurrence`; it uses the dense
  Givens product without rotating the basis, saving one basis vector.
- **Application counts:** `numops` includes CG residual recomputations,
  GMRES recomputed restart residuals and the application starting each cycle,
  so it reports actual map calls that KrylovKit's counts omit.
- **CG stopping:** checks `maxiter` every iteration and stops at exactly zero
  residual, honoring the work limit and avoiding `0/0`.
- **Breakdowns:** CG non-positive curvature is `NotPositiveDefinite`;
  an exactly singular GMRES pivot is `SingularKrylovSpace`, replacing NaNs or
  exceptions with typed failures.

## block_eigsolve

- **Full-basis orthogonalization:** CGS2 or MGS2 replaces the three-term block
  step plus MGS pass, sharing the Arnoldi orthogonalization path without
  access to the previous block.
- **Application counts:** skips the type-probe application; `numops` is lower
  by `1 + dropped` when start QR drops vectors, since it counts actual calls.
- **Diagonal blocks:** use `(M + Mᴴ)/2` rather than just the upper triangle;
  `max_imag` reports the largest discarded anti-Hermitian entry. This makes
  the treatment of non-Hermiticity explicit.
- **Exhaustion:** a fully dropped new block stops iteration instead of
  throwing. Zero vectors are dropped even at `qr_tol == 0`, avoiding division
  by zero. After restart the norm is recomputed for the transformed residual
  block rather than retaining its old norm. An entirely dropped start block
  is `ZeroStartVector`; empty starts and ineffective restart sizes are
  `InvalidArgument`.

## svdsolve

- **Full reorthogonalization on both sides:** each new vector is
  orthogonalized twice against its full basis to control loss of
  orthogonality across restarts. CGS2 uses six reductions per step instead
  of KrylovKit's three (its right step is local, left step has one full pass).
- **Adjoint compatibility:** the failed start check `⟨u, A·v⟩ = ‖Aᴴu‖`
  returns `IncompatibleAdjoint` rather than `ArgumentError`, giving callers
  a distinct failure variant. SVD `max_imag` is always zero.
- **Adjoint-side exhaustion:** when the new right-vector norm is `<= tol`,
  use the `(k+1) × k` bidiagonal projection rather than dividing by that
  norm. The forward residual is zero in this projection; the adjoint
  relation holds up to the dropped norm. Return fewer than `howmany` if
  exhausted, or none if exhausted at the start, avoiding indexing failures.
- **Smallest singular values:** require codomain dimension ≤ domain dimension.
  On tall maps, solve on the adjoint and swap left/right vectors. The caller
  chooses the side because a matrix-free map exposes no dimensions. Otherwise
  codomain null directions can produce spurious near-zero values; krylov
  retains their non-small forward residual rather than marking them converged
  as KrylovKit does. See [tests/svdsolve.rs](../tests/svdsolve.rs).

## Krylov–Schur (`schursolve`, `arnoldi_eigsolve`, `realeigsolve`)

- **Real eigenpairs:** a requested real Schur block coupling above `tol` is
  `ComplexEigenvalue { index }` (0-based), rather than a warning and silent
  loss of its imaginary part. Couplings `<= tol` are dropped; `max_imag`
  records the largest discarded imaginary part (zero for the other entries).
- **Scalar capabilities:** `arnoldi_eigsolve` requires complex vectors;
  `realeigsolve` requires real vectors. Complex eigenvectors need complex
  storage; KrylovKit's `RealVec` for real-linear maps on complex vectors is
  unsupported.
- **Range scaling:** before `hseqr`, scale extreme Rayleigh quotients by
  powers of four as in LAPACK's eigen/Schur drivers, repairing real blocks
  after unscaling. This prevents tiny quotients from deflating to their
  diagonal; in-range inputs are unchanged.
- **Failures and exhaustion:** rejected `trexc` swaps return
  `SchurReorderFailed` rather than `LAPACKException`. `realeigsolve` returns
  available pairs when an invariant subspace is smaller than `howmany`,
  avoiding out-of-bounds indexing.
