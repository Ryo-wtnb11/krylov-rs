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
  MGS2 runs through the required per-column `Basis::{inner_at, axpy_at}`,
  so its `2·len` inner products per step match KrylovKit's
  `ModifiedGramSchmidt2` for every basis implementation.
- **Caller-owned workspace:** iteration and restart code constructs no new
  vectors. Per-call scratch and results are documented on each entry point;
  caller-provided trait methods control their own allocations and placement.
- **Explicit execution context:** every solver takes `cx: &mut V::Context`
  first and passes that one borrow, sequentially, to every vector, basis and
  map operation (KrylovKit has no such argument; Julia arrays carry their
  device). A caller-held engine (device stream, thread pool, communicator)
  therefore stays visible and is never stored by krylov. Host vectors use
  `Context = ()`. `V` and `W` of a rectangular map share one context type.
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

Small projected problems use sequential `colmat` 0.34.0 rather than
KrylovKit's LAPACK kernels, removing external provider linkage. Its
QR/divide-and-conquer eigensolvers and SVD replace STEGR/SYEVR/HEEVR/BDSQR;
Hessenberg Schur, block moves, right eigenvectors, balancing and pivoted LU
use the matching public colmat operations. Signs/phases and unsorted dense
output order can differ. After real QR the reported real parts are taken from
the authoritative T diagonal: colmat's QR eigenvalue array can differ by one ulp
from that diagonal after deflation updates. Stable `Which` sorting, real-pair completion and
existing iteration/application-count contracts remain enforced.

The adapters preserve upper-triangle Hermitian authority by reflecting it
with conjugation into colmat's lower triangle. As LAPACK `xSYEVR`/`xHEEVR` do
internally, a Hermitian matrix whose max-abs entry lies outside
`[sqrt(SAFMIN/EPS), min(sqrt(EPS/SAFMIN), SAFMIN^(-1/4))]` is scaled into that
range before colmat's reduction and its eigenvalues are scaled back; colmat's
reduction alone loses subnormal couplings. Bidiagonal SVD returns V directly;
Schur vectors accumulate into the supplied Z; eigenvectors are right vectors
of T before the caller's back-transformation. Balancing uses colmat's explicit
permutation, scales and half-open active interval; matrix functions are
restored by `X[perm[i],perm[j]] = scale[i]*X_bal[i,j]/scale[j]`. LU rejects
exactly zero pivots explicitly. Dense failures retain `DenseNoConvergence`,
and rejected real block moves retain `SchurReorderFailed`.

Private scratch is pre-sized across supported dimensions and reused for the
whole solver invocation, including exponential trials. Peak scratch is higher
than KrylovKit's LAPACK workspaces: colmat requires additional square
temporaries for Schur vectors, bidiagonal SVD and LU RHS permutation.

Givens rotations, φ functions, restart restoration and the matrix-exponential
driver live in crate code (see [src/dense](../src/dense)).

Householder restarts and `VecBasis::rotate` keep the sign convention
`H·x = +‖x‖e₀`. Input is scaled by powers of two and complex pivots are
safeguarded against underflow/overflow, extending KrylovKit's reflector to
extreme scales. LAPACK's reflector cannot replace it directly because its
sign/phase convention differs. See [householder.rs](../src/dense/householder.rs).

## Lanczos and the exponential integrator

- **Lanczos projected functions:** `exp`/φ use the tridiagonal
  eigendecomposition (colmat tridiagonal EVD) rather than an augmented Padé exponential.
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
- **Scaled residual arithmetic:** KrylovKit squares unscaled norms and
  inner products. Here each residual vector is scaled by a power of two near
  its own norm before the Gram matrix is formed, and each weighted
  combination is evaluated relative to its largest term, so a reported
  residual norm is accurate (to rounding and the Gram formula's cancellation)
  whenever it is a normal floating-point number, whatever the sizes of the
  individual residual vectors. A residual norm above the largest finite
  number is `Error::NonFiniteIterate`; one below the normal range may be
  reported as subnormal or zero. In range the scaling is exact, so results
  equal the unscaled formula bit for bit. Each convergence check costs `b`
  norms and `b` scaled copies into the block scratch besides KrylovKit's
  Gram inner products (here `b(b+1)/2`, using symmetry), `b` the current
  block size.
- **Exhaustion:** a fully dropped new block stops iteration instead of
  throwing. Zero vectors are dropped even at `qr_tol == 0`, avoiding division
  by zero. After restart the norm is recomputed for the transformed residual
  block rather than retaining its old norm. An entirely dropped start block
  is `ZeroStartVector`; empty starts and ineffective restart sizes are
  `InvalidArgument`.

## davidson_eigsolve

KrylovKit v0.10.4 has no Davidson solver. The method follows block2's
`davidson` (`src/core/iterative_matrix_functions.hpp`); the API follows
krylov's other eigensolvers.

- **Corrections:** `Correction::Olsen`, the default, is block2's `Normal`
  (`olsen_precondition`, `iterative_matrix_functions.hpp:93-110`, selected
  at 1086-1087): `t = M·r − ε·M·x` with `ε = ⟨x, M·r⟩/⟨x, M·x⟩`, both
  inner products conjugating `x` as block2's `complex_dot` does.
  `Correction::Plain` is block2's `DavidsonPrecond`, `t = M·r`. `M(θ)` is
  the caller's `Preconditioner`, called on `r` and then on `x` under Olsen;
  krylov owns no elementwise kernel, so the diagonal `tᵢ = rᵢ/(θ − dᵢ)`
  (`rᵢ` kept where `|θ − dᵢ|` is at most the caller's guard, `1e-12` in
  block2) is a documented closure.
- **Olsen denominator:** block2 divides by `⟨x, M·x⟩` unguarded, producing
  ±∞ or NaN at zero. krylov uses `w − (n/d)·M·x` for `|n| <= |d|`, the same
  direction scaled as `(d/n)·w − M·x` for `|n| > |d|` (so `−M·x` at
  `d = 0`) and `w = M·r` itself at `n = d = 0`, with `n = ⟨x, w⟩` and
  `d = ⟨x, M·x⟩`. The quotient is taken after scaling `n` and `d` by a
  common power of two, so it neither over- nor underflows in range.
  Non-finite `n` or `d` is `NonFiniteIterate`.
- **Stagnation:** the plain correction can lie in the current subspace. For
  `A = diag(1, 3)` and `x0 = (1, 1)/√2`, `θ = 2`, `r = (−1, 1)/√2` and
  `(θ − D)⁻¹·r = −x0`, which is dropped; the call returns unconverged
  without an error. A diagonal very close to the spectrum also slows the
  plain correction, because it then mostly reproduces the Ritz vector.
  Olsen's correction avoids the first case; it is not uniformly faster.
- **Storage and work:** the basis, its images and the projected matrix stay
  unrotated between restarts, and one column of `Vᴴ·A·V` is added per
  accepted vector. block2 rotates both stores and recomputes the projection
  on every pass, `Θ(k²)` vector operations instead of
  `O((howmany + blocksize)·k)` per pass without a restart, at subspace size
  `k`. Restarts rotate both stores to the kept Ritz vectors without
  re-applying the map; between restarts at least `⌊(m − r)/b⌋` passes
  follow from dimension `r`, while the initial block can restart at once.
- **Blocks:** up to `blocksize` corrections per Rayleigh–Ritz pass, for the
  leading unconverged Ritz pairs; `blocksize = 1` is block2's one vector per
  pass. A converged root does not cost a pass: the next root is corrected
  in the same pass.
- **Orthogonalization and drops:** CGS2 (or MGS2) instead of one MGS pass,
  with a relative `drop_tol` per candidate instead of block2's absolute
  `1e-14` on `‖b‖²`, and non-finite guards (`NonFiniteIterate`).
- **Parameters and results:** `tol` bounds `‖A·x − θ·x‖` (block2's
  `conv_thrd` applies to its square, so `tol = √conv_thrd` at
  `rel_conv_thrd = 0`). Sizes block2 adjusts silently are `InvalidArgument`;
  non-convergence after `maxiter` restart cycles is reported in `info`
  (block2 asserts). `which` selects among real Ritz values; block2's
  `GreaterThan` is relative to a shift, not descending. No shift, projection
  against other states, harmonic or non-Hermitian variants.

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
