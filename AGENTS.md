# krylov-rs repository policy

- **Scope: algorithms only.** The crate owns iteration control,
  orthogonalization, the small projected dense problems, convergence tests and
  restarts. It never owns large vector arithmetic, linear-map internals,
  backend adapters (symtn, tenferro, CUDA, MPI live with their owners),
  threading, caches or global state.
- **Dependencies.** Runtime: `num-traits`, `num-complex` and `lapack`
  (declarations only). The library never selects a LAPACK provider; the final
  binary links one, and the tests use a target-specific `lapack-src`
  dev-dependency (Accelerate on macOS, system OpenBLAS elsewhere). The small
  dense problems call LAPACK where KrylovKit does; crate code is kept only
  where KrylovKit has its own Julia code (Givens, φ functions, Householder
  restart restoration). Dev-dependencies may include one dense reference
  library for test oracles. Anything else needs explicit approval.
- **Design authority.** The fixed design document (krylov design,
  2026-10-17) defines the traits, error type, result semantics and the
  supported scope. Changing a trait or a result semantic is a design decision, not an
  implementation detail; extend traits only through defaulted methods, new
  traits or defaulted type parameters. The breaking 0.3 execution-context
  API (`Vector::Context`, `cx: &mut V::Context` on every operation touching
  vector data) and the required per-column `Basis::{inner_at, axpy_at}`
  (both approved 2026-10-30) are the amendments to that rule.
- **Reference.** KrylovKit.jl v0.10.4 is the semantic reference. Record every intentional deviation, with its reason,
  in `docs/krylovkit.md` and on the affected item's rustdoc.
- **Tests.** A test's expected value is readable from the test: a dense
  oracle (dev-dependency) on the same small matrix, a residual or other
  identity, a closed form, or a few hardcoded values with a comment naming
  where they were checked (e.g. "checked against KrylovKit.jl v0.10.4").
  Iteration and application counts that follow KrylovKit are a contract,
  pinned on a few representative cases with the same inputs and options.
  No generator scripts, Julia environments or fixture files are committed. Allocation
  contracts are tested by counting vector constructions, not by timing.
- **Gates.** `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`, `cargo doc` with `-D warnings`, and an MSRV (1.85) build.

## Test style

A test exists to protect a stated purpose; write the purpose first, then the
assertion that checks exactly it. Reviewers judge every test by that pair.

These extend the Tests bullet above; they do not replace it.

1. Each test protects one purpose, stated in its name: convergence, residual bound, orthogonality, a KrylovKit count correspondence, an allocation/vector-work budget, or a typed error.
2. Expected values come from a dense oracle on the same small matrix, an identity (residual, orthogonality, reconstruction), a closed form, or a few literals naming where they were checked (for example, KrylovKit.jl v0.10.4).
3. Accuracy bounds are derived from the solver's `tol`, the problem scale (`‖A‖`, `‖x‖`, `|t|`) and `S::Real::epsilon()`. A bare literal such as `1e-10` needs a one-line derivation.
4. Converged results also assert the residual bound KrylovKit promises: `‖A·x − λ·x‖ ≤ tol`, or the linsolve/expintegrator equivalent.
5. Iteration and application counts that follow KrylovKit are pinned only on a few representative cases with the same inputs and options, with a stated slack and its reason (for example, a CGS2 step difference).
6. Vector-work, `similar` and allocation budgets are derived from the outputs or the basis size (`2·howmany`, `K(K+3)/2`). Never assert recorded totals that depend on the convergence path.
7. Allocation tests compare runs (`short == long`, `warm == returned vectors`) and do not time anything.
8. Tests must not depend on a LAPACK provider's defects or exact iteration path. A provider-sensitive test says so and asserts only krylov's own guarantee.
9. Errors are asserted by variant (`matches!`), never by message text.
10. Cover f64, f32, c64 and c32 once per solver, in one parameterized test, not per file.
11. No issue numbers, commit hashes or measured timings in test names or comments.
12. Shared fixtures (`DVec`, `DenseOp`, `SplitMix`, the dense oracles) live in `tests/common`. Do not add another local copy.
