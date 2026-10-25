# krylov-rs repository policy

- **Scope: algorithms only.** The crate owns iteration control,
  orthogonalization, the small projected dense problems, convergence tests and
  restarts. It never owns large vector arithmetic, linear-map internals,
  backend adapters (symtn, tenferro, CUDA, MPI live with their owners),
  threading, caches or global state.
- **Dependencies.** Runtime: `num-traits`, `num-complex` and
  `faer = { package = "t4a-faer", version = "=0.24.9", default-features = false, features = ["std", "linalg"] }`.
  Small dense problems use sequential borrowed faer views and a private
  per-invocation arena; no BLAS/LAPACK provider is linked. Crate code is kept
  where KrylovKit has its own Julia code (Givens, φ functions, Householder
  restart restoration). Dev-dependencies may include one dense reference
  library for test oracles. Anything else needs explicit approval.
- **Design authority.** The fixed design document (krylov design,
  2026-10-17) defines the traits, error type, result semantics and the
  supported scope. Changing a trait or a result semantic is a design decision, not an
  implementation detail; extend traits only through defaulted methods, new
  traits or defaulted type parameters.
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
