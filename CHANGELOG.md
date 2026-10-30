# Changelog

## Unreleased (0.3.0)

### Breaking

- `Vector` gains `type Context: ?Sized`, the caller's execution context
  (`()` for host vectors). Every operation that touches vector data takes
  `cx: &mut V::Context`, and krylov forwards that one borrow sequentially
  without storing it:
  - `Vector::{similar, copy_from, inner, norm, axpby}`;
  - `Basis::{push, project, combine_into, rotate, inner_at, axpy_at}`
    (metadata methods `len`, `is_empty`, `capacity`, `truncate` are unchanged);
  - `LinearMap::apply_into`, `AdjointMap::apply_adjoint_into`, and the
    closure implementation, now `FnMut(&mut V::Context, &V, &mut W)`;
  - `VecBasis::with_capacity`, `Workspace::new`, `Workspace::with_capacity`;
  - every solver, as the first argument: `eigsolve`, `block_eigsolve`,
    `svdsolve`, `linsolve` (and `LinearSolver::solve`), `exponentiate`,
    `expintegrator`, `schursolve`, `arnoldi_eigsolve`, `realeigsolve`.
- `LinearMap<V, W>` and `AdjointMap<V, W>` require `W::Context = V::Context`.
- `Basis::inner_at` and `Basis::axpy_at` are required methods. Their former
  defaults ran a full `project`/`combine_into` per coefficient, which made
  MGS2 cost `O(k³)` instead of `O(k²)` vector operations per Krylov space of
  size `k`; implement them with one inner product and one vector update.
  Their `scratch` parameter is removed: `inner_at(cx, i, x)` and
  `axpy_at(cx, i, a, y)`.

Migration for host vectors: add `type Context = ();`, take `_: &mut ()` in
each vector method and closure map, and pass `&mut ()` to solvers and
workspace constructors. A custom `Basis` implements `inner_at(cx, i, x)`
and `axpy_at(cx, i, a, y)` (now required, without the former `scratch`
argument).

### Fixed

- `block_eigsolve` scales each residual vector by a power of two near its
  own norm before forming the residual Gram matrix and norms, so residual
  norms whose squares leave the floating-point range (e.g. `1e∓200` in
  `f64`, `1e∓30` in `f32`, or residual vectors of very different sizes) are
  no longer reported as zero with a false convergence. A residual norm that
  is not finite, or a non-finite Gram quadratic form, is
  `Error::NonFiniteIterate` instead of an `Ok` result with an infinite norm
  or a norm clamped to zero. Each convergence check adds one norm and one
  scaled copy per residual vector.
