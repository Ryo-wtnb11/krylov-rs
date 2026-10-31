# krylov

[![CI](https://github.com/Ryo-wtnb11/krylov-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ryo-wtnb11/krylov-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/krylov.svg)](https://crates.io/crates/krylov)
[![docs.rs](https://docs.rs/krylov/badge.svg)](https://docs.rs/krylov)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Ryo-wtnb11/krylov-rs/badges/badge.json)](https://github.com/Ryo-wtnb11/krylov-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Ryo-wtnb11/krylov-rs/graph/badge.svg)](https://codecov.io/gh/Ryo-wtnb11/krylov-rs)
[![license](https://img.shields.io/crates/l/krylov.svg)](#license)

Matrix-free Krylov algorithms in Rust for user-defined vector types, inspired
by [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl/tree/v0.10.4).
krylov owns iteration, orthogonalization, small projected problems,
convergence tests and restarts. Callers own vector storage and map operations
through `Vector`, `Basis` and `LinearMap`, run on a caller-held execution
context (`Vector::Context`, `()` for host vectors) passed to every call.

Intentional differences include caller-owned workspaces, explicit cost and
accuracy parameters, CGS2 orthogonalization by default, and typed failures.
See the [KrylovKit correspondence](docs/krylovkit.md) for details and reasons.

## Installation

```toml
[dependencies]
krylov = "0.3"
lapack-src = { version = "0.13", features = ["openblas"] } # "accelerate" on macOS
```

The final binary must link a LAPACK provider; krylov depends only on the
[`lapack`](https://crates.io/crates/lapack) declarations. For the provider above:

```rust
extern crate lapack_src as _; // once, in the binary (or a crate it links)
```

See the [provider notes](docs/guide.md#lapack-provider) for system linking.

## Documentation

- [API documentation and runnable example](https://docs.rs/krylov)
- [User guide](docs/guide.md): solver selection, ownership, workspace sizes,
  convergence and vector contracts.
- [KrylovKit correspondence](docs/krylovkit.md): intentional differences and reasons.

## Acknowledgments

The API, the algorithms and their result semantics are modelled on
[KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl) by Jutho Haegeman and
contributors (reference revision: v0.10.4), and the vector interface on
[VectorInterface.jl](https://github.com/Jutho/VectorInterface.jl).

## Citation

If you use krylov in research, please cite KrylovKit.jl, on which it is
modelled: Jutho Haegeman, *KrylovKit*,
[doi:10.5281/zenodo.10622234](https://doi.org/10.5281/zenodo.10622234) (see
KrylovKit's [citation metadata](https://github.com/Jutho/KrylovKit.jl/blob/master/CITATION.cff)).
To refer to krylov itself, cite this repository and the version used. This
requests scholarly credit; it does not add a condition to the software
license.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
