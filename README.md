# krylov

[![CI](https://github.com/Ryo-wtnb11/krylov-rs/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Ryo-wtnb11/krylov-rs/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/krylov.svg)](https://crates.io/crates/krylov)
[![docs.rs](https://docs.rs/krylov/badge.svg)](https://docs.rs/krylov)
[![coverage](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Ryo-wtnb11/krylov-rs/badges/badge.json)](https://github.com/Ryo-wtnb11/krylov-rs/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Ryo-wtnb11/krylov-rs/graph/badge.svg)](https://codecov.io/gh/Ryo-wtnb11/krylov-rs)
[![license](https://img.shields.io/crates/l/krylov.svg)](#license)

Krylov subspace algorithms in Rust for user-defined vector types, inspired by
KrylovKit.jl.

krylov provides only the procedure: iteration, orthogonalization, the small
projected problem, convergence tests and restarts (eigsolve, svdsolve,
linsolve, exponentiate, …). Vector arithmetic and the linear map stay with the
caller behind three traits (Vector, Basis, LinearMap), so the same solvers run
on Vec<f64>, symmetric tensors, or device and distributed storage.

Unlike KrylovKit, which builds on Julia's generic vector interface, krylov
separates the procedure from the storage: the iteration runs in a
caller-owned workspace and allocates no vectors inside the iteration;
per-call scratch and results are documented on each entry point. Allocation
and data placement (host, device, distributed) stay with the caller.
Differences in detail are listed in [docs/](docs/krylovkit.md).

## Status and installation

Pre-1.0: the API may change between minor versions.

```toml
[dependencies]
krylov = "0.2"
lapack-src = { version = "0.13", features = ["openblas"] } # or "accelerate", "intel-mkl-dynamic-parallel", "netlib"
```

The small projected problems call LAPACK. krylov depends only on the
[`lapack`](https://crates.io/crates/lapack) declarations and never selects an
implementation; the final binary links one, for example by naming
`lapack-src` once:

```rust
extern crate lapack_src as _; // once, in the binary (or a crate it links)
```

Other providers are described in the [user guide](docs/guide.md#lapack-provider).

## Example

```rust
use krylov::{eigsolve, exponentiate, Error, Lanczos, LinearMap, Vector, Which, Workspace};

fn evolve_ground_state<V: Vector<Scalar = f64>>(
    h: &mut impl LinearMap<V>,
    x0: &V,
) -> Result<V, Error<V::Error>> {
    let alg = Lanczos::new(30, 100, 1e-12); // krylovdim, maxiter, tol
    // The only basis allocations: done once, reused while the space is fixed.
    let mut ws = Workspace::with_capacity(x0, alg.get_krylovdim()).map_err(Error::Vector)?;
    let (_energies, mut states, info) = eigsolve(h, x0, 1, Which::SmallestReal, &alg, &mut ws)?;
    assert!(info.converged >= 1);
    // exp(-0.1·H)·ψ with the same workspace.
    let (y, _) = exponentiate(h, -0.1, &states.swap_remove(0), &alg, &mut ws)?;
    Ok(y)
}
```

A runnable version with a `Vec<f64>` vector type is in the
[crate documentation](https://docs.rs/krylov).

## Documentation

- [API documentation](https://docs.rs/krylov)
- [User guide](docs/guide.md): parameters, workspace sizes, LAPACK
  providers, the vector contract and distributed use, low-rank and
  tensor-network vectors.
- [KrylovKit correspondence](docs/krylovkit.md): what is implemented and
  every difference from KrylovKit.jl with its reason.

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
