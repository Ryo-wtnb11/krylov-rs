//! Krylov subspace algorithms over user-defined vector types.
//!
//! krylov owns only the procedure (iteration control, orthogonalization, the
//! small projected problem, convergence tests and restarts). Vector
//! arithmetic and the linear map belong to the caller through three traits:
//! [`Vector`], [`Basis`] and [`LinearMap`]. The API and the algorithms are
//! modelled on [KrylovKit.jl](https://github.com/Jutho/KrylovKit.jl).
//!
//! Stage 1 provides Hermitian Lanczos [`eigsolve`] (thick restart) and
//! [`exponentiate`]; [`expintegrator`] generalizes the latter to
//! `φ`-function combinations. [`linsolve`] solves `(a₀ + a₁·A)·x = b` by
//! [`Cg`] or [`Gmres`]. [`block_eigsolve`] handles degenerate
//! spectra with [`BlockLanczos`].
//!
//! ```
//! use krylov::{eigsolve, Lanczos, Vector, Which, Workspace};
//!
//! // A minimal host vector.
//! #[derive(Debug)]
//! struct V(Vec<f64>);
//! impl Vector for V {
//!     type Scalar = f64;
//!     type Error = core::convert::Infallible;
//!     fn similar(&self) -> Result<Self, Self::Error> { Ok(V(vec![0.0; self.0.len()])) }
//!     fn copy_from(&mut self, x: &Self) -> Result<(), Self::Error> {
//!         self.0.copy_from_slice(&x.0); Ok(())
//!     }
//!     fn inner(&self, y: &Self) -> Result<f64, Self::Error> {
//!         Ok(self.0.iter().zip(&y.0).map(|(a, b)| a * b).sum())
//!     }
//!     fn norm(&self) -> Result<f64, Self::Error> { Ok(self.inner(self)?.sqrt()) }
//!     fn axpby(&mut self, a: f64, x: &Self, b: f64) -> Result<(), Self::Error> {
//!         for (s, x) in self.0.iter_mut().zip(&x.0) {
//!             *s = if b == 0.0 { a * x } else { a * x + b * *s };
//!         }
//!         Ok(())
//!     }
//! }
//!
//! // 1D Laplacian with Dirichlet boundaries.
//! let n = 50;
//! let mut op = |x: &V, y: &mut V| {
//!     for i in 0..n {
//!         let l = if i > 0 { x.0[i - 1] } else { 0.0 };
//!         let r = if i + 1 < n { x.0[i + 1] } else { 0.0 };
//!         y.0[i] = 2.0 * x.0[i] - l - r;
//!     }
//!     Ok(())
//! };
//! let x0 = V((0..n).map(|i| 1.0 + (i as f64).sin()).collect());
//! let alg = Lanczos::new().krylovdim(20).tol(1e-10);
//! let mut ws = Workspace::with_capacity(&x0, alg.get_krylovdim())?;
//! let (vals, _vecs, info) = eigsolve(&mut op, &x0, 1, Which::SmallestReal, &alg, &mut ws)?;
//! let exact = 2.0 - 2.0 * (std::f64::consts::PI / (n as f64 + 1.0)).cos();
//! assert!(info.converged >= 1);
//! assert!((vals[0] - exact).abs() < 1e-9);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![warn(missing_docs)]

mod basis;
mod block_lanczos;
mod dense;
mod eigsolve;
mod error;
mod exponentiate;
mod lanczos;
mod linsolve;
mod orth;
mod scalar;
mod traits;

pub use basis::VecBasis;
pub use block_lanczos::{BlockLanczos, block_eigsolve};
pub use eigsolve::{ConvergenceInfo, Which, eigsolve};
pub use error::Error;
pub use exponentiate::{ExpInfo, expintegrator, exponentiate};
pub use lanczos::Lanczos;
pub use linsolve::{Cg, Gmres, LinearSolver, LinsolveInfo, linsolve};
pub use orth::Orth;
pub use scalar::{MatRef, Real, RealScalar, Scalar};
pub use traits::{Basis, LinearMap, Vector, Workspace};
