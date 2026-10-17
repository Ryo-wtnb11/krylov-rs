//! The krylov error type.

use core::fmt;

/// Failure of a krylov algorithm.
///
/// `E` is the error type of the vector implementation. Non-convergence is
/// not an error: it is reported through the convergence information. The
/// enum is non-exhaustive so that new algorithms can add their own failures.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Error<E> {
    /// A vector or basis operation failed; the error is passed through.
    Vector(E),
    /// The start vector has norm zero.
    ZeroStartVector,
    /// The start vector contains NaN or infinite entries.
    NonFiniteStartVector,
    /// More eigenvalues were requested than the Krylov dimension allows.
    HowmanyTooLarge {
        /// Requested number of eigenvalues.
        howmany: usize,
        /// Configured Krylov dimension.
        krylovdim: usize,
    },
    /// The basis cannot hold the configured Krylov dimension.
    InsufficientCapacity {
        /// Number of basis vectors the algorithm needs.
        needed: usize,
        /// Capacity of the supplied basis.
        capacity: usize,
    },
    /// An algorithm parameter is outside its domain.
    InvalidArgument(&'static str),
    /// The small dense eigenvalue or singular value computation (LAPACK)
    /// failed: no convergence, or non-finite entries in the projected matrix.
    DenseNoConvergence,
    /// Conjugate gradients met a search direction `p` with `Re⟨p, A·p⟩ <= 0`
    /// (or non-finite): the map is not positive definite on the Krylov space.
    NotPositiveDefinite,
    /// The projected least-squares problem of GMRES has an exactly singular
    /// triangular factor: the map is singular on the Krylov space.
    SingularKrylovSpace,
    /// `svdsolve`'s start check `⟨u, A·v⟩ = ‖Aᴴu‖` (with `v = Aᴴu/‖Aᴴu‖`)
    /// failed: the adjoint does not belong to the map.
    IncompatibleAdjoint,
    /// A map application produced a vector with a non-finite norm during
    /// the iteration. In `expintegrator`/`exponentiate` also: the time-step
    /// error estimate stays non-finite at the smallest allowed step, or a
    /// step's update coefficients are not finite (typically `exp(t·A)`
    /// overflows).
    NonFiniteIterate,
    /// Reordering the small Schur form was rejected by LAPACK `xTREXC`
    /// (a swap of nearly equal or ill-conditioned 2×2 blocks).
    SchurReorderFailed,
    /// `realeigsolve`: the eigenvalue at `index` (0-based, in `which` order)
    /// of the requested ones is complex, i.e. its 2×2 Schur block couples by
    /// more than `tol`.
    ComplexEigenvalue {
        /// Position of the first eigenvalue of the complex pair.
        index: usize,
    },
}

impl<E: fmt::Display> fmt::Display for Error<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Vector(e) => write!(f, "vector operation failed: {e}"),
            Error::ZeroStartVector => f.write_str("start vector has norm zero"),
            Error::NonFiniteStartVector => f.write_str("start vector is not finite"),
            Error::HowmanyTooLarge { howmany, krylovdim } => write!(
                f,
                "krylov dimension {krylovdim} too small to compute {howmany} eigenvalues"
            ),
            Error::InsufficientCapacity { needed, capacity } => write!(
                f,
                "basis capacity {capacity} is smaller than the required {needed}"
            ),
            Error::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Error::DenseNoConvergence => {
                f.write_str("small dense eigenvalue or singular value problem did not converge")
            }
            Error::NotPositiveDefinite => {
                f.write_str("linear map is not positive definite on the Krylov space")
            }
            Error::SingularKrylovSpace => f.write_str("linear map is singular on the Krylov space"),
            Error::IncompatibleAdjoint => {
                f.write_str("adjoint map is not compatible with the linear map")
            }
            Error::NonFiniteIterate => f.write_str("linear map produced a non-finite vector"),
            Error::SchurReorderFailed => f.write_str("reordering the small Schur form failed"),
            Error::ComplexEigenvalue { index } => {
                write!(f, "requested eigenvalue {index} is complex, not real")
            }
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for Error<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Vector(e) => Some(e),
            _ => None,
        }
    }
}
