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
    /// The small dense eigenvalue computation did not converge.
    DenseNoConvergence,
    /// Conjugate gradients met a search direction `p` with `Re⟨p, A·p⟩ <= 0`
    /// (or non-finite): the map is not positive definite on the Krylov space.
    NotPositiveDefinite,
    /// The projected least-squares problem of GMRES has an exactly singular
    /// triangular factor: the map is singular on the Krylov space.
    SingularKrylovSpace,
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
                f.write_str("small dense eigenvalue problem did not converge")
            }
            Error::NotPositiveDefinite => {
                f.write_str("linear map is not positive definite on the Krylov space")
            }
            Error::SingularKrylovSpace => f.write_str("linear map is singular on the Krylov space"),
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
