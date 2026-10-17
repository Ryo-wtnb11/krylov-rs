//! The boundary traits between krylov and the caller's vector storage.

use crate::scalar::{MatRef, Real, Scalar};

/// One vector of the space the algorithm works in.
///
/// It may live on the host, on a device, or be distributed; krylov only calls
/// these operations. Independent storage is obtained through
/// [`similar`](Vector::similar), never through `Clone` (a vector type may
/// share storage on clone).
///
/// # Distributed contract
///
/// [`inner`](Vector::inner) and [`norm`](Vector::norm) must return bitwise
/// identical values on every rank, and an error must be returned on every rank,
/// so that all ranks take the same branches.
pub trait Vector: Sized {
    /// Scalar field (`f32`, `f64`, `Complex32`, `Complex64`).
    type Scalar: Scalar;
    /// Error of the vector operations.
    type Error;

    /// A new, independent, writable zero vector of the same space.
    fn similar(&self) -> Result<Self, Self::Error>;
    /// `self ← x`.
    fn copy_from(&mut self, x: &Self) -> Result<(), Self::Error>;
    /// `⟨self, y⟩`, conjugate-linear in `self`.
    fn inner(&self, y: &Self) -> Result<Self::Scalar, Self::Error>;
    /// Euclidean norm induced by [`inner`](Vector::inner).
    fn norm(&self) -> Result<Real<Self>, Self::Error>;
    /// `self ← a·x + b·self`.
    ///
    /// When `b` is zero, `self` must be overwritten without being read (as in
    /// BLAS), so stale non-finite contents do not propagate.
    fn axpby(&mut self, a: Self::Scalar, x: &Self, b: Self::Scalar) -> Result<(), Self::Error>;
}

/// Storage for an orthonormal basis `V = [v₀ … v_{len-1}]`.
///
/// All operations that act on the basis as a block live here, so a device or
/// distributed implementation can map [`project`](Basis::project) to one GEMV
/// and [`rotate`](Basis::rotate) to one GEMM.
pub trait Basis<V: Vector> {
    /// Number of stored basis vectors.
    fn len(&self) -> usize;
    /// `true` when no basis vector is stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Maximum number of basis vectors.
    fn capacity(&self) -> usize;
    /// Appends `a·v`. Normalization is the caller's choice of `a`.
    ///
    /// The caller guarantees `len() < capacity()`.
    fn push(&mut self, v: &V, a: V::Scalar) -> Result<(), V::Error>;
    /// Keeps the first `len` vectors (`0` empties the basis).
    fn truncate(&mut self, len: usize);
    /// `out ← Vᴴx`; `out.len() == len()`.
    fn project(&self, x: &V, out: &mut [V::Scalar]) -> Result<(), V::Error>;
    /// `y ← a·V·c + b·y`; `c.len() == len()`.
    fn combine_into(
        &self,
        c: &[V::Scalar],
        y: &mut V,
        a: V::Scalar,
        b: V::Scalar,
    ) -> Result<(), V::Error>;
    /// `V ← V·Q[:, ..keep]` for `Q` with `len()` rows and orthonormal columns;
    /// afterwards `len() == keep`.
    fn rotate(&mut self, q: MatRef<'_, V::Scalar>, keep: usize) -> Result<(), V::Error>;

    /// `⟨v_i, x⟩`, used by modified Gram–Schmidt.
    ///
    /// The default performs a full [`project`](Basis::project) into
    /// `scratch` (`scratch.len() == len()`), i.e. `len()` inner products;
    /// implementations with per-vector access should override it.
    fn inner_at(&self, i: usize, x: &V, scratch: &mut [V::Scalar]) -> Result<V::Scalar, V::Error> {
        self.project(x, scratch)?;
        Ok(scratch[i])
    }
    /// `y ← a·v_i + y`, used by modified Gram–Schmidt.
    ///
    /// The default performs a full [`combine_into`](Basis::combine_into)
    /// with a unit coefficient vector built in `scratch`
    /// (`scratch.len() == len()`); implementations with per-vector access
    /// should override it.
    fn axpy_at(
        &self,
        i: usize,
        a: V::Scalar,
        y: &mut V,
        scratch: &mut [V::Scalar],
    ) -> Result<(), V::Error> {
        let zero = <V::Scalar as num_traits::Zero>::zero();
        scratch.fill(zero);
        scratch[i] = a;
        self.combine_into(
            scratch,
            y,
            <V::Scalar as num_traits::One>::one(),
            <V::Scalar as num_traits::One>::one(),
        )
    }
}

/// A linear map `A` acting on vectors of type `V`, written into a destination.
pub trait LinearMap<V: Vector> {
    /// `y ← A·x`. `y` holds arbitrary previous contents.
    fn apply_into(&mut self, x: &V, y: &mut V) -> Result<(), V::Error>;
}

impl<V: Vector, F: FnMut(&V, &mut V) -> Result<(), V::Error>> LinearMap<V> for F {
    fn apply_into(&mut self, x: &V, y: &mut V) -> Result<(), V::Error> {
        self(x, y)
    }
}

/// Working storage of one solve: the basis plus the operator input and
/// output vectors (`capacity + 2` vectors in total).
///
/// Create it once per vector space and reuse it across calls; the algorithms
/// allocate no vectors from it during iterations or restarts.
pub struct Workspace<V, B> {
    pub(crate) basis: B,
    pub(crate) v: V,
    pub(crate) w: V,
}

impl<V: Vector, B: Basis<V>> Workspace<V, B> {
    /// Wraps `basis` and allocates the two operator vectors like `x`.
    pub fn new(basis: B, x: &V) -> Result<Self, V::Error> {
        Ok(Self {
            basis,
            v: x.similar()?,
            w: x.similar()?,
        })
    }
    /// The basis left by the last solve.
    pub fn basis(&self) -> &B {
        &self.basis
    }
    /// Returns the basis.
    pub fn into_basis(self) -> B {
        self.basis
    }
}

impl<V: Vector> Workspace<V, crate::VecBasis<V>> {
    /// A workspace with a [`VecBasis`](crate::VecBasis) of `capacity` vectors
    /// in the space of `x`.
    pub fn with_capacity(x: &V, capacity: usize) -> Result<Self, V::Error> {
        Self::new(crate::VecBasis::with_capacity(x, capacity)?, x)
    }
}
