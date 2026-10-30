//! The boundary traits between krylov and the caller's vector storage.

use crate::scalar::{MatRef, Real, Scalar};

/// One vector of the space the algorithm works in.
///
/// It may live on the host, on a device, or be distributed; krylov only calls
/// these operations. Independent storage is obtained through
/// [`similar`](Vector::similar), never through `Clone` (a vector type may
/// share storage on clone).
///
/// # `axpby` with `b = 0` overwrites
///
/// [`axpby`](Vector::axpby) with `b == 0` must write `self ← a·x` **without
/// reading `self`** (BLAS `beta = 0` semantics). krylov relies on this to
/// reuse workspace vectors whose old contents are stale or non-finite; an
/// implementation computing `0·self` would propagate NaN/Inf.
///
/// # Distributed contract
///
/// When vectors are distributed over ranks, krylov's branches stay identical
/// on every rank only if:
///
/// - the results of [`inner`](Vector::inner), [`norm`](Vector::norm) and
///   [`Basis::project`] are bitwise identical on every rank (reduced, then
///   replicated);
/// - every rank runs the same binary with the same libm and the same LAPACK
///   build (provider, version and thread count);
/// - an error is returned on every rank, not only on some (otherwise the
///   others block in the next reduction).
///
/// Under this contract all scalar decisions are deterministic. Outside LAPACK,
/// `eigsolve` branches only on arithmetic and `sqrt`;
/// `exponentiate`'s step acceptance additionally branches on libm `exp`,
/// `ln` and `powf`, which is why the same libm is required.
pub trait Vector: Sized {
    /// Scalar field (`f32`, `f64`, `Complex32`, `Complex64`).
    type Scalar: Scalar;
    /// Error of the vector operations.
    type Error;
    /// Execution context the vector operations run on (a device engine, a
    /// thread pool, a communicator, ...); `()` for vectors that need none.
    ///
    /// The caller owns it and passes `&mut` to every solver call; krylov
    /// forwards that one borrow, sequentially, to every vector, basis and map
    /// operation and never stores it. A context needed by both the vectors and
    /// the map is therefore reached through this one borrow, not captured by
    /// the map.
    ///
    /// A trait object is written `dyn Trait`, which as an associated type
    /// means `dyn Trait + 'static`: code forwarding to such a vector takes
    /// `&mut (dyn Trait + 'static)`. That bounds the object type, not the
    /// borrow, so a locally owned engine coerces to it.
    type Context: ?Sized;

    /// A new, independent, writable zero vector of the same space.
    fn similar(&self, cx: &mut Self::Context) -> Result<Self, Self::Error>;
    /// `self ← x`.
    fn copy_from(&mut self, cx: &mut Self::Context, x: &Self) -> Result<(), Self::Error>;
    /// `⟨self, y⟩`, conjugate-linear in `self`.
    fn inner(&self, cx: &mut Self::Context, y: &Self) -> Result<Self::Scalar, Self::Error>;
    /// Euclidean norm induced by [`inner`](Vector::inner).
    fn norm(&self, cx: &mut Self::Context) -> Result<Real<Self>, Self::Error>;
    /// `self ← a·x + b·self`.
    ///
    /// When `b` is zero, `self` must be overwritten without being read (see
    /// the trait documentation).
    fn axpby(
        &mut self,
        cx: &mut Self::Context,
        a: Self::Scalar,
        x: &Self,
        b: Self::Scalar,
    ) -> Result<(), Self::Error>;
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
    fn push(&mut self, cx: &mut V::Context, v: &V, a: V::Scalar) -> Result<(), V::Error>;
    /// Keeps the first `len` vectors (`0` empties the basis).
    fn truncate(&mut self, len: usize);
    /// `out ← Vᴴx`; `out.len() == len()`.
    fn project(&self, cx: &mut V::Context, x: &V, out: &mut [V::Scalar]) -> Result<(), V::Error>;
    /// `y ← a·V·c + b·y`; `c.len() == len()`.
    fn combine_into(
        &self,
        cx: &mut V::Context,
        c: &[V::Scalar],
        y: &mut V,
        a: V::Scalar,
        b: V::Scalar,
    ) -> Result<(), V::Error>;
    /// `V ← V·Q[:, ..keep]` for `Q` with `len()` rows and orthonormal columns;
    /// afterwards `len() == keep`.
    fn rotate(
        &mut self,
        cx: &mut V::Context,
        q: MatRef<'_, V::Scalar>,
        keep: usize,
    ) -> Result<(), V::Error>;

    /// `⟨v_i, x⟩`, used by modified Gram–Schmidt: one inner product.
    ///
    /// Required so that [`Orth::Mgs2`](crate::Orth::Mgs2) costs `O(len)`
    /// vector operations per orthogonalization; a full
    /// [`project`](Basis::project) here would make it `O(len²)`.
    fn inner_at(&self, cx: &mut V::Context, i: usize, x: &V) -> Result<V::Scalar, V::Error>;
    /// `y ← a·v_i + y`, used by modified Gram–Schmidt and by the Lanczos
    /// three-term recurrence (`β·v_{k-1}`) under both orthogonalisers: one
    /// vector update.
    ///
    /// Required for the same reason as [`inner_at`](Basis::inner_at).
    fn axpy_at(
        &self,
        cx: &mut V::Context,
        i: usize,
        a: V::Scalar,
        y: &mut V,
    ) -> Result<(), V::Error>;
}

/// A linear map `A` from vectors of type `V` (domain) to vectors of type `W`
/// (codomain, `V` by default), written into a destination.
///
/// Every solver except [`svdsolve`](crate::svdsolve) needs a square map and
/// takes `LinearMap<V>`. The two types share their scalar and error types so
/// that one [`Error`](crate::Error) carries the failures of both, and their
/// execution context, which the solver passes to the map as `cx`.
pub trait LinearMap<
    V: Vector,
    W: Vector<Scalar = V::Scalar, Error = V::Error, Context = V::Context> = V,
>
{
    /// `y ← A·x`. `y` holds arbitrary previous contents.
    fn apply_into(&mut self, cx: &mut V::Context, x: &V, y: &mut W) -> Result<(), V::Error>;
}

impl<V, W, F> LinearMap<V, W> for F
where
    V: Vector,
    W: Vector<Scalar = V::Scalar, Error = V::Error, Context = V::Context>,
    F: FnMut(&mut V::Context, &V, &mut W) -> Result<(), V::Error>,
{
    fn apply_into(&mut self, cx: &mut V::Context, x: &V, y: &mut W) -> Result<(), V::Error> {
        self(cx, x, y)
    }
}

/// A linear map `A: V → W` that can also apply its adjoint `Aᴴ: W → V`, as
/// `svdsolve` needs.
///
/// Domain and codomain may be different Rust vector types, or one type for
/// different spaces (e.g. vectors of different lengths, tensors with
/// different legs).
pub trait AdjointMap<
    V: Vector,
    W: Vector<Scalar = V::Scalar, Error = V::Error, Context = V::Context> = V,
>: LinearMap<V, W>
{
    /// `x ← Aᴴ·y`. `x` holds arbitrary previous contents.
    fn apply_adjoint_into(&mut self, cx: &mut V::Context, y: &W, x: &mut V)
    -> Result<(), V::Error>;
}

/// Working storage of one solve: the basis plus the operator input and
/// output vectors, i.e. the basis's vectors `+ 2`. With
/// [`with_capacity`](Workspace::with_capacity) (a [`VecBasis`](crate::VecBasis),
/// which keeps one extra scratch vector for `rotate`) that is `capacity + 3`
/// vectors in total.
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
    pub fn new(cx: &mut V::Context, basis: B, x: &V) -> Result<Self, V::Error> {
        Ok(Self {
            basis,
            v: x.similar(cx)?,
            w: x.similar(cx)?,
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
    /// in the space of `x`: allocates `capacity + 3` vectors (basis, one
    /// rotate scratch, operator input and output).
    pub fn with_capacity(cx: &mut V::Context, x: &V, capacity: usize) -> Result<Self, V::Error> {
        let basis = crate::VecBasis::with_capacity(cx, x, capacity)?;
        Self::new(cx, basis, x)
    }
}
