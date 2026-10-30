//! The Hermitian Lanczos process shared by `eigsolve` and `exponentiate`.

use num_traits::{One, Zero};

use crate::error::Error;
use crate::orth::Orth;
use crate::scalar::sealed::{Ops as _, RealOps as _};
use crate::scalar::{Real, Scalar};
use crate::traits::{Basis, LinearMap, Vector, Workspace};

/// Lanczos algorithm settings for Hermitian (real symmetric or complex
/// Hermitian) linear maps.
///
/// Mirrors KrylovKit's `Lanczos`: `krylovdim` is the maximal Krylov
/// dimension, `maxiter` the number of Krylov subspaces built (restarts + 1),
/// `tol` an absolute residual norm for `eigsolve` and an error per unit time
/// for `exponentiate`, and `eager` (default `false`) evaluates the projected
/// problem after every expansion.
///
/// Each step follows KrylovKit's `lanczosrecurrence`: the three-term
/// recurrence, then one full pass of `orth` against the basis (`k + 1` inner
/// products and about as many axpys at basis size `k`).
///
/// Differs from KrylovKit: `orth` defaults to [`Orth::Cgs2`] (KrylovKit:
/// MGS2). See [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lanczos {
    krylovdim: usize,
    maxiter: usize,
    tol: f64,
    orth: Orth,
    eager: bool,
}

impl Lanczos {
    /// Settings with the given problem-dependent parameters; the variant
    /// options start at their documented defaults. The values are validated
    /// when a solver is called.
    ///
    /// Differs from KrylovKit: there these parameters have keyword defaults
    /// (`krylovdim = 30`, `maxiter = 100`, `tol = 1e-12`); here they are
    /// required because suitable values depend on the problem and the
    /// floating-point precision (see
    /// [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md)).
    pub fn new(krylovdim: usize, maxiter: usize, tol: f64) -> Self {
        Self {
            krylovdim,
            maxiter,
            tol,
            orth: Orth::Cgs2,
            eager: false,
        }
    }
    /// Sets the orthogonalizer.
    pub fn orth(mut self, orth: Orth) -> Self {
        self.orth = orth;
        self
    }
    /// Sets eager evaluation.
    pub fn eager(mut self, eager: bool) -> Self {
        self.eager = eager;
        self
    }
    /// Maximal Krylov dimension.
    pub fn get_krylovdim(&self) -> usize {
        self.krylovdim
    }
    /// Maximal number of Krylov subspaces built.
    pub fn get_maxiter(&self) -> usize {
        self.maxiter
    }
    /// Tolerance.
    pub fn get_tol(&self) -> f64 {
        self.tol
    }
    /// Orthogonalizer.
    pub fn get_orth(&self) -> Orth {
        self.orth
    }
    /// Eager evaluation.
    pub fn get_eager(&self) -> bool {
        self.eager
    }

    pub(crate) fn validate<E, V: Vector, B: Basis<V>>(
        &self,
        ws: &Workspace<V, B>,
    ) -> Result<Real<V>, Error<E>> {
        validate(self.krylovdim, self.maxiter, self.tol, ws)
    }
}

/// Checks the settings shared by Lanczos and Arnoldi against the workspace
/// and returns `tol` in the vector's real type.
pub(crate) fn validate<E, V: Vector, B: Basis<V>>(
    krylovdim: usize,
    maxiter: usize,
    tol: f64,
    ws: &Workspace<V, B>,
) -> Result<Real<V>, Error<E>> {
    if krylovdim == 0 {
        return Err(Error::InvalidArgument("krylovdim must be at least 1"));
    }
    if maxiter == 0 {
        return Err(Error::InvalidArgument("maxiter must be at least 1"));
    }
    if tol.is_nan() || tol < 0.0 {
        return Err(Error::InvalidArgument("tol must be non-negative"));
    }
    let capacity = ws.basis.capacity();
    if capacity < krylovdim {
        return Err(Error::InsufficientCapacity {
            needed: krylovdim,
            capacity,
        });
    }
    Ok(Real::<V>::from_f64(tol))
}

/// Tridiagonal Rayleigh quotient `T` with `A·V = V·T + w·e_kᵀ`, plus the
/// projection coefficients of the last step. `beta[k-1]` is the residual norm `‖w‖`.
pub(crate) struct Fact<S: Scalar> {
    pub(crate) alpha: Vec<S::Real>,
    pub(crate) beta: Vec<S::Real>,
    h: Vec<S>,
    pub(crate) max_imag: S::Real,
    pub(crate) numops: usize,
    orth: Orth,
}

impl<S: Scalar> Fact<S> {
    pub(crate) fn new(krylovdim: usize, orth: Orth) -> Self {
        Self {
            alpha: Vec::with_capacity(krylovdim),
            beta: Vec::with_capacity(krylovdim),
            h: Vec::with_capacity(krylovdim),
            max_imag: S::Real::zero(),
            numops: 0,
            orth,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.alpha.len()
    }

    pub(crate) fn normres(&self) -> S::Real {
        self.beta[self.len() - 1]
    }

    /// Starts a new factorization from `scale·x`, or from `scale·w` when `x`
    /// is `None`.
    pub(crate) fn start<V, B, A>(
        &mut self,
        cx: &mut V::Context,
        ws: &mut Workspace<V, B>,
        op: &mut A,
        x: Option<&V>,
        scale: S::Real,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        self.alpha.clear();
        self.beta.clear();
        ws.basis.truncate(0);
        let src = x.unwrap_or(&ws.w);
        ws.basis
            .push(cx, src, S::from_real(scale))
            .map_err(Error::Vector)?;
        ws.v.axpby(cx, S::from_real(scale), src, S::zero())
            .map_err(Error::Vector)?;
        self.step(cx, ws, op)
    }

    /// Appends `w/‖w‖` and performs one Lanczos step.
    pub(crate) fn expand<V, B, A>(
        &mut self,
        cx: &mut V::Context,
        ws: &mut Workspace<V, B>,
        op: &mut A,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        let inv = S::from_real(S::Real::one() / self.normres());
        ws.basis.push(cx, &ws.w, inv).map_err(Error::Vector)?;
        ws.v.axpby(cx, inv, &ws.w, S::zero())
            .map_err(Error::Vector)?;
        self.step(cx, ws, op)
    }

    /// `w ← A·v` orthogonalized against the basis; records `α`, `β`. A
    /// non-finite coefficient or `β` (NaN or ∞ from the map) is reported as
    /// soon as it is produced.
    fn step<V, B, A>(
        &mut self,
        cx: &mut V::Context,
        ws: &mut Workspace<V, B>,
        op: &mut A,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        op.apply_into(cx, &ws.v, &mut ws.w).map_err(Error::Vector)?;
        self.numops += 1;
        let k = ws.basis.len();
        self.h.clear();
        self.h.resize(k, S::zero());
        let prev = k.checked_sub(2).map(|i| (i, self.beta[i]));
        let (basis, v, w) = (&ws.basis, &ws.v, &mut ws.w);
        let alpha =
            recurrence(cx, basis, v, w, prev, &mut self.h, self.orth).map_err(Error::Vector)?;
        if !alpha.all_finite() {
            return Err(Error::NonFiniteIterate);
        }
        let im = num_traits::Float::abs(alpha.im());
        if im > self.max_imag {
            self.max_imag = im;
        }
        self.alpha.push(alpha.re());
        let beta = ws.w.norm(cx).map_err(Error::Vector)?;
        if !num_traits::Float::is_finite(beta) {
            return Err(Error::NonFiniteIterate);
        }
        self.beta.push(beta);
        Ok(())
    }
}

/// One step of KrylovKit's `lanczosrecurrence` for `w = A·v`, `v = v_{k-1}`:
/// the three-term recurrence removes `β·v_{k-2}` (`prev = (k-2, β)`) and
/// `α·v`, then one full pass of `orth` against the basis removes the rounding
/// that the recurrence leaves along older vectors. Returns `α` including the
/// second pass's `v` component.
///
/// A second full pass in place of the recurrence would double the inner
/// products and axpys per step without improving orthogonality: the
/// recurrence already removes the `O(‖A‖)` components exactly as a first
/// Gram–Schmidt pass would, and the remaining ones are at rounding level.
fn recurrence<V: Vector, B: Basis<V>>(
    cx: &mut V::Context,
    basis: &B,
    v: &V,
    w: &mut V,
    prev: Option<(usize, Real<V>)>,
    h: &mut [V::Scalar],
    orth: Orth,
) -> Result<V::Scalar, V::Error> {
    let one = V::Scalar::one();
    let k = basis.len();
    let subtract_prev = |cx: &mut V::Context, w: &mut V| match prev {
        Some((i, beta)) => basis.axpy_at(cx, i, -V::Scalar::from_real(beta), w),
        None => Ok(()),
    };
    match orth {
        Orth::Cgs2 => {
            let alpha = v.inner(cx, w)?;
            subtract_prev(cx, w)?;
            w.axpby(cx, -alpha, v, one)?;
            basis.project(cx, w, h)?;
            basis.combine_into(cx, h, w, -one, one)?;
            Ok(alpha + h[k - 1])
        }
        Orth::Mgs2 => {
            subtract_prev(cx, w)?;
            let alpha = v.inner(cx, w)?;
            w.axpby(cx, -alpha, v, one)?;
            let mut last = V::Scalar::zero();
            for i in 0..k {
                last = basis.inner_at(cx, i, w)?;
                basis.axpy_at(cx, i, -last, w)?;
            }
            Ok(alpha + last)
        }
    }
}
