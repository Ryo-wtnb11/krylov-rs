//! The Hermitian Lanczos process shared by `eigsolve` and `exponentiate`.

use num_traits::{One, Zero};

use crate::error::Error;
use crate::orth::{Orth, orthogonalize};
use crate::scalar::sealed::RealOps as _;
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
/// Differs from KrylovKit: `orth` defaults to [`Orth::Cgs2`] (KrylovKit:
/// MGS2), and each Lanczos step orthogonalizes against the full basis twice,
/// where KrylovKit's CGS2 recurrence uses the three-term recurrence for its
/// first pass. See [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md).
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

/// Tridiagonal Rayleigh quotient `T` with `A·V = V·T + w·e_kᵀ`, plus scalar
/// scratch. `beta[k-1]` is the residual norm `‖w‖`.
pub(crate) struct Fact<S: Scalar> {
    pub(crate) alpha: Vec<S::Real>,
    pub(crate) beta: Vec<S::Real>,
    h: Vec<S>,
    scratch: Vec<S>,
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
            scratch: Vec::with_capacity(krylovdim),
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
            .push(src, S::from_real(scale))
            .map_err(Error::Vector)?;
        ws.v.axpby(S::from_real(scale), src, S::zero())
            .map_err(Error::Vector)?;
        self.step(ws, op)
    }

    /// Appends `w/‖w‖` and performs one Lanczos step.
    pub(crate) fn expand<V, B, A>(
        &mut self,
        ws: &mut Workspace<V, B>,
        op: &mut A,
    ) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        let inv = S::from_real(S::Real::one() / self.normres());
        ws.basis.push(&ws.w, inv).map_err(Error::Vector)?;
        ws.v.axpby(inv, &ws.w, S::zero()).map_err(Error::Vector)?;
        self.step(ws, op)
    }

    /// `w ← A·v` orthogonalized against the basis; records `α`, `β`. A
    /// non-finite coefficient or `β` (NaN or ∞ from the map) is reported as
    /// soon as it is produced.
    fn step<V, B, A>(&mut self, ws: &mut Workspace<V, B>, op: &mut A) -> Result<(), Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
        A: LinearMap<V>,
    {
        op.apply_into(&ws.v, &mut ws.w).map_err(Error::Vector)?;
        self.numops += 1;
        let k = ws.basis.len();
        self.h.clear();
        self.h.resize(k, S::zero());
        self.scratch.clear();
        self.scratch.resize(k, S::zero());
        let (basis, w, h, s) = (&ws.basis, &mut ws.w, &mut self.h[..], &mut self.scratch[..]);
        orthogonalize(basis, w, h, s, self.orth).map_err(Error::Vector)?;
        if !h.iter().all(|c| c.all_finite()) {
            return Err(Error::NonFiniteIterate);
        }
        let alpha = h[k - 1];
        let im = num_traits::Float::abs(alpha.im());
        if im > self.max_imag {
            self.max_imag = im;
        }
        self.alpha.push(alpha.re());
        let beta = ws.w.norm().map_err(Error::Vector)?;
        if !num_traits::Float::is_finite(beta) {
            return Err(Error::NonFiniteIterate);
        }
        self.beta.push(beta);
        Ok(())
    }
}
