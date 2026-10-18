//! Non-Hermitian eigenvalue problems by the Krylov–Schur method (Arnoldi
//! with thick restarts), after KrylovKit v0.10.4 `src/eigsolve/arnoldi.jl`.

use num_complex::Complex;
use num_traits::{Float, One, Zero};

use crate::dense::{Work, arrowhead_to_band, hschur, permute_schur, schur_eigvals, schur_eigvecs};
use crate::eigsolve::{ConvergenceInfo, Which};
use crate::error::Error;
use crate::orth::{Orth, orthogonalize};
use crate::scalar::sealed::{Ops as _, RealOps as _};
use crate::scalar::{MatRef, Real, RealScalar, Scalar};
use crate::traits::{Basis, LinearMap, Vector, Workspace};

/// Arnoldi settings for general (non-Hermitian) linear maps.
///
/// Mirrors KrylovKit's `Arnoldi`: `krylovdim` is the maximal
/// Krylov dimension, `maxiter` the number of Krylov subspaces
/// built (restarts + 1), `tol` an absolute bound on the
/// residual norm, and `eager` (default `false`) evaluates the projected
/// problem after every expansion. The default orthogonalizer is
/// [`Orth::Cgs2`] (KrylovKit: MGS2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Arnoldi {
    krylovdim: usize,
    maxiter: usize,
    tol: f64,
    orth: Orth,
    eager: bool,
}

impl Arnoldi {
    /// Settings with the given problem-dependent parameters; the variant
    /// options start at their documented defaults. The values are validated
    /// when a solver is called.
    ///
    /// Differs from KrylovKit: there these parameters have keyword defaults
    /// (`krylovdim = 30`, `maxiter = 100`, `tol = 1e-12`); here they are required because suitable values depend on the
    /// problem and the floating-point precision (see
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
}

/// Arnoldi factorization `A·V = V·H + r·e_kᵀ` with the Hessenberg `H` stored
/// as `(kd+1) × kd` (the subdiagonal of the last column is `‖r‖`). `ws.v` is
/// the current basis vector and `ws.w` the residual `r`.
pub(crate) struct Fact<S: Scalar> {
    pub(crate) h: Vec<S>,
    pub(crate) ld: usize,
    pub(crate) k: usize,
    col: Vec<S>,
    scratch: Vec<S>,
    pub(crate) numops: usize,
    orth: Orth,
}

impl<S: Scalar> Fact<S> {
    pub(crate) fn new(kd: usize, orth: Orth) -> Self {
        Self {
            h: vec![S::zero(); (kd + 1) * kd],
            ld: kd + 1,
            k: 0,
            col: Vec::with_capacity(kd),
            scratch: Vec::with_capacity(kd),
            numops: 0,
            orth,
        }
    }

    pub(crate) fn normres(&self) -> S::Real {
        self.h[self.k + (self.k - 1) * self.ld].re()
    }

    /// Appends `scale·src` (`src` = `x`, or the residual when `None`) to the
    /// basis and performs one Arnoldi step.
    pub(crate) fn push<V, B, A>(
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
        let ve = Error::Vector;
        let src = x.unwrap_or(&ws.w);
        ws.basis.push(src, S::from_real(scale)).map_err(ve)?;
        ws.v.axpby(S::from_real(scale), src, S::zero())
            .map_err(ve)?;
        op.apply_into(&ws.v, &mut ws.w).map_err(ve)?;
        self.numops += 1;
        let k = ws.basis.len();
        self.col.clear();
        self.col.resize(k, S::zero());
        self.scratch.clear();
        self.scratch.resize(k, S::zero());
        orthogonalize(
            &ws.basis,
            &mut ws.w,
            &mut self.col,
            &mut self.scratch,
            self.orth,
        )
        .map_err(ve)?;
        let beta = ws.w.norm().map_err(ve)?;
        if !beta.is_finite() || !self.col.iter().all(|c| c.all_finite()) {
            return Err(Error::NonFiniteIterate);
        }
        let j = k - 1;
        self.h[j * self.ld..(j + 1) * self.ld].fill(S::zero());
        self.h[j * self.ld..j * self.ld + k].copy_from_slice(&self.col);
        self.h[k + j * self.ld] = S::from_real(beta);
        self.k = k;
        Ok(())
    }
}

/// Outcome of the Krylov–Schur iteration: the Schur form `t` (`k × k`) of
/// the last projected matrix, its Schur vectors `u`, and the residual norm.
/// The basis is in `ws.basis`, the residual in `ws.w`.
struct Outcome<S: Scalar> {
    t: Vec<S>,
    u: Vec<S>,
    k: usize,
    beta: S::Real,
    converged: usize,
    numiter: usize,
    numops: usize,
}

/// `|λ|`, `Re λ` or `Im λ` keyed so that ascending order is `which` order.
fn sort_key<R: RealScalar>(which: Which, l: Complex<R>) -> R {
    match which {
        Which::LargestMagnitude => -l.re.hypot2(l.im),
        Which::LargestReal => -l.re,
        Which::SmallestReal => l.re,
        Which::LargestImag => -l.im,
        Which::SmallestImag => l.im,
    }
}

/// Port of KrylovKit `_schursolve`.
fn krylov_schur<V, B, A>(
    op: &mut A,
    x0: &V,
    howmany: usize,
    which: Which,
    alg: &Arnoldi,
    ws: &mut Workspace<V, B>,
) -> Result<Outcome<V::Scalar>, Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
{
    let kd = alg.krylovdim;
    if kd == 0 {
        return Err(Error::InvalidArgument("krylovdim must be at least 1"));
    }
    if alg.maxiter == 0 {
        return Err(Error::InvalidArgument("maxiter must be at least 1"));
    }
    if alg.tol.is_nan() || alg.tol < 0.0 {
        return Err(Error::InvalidArgument("tol must be non-negative"));
    }
    if which.is_imag() && !V::Scalar::COMPLEX {
        return Err(Error::InvalidArgument(
            "LargestImag/SmallestImag split complex-conjugate pairs; use a complex vector type",
        ));
    }
    let capacity = ws.basis.capacity();
    if capacity < kd {
        return Err(Error::InsufficientCapacity {
            needed: kd,
            capacity,
        });
    }
    if howmany > kd {
        return Err(Error::HowmanyTooLarge {
            howmany,
            krylovdim: kd,
        });
    }
    let beta0 = x0.norm().map_err(Error::Vector)?;
    if !beta0.is_finite() {
        return Err(Error::NonFiniteStartVector);
    }
    if beta0 == Real::<V>::zero() {
        return Err(Error::ZeroStartVector);
    }
    let tol = Real::<V>::from_f64(alg.tol);
    let s0 = V::Scalar::zero();
    let r0 = Real::<V>::zero();

    let mut fact = Fact::<V::Scalar>::new(kd, alg.orth);
    ws.basis.truncate(0);
    fact.push(ws, op, Some(x0), Real::<V>::one() / beta0)?;
    let mut numiter = 1;

    let mut t = vec![s0; kd * kd];
    let mut u = vec![s0; kd * kd];
    let mut f = vec![s0; kd];
    let mut ev = vec![Complex::new(r0, r0); kd];
    let mut order: Vec<usize> = Vec::with_capacity(kd);
    let mut perm = Vec::with_capacity(kd);
    let mut hb = vec![s0; (kd + 1) * kd];
    let mut refl = vec![s0; kd];
    let mut lw = Work::default();
    let mut converged = 0;
    let mut kk = 1;
    let mut beta_k = fact.normres();

    loop {
        let k = fact.k;
        let beta = fact.normres();
        if k == kd || beta <= tol || (alg.eager && k >= howmany) {
            kk = k;
            beta_k = beta;
            let (t, u) = (&mut t[..k * k], &mut u[..k * k]);
            for j in 0..k {
                t[j * k..(j + 1) * k].copy_from_slice(&fact.h[j * fact.ld..j * fact.ld + k]);
            }
            u.fill(s0);
            for i in 0..k {
                u[i + i * k] = V::Scalar::one();
            }
            hschur(t, u, k, &mut ev[..k], &mut lw).map_err(|()| Error::DenseNoConvergence)?;
            order.clear();
            order.extend(0..k);
            // Stable, as Julia's `sortperm(values; by, rev)`.
            order.sort_by(|&a, &b| {
                sort_key(which, ev[a])
                    .partial_cmp(&sort_key(which, ev[b]))
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            permute_schur(t, u, k, &order, &mut perm, &mut lw)
                .map_err(|()| Error::SchurReorderFailed)?;
            for (i, fi) in f[..k].iter_mut().enumerate() {
                *fi = u[(k - 1) + i * k].conj() * V::Scalar::from_real(beta);
            }
            converged = f[..k].iter().take_while(|x| x.modulus() <= tol).count();
            if 0 < converged && converged < k && t[converged + (converged - 1) * k] != s0 {
                converged -= 1;
            }
            if converged >= howmany || beta <= tol {
                break;
            }
        }
        if k < kd {
            fact.push(ws, op, None, Real::<V>::one() / beta)?;
            continue;
        }
        // Shrink: keep (3·krylovdim + 2·converged)/5 Schur vectors, not
        // splitting a 2×2 block. With krylovdim 1 nothing can be kept.
        if numiter == alg.maxiter {
            break;
        }
        let mut keep = (3 * kd + 2 * converged) / 5;
        if keep == 0 {
            break;
        }
        if t[keep + (keep - 1) * k] != s0 {
            if keep > 1 {
                keep -= 1;
            } else {
                keep += 1;
                if kd == 2 {
                    break;
                }
            }
        }
        // KrylovKit `_restorearnoldiform!`: [T[..keep, ..keep]; fᴴ] back to
        // Hessenberg form with the residual coupling only to column keep.
        let ldk = keep + 1;
        let hb = &mut hb[..ldk * keep];
        for j in 0..keep {
            hb[j * ldk..j * ldk + keep].copy_from_slice(&t[j * k..j * k + keep]);
            hb[keep + j * ldk] = f[j].conj();
        }
        arrowhead_to_band(hb, keep, 1, &mut u[..k * k], k, &mut refl);
        fact.h.fill(s0);
        for j in 0..keep {
            let rows = (j + 2).min(ldk);
            fact.h[j * fact.ld..j * fact.ld + rows].copy_from_slice(&hb[j * ldk..j * ldk + rows]);
        }
        fact.k = keep;
        ws.basis
            .rotate(MatRef::new(&u[..k * keep], k, keep), keep)
            .map_err(Error::Vector)?;
        // The residual direction is unchanged; its new norm is H[keep, keep-1].
        let s = V::Scalar::from_real(fact.normres() / beta);
        ws.w.axpby(V::Scalar::zero(), &ws.v, s)
            .map_err(Error::Vector)?;
        numiter += 1;
    }
    let k = kk;
    t.truncate(k * k);
    u.truncate(k * k);
    Ok(Outcome {
        t,
        u,
        k,
        beta: beta_k,
        converged,
        numiter,
        numops: fact.numops,
    })
}

/// The `n` vectors `V·X[:, i]` and residuals `r·X[k-1, i]` for the `k × n`
/// coefficient matrix `x`, with residual norms `β·|X[k-1, i]|`.
#[allow(clippy::type_complexity)]
fn assemble<V, B>(
    ws: &mut Workspace<V, B>,
    x0: &V,
    x: &[V::Scalar],
    k: usize,
    n: usize,
    beta: Real<V>,
) -> Result<(Vec<V>, Vec<V>, Vec<Real<V>>), Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
{
    let ve = Error::Vector;
    let (mut vectors, mut residual, mut normres) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for col in x[..k * n].chunks_exact(k) {
        let mut v = x0.similar().map_err(ve)?;
        ws.basis
            .combine_into(col, &mut v, V::Scalar::one(), V::Scalar::zero())
            .map_err(ve)?;
        vectors.push(v);
        let mut r = x0.similar().map_err(ve)?;
        r.axpby(col[k - 1], &ws.w, V::Scalar::zero()).map_err(ve)?;
        residual.push(r);
        normres.push(beta * col[k - 1].modulus());
    }
    Ok((vectors, residual, normres))
}

/// KrylovKit's `howmany′`: one more to complete a real 2×2 pair, fewer on an
/// invariant subspace, and all converged ones when more converged.
fn n_returned<S: Scalar>(o: &Outcome<S>, howmany: usize) -> usize {
    let k = o.k;
    let mut n = howmany;
    if !S::COMPLEX && howmany > 0 && howmany < k && o.t[howmany + (howmany - 1) * k] != S::zero() {
        n += 1;
    } else if k < howmany {
        n = k;
    }
    if o.converged > howmany {
        n = o.converged;
    }
    n
}

fn info<V: Vector>(
    o: &Outcome<V::Scalar>,
    converged: usize,
    residual: Vec<V>,
    normres: Vec<Real<V>>,
    max_imag: Real<V>,
) -> ConvergenceInfo<V> {
    ConvergenceInfo {
        converged,
        residual,
        normres,
        numiter: o.numiter,
        numops: o.numops,
        max_imag,
    }
}

/// Partial Schur decomposition `A·Q = Q·T` with `howmany` eigenvalues of the
/// general map `op` selected by `which`, by the Krylov–Schur method from
/// `x0`.
///
/// Follows KrylovKit v0.10.4 `schursolve(A, x₀, howmany, which, ::Arnoldi)`.
/// Returns `(T, Q, λ, info)`: the `n × n` Schur factor `T` (column-major;
/// triangular for complex vectors, quasi-triangular with standardized 2×2
/// blocks for real ones), the `n` orthonormal Schur vectors, the `n`
/// eigenvalues (always complex) and the convergence information, whose
/// residuals are `A·qᵢ - Σ_j q_j·T[j, i]` (norms from the dense problem).
/// `n` is `howmany`, one more to complete a real complex-conjugate pair,
/// fewer on an invariant subspace, or the number converged when larger.
///
/// `LargestImag`/`SmallestImag` need a complex vector type. The workspace
/// basis needs capacity `>= alg.get_krylovdim()`; only the `2·n` returned
/// vectors are allocated.
///
/// Differs from KrylovKit (see [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md)); the same holds for
/// [`arnoldi_eigsolve`] and [`realeigsolve`]:
/// - Before LAPACK `hseqr` the Rayleigh quotient is scaled by a power of four
///   into `[sqrt(SAFMIN)/ULP, its inverse]`, as LAPACK's `xGEEV`/`xGEES`
///   drivers do; KrylovKit calls `hseqr` unscaled, which deflates a quotient
///   with entries near `1e-300` to its diagonal. In-range quotients are
///   passed to LAPACK unchanged.
/// - A reordering swap rejected by LAPACK `trexc` is
///   [`Error::SchurReorderFailed`] (KrylovKit throws `LAPACKException`).
/// - With `krylovdim == 1` the first subspace is final (KrylovKit indexes out
///   of bounds).
/// - `LargestImag`/`SmallestImag` with real vectors are an
///   [`Error::InvalidArgument`]; KrylovKit's `schursolve` fails later trying
///   to split a 2×2 block.
#[allow(clippy::type_complexity)]
pub fn schursolve<V, B, A>(
    op: &mut A,
    x0: &V,
    howmany: usize,
    which: Which,
    alg: &Arnoldi,
    ws: &mut Workspace<V, B>,
) -> Result<
    (
        Vec<V::Scalar>,
        Vec<V>,
        Vec<Complex<Real<V>>>,
        ConvergenceInfo<V>,
    ),
    Error<V::Error>,
>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
{
    let o = krylov_schur(op, x0, howmany, which, alg, ws)?;
    let (k, n) = (o.k, n_returned(&o, howmany));
    let mut tt = vec![V::Scalar::zero(); n * n];
    for j in 0..n {
        tt[j * n..(j + 1) * n].copy_from_slice(&o.t[j * k..j * k + n]);
    }
    let mut values = Vec::with_capacity(n);
    schur_eigvals(&tt, n, n, &mut values);
    let (vectors, residual, normres) = assemble(ws, x0, &o.u, k, n, o.beta)?;
    let info = info(&o, o.converged, residual, normres, Real::<V>::zero());
    Ok((tt, vectors, values, info))
}

/// Eigenpairs of the general map `op` for a complex vector type, by the
/// Krylov–Schur method (KrylovKit v0.10.4 `eigsolve(A, x₀, howmany, which,
/// ::Arnoldi)`).
///
/// Returns `n` eigenvalues and unit-norm eigenvectors (`n` as in
/// [`schursolve`]) with residuals `A·xᵢ - λᵢ·xᵢ`. The eigenvectors of the
/// Schur factor come from LAPACK `xTREVC`. Real maps whose eigenvectors are
/// complex need a complex vector type; for real eigenvalues of a real map
/// see [`realeigsolve`].
///
/// The workspace basis needs capacity `>= alg.get_krylovdim()`; only the
/// `2·n` returned vectors are allocated. The differences from KrylovKit
/// listed on [`schursolve`] apply.
#[allow(clippy::type_complexity)]
pub fn arnoldi_eigsolve<V, B, A, R>(
    op: &mut A,
    x0: &V,
    howmany: usize,
    which: Which,
    alg: &Arnoldi,
    ws: &mut Workspace<V, B>,
) -> Result<(Vec<Complex<R>>, Vec<V>, ConvergenceInfo<V>), Error<V::Error>>
where
    V: Vector<Scalar = Complex<R>>,
    R: RealScalar,
    Complex<R>: Scalar<Real = R>,
    B: Basis<V>,
    A: LinearMap<V>,
{
    let o = krylov_schur(op, x0, howmany, which, alg, ws)?;
    let (k, n) = (o.k, n_returned(&o, howmany));
    let zero = Complex::new(R::zero(), R::zero());
    let mut tt = vec![zero; n * n];
    for j in 0..n {
        tt[j * n..(j + 1) * n].copy_from_slice(&o.t[j * k..j * k + n]);
    }
    let values: Vec<Complex<R>> = (0..n).map(|i| tt[i + i * n]).collect();
    let x = eigvec_coefficients(&mut tt, &o.u, k, n)?;
    let (vectors, residual, normres) = assemble(ws, x0, &x, k, n, o.beta)?;
    let info = info(&o, o.converged, residual, normres, R::zero());
    Ok((values, vectors, info))
}

/// `U[:, ..n]·X_T` (`k × n`) with `X_T` the unit eigenvectors of the
/// triangular `tt` (`n × n`).
fn eigvec_coefficients<S: Scalar, E>(
    tt: &mut [S],
    u: &[S],
    k: usize,
    n: usize,
) -> Result<Vec<S>, Error<E>> {
    let mut vr = vec![S::zero(); n * n];
    schur_eigvecs(tt, n, &mut vr, &mut Work::default()).map_err(|()| Error::DenseNoConvergence)?;
    let mut x = vec![S::zero(); k * n];
    for j in 0..n {
        for l in 0..n {
            let c = vr[l + j * n];
            if c == S::zero() {
                continue;
            }
            for i in 0..k {
                x[i + j * k] += u[i + l * k] * c;
            }
        }
    }
    Ok(x)
}

/// Real eigenpairs of a real map (real vector type), by the Krylov–Schur
/// method (KrylovKit v0.10.4 `realeigsolve(A, x₀, howmany, which,
/// ::Arnoldi)`).
///
/// Returns `n ≥ howmany` real eigenvalues (more when more converged and are
/// real, fewer on an invariant subspace) with unit-norm real eigenvectors and
/// residuals `A·xᵢ - λᵢ·xᵢ`. A 2×2 Schur block among the first `howmany`
/// that couples by more than `tol` is a complex pair:
/// [`Error::ComplexEigenvalue`] (KrylovKit warns and drops the imaginary
/// part). Couplings `≤ tol` are dropped as in KrylovKit; the largest
/// imaginary part discarded that way is reported as `info.max_imag`.
/// `LargestImag`/`SmallestImag` are rejected (KrylovKit fails later trying
/// to split a 2×2 block).
///
/// The vector type must be real: KrylovKit's `RealVec` (real-linear maps on
/// complex vectors) is not supported (see [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md)).
///
/// The workspace basis needs capacity `>= alg.get_krylovdim()`; only the
/// `2·n` returned vectors are allocated. The differences from KrylovKit
/// listed on [`schursolve`] apply; on an invariant subspace smaller than
/// `howmany` it returns what it has.
#[allow(clippy::type_complexity)]
pub fn realeigsolve<V, B, A>(
    op: &mut A,
    x0: &V,
    howmany: usize,
    which: Which,
    alg: &Arnoldi,
    ws: &mut Workspace<V, B>,
) -> Result<(Vec<Real<V>>, Vec<V>, ConvergenceInfo<V>), Error<V::Error>>
where
    V: Vector,
    V::Scalar: RealScalar,
    B: Basis<V>,
    A: LinearMap<V>,
{
    let mut o = krylov_schur(op, x0, howmany, which, alg, ws)?;
    let k = o.k;
    let tol = V::Scalar::from_f64(alg.tol);
    let zero = V::Scalar::zero();
    let mut max_imag = Real::<V>::zero();
    // T[i, i-1] (0-based) couples eigenvalues i-1 and i.
    let mut drop = |t: &mut [V::Scalar], i: usize| {
        let (c, b) = (t[i + (i - 1) * k], t[(i - 1) + i * k]);
        if c != zero {
            let im = c.modulus().sqrt() * b.modulus().sqrt();
            max_imag = max_imag.max(im.re());
        }
        t[i + (i - 1) * k] = zero;
    };
    let mut i = 0;
    while i < howmany {
        i += 1;
        if i < k {
            if o.t[i + (i - 1) * k].modulus() > tol.re() {
                return Err(Error::ComplexEigenvalue { index: i - 1 });
            }
            drop(&mut o.t, i);
        }
    }
    while i < o.converged {
        i += 1;
        if i < k {
            if o.t[i + (i - 1) * k].modulus() <= tol.re() {
                drop(&mut o.t, i);
            } else {
                i -= 1;
                break;
            }
        }
    }
    // KrylovKit indexes past the factorization when it is shorter than
    // howmany; return what exists.
    let n = i.min(k);
    let converged = o.converged.min(n);
    let mut tt = vec![zero; n * n];
    for j in 0..n {
        tt[j * n..(j + 1) * n].copy_from_slice(&o.t[j * k..j * k + n]);
    }
    let values: Vec<Real<V>> = (0..n).map(|i| tt[i + i * n].re()).collect();
    let x = eigvec_coefficients(&mut tt, &o.u, k, n)?;
    let (vectors, residual, normres) = assemble(ws, x0, &x, k, n, o.beta)?;
    let info = info(&o, converged, residual, normres, max_imag);
    Ok((values, vectors, info))
}
