//! Hermitian eigenvalue problems by the preconditioned (block) Davidson
//! method.

use num_traits::{Float, One, Zero};

use crate::dense::{Problem, Work};
use crate::eigsolve::{ConvergenceInfo, Which, real_spectrum, sort_eig};
use crate::error::Error;
use crate::orth::{Orth, orthogonalize};
use crate::scalar::sealed::{Ops as _, RealOps as _};
use crate::scalar::{MatRef, Real};
use crate::traits::{Basis, LinearMap, Vector, Workspace};

/// Davidson settings for Hermitian maps.
///
/// KrylovKit v0.10.4 has no Davidson solver; the parameters follow krylov's
/// other eigensolvers:
///
/// - `krylovdim` (`m`): maximal subspace size.
/// - `restartdim` (`r`): number of Ritz vectors kept at a restart
///   (`howmany <= r`, `blocksize <= m − r`).
/// - `blocksize` (`b`): maximal number of corrections added per
///   Rayleigh–Ritz pass; `1` is the classical one-vector method.
/// - `maxiter`: maximal number of subspaces built (restart cycles), as in
///   [`Lanczos`](crate::Lanczos); `1` means no restart.
/// - `tol`: absolute residual norm `‖A·x − θ·x‖ <= tol`.
/// - `drop_tol`: a candidate whose norm after orthogonalization is exactly
///   zero or `<= drop_tol ×` its norm before is dropped (`0 <= drop_tol < 1`
///   after conversion to the real type of the vector scalar).
///
/// A restart rotates the basis and its images to the `r` kept Ritz vectors,
/// at `O(m·r)` vector updates. Between restarts, starting from dimension
/// `r`, a restart follows at least `⌊(m − r)/b⌋` passes; the initial block
/// can trigger a restart in the first pass. Small `m − r` makes restarts
/// frequent: `r = m − 1` with `b = 1` restarts after every expansion.
///
/// `orth` defaults to [`Orth::Cgs2`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Davidson {
    krylovdim: usize,
    restartdim: usize,
    blocksize: usize,
    maxiter: usize,
    tol: f64,
    drop_tol: f64,
    orth: Orth,
}

impl Davidson {
    /// Settings with the given cost and accuracy parameters; the
    /// orthogonalizer starts at [`Orth::Cgs2`]. The values are validated
    /// when [`davidson_eigsolve`] is called.
    pub fn new(
        krylovdim: usize,
        restartdim: usize,
        blocksize: usize,
        maxiter: usize,
        tol: f64,
        drop_tol: f64,
    ) -> Self {
        Self {
            krylovdim,
            restartdim,
            blocksize,
            maxiter,
            tol,
            drop_tol,
            orth: Orth::Cgs2,
        }
    }
    /// Sets the orthogonalizer.
    pub fn orth(mut self, orth: Orth) -> Self {
        self.orth = orth;
        self
    }
    /// Maximal subspace size.
    pub fn get_krylovdim(&self) -> usize {
        self.krylovdim
    }
    /// Number of Ritz vectors kept at a restart.
    pub fn get_restartdim(&self) -> usize {
        self.restartdim
    }
    /// Maximal number of corrections per Rayleigh–Ritz pass.
    pub fn get_blocksize(&self) -> usize {
        self.blocksize
    }
    /// Maximal number of subspaces built (restart cycles).
    pub fn get_maxiter(&self) -> usize {
        self.maxiter
    }
    /// Residual tolerance.
    pub fn get_tol(&self) -> f64 {
        self.tol
    }
    /// Relative tolerance below which a candidate is dropped.
    pub fn get_drop_tol(&self) -> f64 {
        self.drop_tol
    }
    /// Orthogonalizer.
    pub fn get_orth(&self) -> Orth {
        self.orth
    }
}

/// A Davidson preconditioner: `t ← M(θ)·r` for the residual `r` of the Ritz
/// pair with value `θ`.
///
/// `M(θ)` must be linear in `r`. `t` holds arbitrary previous contents and
/// is overwritten. Under the distributed contract of [`Vector`] the result
/// must be consistent, and an error returned, on every rank.
///
/// Closures `FnMut(&mut V::Context, Real<V>, &V, &mut V) -> Result<(), V::Error>`
/// implement this trait. The identity is `|cx, _, r, t| t.copy_from(cx, r)`.
///
/// # Diagonal (Jacobi) preconditioner
///
/// krylov owns no elementwise arithmetic, so the diagonal preconditioner is
/// the caller's closure over its vector storage. With `d` the diagonal of
/// `A` and a caller-chosen guard `g > 0` (block2 uses `1e-12`), the
/// convention is block2's:
///
/// `t_i = r_i / (θ − d_i)` if `|θ − d_i| > g`, otherwise `t_i = r_i`.
///
/// The sign is `(θ − d)⁻¹`, not `(d − θ)⁻¹`; with skipped entries the two
/// are not related by a global sign flip. See [`davidson_eigsolve`] for a
/// runnable closure.
///
/// The plain correction can stagnate: for `A = diag(1, 3)` and
/// `x0 = (1, 1)/√2`, `θ = 2`, `r = (−1, 1)/√2` and `(θ − D)⁻¹·r = −x0`,
/// which lies in the current subspace and is dropped. The call then returns
/// unconverged, without an error. block2's default (`Normal`, Olsen's
/// correction) is not this method.
pub trait Preconditioner<V: Vector> {
    /// `t ← M(θ)·r`. `t` holds arbitrary previous contents.
    fn apply_into(
        &mut self,
        cx: &mut V::Context,
        theta: Real<V>,
        r: &V,
        t: &mut V,
    ) -> Result<(), V::Error>;
}

impl<V, F> Preconditioner<V> for F
where
    V: Vector,
    F: FnMut(&mut V::Context, Real<V>, &V, &mut V) -> Result<(), V::Error>,
{
    fn apply_into(
        &mut self,
        cx: &mut V::Context,
        theta: Real<V>,
        r: &V,
        t: &mut V,
    ) -> Result<(), V::Error> {
        self(cx, theta, r, t)
    }
}

/// Computes `howmany` eigenpairs of the Hermitian map `op` selected by
/// `which` with the preconditioned block Davidson method, starting from the
/// orthogonalized span of `x0`.
///
/// Each Rayleigh–Ritz pass scans the Ritz pairs in `which` order and, for
/// up to `blocksize` of those with residual norm above `tol`, adds the
/// orthogonalized correction `M(θ)·r` from `precond` (see [`Preconditioner`]).
/// Once the subspace exceeds `krylovdim − blocksize` it is restarted to the
/// `restartdim` best Ritz vectors; their images are rotated, not re-applied.
/// No KrylovKit counterpart exists; the method follows block2's `davidson`
/// with the plain (not Olsen) correction, extended to blocks (see
/// [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md)).
///
/// Result: `nret = min(howmany, q)` pairs in `which` order, `q` the size of
/// the final Ritz decomposition, so fewer than `howmany` are returned when
/// the start block or an invariant subspace is too small; converged roots
/// beyond `howmany` are not returned (unlike [`eigsolve`](crate::eigsolve)).
/// [`ConvergenceInfo::converged`] is the leading prefix with residual norm
/// `<= tol`; `numiter` counts restart cycles (restarts + 1), `numops` map
/// applications, and `max_imag` the largest `|Im ⟨vᵢ, A·vᵢ⟩|`. Stagnation
/// (every correction of a pass dropped) ends the call without an error.
/// The residuals are formed from the stored images `A·V`.
///
/// Workspaces: `ws` holds the orthonormal basis `V` and two scratch vectors;
/// `ws_av` holds the images `A·V` (not orthonormal; its scratch vectors are
/// unused). Both bases need capacity `>= krylovdim`. Each call starts from
/// `x0`: the workspaces keep allocation, not state, across calls. Passes and
/// restarts construct no vectors; a call constructs the `2·nret` returned
/// eigenvectors and residuals.
///
/// Each pass without a restart costs `O((howmany + blocksize)·k)` vector
/// operations at subspace size `k`. In detail, with
/// `s <= howmany + blocksize` scanned roots, `a <= blocksize` attempted and
/// `c` accepted corrections: `2k` vector updates and one norm per scanned
/// root, `2k` inner products, `2k`
/// updates, two norms and one preconditioner call per attempt, and `k + 1`
/// inner products, three updates and one map application per accepted
/// correction, plus an `O(k³)` dense eigenproblem.
///
/// Errors: [`Error::InvalidArgument`] for an empty `x0`, `blocksize`,
/// `maxiter` or `howmany` of zero, a negative or NaN `tol`, `drop_tol`
/// outside `[0, 1)`, `restartdim < howmany`, `restartdim > krylovdim`,
/// `blocksize > krylovdim − restartdim`, `x0.len() > krylovdim`,
/// `krylovdim²` scalars beyond the addressable size, or an imaginary
/// `which`; [`Error::InsufficientCapacity`] for a basis smaller than
/// `krylovdim`; [`Error::NonFiniteStartVector`] for a start vector with NaN
/// or ±∞; [`Error::ZeroStartVector`] when every start vector is dropped;
/// [`Error::NonFiniteIterate`] for NaN or ±∞ produced by the map, the
/// preconditioner or a residual; [`Error::DenseNoConvergence`]; and
/// [`Error::Vector`] for any failing vector, basis, map or preconditioner
/// call.
///
/// ```
/// use krylov::{Davidson, Vector, Which, Workspace, davidson_eigsolve};
///
/// #[derive(Debug)]
/// struct V(Vec<f64>);
/// impl Vector for V {
///     type Scalar = f64;
///     type Error = core::convert::Infallible;
///     type Context = ();
///     fn similar(&self, _: &mut ()) -> Result<Self, Self::Error> { Ok(V(vec![0.0; self.0.len()])) }
///     fn copy_from(&mut self, _: &mut (), x: &Self) -> Result<(), Self::Error> {
///         self.0.copy_from_slice(&x.0); Ok(())
///     }
///     fn inner(&self, _: &mut (), y: &Self) -> Result<f64, Self::Error> {
///         Ok(self.0.iter().zip(&y.0).map(|(a, b)| a * b).sum())
///     }
///     fn norm(&self, cx: &mut ()) -> Result<f64, Self::Error> { Ok(self.inner(cx, self)?.sqrt()) }
///     fn axpby(&mut self, _: &mut (), a: f64, x: &Self, b: f64) -> Result<(), Self::Error> {
///         for (s, x) in self.0.iter_mut().zip(&x.0) {
///             *s = if b == 0.0 { a * x } else { a * x + b * *s };
///         }
///         Ok(())
///     }
/// }
///
/// // Diagonally dominant: A = diag(1, 2, …, n) + 0.1·(nearest-neighbour hopping).
/// let n = 100;
/// let d: Vec<f64> = (1..=n).map(f64::from).collect();
/// let mut op = |_: &mut (), x: &V, y: &mut V| {
///     for i in 0..n as usize {
///         let l = if i > 0 { x.0[i - 1] } else { 0.0 };
///         let r = if i + 1 < n as usize { x.0[i + 1] } else { 0.0 };
///         y.0[i] = d[i] * x.0[i] + 0.1 * (l + r);
///     }
///     Ok(())
/// };
/// // Jacobi: t_i = r_i/(θ − d_i), left unchanged where |θ − d_i| <= guard.
/// let guard = 1e-12;
/// let mut jacobi = |_: &mut (), theta: f64, r: &V, t: &mut V| {
///     for ((ti, &ri), &di) in t.0.iter_mut().zip(&r.0).zip(&d) {
///         let den = theta - di;
///         *ti = if den.abs() > guard { ri / den } else { ri };
///     }
///     Ok(())
/// };
/// let x0 = [V((0..n).map(|i| if i == 0 { 1.0 } else { 0.01 }).collect())];
/// let alg = Davidson::new(20, 5, 1, 50, 1e-10, 1e-12);
/// let mut cx = ();
/// let mut ws = Workspace::with_capacity(&mut cx, &x0[0], alg.get_krylovdim())?;
/// let mut ws_av = Workspace::with_capacity(&mut cx, &x0[0], alg.get_krylovdim())?;
/// let (vals, _vecs, info) = davidson_eigsolve(
///     &mut cx, &mut op, &mut jacobi, &x0, 1, Which::SmallestReal, &alg, &mut ws, &mut ws_av,
/// )?;
/// assert_eq!(info.converged, 1);
/// assert!(info.normres[0] <= 1e-10);
/// assert!((vals[0] - 1.0).abs() < 0.1);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn davidson_eigsolve<V, B, BA, A, P>(
    cx: &mut V::Context,
    op: &mut A,
    precond: &mut P,
    x0: &[V],
    howmany: usize,
    which: Which,
    alg: &Davidson,
    ws: &mut Workspace<V, B>,
    ws_av: &mut Workspace<V, BA>,
) -> Result<(Vec<Real<V>>, Vec<V>, ConvergenceInfo<V>), Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
    BA: Basis<V>,
    A: LinearMap<V>,
    P: Preconditioner<V>,
{
    let (m, r, b, h) = (alg.krylovdim, alg.restartdim, alg.blocksize, howmany);
    if x0.is_empty() {
        return Err(Error::InvalidArgument("start block must not be empty"));
    }
    if b == 0 {
        return Err(Error::InvalidArgument("blocksize must be at least 1"));
    }
    if alg.maxiter == 0 {
        return Err(Error::InvalidArgument("maxiter must be at least 1"));
    }
    if h == 0 {
        return Err(Error::InvalidArgument("howmany must be at least 1"));
    }
    if alg.tol.is_nan() || alg.tol < 0.0 {
        return Err(Error::InvalidArgument("tol must be non-negative"));
    }
    let drop_tol = Real::<V>::from_f64(alg.drop_tol);
    let zero = Real::<V>::zero();
    if !(drop_tol >= zero && drop_tol < Real::<V>::one()) {
        return Err(Error::InvalidArgument("drop_tol must lie in [0, 1)"));
    }
    if r < h {
        return Err(Error::InvalidArgument(
            "restartdim must be at least howmany",
        ));
    }
    // `r <= m` first, so that `m - r` cannot wrap.
    if r > m {
        return Err(Error::InvalidArgument(
            "restartdim must not exceed krylovdim",
        ));
    }
    if b > m - r {
        return Err(Error::InvalidArgument(
            "blocksize must not exceed krylovdim - restartdim",
        ));
    }
    if x0.len() > m {
        return Err(Error::InvalidArgument(
            "start block must not exceed krylovdim",
        ));
    }
    for capacity in [ws.basis.capacity(), ws_av.basis.capacity()] {
        if capacity < m {
            return Err(Error::InsufficientCapacity {
                needed: m,
                capacity,
            });
        }
    }
    // One length for the three `m × m` projected buffers, checked before
    // any workspace is touched: a basis need not preallocate `m` vectors.
    let mm = m
        .checked_mul(m)
        .filter(|&mm| {
            mm.checked_mul(core::mem::size_of::<V::Scalar>())
                .is_some_and(|bytes| bytes <= isize::MAX as usize)
        })
        .ok_or(Error::InvalidArgument(
            "krylovdim² scalars exceed the addressable size",
        ))?;
    real_spectrum(which)?;
    for xi in x0 {
        if !xi.norm(cx).map_err(Error::Vector)?.is_finite() {
            return Err(Error::NonFiniteStartVector);
        }
    }
    let tol = Real::<V>::from_f64(alg.tol);
    let szero = V::Scalar::zero();
    let sone = V::Scalar::one();
    let ve = Error::Vector;

    // Per-call reset: the workspaces carry allocation, never state.
    ws.basis.truncate(0);
    ws_av.basis.truncate(0);
    let mut st = State {
        hm: vec![szero; mm],
        ld: m,
        h: Vec::with_capacity(m),
        s: Vec::with_capacity(m),
        drop_tol,
        orth: alg.orth,
        numops: 0,
        max_imag: zero,
    };
    for xi in x0 {
        ws.w.copy_from(cx, xi).map_err(ve)?;
        st.attempt(cx, op, ws, ws_av)?;
    }
    if ws.basis.is_empty() {
        return Err(Error::ZeroStartVector);
    }

    let mut a = vec![szero; mm];
    let mut y = vec![szero; mm]; // Ritz coefficients, `q × q` (leading dim `q`)
    let mut theta = vec![zero; m];
    let mut rho = vec![zero; m];
    let mut yhat = Vec::with_capacity(m);
    let mut perm = Vec::with_capacity(m);
    let mut lw = Work::new(m, Problem::Hermitian);
    let mut numiter = 1;
    let mut q;
    let mut converged;

    loop {
        // Rayleigh–Ritz on the upper triangle of H; the rest of the dense
        // input is zero because the backend checks all of it for finiteness.
        let k = ws.basis.len();
        for j in 0..k {
            for i in 0..k {
                a[i + j * k] = if i <= j { st.hm[i + j * m] } else { szero };
            }
        }
        V::Scalar::heig(
            &mut a[..k * k],
            k,
            &mut theta[..k],
            &mut y[..k * k],
            &mut lw,
        )
        .map_err(|()| Error::DenseNoConvergence)?;
        sort_eig(which, &mut theta[..k], &mut y[..k * k], &mut perm);
        q = k;

        let mut last = false;
        if k > m - b {
            if numiter == alg.maxiter {
                last = true;
            } else {
                let yr = MatRef::new(&y[..k * r], k, r);
                ws.basis.rotate(cx, yr, r).map_err(ve)?;
                ws_av.basis.rotate(cx, yr, r).map_err(ve)?;
                st.hm.fill(szero);
                y[..r * r].fill(szero);
                for j in 0..r {
                    st.hm[j + j * m] = V::Scalar::from_real(theta[j]);
                    y[j + j * r] = sone;
                }
                q = r;
                numiter += 1;
            }
        }

        // Scan the Ritz pairs `i < q`; corrections accepted earlier in this
        // pass are already in V, so later ones are orthogonalized against them.
        let (mut att, mut acc, mut lead) = (0, 0, 0);
        for i in 0..q {
            ritz_coeffs(&mut yhat, &y[i * q..(i + 1) * q], ws.basis.len());
            ws_av
                .basis
                .combine_into(cx, &yhat, &mut ws.v, sone, szero)
                .map_err(ve)?;
            ws.basis
                .combine_into(cx, &yhat, &mut ws.v, -V::Scalar::from_real(theta[i]), sone)
                .map_err(ve)?;
            let rn = ws.v.norm(cx).map_err(ve)?;
            if !rn.is_finite() {
                return Err(Error::NonFiniteIterate);
            }
            rho[i] = rn;
            if rn <= tol {
                if lead == i {
                    lead += 1;
                }
            } else if !last && att < b {
                att += 1;
                precond
                    .apply_into(cx, theta[i], &ws.v, &mut ws.w)
                    .map_err(ve)?;
                if st.attempt(cx, op, ws, ws_av)? {
                    acc += 1;
                }
            }
            let n = i + 1;
            if n == q || (n >= h && (lead >= h || att == b || last)) || n >= h + b {
                break;
            }
        }
        converged = lead.min(h);
        // Nothing was pushed in a finishing pass, so `y`/`q` describe V.
        if converged == h || last || acc == 0 {
            break;
        }
    }

    let nret = h.min(q);
    let mut vectors = Vec::with_capacity(nret);
    let mut residual = Vec::with_capacity(nret);
    for i in 0..nret {
        ritz_coeffs(&mut yhat, &y[i * q..(i + 1) * q], ws.basis.len());
        let mut x = x0[0].similar(cx).map_err(ve)?;
        ws.basis
            .combine_into(cx, &yhat, &mut x, sone, szero)
            .map_err(ve)?;
        let mut res = x0[0].similar(cx).map_err(ve)?;
        ws_av
            .basis
            .combine_into(cx, &yhat, &mut res, sone, szero)
            .map_err(ve)?;
        res.axpby(cx, -V::Scalar::from_real(theta[i]), &x, sone)
            .map_err(ve)?;
        vectors.push(x);
        residual.push(res);
    }
    let info = ConvergenceInfo {
        converged,
        residual,
        normres: rho[..nret].to_vec(),
        numiter,
        numops: st.numops,
        max_imag: st.max_imag,
    };
    Ok((theta[..nret].to_vec(), vectors, info))
}

/// `out ← y` zero-padded to `len`: the same Ritz vector in a basis that has
/// grown by the corrections accepted after the decomposition.
fn ritz_coeffs<S: Copy + Zero>(out: &mut Vec<S>, y: &[S], len: usize) {
    out.clear();
    out.extend_from_slice(y);
    out.resize(len, S::zero());
}

/// The projected matrix and counters of one call.
struct State<S, R> {
    /// Upper triangle of `H = Vᴴ·A·V` (`ld × ld`, column-major).
    hm: Vec<S>,
    ld: usize,
    /// Orthogonalization coefficients (discarded) of both passes.
    h: Vec<S>,
    s: Vec<S>,
    drop_tol: R,
    orth: Orth,
    numops: usize,
    max_imag: R,
}

impl<S: crate::scalar::Scalar> State<S, S::Real> {
    /// Attempts to add the candidate in `ws.w` (start vector or correction):
    /// orthogonalize, drop it when its norm falls to zero or to `drop_tol`
    /// times its norm before, else normalize, push it, apply the map, push
    /// the image and fill the new column of `H`. Returns whether it was
    /// accepted.
    fn attempt<V, B, BA, A>(
        &mut self,
        cx: &mut V::Context,
        op: &mut A,
        ws: &mut Workspace<V, B>,
        ws_av: &mut Workspace<V, BA>,
    ) -> Result<bool, Error<V::Error>>
    where
        V: Vector<Scalar = S>,
        B: Basis<V>,
        BA: Basis<V>,
        A: LinearMap<V>,
    {
        let ve = Error::Vector;
        let zero = S::Real::zero();
        let nu0 = ws.w.norm(cx).map_err(ve)?;
        if !nu0.is_finite() {
            return Err(Error::NonFiniteIterate);
        }
        if nu0 == zero {
            return Ok(false);
        }
        let k = ws.basis.len();
        self.h.clear();
        self.h.resize(k, S::zero());
        self.s.clear();
        self.s.resize(k, S::zero());
        orthogonalize(
            cx,
            &ws.basis,
            &mut ws.w,
            &mut self.h,
            &mut self.s,
            self.orth,
        )
        .map_err(ve)?;
        let nu = ws.w.norm(cx).map_err(ve)?;
        if !nu.is_finite() {
            return Err(Error::NonFiniteIterate);
        }
        // `drop_tol >= 0` and `ν₀ > 0`, so this also drops an exact zero.
        if nu <= self.drop_tol * nu0 {
            return Ok(false);
        }
        ws.v.axpby(cx, S::from_real(nu.recip()), &ws.w, S::zero())
            .map_err(ve)?;
        ws.basis.push(cx, &ws.v, S::one()).map_err(ve)?;
        op.apply_into(cx, &ws.v, &mut ws.w).map_err(ve)?;
        self.numops += 1;
        ws_av.basis.push(cx, &ws.w, S::one()).map_err(ve)?;
        let col = &mut self.h;
        col.clear();
        col.resize(k + 1, S::zero());
        ws.basis.project(cx, &ws.w, col).map_err(ve)?;
        if !col.iter().all(|c| c.all_finite()) {
            return Err(Error::NonFiniteIterate);
        }
        let diag = col[k];
        let imag = diag.im().abs();
        if imag > self.max_imag {
            self.max_imag = imag;
        }
        // The Rayleigh quotient of a Hermitian map is real.
        col[k] = S::from_real(diag.re());
        self.hm[k * self.ld..k * self.ld + k + 1].copy_from_slice(col);
        Ok(true)
    }
}
