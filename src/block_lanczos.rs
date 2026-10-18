//! Hermitian eigenvalue problems with degenerate spectra by thick-restart
//! block Lanczos.

use num_traits::{Float, One, Zero};

use crate::dense::{Work, arrowhead_to_band};
use crate::eigsolve::{ConvergenceInfo, Which, sort_eig};
use crate::error::Error;
use crate::orth::{Orth, orthogonalize};
use crate::scalar::sealed::{Ops as _, RealOps as _};
use crate::scalar::{MatRef, Real, Scalar};
use crate::traits::{Basis, LinearMap, Vector, Workspace};

/// Block Lanczos settings for Hermitian maps with (nearly) repeated
/// eigenvalues.
///
/// Mirrors KrylovKit's `BlockLanczos`: `krylovdim` is the maximal Krylov
/// dimension before a restart, `maxiter` the number of Krylov subspaces built,
/// `tol` an absolute residual norm, `qr_tol` the norm below which a vector of
/// a new block counts as linearly dependent and is dropped, and `eager`
/// (default `false`) evaluates the projected problem after every block step.
/// The start block size bounds the multiplicity that can be resolved.
///
/// Differs from KrylovKit: `orth` defaults to [`Orth::Cgs2`] (KrylovKit
/// accepts only MGS2), and every new vector is orthogonalized against the
/// full basis instead of KrylovKit's three-term block step plus one MGS pass.
/// See [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockLanczos {
    krylovdim: usize,
    maxiter: usize,
    tol: f64,
    qr_tol: f64,
    orth: Orth,
    eager: bool,
}

impl BlockLanczos {
    /// Settings with the given problem-dependent parameters; the variant
    /// options start at their documented defaults. The values are validated
    /// when a solver is called.
    ///
    /// Differs from KrylovKit: there these parameters have keyword defaults
    /// (`krylovdim = 100`, `maxiter = 100`, `tol = 1e-12`); here they are required because suitable values depend on the
    /// problem and the floating-point precision (see
    /// [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md)).
    pub fn new(krylovdim: usize, maxiter: usize, tol: f64, qr_tol: f64) -> Self {
        Self {
            krylovdim,
            maxiter,
            tol,
            qr_tol,
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
    /// Residual tolerance.
    pub fn get_tol(&self) -> f64 {
        self.tol
    }
    /// Rank tolerance of the block QR.
    pub fn get_qr_tol(&self) -> f64 {
        self.qr_tol
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

/// Computes `howmany` eigenpairs of the Hermitian map `op` selected by
/// `which`, starting from the block `x0`, with thick-restart block Lanczos.
///
/// Follows KrylovKit v0.10.4 `eigsolve(A, x₀::Block, howmany, which,
/// ::BlockLanczos)`: an eigenvalue of multiplicity up to `x0.len()` is found
/// with all its copies. The result holds `max(howmany, converged)` pairs, or
/// fewer when the Krylov space is exhausted; values are ordered by `which`;
/// residual norms are `‖A·xᵢ - λᵢ·xᵢ‖`, and [`ConvergenceInfo::max_imag`] is
/// the largest anti-Hermitian entry discarded from the diagonal blocks.
/// Vectors of a new block with norm below `qr_tol` (or exactly zero) are
/// dropped, so the block may shrink; when a whole block is dropped the
/// iteration stops (invariant subspace). `krylovdim` must exceed `x0.len()`.
///
/// The workspace basis needs capacity `>= alg.get_krylovdim() + x0.len() - 1`;
/// its operator vectors are unused. A call allocates `2·x0.len()` scratch
/// vectors plus the returned eigenvectors and residuals.
///
/// Differs from KrylovKit (see [docs/krylovkit.md](https://github.com/Ryo-wtnb11/krylov-rs/blob/main/docs/krylovkit.md)):
/// - [`ConvergenceInfo::numops`] is one lower, because KrylovKit's type-probe
///   application is skipped (lower by `1 + dropped` when the QR drops start
///   vectors, since KrylovKit counts the input block size).
/// - Diagonal blocks of the Rayleigh quotient are symmetrized,
///   `(M + Mᴴ)/2`, where KrylovKit reads the upper triangle.
/// - When a whole new block is dropped the iteration stops as an invariant
///   subspace (KrylovKit throws `ArgumentError`), and exactly zero vectors
///   are dropped even with `qr_tol == 0`.
/// - After a restart the residual norm is that of the transformed residual
///   block (KrylovKit keeps the pre-restart norm).
#[allow(clippy::type_complexity)]
pub fn block_eigsolve<V, B, A>(
    op: &mut A,
    x0: &[V],
    howmany: usize,
    which: Which,
    alg: &BlockLanczos,
    ws: &mut Workspace<V, B>,
) -> Result<(Vec<Real<V>>, Vec<V>, ConvergenceInfo<V>), Error<V::Error>>
where
    V: Vector,
    B: Basis<V>,
    A: LinearMap<V>,
{
    let bs0 = x0.len();
    let kd = alg.krylovdim;
    if bs0 == 0 {
        return Err(Error::InvalidArgument("start block must not be empty"));
    }
    if kd <= bs0 {
        // keep would be a multiple of bs₀ ≥ krylovdim: every restart a no-op.
        return Err(Error::InvalidArgument(
            "krylovdim must exceed the start block size",
        ));
    }
    if alg.maxiter == 0 {
        return Err(Error::InvalidArgument("maxiter must be at least 1"));
    }
    if alg.tol.is_nan() || alg.tol < 0.0 {
        return Err(Error::InvalidArgument("tol must be non-negative"));
    }
    if alg.qr_tol.is_nan() || alg.qr_tol < 0.0 {
        return Err(Error::InvalidArgument("qr_tol must be non-negative"));
    }
    let kmax = kd + bs0 - 1;
    let capacity = ws.basis.capacity();
    if capacity < kmax {
        return Err(Error::InsufficientCapacity {
            needed: kmax,
            capacity,
        });
    }
    crate::eigsolve::real_spectrum(which)?;
    if howmany > kd {
        return Err(Error::HowmanyTooLarge {
            howmany,
            krylovdim: kd,
        });
    }
    for xi in x0 {
        if !xi.norm().map_err(Error::Vector)?.is_finite() {
            return Err(Error::NonFiniteStartVector);
        }
    }
    let tol = Real::<V>::from_f64(alg.tol);
    let zero = Real::<V>::zero();
    let szero = V::Scalar::zero();

    let mut st = State {
        x: new_block(&x0[0], bs0).map_err(Error::Vector)?,
        r: new_block(&x0[0], bs0).map_err(Error::Vector)?,
        dummy: &x0[0],
        hm: vec![szero; kmax * kmax],
        ldh: kmax,
        bq: vec![szero; bs0 * bs0],
        m: vec![szero; bs0 * bs0],
        h: Vec::with_capacity(kmax),
        s: Vec::with_capacity(kmax),
        rs: 0,
        numops: 0,
        max_imag: zero,
        qr_tol: Real::<V>::from_f64(alg.qr_tol),
        orth: alg.orth,
    };
    for (xi, x0i) in st.x.iter_mut().zip(x0) {
        xi.copy_from(x0i).map_err(Error::Vector)?;
    }
    let (nb, _) =
        block_qr(&mut st.x, st.dummy, st.qr_tol, &mut st.bq, bs0).map_err(Error::Vector)?;
    if nb == 0 {
        return Err(Error::ZeroStartVector);
    }
    ws.basis.truncate(0);
    st.recurrence(op, ws, nb)?;
    let mut norm_r = frobenius(&st.r[..st.rs]).map_err(Error::Vector)?;

    let ld = kmax;
    let mut d = vec![zero; kmax];
    let mut tmp = vec![szero; kmax * kmax];
    let mut perm = Vec::with_capacity(kmax);
    let mut a = vec![szero; kmax * kmax];
    let mut u = vec![szero; kmax * kmax];
    let mut xs = vec![szero; kmax];
    let mut lw = Work::default();
    let mut gram = vec![szero; bs0 * bs0];
    let mut normres = vec![zero; kmax];
    let mut hh = vec![szero; (kmax + bs0) * kmax];
    let mut col = vec![szero; bs0];
    let mut numiter = 1;
    let mut converged = 0;
    let mut kk = 0;
    let mut exhausted = false;

    loop {
        let k = ws.basis.len();
        if k >= kd || norm_r <= tol || exhausted || (alg.eager && k >= howmany) {
            kk = k;
            for j in 0..k {
                a[j * k..(j + 1) * k].copy_from_slice(&st.hm[j * ld..j * ld + k]);
            }
            V::Scalar::heig(&mut a[..k * k], k, &mut d[..k], &mut u[..k * k], &mut lw)
                .map_err(|()| Error::DenseNoConvergence)?;
            sort_eig(which, &mut d[..k], &mut u[..k * k], &mut perm, &mut tmp);
            let rs = st.rs;
            for c in 0..rs {
                for i in 0..=c {
                    let g = st.r[i].inner(&st.r[c]).map_err(Error::Vector)?;
                    gram[i + c * rs] = g;
                    gram[c + i * rs] = g.conj();
                }
            }
            for (i, nr) in normres[..k].iter_mut().enumerate() {
                *nr = quad(&gram[..rs * rs], &u[i * k + k - rs..(i + 1) * k]);
            }
            converged = normres[..k].iter().take_while(|&&x| x <= tol).count();
            if converged >= howmany || norm_r <= tol || exhausted {
                break;
            }
        }
        if k < kd {
            exhausted = !st.expand(op, ws)?;
            if !exhausted {
                norm_r = frobenius(&st.r[..st.rs]).map_err(Error::Vector)?;
            }
            continue;
        }
        if numiter >= alg.maxiter {
            break;
        }
        // Thick restart (KrylovKit's choice of `keep`, a multiple of bs₀).
        let keep = ((3 * kd + 2 * converged) / (5 * bs0)).max(1) * bs0;
        let ldk = keep + bs0;
        let hb = &mut hh[..ldk * keep];
        hb.fill(szero);
        for j in 0..keep {
            hb[j + j * ldk] = V::Scalar::from_real(d[j]);
            for i in 0..bs0 {
                hb[keep + i + j * ldk] = u[(k - bs0 + i) + j * k];
            }
        }
        arrowhead_to_band(hb, keep, bs0, &mut u[..k * k], k, &mut xs);
        let half = V::Scalar::from_real(Real::<V>::from_f64(0.5));
        st.hm.fill(szero);
        for j in 0..keep {
            for i in 0..keep {
                st.hm[i + j * ld] = (hb[i + j * ldk] + hb[j + i * ldk].conj()) * half;
            }
        }
        ws.basis
            .rotate(MatRef::new(&u[..k * keep], k, keep), keep)
            .map_err(Error::Vector)?;
        // New residual R·C, C the upper-triangular coupling corner; its norm
        // follows from the Gram matrix without further reductions.
        let rs = st.rs;
        let corner = |i: usize, l: usize| hb[(keep + bs0 - rs + i) + (keep - rs + l) * ldk];
        let mut nr2 = zero;
        let col = &mut col[..rs];
        for l in (0..rs).rev() {
            for (i, ci) in col.iter_mut().enumerate() {
                *ci = if i <= l { corner(i, l) } else { szero };
            }
            let q = quad(&gram[..rs * rs], col);
            nr2 += q * q;
            let (head, tail) = st.r.split_at_mut(l);
            tail[0]
                .axpby(szero, st.dummy, corner(l, l))
                .map_err(Error::Vector)?;
            for (i, ri) in head.iter().enumerate() {
                tail[0]
                    .axpby(corner(i, l), ri, V::Scalar::one())
                    .map_err(Error::Vector)?;
            }
        }
        norm_r = nr2.sqrt();
        numiter += 1;
    }

    let k = kk;
    let rs = st.rs;
    let nret = if converged > howmany {
        converged
    } else {
        howmany.min(k)
    };
    let mut vectors = Vec::with_capacity(nret);
    let mut residual = Vec::with_capacity(nret);
    for i in 0..nret {
        let ui = &u[i * k..(i + 1) * k];
        let mut x = x0[0].similar().map_err(Error::Vector)?;
        ws.basis
            .combine_into(ui, &mut x, V::Scalar::one(), szero)
            .map_err(Error::Vector)?;
        vectors.push(x);
        let mut res = x0[0].similar().map_err(Error::Vector)?;
        for (j, rj) in st.r[..rs].iter().enumerate() {
            let b = if j == 0 { szero } else { V::Scalar::one() };
            res.axpby(ui[k - rs + j], rj, b).map_err(Error::Vector)?;
        }
        residual.push(res);
    }
    let info = ConvergenceInfo {
        converged,
        residual,
        normres: normres[..nret].to_vec(),
        numiter,
        numops: st.numops,
        max_imag: st.max_imag,
    };
    Ok((d[..nret].to_vec(), vectors, info))
}

/// Factorization state: `A·V = V·H + R·[0 I]` with `R = r[..rs]`.
struct State<'a, V: Vector> {
    /// The next block (a QR'd copy of the residual).
    x: Vec<V>,
    /// The residual block.
    r: Vec<V>,
    /// A finite vector multiplied by zero where `axpby` only scales.
    dummy: &'a V,
    /// Block-tridiagonal `H` (`ldh × ldh`, column-major).
    hm: Vec<V::Scalar>,
    ldh: usize,
    /// R factor of the block QR, then the coupling `B` (`bs₀ × bs₀`).
    bq: Vec<V::Scalar>,
    /// Coefficients of the newest block, `M = Xᴴ·A·X`.
    m: Vec<V::Scalar>,
    h: Vec<V::Scalar>,
    s: Vec<V::Scalar>,
    rs: usize,
    numops: usize,
    max_imag: Real<V>,
    qr_tol: Real<V>,
    orth: Orth,
}

impl<V: Vector> State<'_, V> {
    /// Port of KrylovKit `expand!(::BlockLanczosIterator, …)`: QR of the
    /// residual into the next block, `B`, then [`recurrence`](Self::recurrence).
    /// Returns `false` (nothing changed) when every vector is dropped.
    fn expand<B: Basis<V>, A: LinearMap<V>>(
        &mut self,
        op: &mut A,
        ws: &mut Workspace<V, B>,
    ) -> Result<bool, Error<V::Error>> {
        let ve = Error::Vector;
        let (rs, ldb) = (self.rs, self.x.len());
        for (xi, ri) in self.x.iter_mut().zip(&self.r[..rs]) {
            xi.copy_from(ri).map_err(ve)?;
        }
        let (mut nb, drift) = block_qr(
            &mut self.x[..rs],
            self.dummy,
            self.qr_tol,
            &mut self.bq,
            ldb,
        )
        .map_err(ve)?;
        if drift {
            // A small pivot may let the block drift into span(V): orthogonalize
            // against the basis, redo the QR and recompute B = Xᴴ·R exactly.
            for xi in &mut self.x[..nb] {
                self.h.clear();
                self.h.resize(ws.basis.len(), V::Scalar::zero());
                self.s.clear();
                self.s.resize(ws.basis.len(), V::Scalar::zero());
                orthogonalize(&ws.basis, xi, &mut self.h, &mut self.s, self.orth).map_err(ve)?;
            }
            nb = block_qr(
                &mut self.x[..nb],
                self.dummy,
                self.qr_tol,
                &mut self.bq,
                ldb,
            )
            .map_err(ve)?
            .0;
            for c in 0..rs {
                for a in 0..nb {
                    self.bq[a + c * ldb] = self.x[a].inner(&self.r[c]).map_err(ve)?;
                }
            }
        }
        if nb == 0 {
            return Ok(false);
        }
        let k = ws.basis.len();
        let ld = self.ldh;
        for c in 0..rs {
            for a in 0..nb {
                let b = self.bq[a + c * ldb];
                self.hm[(k + a) + (k - rs + c) * ld] = b;
                self.hm[(k - rs + c) + (k + a) * ld] = b.conj();
            }
        }
        self.recurrence(op, ws, nb)?;
        Ok(true)
    }

    /// Pushes `x[..nb]` and forms `R = (I - V·Vᴴ)·A·X` with full
    /// orthogonalization of each vector; the coefficients on `X` are the
    /// diagonal block `M` of `H`.
    fn recurrence<B: Basis<V>, A: LinearMap<V>>(
        &mut self,
        op: &mut A,
        ws: &mut Workspace<V, B>,
        nb: usize,
    ) -> Result<(), Error<V::Error>> {
        let ve = Error::Vector;
        let k0 = ws.basis.len();
        for xi in &self.x[..nb] {
            ws.basis.push(xi, V::Scalar::one()).map_err(ve)?;
        }
        let k = k0 + nb;
        for (i, (xi, ri)) in self.x[..nb].iter().zip(&mut self.r[..nb]).enumerate() {
            op.apply_into(xi, ri).map_err(ve)?;
            self.numops += 1;
            self.h.clear();
            self.h.resize(k, V::Scalar::zero());
            self.s.clear();
            self.s.resize(k, V::Scalar::zero());
            orthogonalize(&ws.basis, ri, &mut self.h, &mut self.s, self.orth).map_err(ve)?;
            if !self.h.iter().all(|c| c.all_finite()) {
                return Err(Error::NonFiniteIterate);
            }
            self.m[i * nb..(i + 1) * nb].copy_from_slice(&self.h[k0..k]);
        }
        let half = V::Scalar::from_real(Real::<V>::from_f64(0.5));
        let ld = self.ldh;
        for j in 0..nb {
            for i in 0..nb {
                let (mij, mji) = (self.m[i + j * nb], self.m[j + i * nb].conj());
                let anti = ((mij - mji) * half).modulus();
                if anti > self.max_imag {
                    self.max_imag = anti;
                }
                self.hm[(k0 + i) + (k0 + j) * ld] = (mij + mji) * half;
            }
        }
        self.rs = nb;
        Ok(())
    }
}

fn new_block<V: Vector>(x: &V, n: usize) -> Result<Vec<V>, V::Error> {
    (0..n).map(|_| x.similar()).collect()
}

/// In-place QR of the block `x` by modified Gram–Schmidt with DGKS
/// reorthogonalization (port of KrylovKit `block_qr!`): vectors with norm
/// below `tol` (for the first: not above) are dropped and the kept ones
/// compacted to `x[..nb]`; `b[..nb, ..x.len()]` (column-major, leading
/// dimension `ldb`) receives the corresponding rows of the R factor.
/// Returns `(nb, drift)`, `drift` when DGKS was needed.
fn block_qr<V: Vector>(
    x: &mut [V],
    dummy: &V,
    tol: Real<V>,
    b: &mut [V::Scalar],
    ldb: usize,
) -> Result<(usize, bool), V::Error> {
    let n = x.len();
    let one = V::Scalar::one();
    let hundred = Real::<V>::from_f64(100.0);
    b.fill(V::Scalar::zero());
    let mut nb = 0;
    let mut drift = false;
    for j in 0..n {
        x.swap(nb, j);
        let (q, rest) = x.split_at_mut(nb);
        let xj = &mut rest[0];
        for (i, qi) in q.iter().enumerate() {
            let c = qi.inner(xj)?;
            b[i + j * ldb] = c;
            xj.axpby(-c, qi, one)?;
        }
        let mut beta = xj.norm()?;
        if j > 0 && tol < beta && beta < hundred * tol {
            drift = true;
            for (i, qi) in q.iter().enumerate() {
                let c = qi.inner(xj)?;
                b[i + j * ldb] += c;
                xj.axpby(-c, qi, one)?;
            }
            beta = xj.norm()?;
        }
        // `β > 0` keeps `qr_tol == 0` from normalizing a zero vector.
        let kept = if j == 0 {
            beta > tol
        } else {
            beta >= tol && beta > Real::<V>::zero()
        };
        if kept {
            b[nb + j * ldb] = V::Scalar::from_real(beta);
            xj.axpby(V::Scalar::zero(), dummy, V::Scalar::from_real(beta.recip()))?;
            nb += 1;
        }
    }
    Ok((nb, drift))
}

/// `sqrt(max(Re cᴴ·G·c, 0))`.
fn quad<S: Scalar>(g: &[S], c: &[S]) -> S::Real {
    let n = c.len();
    let mut acc = S::zero();
    for (j, &cj) in c.iter().enumerate() {
        let mut gc = S::zero();
        for (i, &ci) in c.iter().enumerate() {
            gc += ci.conj() * g[i + j * n];
        }
        acc += gc * cj;
    }
    acc.re().max(S::Real::zero()).sqrt()
}

/// Frobenius norm of a block.
fn frobenius<V: Vector>(r: &[V]) -> Result<Real<V>, V::Error> {
    let mut s = Real::<V>::zero();
    for ri in r {
        let n = ri.norm()?;
        s += n * n;
    }
    Ok(s.sqrt())
}
