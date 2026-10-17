//! The default `Vec`-backed basis.

use num_traits::{One, Zero};

use crate::dense::householder;
use crate::scalar::sealed::Ops as _;
use crate::scalar::{MatRef, Scalar};
use crate::traits::{Basis, Vector};

/// Default [`Basis`]: `capacity` preallocated vectors plus one scratch vector,
/// with every block operation a loop of per-vector operations.
pub struct VecBasis<V: Vector> {
    // `capacity` basis slots followed by one scratch vector for `rotate`.
    vecs: Vec<V>,
    len: usize,
    // Copy of `Q[:, ..keep]`, reduced in place by `rotate`.
    q: Vec<V::Scalar>,
}

impl<V: Vector> VecBasis<V> {
    /// Allocates `capacity + 1` vectors like `x` (one is scratch for
    /// [`rotate`](Basis::rotate)); inside a [`Workspace`](crate::Workspace)
    /// the total is `capacity + 3`. The basis starts empty.
    pub fn with_capacity(x: &V, capacity: usize) -> Result<Self, V::Error> {
        let vecs = (0..=capacity)
            .map(|_| x.similar())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            vecs,
            len: 0,
            q: Vec::with_capacity(capacity * capacity),
        })
    }
    /// The stored basis vectors.
    pub fn vectors(&self) -> &[V] {
        &self.vecs[..self.len]
    }
}

impl<V: Vector> Basis<V> for VecBasis<V> {
    fn len(&self) -> usize {
        self.len
    }
    fn capacity(&self) -> usize {
        self.vecs.len() - 1
    }
    /// # Panics
    ///
    /// Panics if the basis is full.
    fn push(&mut self, v: &V, a: V::Scalar) -> Result<(), V::Error> {
        assert!(self.len < self.capacity(), "VecBasis::push: basis is full");
        self.vecs[self.len].axpby(a, v, V::Scalar::zero())?;
        self.len += 1;
        Ok(())
    }
    fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }
    fn project(&self, x: &V, out: &mut [V::Scalar]) -> Result<(), V::Error> {
        debug_assert_eq!(out.len(), self.len);
        for (o, v) in out.iter_mut().zip(self.vectors()) {
            *o = v.inner(x)?;
        }
        Ok(())
    }
    fn combine_into(
        &self,
        c: &[V::Scalar],
        y: &mut V,
        a: V::Scalar,
        b: V::Scalar,
    ) -> Result<(), V::Error> {
        debug_assert_eq!(c.len(), self.len);
        let Some((&c0, rest)) = c.split_first() else {
            // Empty basis: y ← b·y, with a finite dummy `x` scaled by zero.
            return y.axpby(V::Scalar::zero(), &self.vecs[0], b);
        };
        y.axpby(a * c0, &self.vecs[0], b)?;
        for (&ci, v) in rest.iter().zip(&self.vecs[1..]) {
            y.axpby(a * ci, v, V::Scalar::one())?;
        }
        Ok(())
    }
    /// Applies `Q` as a product of Householder reflections, in place with the
    /// one scratch vector (`O(len·keep)` axpbys, no allocation).
    ///
    /// # Panics
    ///
    /// Panics if `q.nrows() != len()` or `keep > min(q.ncols(), len())`.
    fn rotate(&mut self, q: MatRef<'_, V::Scalar>, keep: usize) -> Result<(), V::Error> {
        let m = self.len;
        assert!(
            q.nrows() == m && keep <= q.ncols(),
            "VecBasis::rotate: shape mismatch"
        );
        self.q.clear();
        for j in 0..keep {
            self.q.extend_from_slice(q.col(j));
        }
        let cap = self.vecs.len() - 1;
        let (basis, scratch) = self.vecs.split_at_mut(cap);
        let y = &mut scratch[0];
        // Reduce Q[:, ..keep] to [I; 0] by reflectors h_j from the left; then
        // Q = h_1ᴴ ⋯ h_keepᴴ [I; 0], so V·Q is V·h_1ᴴ ⋯ h_keepᴴ restricted to
        // its first `keep` columns.
        for j in 0..keep {
            let (head, tail) = self.q.split_at_mut((j + 1) * m);
            let col = &mut head[j * m + j..];
            let (_, Some(beta)) = householder(col) else {
                continue;
            };
            let u = &*col; // u[0] = 1, reflector h = I - β u uᴴ on rows j..m
            for k in 0..keep - j - 1 {
                let c = &mut tail[k * m + j..(k + 1) * m];
                let mut mu = V::Scalar::zero();
                for (ui, ci) in u.iter().zip(c.iter()) {
                    mu += ui.conj() * *ci;
                }
                mu *= beta;
                for (ui, ci) in u.iter().zip(c.iter_mut()) {
                    *ci -= mu * *ui;
                }
            }
            // V ← V·hᴴ on columns j..m: y = Σ u_i v_i; v_k -= conj(β)·conj(u_k)·y.
            y.axpby(u[0], &basis[j], V::Scalar::zero())?;
            for (ui, v) in u[1..].iter().zip(&basis[j + 1..m]) {
                y.axpby(*ui, v, V::Scalar::one())?;
            }
            for (ui, v) in u.iter().zip(&mut basis[j..m]) {
                v.axpby(-(beta.conj() * ui.conj()), y, V::Scalar::one())?;
            }
        }
        self.len = keep;
        Ok(())
    }

    fn inner_at(&self, i: usize, x: &V, _: &mut [V::Scalar]) -> Result<V::Scalar, V::Error> {
        self.vectors()[i].inner(x)
    }
    fn axpy_at(
        &self,
        i: usize,
        a: V::Scalar,
        y: &mut V,
        _: &mut [V::Scalar],
    ) -> Result<(), V::Error> {
        y.axpby(a, &self.vectors()[i], V::Scalar::one())
    }
}

impl<S: Scalar, V: Vector<Scalar = S>> core::fmt::Debug for VecBasis<V> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("VecBasis")
            .field("len", &self.len)
            .field("capacity", &self.capacity())
            .finish()
    }
}
