# `svdsolve` (GKL) cases; included by generate.jl, which defines `out` and
# the shared input generators.

# General m×n matrix, filled column by column.
function rectangular(g, m, n, iscomplex)
    A = zeros(iscomplex ? ComplexF64 : Float64, m, n)
    for j in 1:n, i in 1:m
        A[i, j] = iscomplex ? complex(unif(g), unif(g)) : unif(g)
    end
    return A
end

function svdcase(out, name, seed, m, n, iscomplex, howmany, which, krylovdim, tol, eager)
    g = SplitMix(seed)
    A = rectangular(g, m, n, iscomplex)
    x0 = startvec(g, m, iscomplex)
    alg = GKL(; krylovdim, tol, maxiter = 100, eager, orth = ClassicalGramSchmidt2())
    vals, lvecs, rvecs, info = svdsolve(A, x0, howmany, which, alg)
    println(out, "case ", name)
    println(out, "kind svd")
    println(out, "complex ", Int(iscomplex))
    println(out, "seed ", seed, "\nm ", m, "\nn ", n, "\nhowmany ", howmany, "\nwhich ", which)
    println(out, "krylovdim ", krylovdim, "\ntol ", repr(tol), "\neager ", Int(eager))
    println(out, "values ", fmt(vals))
    println(out, "normres ", fmt(info.normres))
    println(out, "converged ", info.converged, "\nnumiter ", info.numiter, "\nnumops ", info.numops)
    return println(out, "end")
end

svdcase(out, "svd_real_lr", 21, 90, 70, false, 3, :LR, 20, 1e-10, false)
svdcase(out, "svd_complex_lr", 22, 60, 80, true, 2, :LR, 16, 1e-10, false)
svdcase(out, "svd_real_sr_square", 23, 40, 40, false, 2, :SR, 20, 1e-9, false)
svdcase(out, "svd_complex_sr_wide_eager", 24, 40, 50, true, 2, :SR, 24, 1e-9, true)
# Tall maps with SmallestReal are left out: KrylovKit then reports a spurious
# converged σ ≈ 0 (see tests/svdsolve.rs).
