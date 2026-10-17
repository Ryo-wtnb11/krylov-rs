# Hermitian Lanczos `eigsolve` cases; included by generate.jl, which defines
# `out` and the shared input generators.
function eigcase(out, name, seed, n, iscomplex, howmany, which, krylovdim, tol, eager)
    g = SplitMix(seed)
    A = hermitian(g, n, iscomplex)
    x0 = startvec(g, n, iscomplex)
    alg = Lanczos(; krylovdim, tol, maxiter = 100, eager, orth = ClassicalGramSchmidt2())
    vals, vecs, info = eigsolve(A, x0, howmany, which, alg)
    println(out, "case ", name)
    println(out, "kind eig")
    println(out, "complex ", Int(iscomplex))
    println(out, "seed ", seed, "\nn ", n, "\nhowmany ", howmany, "\nwhich ", which)
    println(out, "krylovdim ", krylovdim, "\ntol ", repr(tol), "\neager ", Int(eager))
    println(out, "values ", fmt(vals))
    println(out, "normres ", fmt(info.normres))
    println(out, "converged ", info.converged, "\nnumiter ", info.numiter, "\nnumops ", info.numops)
    return println(out, "end")
end

eigcase(out, "eig_real_sr", 1, 120, false, 3, :SR, 20, 1e-10, false)
eigcase(out, "eig_complex_lm", 2, 100, true, 2, :LM, 16, 1e-10, false)
eigcase(out, "eig_real_lr_eager", 3, 80, false, 2, :LR, 30, 1e-11, true)
eigcase(out, "eig_complex_sr_small_k", 4, 60, true, 4, :SR, 8, 1e-9, false)
