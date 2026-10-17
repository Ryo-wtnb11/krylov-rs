# BlockLanczos `eigsolve` cases (KrylovKit supports only MGS2 here); included
# by generate.jl, which defines `out` and the shared input generators.
# `copies > 1` builds kron(I, H), so every eigenvalue is `copies`-fold.
function blockcase(out, name, seed, m, copies, iscomplex, bs, howmany, which, krylovdim, tol, eager)
    g = SplitMix(seed)
    H = hermitian(g, m, iscomplex)
    A = kron(Matrix{eltype(H)}(I, copies, copies), H)
    n = m * copies
    x0 = Block([startvec(g, n, iscomplex) for _ in 1:bs])
    alg = BlockLanczos(; krylovdim, tol, maxiter = 100, eager, orth = ModifiedGramSchmidt2())
    vals, vecs, info = eigsolve(A, x0, howmany, which, alg)
    println(out, "case ", name)
    println(out, "kind block")
    println(out, "complex ", Int(iscomplex))
    println(out, "seed ", seed, "\nm ", m, "\ncopies ", copies, "\nbs ", bs)
    println(out, "howmany ", howmany, "\nwhich ", which)
    println(out, "krylovdim ", krylovdim, "\ntol ", repr(tol), "\neager ", Int(eager))
    println(out, "values ", fmt(vals))
    println(out, "normres ", fmt(info.normres))
    println(out, "converged ", info.converged, "\nnumiter ", info.numiter, "\nnumops ", info.numops)
    return println(out, "end")
end

blockcase(out, "block_real_double_sr", 21, 30, 2, false, 2, 4, :SR, 20, 1e-10, false)
blockcase(out, "block_complex_triple_lr", 22, 25, 3, true, 3, 3, :LR, 30, 1e-10, false)
blockcase(out, "block_real_generic_sr_eager", 23, 60, 1, false, 2, 2, :SR, 24, 1e-10, true)
blockcase(out, "block_complex_double_restart", 24, 40, 2, true, 2, 4, :SR, 16, 1e-9, false)
