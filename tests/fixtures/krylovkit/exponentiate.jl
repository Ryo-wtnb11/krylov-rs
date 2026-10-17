# Lanczos `exponentiate` cases; included by generate.jl, which defines
# `out` and the shared input generators.
function expcase(out, name, seed, n, iscomplex, t, krylovdim, tol, eager)
    g = SplitMix(seed)
    A = hermitian(g, n, iscomplex)
    x0 = startvec(g, n, iscomplex)
    alg = Lanczos(; krylovdim, tol, maxiter = 100, eager, orth = ClassicalGramSchmidt2())
    y, info = exponentiate(A, t, x0, alg)
    println(out, "case ", name)
    println(out, "kind exp")
    println(out, "complex ", Int(iscomplex))
    println(out, "seed ", seed, "\nn ", n)
    println(out, "t ", repr(real(t)), " ", repr(imag(t)))
    println(out, "krylovdim ", krylovdim, "\ntol ", repr(tol), "\neager ", Int(eager))
    println(out, "y_re ", fmt(real.(y)))
    println(out, "y_im ", fmt(imag.(y)))
    println(out, "normres ", repr(info.normres))
    println(out, "converged ", info.converged, "\nnumiter ", info.numiter, "\nnumops ", info.numops)
    return println(out, "end")
end

expcase(out, "exp_real_imag_time", 5, 60, false, -2.0, 12, 1e-12, false)
expcase(out, "exp_complex_real_time", 6, 60, true, -5.0im, 15, 1e-12, false)
expcase(out, "exp_complex_imag_time_eager", 7, 50, true, complex(-1.5), 20, 1e-12, true)
