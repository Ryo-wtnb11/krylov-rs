# `linsolve` (CG, GMRES) cases; included by generate.jl, which defines
# `out` and the shared input generators.

# General (non-Hermitian) matrix, filled column by column.
function general(g, n, iscomplex)
    A = zeros(iscomplex ? ComplexF64 : Float64, n, n)
    for j in 1:n, i in 1:n
        A[i, j] = iscomplex ? complex(unif(g), unif(g)) : unif(g)
    end
    return A
end

function lincase(out, name, seed, n, iscomplex, kind, a0, a1, krylovdim, maxiter, tol)
    g = SplitMix(seed)
    A = kind == :cg || kind == :gmres_herm ? hermitian(g, n, iscomplex) : general(g, n, iscomplex)
    b = startvec(g, n, iscomplex)
    x0 = startvec(g, n, iscomplex)
    alg = kind == :cg ? CG(; maxiter, tol) :
        GMRES(; krylovdim, maxiter, tol, orth = ClassicalGramSchmidt2())
    x, info = linsolve(A, b, x0, alg, a0, a1)
    println(out, "case ", name)
    println(out, "kind ", kind == :cg ? "cg" : "gmres")
    println(out, "matrix ", kind == :gmres ? "general" : "hermitian")
    println(out, "complex ", Int(iscomplex))
    println(out, "seed ", seed, "\nn ", n)
    println(out, "a0 ", repr(Float64(real(a0))), " ", repr(Float64(imag(a0))))
    println(out, "a1 ", repr(Float64(real(a1))), " ", repr(Float64(imag(a1))))
    println(out, "krylovdim ", krylovdim, "\nmaxiter ", maxiter, "\ntol ", repr(tol))
    println(out, "x_re ", fmt(real.(x)))
    println(out, "x_im ", fmt(imag.(x)))
    println(out, "normres ", repr(info.normres))
    println(out, "converged ", info.converged, "\nnumiter ", info.numiter, "\nnumops ", info.numops)
    return println(out, "end")
end

lincase(out, "cg_real_shift", 11, 80, false, :cg, 15.0, 1.0, 0, 200, 1e-10)
lincase(out, "cg_complex_scaled", 12, 60, true, :cg, 10.0, 0.5, 0, 200, 1e-10)
lincase(out, "gmres_real_restart", 13, 80, false, :gmres, 8.0, 1.0, 10, 100, 1e-10)
lincase(out, "gmres_complex_shift", 14, 60, true, :gmres, complex(10.0, 2.0), 1.0, 30, 100, 1e-10)
lincase(out, "gmres_hermitian_indefinite", 15, 50, false, :gmres_herm, 0.37, 1.0, 50, 100, 1e-9)
