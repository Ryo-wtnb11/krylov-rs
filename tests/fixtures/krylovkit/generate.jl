# Generates `fixtures.txt` for tests/krylovkit.rs.
#
#   julia --project=tests/fixtures/krylovkit tests/fixtures/krylovkit/generate.jl
#
# Matrices and start vectors come from SplitMix64 mapped exactly to
# k·2⁻⁵² - 1, so the Rust test rebuilds bit-identical inputs.
using KrylovKit, LinearAlgebra, Pkg

mutable struct SplitMix
    s::UInt64
end
function next!(g::SplitMix)
    g.s += 0x9e3779b97f4a7c15
    z = g.s
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end
unif(g) = Float64(next!(g) >> 11) * 2.0^-52 - 1.0

function hermitian(g, n, iscomplex)
    A = zeros(iscomplex ? ComplexF64 : Float64, n, n)
    for j in 1:n, i in 1:j
        if i == j || !iscomplex
            a = unif(g)
            A[i, j] = a
            A[j, i] = a
        else
            a = complex(unif(g), unif(g))
            A[i, j] = a
            A[j, i] = conj(a)
        end
    end
    return A
end
startvec(g, n, iscomplex) = iscomplex ? [complex(unif(g), unif(g)) for _ in 1:n] : [unif(g) for _ in 1:n]

out = IOBuffer()
info = Pkg.dependencies()[Base.UUID("0b1a1467-8014-51b9-945f-bf0ae24f4b77")]
println(out, "# KrylovKit ", info.version, " git-tree-sha1 ", info.tree_hash)
fmt(xs) = join((repr(Float64(x)) for x in xs), " ")

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

eigcase(out, "eig_real_sr", 1, 120, false, 3, :SR, 20, 1e-10, false)
eigcase(out, "eig_complex_lm", 2, 100, true, 2, :LM, 16, 1e-10, false)
eigcase(out, "eig_real_lr_eager", 3, 80, false, 2, :LR, 30, 1e-11, true)
eigcase(out, "eig_complex_sr_small_k", 4, 60, true, 4, :SR, 8, 1e-9, false)
expcase(out, "exp_real_imag_time", 5, 60, false, -2.0, 12, 1e-12, false)
expcase(out, "exp_complex_real_time", 6, 60, true, -5.0im, 15, 1e-12, false)
expcase(out, "exp_complex_imag_time_eager", 7, 50, true, complex(-1.5), 20, 1e-12, true)

write(joinpath(@__DIR__, "fixtures.txt"), take!(out))
