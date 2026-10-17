# Dense `exp!` (Julia LinearAlgebra, which KrylovKit's expintegrator calls on
# its augmented matrix) cases for src/dense/expm.rs; included by generate.jl,
# which defines `out` and the shared input generators.

function expmcase(out, name, seed, n, iscomplex, kind, scale)
    g = SplitMix(seed)
    T = iscomplex ? ComplexF64 : Float64
    A = zeros(T, n, n)
    for j in 1:n, i in 1:n
        A[i, j] = iscomplex ? complex(unif(g), unif(g)) : unif(g)
    end
    if kind == :nilpotent
        A = triu(A, 1)
    elseif kind == :graded # exact similarity by powers of two
        for j in 1:n, i in 1:n
            A[i, j] *= 2.0^(4 * (i - j))
        end
    end
    A .*= scale
    X = LinearAlgebra.exp!(copy(A))
    println(out, "case ", name)
    println(out, "kind ", kind, "\ncomplex ", Int(iscomplex), "\nseed ", seed, "\nn ", n)
    println(out, "scale ", repr(scale), "\njulia ", VERSION)
    println(out, "x_re ", fmt(real.(vec(X))))
    println(out, "x_im ", fmt(imag.(vec(X))))
    return println(out, "end")
end

expmcase(out, "expm_deg3", 31, 8, false, :random, 0.001)
expmcase(out, "expm_deg5", 32, 8, true, :random, 0.01)
expmcase(out, "expm_deg7", 33, 8, false, :random, 0.06)
expmcase(out, "expm_deg9", 34, 8, true, :random, 0.2)
expmcase(out, "expm_deg13", 35, 8, false, :random, 0.8)
expmcase(out, "expm_squaring", 36, 10, true, :random, 6.0)
expmcase(out, "expm_squaring_real", 37, 10, false, :random, 40.0)
expmcase(out, "expm_graded", 38, 7, true, :graded, 0.7)
expmcase(out, "expm_nilpotent", 39, 6, false, :nilpotent, 1e4)
