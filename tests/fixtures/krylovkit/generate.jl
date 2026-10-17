# Generates the KrylovKit fixtures for tests/krylovkit.rs.
#
#   julia --project=tests/fixtures/krylovkit tests/fixtures/krylovkit/generate.jl
#
# Each algorithm family lives in its own `<family>.jl`, which defines the
# case functions and `cases(out)`; its fixtures go to `<family>.txt`.
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

info = Pkg.dependencies()[Base.UUID("0b1a1467-8014-51b9-945f-bf0ae24f4b77")]
fmt(xs) = join((repr(Float64(x)) for x in xs), " ")

for family in ("eigsolve", "exponentiate")
    global out = IOBuffer()
    println(out, "# KrylovKit ", info.version, " git-tree-sha1 ", info.tree_hash)
    include(joinpath(@__DIR__, family * ".jl"))
    write(joinpath(@__DIR__, family * ".txt"), take!(out))
end
