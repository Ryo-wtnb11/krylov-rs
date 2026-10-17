# Generates hseqr.txt: eigenvalues, in LAPACK order, of fixed upper
# Hessenberg matrices, from Julia's LAPACK.hseqr! (job 'S', compz 'I').
# Read by src/dense/schur.rs tests (tolerance-based, order-checked).
#
#   julia tests/fixtures/lapack/hseqr.jl
#
# Entries are small multiples of 1/4, so the Rust test rebuilds them exactly.
using LinearAlgebra

re(i, j) = (mod(7i + 3j * j, 11) - 5) / 4
im_(i, j) = (mod(5i + 2j, 7) - 3) / 2
hess(n, f) = [i <= j + 1 ? f(i, j) : zero(f(1, 1)) for i in 1:n, j in 1:n]
hex(x::Float64) = string(reinterpret(UInt64, x); base = 16, pad = 16)

out = IOBuffer()
println(out, "# Julia ", VERSION, " LAPACK ", LinearAlgebra.LAPACK.version())
for n in (6, 12)
    H = hess(n, re)
    _, _, w = LAPACK.hseqr!('S', 'I', 1, n, copy(H), zeros(n, n))
    println(out, "real ", n, " ", join((hex(real(x)) * ":" * hex(imag(x)) for x in w), " "))
end
for n in (5, 10)
    H = hess(n, (i, j) -> complex(re(i, j), im_(i, j)))
    _, _, w = LAPACK.hseqr!('S', 'I', 1, n, copy(H), zeros(ComplexF64, n, n))
    println(out, "complex ", n, " ", join((hex(real(x)) * ":" * hex(imag(x)) for x in w), " "))
end
write(joinpath(@__DIR__, "hseqr.txt"), take!(out))
