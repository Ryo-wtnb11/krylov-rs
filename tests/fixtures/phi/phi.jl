# Generates phi.txt, the high-precision oracle for the φ-function tests in
# src/dense/mod.rs (no packages needed):
#
#   julia tests/fixtures/phi/phi.jl > tests/fixtures/phi/phi.txt
#
# φ_j(x) = Σ_n xⁿ/(n+j)! summed in 1024-bit BigFloat until the terms fall
# below 2⁻⁹⁰⁰ of the running sum; the series is exact for every x, and the
# working precision covers its cancellation for |x| ≤ 60 (e⁶⁰ ≈ 2⁸⁷). Each
# line is `bits re(x) im(x) j re(φ_j) im(φ_j)` with x exactly representable
# in Float`bits` (64 or 32), values rounded to Float64 with 17 significant
# digits.
using Printf
setprecision(BigFloat, 1024)

function phi(x::Complex{BigFloat}, j)
    t = one(x) / factorial(big(j))
    s = zero(x)
    n = 0
    while true
        s += t
        n += 1
        t = t * x / (n + j)
        (n > abs(x) + 10 && abs(t) < abs(s) * big(2.0)^-900) && break
    end
    return s
end

reals = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0, 1.01, 1.1, 1.3, 1.5, 2.0, 2.5,
         3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 30.0, 40.0]
xs = Complex{Float64}[]
for r in reals
    push!(xs, complex(r), complex(-r))
end
for r in [0.5, 0.9, 1.1, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 15.0, 20.0],
    a in [0.1, pi / 6, pi / 3, pi / 2, 2pi / 3, 5pi / 6, pi - 0.1]
    push!(xs, r * cis(a))
end
for (bits, T) in [(64, Float64), (32, Float32)], x0 in xs, j in 1:9
    x = Complex{Float64}(Complex{T}(x0))
    v = phi(Complex{BigFloat}(x), j)
    @printf("%d %.17e %.17e %d %.17e %.17e\n", bits, real(x), imag(x), j,
            Float64(real(v)), Float64(imag(v)))
end
