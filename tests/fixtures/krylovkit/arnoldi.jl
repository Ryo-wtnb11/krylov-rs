# Arnoldi (Krylov–Schur) cases: `schursolve`, `eigsolve(…, ::Arnoldi)` and
# `realeigsolve`; included by generate.jl, which defines `out` and the shared
# input generators.

# G/√n (spectrum roughly in the unit disk) plus outliers on the diagonal:
# real 3, -2.5, 2 and a 2×2 rotation block 1.5 ± 1.2i; complex 3i, -2.5,
# 2 + 1i and -1.5 - 1.5i.
function outliers(g, n, iscomplex)
    A = zeros(iscomplex ? ComplexF64 : Float64, n, n)
    for j in 1:n, i in 1:n
        A[i, j] = (iscomplex ? complex(unif(g), unif(g)) : unif(g)) / sqrt(n)
    end
    if iscomplex
        A[1, 1] += 3im
        A[2, 2] += -2.5
        A[3, 3] += 2 + 1im
        A[4, 4] += -1.5 - 1.5im
    else
        A[1, 1] += 3
        A[2, 2] += -2.5
        A[3, 3] += 2
        A[4, 4] += 1.5
        A[5, 5] += 1.5
        A[4, 5] += 1.2
        A[5, 4] -= 1.2
    end
    return A
end

# Real spectrum: a random upper triangular T0 conjugated by a permutation
# (exact in floating point), so the matrix is non-normal with known real
# eigenvalues diag(T0).
function permuted_triangular(g, n)
    T0 = zeros(n, n)
    for j in 1:n, i in 1:j
        T0[i, j] = i == j ? 4 * unif(g) : unif(g)
    end
    p = sortperm([unif(g) for _ in 1:n])
    return T0[p, p]
end

function arncase(out, name, kind, seed, n, iscomplex, howmany, which, krylovdim, tol, eager)
    g = SplitMix(seed)
    A = kind == :real ? permuted_triangular(g, n) : outliers(g, n, iscomplex)
    x0 = startvec(g, n, iscomplex)
    alg = Arnoldi(; krylovdim, tol, maxiter = 100, eager, orth = ClassicalGramSchmidt2())
    if kind == :schur
        _, _, vals, info = schursolve(A, x0, howmany, which, alg)
    elseif kind == :eig
        vals, _, info = eigsolve(A, x0, howmany, which, alg)
    else
        vals, _, info = realeigsolve(A, x0, howmany, which, alg)
    end
    println(out, "case ", name)
    println(out, "kind ", kind)
    println(out, "complex ", Int(iscomplex))
    println(out, "seed ", seed, "\nn ", n, "\nhowmany ", howmany, "\nwhich ", which)
    println(out, "krylovdim ", krylovdim, "\ntol ", repr(tol), "\neager ", Int(eager))
    println(out, "values_re ", fmt(real.(vals)))
    println(out, "values_im ", fmt(imag.(vals)))
    println(out, "normres ", fmt(info.normres))
    println(out, "converged ", info.converged, "\nnumiter ", info.numiter, "\nnumops ", info.numops)
    return println(out, "end")
end

arncase(out, "schur_real_lm_pair", :schur, 31, 80, false, 4, :LM, 20, 1e-10, false)
arncase(out, "schur_real_lr_eager", :schur, 32, 60, false, 3, :LR, 16, 1e-10, true)
arncase(out, "schur_real_sr", :schur, 33, 60, false, 2, :SR, 12, 1e-10, false)
arncase(out, "schur_complex_li", :schur, 34, 60, true, 3, :LI, 20, 1e-10, false)
arncase(out, "eig_complex_lm", :eig, 35, 70, true, 4, :LM, 20, 1e-10, false)
arncase(out, "eig_complex_si_eager", :eig, 36, 50, true, 2, :SI, 14, 1e-10, true)
arncase(out, "real_lr", :real, 37, 60, false, 3, :LR, 20, 1e-10, false)
arncase(out, "real_sr_eager", :real, 38, 50, false, 2, :SR, 16, 1e-10, true)
