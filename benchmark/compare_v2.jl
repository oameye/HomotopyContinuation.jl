# Compare Phase 1 primitives: HomotopyContinuationNext vs HomotopyContinuation (v2)
# Run: julia --project=benchmark benchmark/compare_v2.jl
#
# Every benchmark here exercises a distinct codepath or scaling regime.
# Normal-range benchmarks serve as baselines for the extreme-case counterparts.

using BenchmarkTools
using LinearAlgebra: LinearAlgebra, I, norm, ldiv!, diagm

using HomotopyContinuationNext
using HomotopyContinuation

const Next = HomotopyContinuationNext
const HC = HomotopyContinuation

using FixedSizeArrays: FixedSizeArray
const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}

println("="^72)
println("  Phase 1 Primitives: HomotopyContinuationNext vs HomotopyContinuation")
println("="^72)

# ─────────────────────────────────────────────────────────────────────────
# 1. inf_norm: normal (abs2 fast path) vs overflow (isinf fallback to abs)
# ─────────────────────────────────────────────────────────────────────────
println("\n── inf_norm: normal range ──")

for n in [4, 16, 64]
    x_data = rand(ComplexF64, n)
    x_fs = FSVec{ComplexF64}(x_data)
    x_vec = Vector{ComplexF64}(x_data)

    t_next = @belapsed Next.inf_norm($x_fs)
    hc_inf = HC.InfNorm()
    t_hc = @belapsed $hc_inf($x_vec)
    ratio = t_hc / t_next
    println("  n=$n:  Next=$(round(t_next * 1.0e9; digits = 1))ns  HC=$(round(t_hc * 1.0e9; digits = 1))ns  ratio=$(round(ratio; digits = 2))x")
end

println("\n── inf_norm: overflow (exp2(700), triggers isinf fallback) ──")

for n in [4, 16, 64]
    x_data = exp2(700) .* rand(ComplexF64, n)
    x_fs = FSVec{ComplexF64}(x_data)
    x_vec = Vector{ComplexF64}(x_data)

    t_next = @belapsed Next.inf_norm($x_fs)
    hc_inf = HC.InfNorm()
    t_hc = @belapsed $hc_inf($x_vec)
    ratio = t_hc / t_next
    println("  n=$n:  Next=$(round(t_next * 1.0e9; digits = 1))ns  HC=$(round(t_hc * 1.0e9; digits = 1))ns  ratio=$(round(ratio; digits = 2))x")
end

# ─────────────────────────────────────────────────────────────────────────
# 2. LU + ldiv!: well-conditioned vs ill-conditioned (different pivot patterns)
# ─────────────────────────────────────────────────────────────────────────
println("\n── LU + ldiv!: well-conditioned ──")

for n in [4, 8, 16]
    A_data = rand(ComplexF64, n, n) + 5.0I
    b_data = rand(ComplexF64, n)

    WS_next = Next.MatrixWorkspace(n, n)
    copyto!(WS_next.A, A_data)
    Next.updated!(WS_next)
    x_next = FSVec{ComplexF64}(zeros(ComplexF64, n))
    b_next = FSVec{ComplexF64}(b_data)
    t_next = @belapsed begin
        copyto!($WS_next.A, $A_data)
        Next.updated!($WS_next)
        ldiv!($x_next, $WS_next, $b_next)
    end

    WS_hc = HC.MatrixWorkspace(n, n)
    copyto!(WS_hc.A, A_data)
    HC.updated!(WS_hc)
    x_hc = zeros(ComplexF64, n)
    b_hc = Vector{ComplexF64}(b_data)
    t_hc = @belapsed begin
        copyto!($WS_hc.A, $A_data)
        HC.updated!($WS_hc)
        ldiv!($x_hc, $WS_hc, $b_hc)
    end
    ratio = t_hc / t_next
    println("  n=$n:  Next=$(round(t_next * 1.0e9; digits = 1))ns  HC=$(round(t_hc * 1.0e9; digits = 1))ns  ratio=$(round(ratio; digits = 2))x")
end

println("\n── LU + ldiv!: ill-conditioned (cond ~ 1e12, different pivot patterns) ──")

for n in [4, 8, 16]
    d = exp10.(range(-6; stop = 6, length = n))
    A_data = ComplexF64.(diagm(d) * randn(n, n) + 0.01I)
    b_data = rand(ComplexF64, n)

    WS_next = Next.MatrixWorkspace(n, n)
    copyto!(WS_next.A, A_data)
    Next.updated!(WS_next)
    x_next = FSVec{ComplexF64}(zeros(ComplexF64, n))
    b_next = FSVec{ComplexF64}(b_data)
    t_next = @belapsed begin
        copyto!($WS_next.A, $A_data)
        Next.updated!($WS_next)
        ldiv!($x_next, $WS_next, $b_next)
    end

    WS_hc = HC.MatrixWorkspace(n, n)
    copyto!(WS_hc.A, A_data)
    HC.updated!(WS_hc)
    x_hc = zeros(ComplexF64, n)
    b_hc = Vector{ComplexF64}(b_data)
    t_hc = @belapsed begin
        copyto!($WS_hc.A, $A_data)
        HC.updated!($WS_hc)
        ldiv!($x_hc, $WS_hc, $b_hc)
    end
    ratio = t_hc / t_next
    println("  n=$n:  Next=$(round(t_next * 1.0e9; digits = 1))ns  HC=$(round(t_hc * 1.0e9; digits = 1))ns  ratio=$(round(ratio; digits = 2))x")
end

println("\n" * "="^72)
println("  ratio > 1.0 means Next is faster")
println("="^72)
