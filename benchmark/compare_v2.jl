# Compare HomotopyContinuationNext vs HomotopyContinuation (v2)
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
println("  HomotopyContinuationNext vs HomotopyContinuation (v2)")
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

# ─────────────────────────────────────────────────────────────────────────
# 3. Interpreter: evaluate! and evaluate_and_jacobian! on same systems
# ─────────────────────────────────────────────────────────────────────────

using DynamicPolynomials: @polyvar
using HomotopyContinuationNext: build_interpreter, build_jacobian_interpreter
using HomotopyContinuation.ModelKit: @var, System, InterpretedSystem

println("\n── Interpreter: evaluate! (same polynomial, different tape) ──")

# Build same system in both APIs
# Katsura-3
@polyvar px0 px1 px2 px3
F_next_polys = [
    px0 + 2px1 + 2px2 + 2px3 - 1,
    px0^2 + 2px1^2 + 2px2^2 + 2px3^2 - px0,
    2px0 * px1 + 2px1 * px2 + 2px2 * px3 - px1,
    px1^2 + 2px0 * px2 + 2px1 * px3 - px2,
]

@var vx0 vx1 vx2 vx3
F_hc_sys = System([
    vx0 + 2vx1 + 2vx2 + 2vx3 - 1,
    vx0^2 + 2vx1^2 + 2vx2^2 + 2vx3^2 - vx0,
    2vx0 * vx1 + 2vx1 * vx2 + 2vx2 * vx3 - vx1,
    vx1^2 + 2vx0 * vx2 + 2vx1 * vx3 - vx2,
])

# Next: build_interpreter → execute!
I_next = build_interpreter(F_next_polys)
u_next = zeros(ComplexF64, 4)
x_val = ComplexF64.(randn(4))

# HC: InterpretedSystem → evaluate!
IS_hc = InterpretedSystem(F_hc_sys)
u_hc = zeros(ComplexF64, 4)

t_next = @belapsed Next.execute!($u_next, $I_next, $x_val, $(ComplexF64[]))
t_hc = @belapsed HC.ModelKit.evaluate!($u_hc, $IS_hc, $x_val)
ratio = t_hc / t_next
println("  katsura3 eval:  Next=$(round(t_next * 1e9; digits=1))ns  HC=$(round(t_hc * 1e9; digits=1))ns  ratio=$(round(ratio; digits=2))x")

# With Jacobian
I_jac_next = build_jacobian_interpreter(F_next_polys)
U_next = zeros(ComplexF64, 4, 4)
U_hc = zeros(ComplexF64, 4, 4)

t_next = @belapsed Next.execute!($u_next, $U_next, $I_jac_next, $x_val, $(ComplexF64[]))
t_hc = @belapsed HC.ModelKit.evaluate_and_jacobian!($u_hc, $U_hc, $IS_hc, $x_val)
ratio = t_hc / t_next
println("  katsura3 eval+jac:  Next=$(round(t_next * 1e9; digits=1))ns  HC=$(round(t_hc * 1e9; digits=1))ns  ratio=$(round(ratio; digits=2))x")

# Cyclic-5
@polyvar cy1 cy2 cy3 cy4 cy5
F_next_cyclic = [
    cy1 + cy2 + cy3 + cy4 + cy5,
    cy1 * cy2 + cy2 * cy3 + cy3 * cy4 + cy4 * cy5 + cy5 * cy1,
    cy1 * cy2 * cy3 + cy2 * cy3 * cy4 + cy3 * cy4 * cy5 + cy4 * cy5 * cy1 + cy5 * cy1 * cy2,
    cy1 * cy2 * cy3 * cy4 + cy2 * cy3 * cy4 * cy5 + cy3 * cy4 * cy5 * cy1 +
    cy4 * cy5 * cy1 * cy2 + cy5 * cy1 * cy2 * cy3,
    cy1 * cy2 * cy3 * cy4 * cy5 - 1,
]

@var vy1 vy2 vy3 vy4 vy5
F_hc_cyclic = System([
    vy1 + vy2 + vy3 + vy4 + vy5,
    vy1 * vy2 + vy2 * vy3 + vy3 * vy4 + vy4 * vy5 + vy5 * vy1,
    vy1 * vy2 * vy3 + vy2 * vy3 * vy4 + vy3 * vy4 * vy5 + vy4 * vy5 * vy1 + vy5 * vy1 * vy2,
    vy1 * vy2 * vy3 * vy4 + vy2 * vy3 * vy4 * vy5 + vy3 * vy4 * vy5 * vy1 +
    vy4 * vy5 * vy1 * vy2 + vy5 * vy1 * vy2 * vy3,
    vy1 * vy2 * vy3 * vy4 * vy5 - 1,
])

I_next_cyc = build_interpreter(F_next_cyclic)
u_next_cyc = zeros(ComplexF64, 5)
x_cyc = ComplexF64.(randn(5))

IS_hc_cyc = InterpretedSystem(F_hc_cyclic)
u_hc_cyc = zeros(ComplexF64, 5)

t_next = @belapsed Next.execute!($u_next_cyc, $I_next_cyc, $x_cyc, $(ComplexF64[]))
t_hc = @belapsed HC.ModelKit.evaluate!($u_hc_cyc, $IS_hc_cyc, $x_cyc)
ratio = t_hc / t_next
println("  cyclic5 eval:  Next=$(round(t_next * 1e9; digits=1))ns  HC=$(round(t_hc * 1e9; digits=1))ns  ratio=$(round(ratio; digits=2))x")

println("\n" * "="^72)
println("  ratio > 1.0 means Next is faster")
println("="^72)
