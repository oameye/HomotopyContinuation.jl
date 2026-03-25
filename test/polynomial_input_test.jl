using Test
import HomotopyContinuationNext
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: differentiate as mp_diff
using LinearAlgebra: norm
using HomotopyContinuationNext: build_interpreter, build_jacobian_interpreter,
    build_taylor_interpreter, build_df64_interpreter,
    execute!, execute_taylor!, TaylorVector, TruncatedTaylorSeries,
    DoubleF64, ComplexDF64, OpType

# Helper: evaluate polynomial system via MP substitution
function eval_mp(F, vars, x_vals)
    return ComplexF64[p(vars => x_vals) for p in F]
end

function eval_mp_jacobian(F, vars, x_vals)
    m, n = length(F), length(vars)
    J = zeros(ComplexF64, m, n)
    for j in 1:n, i in 1:m
        dp = mp_diff(F[i], vars[j])
        J[i, j] = dp(vars => x_vals)
    end
    return J
end

# ─────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────

@testset "Polynomial input: edge cases" begin
    @testset "with parameters" begin
        @polyvar x y a b
        F = [x^2 + a * y, x * y - b]
        I = build_interpreter(F; parameters = [a, b])
        u = zeros(ComplexF64, 2)
        execute!(u, I, ComplexF64[2.0, 3.0], ComplexF64[1.0, 1.0])
        @test u[1] ≈ 7.0 + 0im
        @test u[2] ≈ 5.0 + 0im
    end

    @testset "single variable" begin
        @polyvar x
        F = [x^3 - 1]
        I = build_interpreter(F)
        u = zeros(ComplexF64, 1)
        execute!(u, I, ComplexF64[2.0])
        @test u[1] ≈ 7.0 + 0im
    end

    @testset "constant polynomial" begin
        @polyvar x y
        F = [x + y, x * y, x - y + 3]
        I = build_interpreter(F)
        u = zeros(ComplexF64, 3)
        x_val = ComplexF64[1.0, 2.0]
        execute!(u, I, x_val)
        @test u ≈ ComplexF64[3.0, 2.0, 2.0]
    end

    @testset "high-degree monomial" begin
        @polyvar x
        F = [x^7 - x^3 + x - 1]
        I = build_interpreter(F)
        u = zeros(ComplexF64, 1)
        x_val = ComplexF64[0.5]
        execute!(u, I, x_val)
        expected = 0.5^7 - 0.5^3 + 0.5 - 1.0
        @test u[1] ≈ expected + 0im
    end

    @testset "pure quadratic (no linear)" begin
        @polyvar x y
        F = [x^2 + y^2, x^2 - y^2]
        I = build_interpreter(F)
        u = zeros(ComplexF64, 2)
        x_val = ComplexF64[3.0, 4.0]
        execute!(u, I, x_val)
        @test u[1] ≈ 25.0 + 0im
        @test u[2] ≈ -7.0 + 0im
    end

    @testset "complex coefficients" begin
        @polyvar x y
        F = [(1.0 + 2.0im) * x + (3.0 - 1.0im) * y, x * y - (2.0im) * x]
        I = build_interpreter(F)
        u = zeros(ComplexF64, 2)
        x_val = ComplexF64[1.0 + 1.0im, 2.0]
        execute!(u, I, x_val)
        expected = eval_mp(F, [x, y], x_val)
        @test u ≈ expected
    end
end

# ─────────────────────────────────────────────────────────────────────────
# Jacobian correctness
# ─────────────────────────────────────────────────────────────────────────

@testset "Jacobian via build_jacobian_interpreter" begin
    @testset "Katsura-3" begin
        @polyvar x0 x1 x2 x3
        vars = [x0, x1, x2, x3]
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]

        x_val = ComplexF64.(randn(4))
        u = zeros(ComplexF64, 4)
        U = zeros(ComplexF64, 4, 4)

        IJ = build_jacobian_interpreter(F)
        execute!(u, U, IJ, x_val)

        @test u ≈ eval_mp(F, vars, x_val) rtol = 1e-12
        @test U ≈ eval_mp_jacobian(F, vars, x_val) rtol = 1e-12
    end

    @testset "with parameters" begin
        @polyvar x y a b
        F = [a * x^2 + y, x * y - b]
        IJ = build_jacobian_interpreter(F; parameters = [a, b])
        x_val = ComplexF64[2.0, 3.0]
        p_val = ComplexF64[1.5, 0.5]
        u = zeros(ComplexF64, 2)
        U = zeros(ComplexF64, 2, 2)
        execute!(u, U, IJ, x_val, p_val)

        @test u[1] ≈ 1.5 * 4.0 + 3.0
        @test u[2] ≈ 6.0 - 0.5
        @test U[1, 1] ≈ 1.5 * 2.0 * 2.0  # dF1/dx = 2*a*x
        @test U[1, 2] ≈ 1.0               # dF1/dy = 1
        @test U[2, 1] ≈ 3.0               # dF2/dx = y
        @test U[2, 2] ≈ 2.0               # dF2/dy = x
    end
end

# ─────────────────────────────────────────────────────────────────────────
# Repeated shared refs across outputs
# ─────────────────────────────────────────────────────────────────────────

@testset "Repeated shared refs across outputs" begin
    @polyvar x y
    I = build_interpreter([2x * y, 2x * y])
    IJ = build_jacobian_interpreter([2x * y, 2x * y])

    xval = ComplexF64.(randn(2))
    u = zeros(ComplexF64, 2)
    U = zeros(ComplexF64, 2, 2)

    execute!(u, I, xval)
    execute!(u, U, IJ, xval)

    expected_u = ComplexF64[2xval[1] * xval[2], 2xval[1] * xval[2]]
    expected_U = ComplexF64[
        2xval[2] 2xval[1]
        2xval[2] 2xval[1]
    ]

    @test u ≈ expected_u rtol = 1.0e-12
    @test U ≈ expected_U rtol = 1.0e-12
end

# ─────────────────────────────────────────────────────────────────────────
# Instruction count bounds (smoke check — detailed parity in instruction_count_test.jl)
# ─────────────────────────────────────────────────────────────────────────

@testset "Instruction count bounds" begin
    @testset "Katsura-3 eval tape" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        seq = build_interpreter(F).sequence
        ops = [instr.op for instr in seq.instructions]

        @test length(seq.instructions) - 1 <= 23
        @test OpType.OP_MULADD in ops || OpType.OP_MULMULADD in ops
    end

    @testset "Cyclic-5 eval tape" begin
        @polyvar x1 x2 x3 x4 x5
        F = [
            x1 + x2 + x3 + x4 + x5,
            x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1,
            x1 * x2 * x3 + x2 * x3 * x4 + x3 * x4 * x5 + x4 * x5 * x1 + x5 * x1 * x2,
            x1 * x2 * x3 * x4 + x2 * x3 * x4 * x5 + x3 * x4 * x5 * x1 +
                x4 * x5 * x1 * x2 + x5 * x1 * x2 * x3,
            x1 * x2 * x3 * x4 * x5 - 1,
        ]
        seq = build_interpreter(F).sequence
        @test length(seq.instructions) - 1 <= 25
    end
end

# ─────────────────────────────────────────────────────────────────────────
# Taylor via finite differences
# ─────────────────────────────────────────────────────────────────────────

@testset "Taylor via build_taylor_interpreter" begin
    @polyvar x y
    F = [x^3 + x * y - 2, y^2 - x]

    x0 = ComplexF64[1.5, -0.5]
    v = ComplexF64[0.3, 0.7]
    ε = 1.0e-7

    I = build_interpreter(F)
    u0 = zeros(ComplexF64, 2)
    u1 = zeros(ComplexF64, 2)
    execute!(u0, I, x0, ComplexF64[])
    execute!(u1, I, x0 .+ ε .* v, ComplexF64[])
    fd_deriv = (u1 .- u0) ./ ε

    I_t1 = build_taylor_interpreter(F, Val(1))
    tx = TaylorVector{2, ComplexF64}(2)
    tx[1] = TruncatedTaylorSeries((x0[1], v[1]))
    tx[2] = TruncatedTaylorSeries((x0[2], v[2]))
    u_taylor = zeros(ComplexF64, 2)
    execute_taylor!(u_taylor, Val(1), I_t1, tx, ComplexF64[])

    @test u_taylor ≈ fd_deriv rtol = 1.0e-5
end

# ─────────────────────────────────────────────────────────────────────────
# DF64
# ─────────────────────────────────────────────────────────────────────────

@testset "DF64 via build_df64_interpreter" begin
    @polyvar x y
    F = [x^3 + x * y - 2, y^2 - x]

    x_val = ComplexF64[1.5, -0.5]

    I = build_interpreter(F)
    u_f64 = zeros(ComplexF64, 2)
    execute!(u_f64, I, x_val, ComplexF64[])

    I_df64 = build_df64_interpreter(F)
    u_df64 = zeros(ComplexDF64, 2)
    execute!(u_df64, I_df64, ComplexDF64.(x_val), ComplexDF64[])

    @test Float64.(real.(u_df64)) ≈ Float64.(real.(u_f64)) rtol = 1.0e-14
    @test Float64.(imag.(u_df64)) ≈ Float64.(imag.(u_f64)) rtol = 1.0e-14
end
