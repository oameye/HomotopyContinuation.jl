using Test
import HomotopyContinuationNext
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: differentiate as mp_diff
using LinearAlgebra: norm
using HomotopyContinuationNext: System, execute!, execute_taylor!,
    TaylorVector, TruncatedTaylorSeries, DoubleF64, ComplexDF64, OpType, instruction_op,
    Interpreter, _build_instruction_sequence_direct, _build_instruction_sequence_via_sexpr,
    _effective_variables, _empty_vars

# Helpers: build a System through the public polynomial input path
function test_system(F; parameters = nothing, variables = nothing)
    if parameters === nothing
        if variables === nothing
            return System(F)
        end
        return System(F; variables = variables)
    elseif variables === nothing
        return System(F; parameters = parameters)
    else
        return System(F; parameters = parameters, variables = variables)
    end
end

eval_interp(F; parameters = nothing, variables = nothing) =
    test_system(F; parameters = parameters, variables = variables)._interp_f64
jac_interp(F; parameters = nothing, variables = nothing) =
    test_system(F; parameters = parameters, variables = variables)._interp_jac
df64_interp(F; parameters = nothing, variables = nothing) =
    test_system(F; parameters = parameters, variables = variables)._interp_df64
eval_sequence(F; parameters = nothing, variables = nothing) =
    test_system(F; parameters = parameters, variables = variables)._interp_f64.sequence

function taylor_interp(F, ::Val{K}; parameters = nothing, variables = nothing) where {K}
    sys = test_system(F; parameters = parameters, variables = variables)
    if K == 1
        return sys._interp_t1
    elseif K == 2
        return sys._interp_t2
    elseif K == 3
        return sys._interp_t3
    end
    error("unsupported Taylor order $K")
end

# Helpers: evaluate polynomial system via MP substitution
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

# Helper: test eval + jacobian against MP ground truth at multiple random points
function test_eval_and_jac(F, vars; npoints = 3)
    n = length(vars)
    m = length(F)
    sys = test_system(F)
    I_eval = sys._interp_f64
    I_jac = sys._interp_jac
    for _ in 1:npoints
        x = ComplexF64.(randn(n))
        u = zeros(ComplexF64, m)
        U = zeros(ComplexF64, m, n)
        execute!(u, I_eval, x)
        @test u ≈ eval_mp(F, vars, x) rtol = 1.0e-10
        execute!(u, U, I_jac, x)
        @test u ≈ eval_mp(F, vars, x) rtol = 1.0e-10
        @test U ≈ eval_mp_jacobian(F, vars, x) rtol = 1.0e-10
    end
    return
end

# ─────────────────────────────────────────────────────────────────────────
# Edge cases: eval
# ─────────────────────────────────────────────────────────────────────────

@testset "Eval: edge cases" begin
    @testset "single variable" begin
        @polyvar x
        test_eval_and_jac([x^3 - 1], [x])
    end

    @testset "constant polynomial term" begin
        @polyvar x y
        F = [x + y, x * y, x - y + 3]
        I = eval_interp(F)
        u = zeros(ComplexF64, 3)
        execute!(u, I, ComplexF64[1.0, 2.0])
        @test u ≈ ComplexF64[3.0, 2.0, 2.0]
    end

    @testset "high-degree monomial" begin
        @polyvar x
        test_eval_and_jac([x^7 - x^3 + x - 1], [x])
    end

    @testset "pure quadratic (no linear terms)" begin
        @polyvar x y
        test_eval_and_jac([x^2 + y^2, x^2 - y^2], [x, y])
    end

    @testset "purely linear system" begin
        @polyvar x y z
        F = [2x + 3y - z + 1, x - y + 2z - 3, -x + y + z]
        test_eval_and_jac(F, [x, y, z])
    end

    @testset "single polynomial (m=1)" begin
        @polyvar x y z
        test_eval_and_jac([x^2 * y + y^2 * z + z^2 * x - 1], [x, y, z])
    end

    @testset "negative coefficients (-1 handling)" begin
        @polyvar x y
        F = [-x * y, -x^2 - y^2 + 1, -2x + 3y - 1]
        test_eval_and_jac(F, [x, y])
    end

    @testset "complex coefficients" begin
        @polyvar x y
        F = [(1.0 + 2.0im) * x + (3.0 - 1.0im) * y, x * y - (2.0im) * x]
        test_eval_and_jac(F, [x, y])
    end

    @testset "mixed real and complex coefficients" begin
        @polyvar x y
        F = [x^2 + (1.0im) * x * y + y^2, 2x - (3.0 + 1.0im) * y + 1]
        test_eval_and_jac(F, [x, y])
    end

    @testset "large shared subexpressions" begin
        @polyvar x y z
        # x*y*z appears in multiple polynomials → CSE should share it
        F = [x * y * z + x + y, x * y * z + y + z, x * y * z + z + x]
        test_eval_and_jac(F, [x, y, z])
    end

    @testset "identical polynomials (repeated outputs)" begin
        @polyvar x y
        F = [2x * y, 2x * y]
        sys = test_system(F)
        I = sys._interp_f64
        IJ = sys._interp_jac
        xval = ComplexF64.(randn(2))
        u = zeros(ComplexF64, 2)
        U = zeros(ComplexF64, 2, 2)
        execute!(u, I, xval)
        execute!(u, U, IJ, xval)
        @test u ≈ eval_mp(F, [x, y], xval) rtol = 1.0e-12
        @test U ≈ eval_mp_jacobian(F, [x, y], xval) rtol = 1.0e-12
    end

    @testset "coefficient 2 (tests 2*x → x+x optimization)" begin
        @polyvar x y
        F = [2x + 2y, 2x * y, 2x^2 + 2y^2]
        test_eval_and_jac(F, [x, y])
    end
end

# ─────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────

@testset "Systems with parameters" begin
    @testset "eval with parameters" begin
        @polyvar x y a b
        F = [x^2 + a * y, x * y - b]
        I = eval_interp(F; parameters = [a, b])
        u = zeros(ComplexF64, 2)
        execute!(u, I, ComplexF64[2.0, 3.0], ComplexF64[1.0, 1.0])
        @test u[1] ≈ 7.0 + 0im
        @test u[2] ≈ 5.0 + 0im
    end

    @testset "Jacobian with parameters" begin
        @polyvar x y a b
        F = [a * x^2 + y, x * y - b]
        IJ = jac_interp(F; parameters = [a, b])
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

    @testset "Jacobian with parameters vs ground truth" begin
        @polyvar x y z p q
        vars = [x, y, z]
        params = [p, q]
        F = [p * x^2 + q * y * z, x * y - p * z^2 + q, x + y + z - p * q]
        IJ = jac_interp(F; parameters = params)
        x_val = ComplexF64.(randn(3))
        p_val = ComplexF64[1.5, -0.7]  # real values to keep MP substitution simple
        u = zeros(ComplexF64, 3)
        U = zeros(ComplexF64, 3, 3)
        execute!(u, U, IJ, x_val, p_val)
        # Compute ground truth by direct substitution
        u_expected = ComplexF64[f(vars => x_val, params => Float64.(real.(p_val))) for f in F]
        @test u ≈ u_expected rtol = 1.0e-10
        J_expected = zeros(ComplexF64, 3, 3)
        for j in 1:3, i in 1:3
            dp = mp_diff(F[i], vars[j])
            J_expected[i, j] = dp(vars => x_val, params => Float64.(real.(p_val)))
        end
        @test U ≈ J_expected rtol = 1.0e-10
    end
end

# ─────────────────────────────────────────────────────────────────────────
# Direct polynomial compiler parity
# ─────────────────────────────────────────────────────────────────────────

@testset "Direct polynomial compiler parity" begin
    @testset "eval and jacobian agree with symbolic compiler" begin
        @polyvar x y z a
        F = [a * x^2 + y * z - 1, x * y + z^2 - a]
        vars = _effective_variables(F, [a])
        params = [a]
        seq_symbolic = _build_instruction_sequence_via_sexpr(F, vars, params, true)
        seq_direct = _build_instruction_sequence_direct(F, vars, params, true)
        I_symbolic = Interpreter(Vector{ComplexF64}, seq_symbolic)
        I_direct = Interpreter(Vector{ComplexF64}, seq_direct)

        x_val = ComplexF64[0.3 - 0.2im, -0.7 + 0.1im, 0.5 + 0.4im]
        p_val = ComplexF64[1.2 - 0.3im]
        u_symbolic = zeros(ComplexF64, 2)
        U_symbolic = zeros(ComplexF64, 2, 3)
        u_direct = zeros(ComplexF64, 2)
        U_direct = zeros(ComplexF64, 2, 3)

        execute!(u_symbolic, U_symbolic, I_symbolic, x_val, p_val)
        execute!(u_direct, U_direct, I_direct, x_val, p_val)

        @test u_direct ≈ u_symbolic rtol = 1.0e-12
        @test U_direct ≈ U_symbolic rtol = 1.0e-12
    end
end

# ─────────────────────────────────────────────────────────────────────────
# Real polynomial systems (ground truth correctness)
# ─────────────────────────────────────────────────────────────────────────

@testset "Real systems: ground truth" begin
    @testset "Katsura-3" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        test_eval_and_jac(F, [x0, x1, x2, x3]; npoints = 5)
    end

    @testset "Cyclic-5" begin
        @polyvar x1 x2 x3 x4 x5
        vars = [x1, x2, x3, x4, x5]
        F = [
            x1 + x2 + x3 + x4 + x5,
            x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1,
            x1 * x2 * x3 + x2 * x3 * x4 + x3 * x4 * x5 + x4 * x5 * x1 + x5 * x1 * x2,
            x1 * x2 * x3 * x4 + x2 * x3 * x4 * x5 + x3 * x4 * x5 * x1 +
                x4 * x5 * x1 * x2 + x5 * x1 * x2 * x3,
            x1 * x2 * x3 * x4 * x5 - 1,
        ]
        test_eval_and_jac(F, vars; npoints = 5)
    end

    @testset "Cubic with negative and unit coefficients" begin
        @polyvar x y
        F = [x^3 - 3x * y^2 + y^3, -x^3 + x * y^2 - y]
        test_eval_and_jac(F, [x, y]; npoints = 5)
    end
end

# ─────────────────────────────────────────────────────────────────────────
# Instruction count bounds (smoke check)
# ─────────────────────────────────────────────────────────────────────────

@testset "Instruction count bounds" begin
    @testset "Katsura-3 eval uses fused ops" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        seq = eval_sequence(F)
        ops = [instruction_op(instr) for instr in seq.instructions]
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
        seq = eval_sequence(F)
        @test length(seq.instructions) - 1 <= 25
    end
end

# ─────────────────────────────────────────────────────────────────────────
# Taylor via finite differences
# ─────────────────────────────────────────────────────────────────────────

@testset "Taylor via System interpreters" begin
    @testset "simple 2-var system" begin
        @polyvar x y
        F = [x^3 + x * y - 2, y^2 - x]
        x0 = ComplexF64[1.5, -0.5]
        v = ComplexF64[0.3, 0.7]
        ε = 1.0e-7

        I = eval_interp(F)
        u0 = zeros(ComplexF64, 2)
        u1 = zeros(ComplexF64, 2)
        execute!(u0, I, x0, ComplexF64[])
        execute!(u1, I, x0 .+ ε .* v, ComplexF64[])
        fd_deriv = (u1 .- u0) ./ ε

        I_t1 = taylor_interp(F, Val(1))
        tx = TaylorVector{2, ComplexF64}(2)
        tx[1] = TruncatedTaylorSeries((x0[1], v[1]))
        tx[2] = TruncatedTaylorSeries((x0[2], v[2]))
        u_taylor = zeros(ComplexF64, 2)
        execute_taylor!(u_taylor, Val(1), I_t1, tx, ComplexF64[])

        @test u_taylor ≈ fd_deriv rtol = 1.0e-5
    end

    @testset "Katsura-3 Taylor order 1" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        pt = ComplexF64.(randn(4))
        v = ComplexF64.(randn(4))
        ε = 1.0e-7

        I = eval_interp(F)
        u0 = zeros(ComplexF64, 4)
        u1 = zeros(ComplexF64, 4)
        execute!(u0, I, pt, ComplexF64[])
        execute!(u1, I, pt .+ ε .* v, ComplexF64[])
        fd_deriv = (u1 .- u0) ./ ε

        I_t = taylor_interp(F, Val(1))
        tx = TaylorVector{2, ComplexF64}(4)
        for i in 1:4
            tx[i] = TruncatedTaylorSeries((pt[i], v[i]))
        end
        u_taylor = zeros(ComplexF64, 4)
        execute_taylor!(u_taylor, Val(1), I_t, tx, ComplexF64[])

        @test u_taylor ≈ fd_deriv rtol = 1.0e-5
    end
end

# ─────────────────────────────────────────────────────────────────────────
# DF64
# ─────────────────────────────────────────────────────────────────────────

@testset "DF64 via System interpreters" begin
    @testset "simple system" begin
        @polyvar x y
        F = [x^3 + x * y - 2, y^2 - x]
        x_val = ComplexF64[1.5, -0.5]

        I = eval_interp(F)
        u_f64 = zeros(ComplexF64, 2)
        execute!(u_f64, I, x_val, ComplexF64[])

        I_df64 = df64_interp(F)
        u_df64 = zeros(ComplexDF64, 2)
        execute!(u_df64, I_df64, ComplexDF64.(x_val), ComplexDF64[])

        @test Float64.(real.(u_df64)) ≈ Float64.(real.(u_f64)) rtol = 1.0e-14
        @test Float64.(imag.(u_df64)) ≈ Float64.(imag.(u_f64)) rtol = 1.0e-14
    end

    @testset "complex coefficients" begin
        @polyvar x y
        F = [(1.0 + 2.0im) * x^2 + (3.0 - 1.0im) * y, x * y - (2.0im)]
        x_val = ComplexF64[0.7 + 0.3im, -0.5 + 0.2im]

        I = eval_interp(F)
        u_f64 = zeros(ComplexF64, 2)
        execute!(u_f64, I, x_val, ComplexF64[])

        I_df64 = df64_interp(F)
        u_df64 = zeros(ComplexDF64, 2)
        execute!(u_df64, I_df64, ComplexDF64.(x_val), ComplexDF64[])

        @test Float64.(real.(u_df64)) ≈ Float64.(real.(u_f64)) rtol = 1.0e-14
        @test Float64.(imag.(u_df64)) ≈ Float64.(imag.(u_f64)) rtol = 1.0e-14
    end
end
