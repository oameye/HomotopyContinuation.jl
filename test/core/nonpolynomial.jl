using Test
using Random: MersenneTwister, rand
using HomotopyContinuation

include("../helpers/test_systems.jl")

const NONPOLY_MODES = (
    CompileMode.INTERPRETED,
    CompileMode.COMPILED,
    CompileMode.COMPILED_ALL,
)

function finite_difference_jacobian(f, x; h = 1.0e-6)
    y = f(x)
    J = zeros(ComplexF64, length(y), length(x))
    for j in eachindex(x)
        step = zeros(ComplexF64, length(x))
        step[j] = h
        J[:, j] .= (f(x .+ step) .- f(x .- step)) ./ (2h)
    end
    return J
end

include("../helpers/solution_sets.jl")


@testset "non-polynomial systems through the public API" begin
    @testset "analytic system collection: $name" for (name, exprs, vars, params, reference) in
        NONPOLYNOMIAL_SYSTEM_COLLECTION

        rng = MersenneTwister(0x00e8b1a5 + length(name))
        x = ComplexF64.(0.7 .+ rand(rng, length(vars)))
        p = ComplexF64.(0.7 .+ rand(rng, length(params)))
        expected = reference(x, p)
        expected_jacobian = finite_difference_jacobian(z -> reference(z, p), x)

        systems = [
            System(exprs; variables = vars, parameters = params, compile = mode)
                for mode in NONPOLY_MODES
        ]
        for F in systems
            @test size(F) == (length(exprs), length(vars))
            @test evaluate(F, x, p) ≈ expected rtol = 1.0e-10
            @test jacobian(F, x, p) ≈ expected_jacobian rtol = 1.0e-5
        end

        values = [evaluate(F, x, p) for F in systems]
        jacobians = [jacobian(F, x, p) for F in systems]
        @test all(v -> isapprox(v, first(values); rtol = 1.0e-12), values)
        @test all(J -> isapprox(J, first(jacobians); rtol = 1.0e-12), jacobians)
    end

    @testset "any Expression vector builds a System" begin
        @var x y
        exprs = [x^2 - 1, y / x - 2]
        reference = System(exprs)
        point = ComplexF64[2, 3]
        for input in (view(exprs, :), Expression[e for e in exprs])
            @test evaluate(System(input), point) ≈ evaluate(reference, point)
        end
    end

    @testset "rational system has the expected values and Jacobian" begin
        @var x y u[1:4]
        F = System(
            [u[1] / x^2 + u[2], u[3] / y^2 + u[4]];
            variables = [x, y], parameters = u,
        )
        point = ComplexF64[1.4 + 0.3im, -0.8 + 0.5im]
        p = ComplexF64[2.0, -1.0, 3.0, 0.5]

        expected = ComplexF64[p[1] / point[1]^2 + p[2], p[3] / point[2]^2 + p[4]]
        expected_jacobian = ComplexF64[
            -2p[1] / point[1]^3 0
            0 -2p[3] / point[2]^3
        ]
        @test evaluate(F, point, p) ≈ expected
        @test jacobian(F, point, p) ≈ expected_jacobian
    end

    @testset "sqrt-parameter system is backend independent" begin
        @var x y a b
        equations = [sqrt(a + b) * x^2 - y, (x * y + a - sqrt(b))^2 - 3]
        point = ComplexF64[1.3 + 0.2im, -0.7 + 0.4im]
        p = ComplexF64[2, 3]
        reference(z) = ComplexF64[
            sqrt(p[1] + p[2]) * z[1]^2 - z[2],
            (z[1] * z[2] + p[1] - sqrt(p[2]))^2 - 3,
        ]
        expected_jacobian = finite_difference_jacobian(reference, point)

        systems = [
            System(equations; variables = [x, y], parameters = [a, b], compile = mode)
                for mode in NONPOLY_MODES
        ]
        for F in systems
            @test evaluate(F, point, p) ≈ reference(point)
            @test jacobian(F, point, p) ≈ expected_jacobian atol = 1.0e-6
        end
    end

    @testset "transcendental differentiation agrees with analysis" begin
        @var x
        expressions = [
            sin(x), cos(x), exp(x), tan(x), asin(x), acos(x), sinh(x), cosh(x), tanh(x),
        ]
        derivatives = [
            cos(x),
            -sin(x),
            exp(x),
            1 + tan(x)^2,
            1 / sqrt(1 - x^2),
            -1 / sqrt(1 - x^2),
            cosh(x),
            sinh(x),
            1 - tanh(x)^2,
        ]
        @test expand.(differentiate(expressions, x) - derivatives) == fill(zero(Expression), 9)

        x0 = 0.1
        expected = ComplexF64[
            sin(x0), cos(x0), exp(x0), tan(x0), asin(x0), acos(x0),
            sinh(x0), cosh(x0), tanh(x0),
        ]
        expected_derivative = ComplexF64[
            cos(x0), -sin(x0), exp(x0), 1 + tan(x0)^2,
            1 / sqrt(1 - x0^2), -1 / sqrt(1 - x0^2),
            cosh(x0), sinh(x0), 1 - tanh(x0)^2,
        ]
        for mode in NONPOLY_MODES
            F = System(expressions; variables = [x], compile = mode)
            @test evaluate(F, [x0]) ≈ expected
            @test vec(jacobian(F, [x0])) ≈ expected_derivative
        end
    end

    @testset "fractional powers evaluate and differentiate correctly" begin
        @var x p
        F = System([(x + 1)^(3 // 2) - p]; variables = [x], parameters = [p])
        x0 = ComplexF64[-0.5]
        p0 = ComplexF64[0.1]
        @test only(evaluate(F, x0, p0)) ≈ (x0[1] + 1)^(3 / 2) - p0[1]
        @test only(jacobian(F, x0, p0)) ≈ (3 / 2) * sqrt(x0[1] + 1)

        negative = System([(x + 4)^(-4 / 3)]; variables = [x])
        complex_power = System([(x + 4)^(1.5im)]; variables = [x])
        @test only(evaluate(negative, [1.0])) ≈ 5.0^(-4 / 3)
        @test only(evaluate(complex_power, [1.0])) ≈ ComplexF64(5)^1.5im
    end

    @testset "DynamicPolynomials and Expression rational frontends agree" begin
        @polyvar px py pu[1:4]
        polynomial_frontend = System(
            [pu[1] / px^2 + pu[2], pu[3] / py^2 + pu[4]];
            variables = [px, py], parameters = pu,
        )
        @var x y u[1:4]
        expression_frontend = System(
            [u[1] / x^2 + u[2], u[3] / y^2 + u[4]];
            variables = [x, y], parameters = u,
        )
        point = ComplexF64[1.4 + 0.3im, -0.8 + 0.5im]
        p = ComplexF64[2.0, -1.0, 3.0, 0.5]
        @test evaluate(polynomial_frontend, point, p) ≈ evaluate(expression_frontend, point, p)
        @test jacobian(polynomial_frontend, point, p) ≈ jacobian(expression_frontend, point, p)
    end
end
