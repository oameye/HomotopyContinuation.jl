using Test
using Random: MersenneTwister, rand
using HomotopyContinuation

include("test_systems.jl")

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

function same_solution_set(a, b; atol = 1.0e-7)
    length(a) == length(b) || return false
    return all(sa -> any(sb -> maximum(abs.(sa .- sb)) < atol, b), a)
end

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

    @testset "transcendental parameter continuation follows a known curve" begin
        @var x[1:9] t
        functions = [
            sin(t), cos(t), exp(t), tan(t), asin(t), acos(t), sinh(t), cosh(t), tanh(t),
        ]
        at(τ) = ComplexF64[
            sin(τ), cos(τ), exp(τ), tan(τ), asin(τ), acos(τ), sinh(τ), cosh(τ), tanh(τ),
        ]
        F = System(x - functions; variables = x, parameters = [t])
        result = solve(
            F, [at(π / 4)], ComplexF64[π / 4], ComplexF64[π / 8],
            Continuation(; show_progress = false), Serial(),
        )
        @test nsolutions(result) == 1
        @test only(solutions(result)) ≈ at(π / 8)
    end

    @testset "sqrt-parameter continuation reaches the analytic target roots" begin
        @var x y a
        F = System([sqrt(a) * x^2 + x - 1, x + y - 1]; variables = [x, y], parameters = [a])
        roots(s) = [(-1 + sign * sqrt(1 + 4s)) / (2s) for sign in (1, -1)]
        starts = [ComplexF64[r, 1 - r] for r in roots(2.0)]

        result = solve(
            F, starts, ComplexF64[4], ComplexF64[9],
            Continuation(; show_progress = false), Serial(),
        )
        @test nfailed(result) == 0
        @test nsolutions(result) == 2
        @test sort(real.(first.(solutions(result)))) ≈ sort(roots(3.0)) atol = 1.0e-8
    end

    @testset "rational monodromy is backend independent" begin
        @var x y u[1:4]
        for mode in (CompileMode.INTERPRETED, CompileMode.COMPILED_ALL)
            F = System(
                [u[1] / x^2 + u[2], u[3] / y^2 + u[4]];
                variables = [x, y], parameters = u, compile = mode,
            )
            result = solve(
                F,
                Monodromy(;
                    target_solutions_count = 4,
                    max_loops_no_progress = 100,
                    show_progress = false,
                ),
                Serial(),
            )
            @test nsolutions(result) == 4
        end
    end

    @testset "rational triangulation has six critical points" begin
        @var A1[1:3, 1:4] A2[1:3, 1:4]
        @var x[1:3] u1[1:2] u2[1:2]
        y1 = A1 * [x; 1]
        y2 = A2 * [x; 1]
        objective = sum((u1 - y1[1:2] ./ y1[3]) .^ 2) +
            sum((u2 - y2[1:2] ./ y2[3]) .^ 2)
        F = System(
            differentiate(objective, x);
            variables = x, parameters = [u1; u2; vec(A1); vec(A2)],
        )
        result = solve(
            F,
            Monodromy(;
                max_loops_no_progress = 100,
                target_solutions_count = 6,
                show_progress = false,
            ),
            Serial(),
        )
        @test nsolutions(result) == 6
    end

    @testset "completeness detects a missing expression-system solution" begin
        @var x y a b c
        F = System([x^2 + y^2 - 1, a * x + b * y + c]; variables = [x, y], parameters = [a, b, c])
        p = ComplexF64[1, 2, 3]
        roots = [
            ComplexF64[-0.6 - 0.8im, -1.2 + 0.4im],
            ComplexF64[-0.6 + 0.8im, -1.2 - 0.4im],
        ]
        algorithm = Monodromy(; show_progress = false)
        @test verify_solution_completeness(F, roots, p, algorithm) == Completeness.COMPLETE
        @test verify_solution_completeness(F, roots[1:1], p, algorithm) != Completeness.COMPLETE
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

    @testset "polynomial-only start methods reject non-polynomial equations" begin
        @var x y
        F = System([x / y - 2, x^2 + y^2 - 5]; variables = [x, y])
        @test_throws ArgumentError solve(F, TotalDegree(; show_progress = false), Serial())
        @test_throws ArgumentError solve(F, Polyhedral(; show_progress = false), Serial())

        L = LinearSubspace(ComplexF64[1 1], ComplexF64[1])
        @test_throws ArgumentError solve(F, L, TotalDegree(; show_progress = false), Serial())
        @test_throws ArgumentError solve(
            System([x / y - 1]; variables = [x, y]),
            Witness(; show_progress = false), Serial(),
        )
    end

    @testset "polyhedral solve agrees between expression and polynomial frontends" begin
        @var x y
        @polyvar u v
        expression_system = System([x^2 + y - 1, x + y^2 - 1]; variables = [x, y])
        polynomial_system = System([u^2 + v - 1, u + v^2 - 1])
        expr_result = solve(expression_system, Polyhedral(; show_progress = false), Serial())
        poly_result = solve(polynomial_system, Polyhedral(; show_progress = false), Serial())
        @test same_solution_set(solutions(expr_result), solutions(poly_result))

        sparse = System([x^3 * y^2 - 3, x^2 * y^3 - 5]; variables = [x, y])
        polyhedral = solve(sparse, Polyhedral(; show_progress = false), Serial())
        total_degree = solve(sparse, TotalDegree(; show_progress = false), Serial())
        @test nsolutions(polyhedral) == 5
        @test same_solution_set(solutions(polyhedral), solutions(total_degree))
    end

    @testset "overdetermined expression system is squared up correctly" begin
        @var x y
        F = System([x^2 + y^2 - 1, x - y, x^3 - y^3]; variables = [x, y])
        result = solve(F, Polyhedral(; show_progress = false), Serial())
        root = inv(sqrt(2))
        @test nsolutions(result) == 2
        @test same_solution_set(
            solutions(result),
            [ComplexF64[root, root], ComplexF64[-root, -root]],
        )
    end

    @testset "sliced solve substitutes parameters through the Expression frontend" begin
        @var x y a
        @polyvar u v b
        L = LinearSubspace(ComplexF64[1 1], ComplexF64[1])
        F = System([a * x^2 + y^2 - 1]; variables = [x, y], parameters = [a])
        G = System([b * u^2 + v^2 - 1]; variables = [u, v], parameters = [b])
        expr_result = solve(
            fix_parameters(F, [2.0]), L,
            TotalDegree(; show_progress = false), Serial(),
        )
        poly_result = solve(
            fix_parameters(G, [2.0]), L,
            TotalDegree(; show_progress = false), Serial(),
        )
        @test same_solution_set(solutions(expr_result), solutions(poly_result))

        H = fix_parameters(
            System([a / x + y - 2]; variables = [x, y], parameters = [a]),
            ComplexF64[3],
        )
        @test evaluate(H, ComplexF64[2, 0.5]) ≈ ComplexF64[0]
    end

    @testset "regeneration supports rational expressions and rejects nonalgebraic variables" begin
        @var x y z
        polynomial = System([x^2 + y^2 - z, x + y + z - 1]; variables = [x, y, z])
        regeneration = solve(polynomial, Regeneration(; show_progress = false))
        @test degree.(regeneration) == [2]
        @test ncomponents(solve(polynomial, Decomposition(; show_progress = false))) == 1
        witness = solve(polynomial, Witness(; show_progress = false))
        @test degree(witness) == 2

        rational = System(
            [x^2 + y^2 - z, x / (y - 1) + y + z - 1];
            variables = [x, y, z],
        )
        @test degree.(solve(rational, Regeneration(; show_progress = false))) == [4]

        for equation in (
                sqrt(x) + y - 1,
                1 / sqrt(x) + y - 1,
                x / (1 + sqrt(x)) + y - 1,
                1 / sin(x) + y - 1,
            )
            F = System([equation, x * y - z]; variables = [x, y, z])
            @test_throws ArgumentError solve(F, Regeneration(; show_progress = false))
            @test_throws ArgumentError solve(F, Decomposition(; show_progress = false))
        end

        W = first(regeneration)
        @test_throws ArgumentError intersect(W, 1 / sqrt(x), Intersection(; show_progress = false))
        @test_throws ArgumentError intersect(W, x / (1 + sqrt(y)), Intersection(; show_progress = false))
    end
end
