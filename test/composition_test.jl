using Test
using Random: MersenneTwister, randn
using LinearAlgebra: norm
using HomotopyContinuation

composition_relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), 1.0)

function same_solution_set(a, b; atol = 1.0e-8)
    length(a) == length(b) || return false
    return all(sa -> any(sb -> maximum(abs.(sa .- sb)) < atol, b), a)
end

@testset "CompositionSystem public mathematics" begin
    @testset "shape, variables and parameters compose predictably" begin
        @var x y a b
        inner = System([y^2 + 2x + 3, x - 1]; variables = [x, y])
        outer = System([x + y * a, x - b]; variables = [x, y], parameters = [a, b])
        C = outer ∘ inner

        @test C isa CompositionSystem
        @test size(C) == (2, 2)
        @test nvariables(C) == 2
        @test variables(C) == [x, y]
        @test parameters(C) == [a, b]
        @test nparameters(C) == 2
        @test parameters(inner ∘ outer) == [a, b]
        @test parameters(inner ∘ inner) == Expression[]
        @test size((outer ∘ inner) ∘ inner) == (2, 2)
        @test size(outer ∘ (inner ∘ inner)) == (2, 2)
        @test occursin("CompositionSystem", sprint(show, C))
    end

    @testset "incompatible stages are rejected" begin
        @var x y z a c
        f = System([y^2 + 2x + 3, x - 1]; variables = [x, y])
        wide = System([x + y + z]; variables = [x, y, z])
        ga = System([x + y * a, x - a]; variables = [x, y], parameters = [a])
        gc = System([x + y * c, x - c]; variables = [x, y], parameters = [c])
        @test_throws ArgumentError wide ∘ f
        @test_throws ArgumentError ga ∘ gc
    end

    @testset "composition equals symbolic substitution" begin
        rng = MersenneTwister(0x5cbb)
        @var x y a b
        inner_exprs = [y^2 + 2x + 3, x - 1]
        outer_exprs = [(x^2 + y * a)^2 - 3y, x - b^2]
        inner = System(inner_exprs; variables = [x, y])
        outer = System(outer_exprs; variables = [x, y], parameters = [a, b])
        C = outer ∘ inner
        reference = System(
            subs(outer_exprs, [x, y] => inner_exprs);
            variables = [x, y], parameters = [a, b],
        )
        scale = equation_scales(reference) ./ equation_scales(outer)

        for _ in 1:5
            point = randn(rng, ComplexF64, 2)
            p = randn(rng, ComplexF64, 2)
            @test composition_relerr(evaluate(C, point, p) ./ scale, evaluate(reference, point, p)) < 1.0e-12
            @test composition_relerr(jacobian(C, point, p) ./ scale, jacobian(reference, point, p)) < 1.0e-12
        end
    end

    @testset "parameter-free outer stage equals direct substitution" begin
        rng = MersenneTwister(0x5cc0)
        @var x y a b
        outer_exprs = [y^2 + 2x + 3, x - 1]
        inner_exprs = [x + y * a, x - b^2]
        outer = System(outer_exprs; variables = [x, y])
        inner = System(inner_exprs; variables = [x, y], parameters = [a, b])
        C = outer ∘ inner
        reference = System(
            subs(outer_exprs, [x, y] => inner_exprs);
            variables = [x, y], parameters = [a, b],
        )
        scale = equation_scales(reference) ./ equation_scales(outer)
        point = randn(rng, ComplexF64, 2)
        p = randn(rng, ComplexF64, 2)
        @test composition_relerr(evaluate(C, point, p) ./ scale, evaluate(reference, point, p)) < 1.0e-12
        @test composition_relerr(jacobian(C, point, p) ./ scale, jacobian(reference, point, p)) < 1.0e-12
    end

    @testset "materializing a composition preserves its zero set" begin
        @var x y
        inner = System([y^2 + 2x + 3, x - 1]; variables = [x, y])
        outer = System([(x^2 + y)^2 - 3y, x - 4]; variables = [x, y])
        C = outer ∘ inner
        S = System(C)

        composed = solve(C, TotalDegree(; show_progress = false), Serial())
        materialized = solve(S, TotalDegree(; show_progress = false), Serial())
        @test nfailed(composed) == 0
        @test nfailed(materialized) == 0
        @test same_solution_set(solutions(composed), solutions(materialized))
        @test all(s -> maximum(abs.(evaluate(C, s))) < 1.0e-8, solutions(composed))
        @test all(s -> maximum(abs.(evaluate(S, s))) < 1.0e-8, solutions(materialized))
    end

    @testset "outer equation scaling preserves the composed zero set" begin
        @var x y
        inner = System([x + y, x - y]; variables = [x, y])
        outer = System([x + y - 1, x * y - 2]; variables = [x, y])
        scaled_outer = System([1.0e9 * (x + y - 1), 1.0e-9 * (x * y - 2)]; variables = [x, y])
        reference = outer ∘ inner
        scaled = scaled_outer ∘ inner

        reference_result = solve(reference, TotalDegree(; show_progress = false), Serial())
        scaled_result = solve(scaled, TotalDegree(; show_progress = false), Serial())
        @test nfailed(reference_result) == 0
        @test nfailed(scaled_result) == 0
        @test nsolutions(reference_result) == 2
        @test nsolutions(scaled_result) == 2
        @test same_solution_set(solutions(scaled_result), solutions(reference_result); atol = 1.0e-8)
        @test all(s -> maximum(abs.(evaluate(reference, s))) < 1.0e-8, solutions(reference_result))
        @test all(s -> maximum(abs.(evaluate(scaled, s))) < 1.0e-8, solutions(scaled_result))
    end

    @testset "TotalDegree and Polyhedral solve a composed polynomial system" begin
        @var a b c x y u v
        outer = System([u + 1, v - 2]; variables = [u, v])
        middle = System([a * b - 2, a * c - 1]; variables = [a, b, c])
        inner = System([x + y, y + 3, x + 2]; variables = [x, y])
        C = outer ∘ middle ∘ inner

        total_degree = solve(C, TotalDegree(; show_progress = false), Serial())
        polyhedral = solve(C, Polyhedral(; show_progress = false), Serial())
        @test nfailed(total_degree) == 0
        @test nfailed(polyhedral) == 0
        @test nsolutions(total_degree) == 2
        @test nsolutions(polyhedral) == 2
        @test same_solution_set(solutions(total_degree), solutions(polyhedral))
        @test all(s -> maximum(abs.(evaluate(C, s))) < 1.0e-9, solutions(total_degree))

        @var s t
        nonpolynomial = System([s^2, 1 / t]; variables = [s, t]) ∘
            System([s + t, s - t]; variables = [s, t])
        @test_throws ArgumentError solve(
            System([s * t, s^2]; variables = [s, t]) ∘ nonpolynomial,
            TotalDegree(; show_progress = false), Serial(),
        )
    end

    @testset "Newton and parameter continuation commute with an affine inner map" begin
        rng = MersenneTwister(0x5cbd)
        @var x y q1 q2
        family = System(
            [x^2 + y^2 - q1, x + y - q2];
            variables = [x, y], parameters = [q1, q2],
        )
        A = randn(rng, ComplexF64, 2, 2)
        c = randn(rng, ComplexF64, 2)
        C = family ∘ System(A * [x, y] + c; variables = [x, y])

        p0 = ComplexF64[3, 1]
        target_system = System([x^2 + y^2 - 3, x + y - 1]; variables = [x, y])
        target_roots = solutions(solve(target_system, TotalDegree(; show_progress = false), Serial()))
        starts = [A \ (s - c) for s in target_roots]

        corrected = newton(C, starts[1] .+ 1.0e-6; p = p0)
        @test is_success(corrected)
        @test norm(A * solution(corrected) + c - target_roots[1], Inf) < 1.0e-8

        q = ComplexF64[5, 2]
        result = solve(C, starts, p0, q, Continuation(; show_progress = false), Serial())
        @test nfailed(result) == 0
        @test nsolutions(result) == 2
        for s in solutions(result)
            z = A * s + c
            @test abs(z[1]^2 + z[2]^2 - q[1]) < 1.0e-8
            @test abs(z[1] + z[2] - q[2]) < 1.0e-8
        end
    end

    @testset "monodromy through a composition recovers both roots" begin
        rng = MersenneTwister(0x5cbe)
        @var x y q1 q2
        family = System(
            [x^2 + y^2 - q1, x + y - q2];
            variables = [x, y], parameters = [q1, q2],
        )
        A = randn(rng, ComplexF64, 2, 2)
        c = randn(rng, ComplexF64, 2)
        C = family ∘ System(A * [x, y] + c; variables = [x, y])
        p0 = ComplexF64[3, 1]
        reference = System([x^2 + y^2 - 3, x + y - 1]; variables = [x, y])
        truth = solutions(solve(reference, TotalDegree(; show_progress = false), Serial()))
        start = A \ (truth[1] - c)

        result = solve(
            C, [start], p0,
            Monodromy(; target_solutions_count = 2, show_progress = false, seed = UInt32(0x5cbe)),
            Serial(),
        )
        @test nsolutions(result) == 2
        @test same_solution_set([A * s + c for s in solutions(result)], truth)
    end

    @testset "find_start_pair produces an actual zero" begin
        rng = MersenneTwister(0x5cc2)
        @var x y q1 q2
        family = System(
            [x^2 + y^2 - q1, x + y - q2];
            variables = [x, y], parameters = [q1, q2],
        )
        A = randn(rng, ComplexF64, 2, 2)
        c = randn(rng, ComplexF64, 2)
        C = family ∘ System(A * [x, y] + c; variables = [x, y])

        pair = find_start_pair(C)
        @test pair.found
        @test length(pair.x) == 2
        @test length(pair.p) == 2
        @test maximum(abs.(evaluate(C, pair.x, pair.p))) < 1.0e-9

        result = solve(
            C,
            Monodromy(; target_solutions_count = 2, show_progress = false, seed = UInt32(0x5cc2)),
            Serial(),
        )
        @test nsolutions(result) == 2

        parameter_free = System([x^2 + y^2 - 3, x + y - 1]; variables = [x, y]) ∘
            System(A * [x, y] + c; variables = [x, y])
        free_pair = find_start_pair(parameter_free)
        @test free_pair.found
        @test isempty(free_pair.p)
        @test maximum(abs.(evaluate(parameter_free, free_pair.x))) < 1.0e-9
    end
end
