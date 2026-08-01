using Test
using HomotopyContinuationNext: SemialgebraicSetsHCSolver, System, TotalDegree,
    Polyhedral, Serial, CompileMode, nsolutions, nexcess_solutions, real_solutions,
    excess_residual_tol
using HomotopyContinuationNext: solve as hc_solve
using SemialgebraicSets: SemialgebraicSets, @set, algebraicset
using DynamicPolynomials: @polyvar

# An overdetermined set whose equations are consistent only up to `ε`.
_perturbed_set(x, y, ε::Float64, solver) = algebraicset(
    [
        -x - y + (1 + ε), -(1 + ε) * x * y + (1 + ε) * y^2 - y,
        -(1 + ε) * x^2 + y^2 - 2y + (1 + ε),
    ],
    solver,
)

@testset "SemialgebraicSets" begin
    @polyvar x y

    @testset "square set" begin
        solver = SemialgebraicSetsHCSolver()
        V = @set x^2 == 1 && y^2 == 2 solver
        S = sort!(map(s -> round.(s; digits = 2), collect(V)))
        @test S == [[-1.0, -1.41], [-1.0, 1.41], [1.0, -1.41], [1.0, 1.41]]
        @test eltype(S) == Vector{Float64}
        @test eltype(V) == Vector{Float64}
    end

    @testset "show" begin
        @test sprint(show, SemialgebraicSetsHCSolver()) ==
            "SemialgebraicSetsHCSolver(; algorithm = TotalDegree, " *
            "executor = Threaded, real_tol = 1.0e-6, compile = INTERPRETED)"
        @test sprint(
            show,
            SemialgebraicSetsHCSolver(;
                executor = Serial(), real_tol = 1.0e-4,
                compile = CompileMode.COMPILED,
            ),
        ) ==
            "SemialgebraicSetsHCSolver(; algorithm = TotalDegree, " *
            "executor = Serial, real_tol = 0.0001, compile = COMPILED)"
    end

    @testset "System from an algebraic set" begin
        V = @set x^2 == 1 && y^2 == 2
        F = System(V)
        @test size(F) == (2, 2)
        @test nsolutions(hc_solve(F, TotalDegree(; show_progress = false))) == 4

        # A set with no equalities is the whole space, not a system.
        @test_throws ArgumentError System(SemialgebraicSets.FullSpace())
    end

    @testset "solve and real_solutions on a set" begin
        V = @set x^2 == 1 && y^2 == 2
        result = hc_solve(V, TotalDegree(; show_progress = false))
        @test nsolutions(result) == 4

        S = sort!(map(s -> round.(s; digits = 2), real_solutions(V)))
        @test S == [[-1.0, -1.41], [-1.0, 1.41], [1.0, -1.41], [1.0, 1.41]]

        @test_throws ArgumentError real_solutions(@set x * y == 1)
    end

    @testset "options are forwarded" begin
        solver = SemialgebraicSetsHCSolver(;
            algorithm = Polyhedral(; show_progress = false),
            executor = Serial(),
            compile = CompileMode.COMPILED,
        )
        V = @set x^2 == 1 && y^2 == 2 solver
        @test length(collect(V)) == 4
    end

    @testset "only real solutions are returned" begin
        solver = SemialgebraicSetsHCSolver()
        V = @set x^2 == -1 && y == 1 solver
        @test isempty(collect(V))

        # real_tol admits a solution whose imaginary part is below it.
        loose = SemialgebraicSetsHCSolver(; real_tol = 2.0)
        @test length(collect(@set x^2 == -1 && y == 1 loose)) == 2
    end

    @testset "underdetermined sets are not solved" begin
        V = @set x * y == 1 SemialgebraicSetsHCSolver()
        @test SemialgebraicSets.solve(V, SemialgebraicSetsHCSolver()) === nothing
    end

    # The excess-solution filter rejects both roots of the perturbed set unless
    # the tolerance readmits them.
    @testset "excess_residual_tol" begin
        for (ε, atol) in ((1.0e-5, 1.0e-4), (1.0e-4, 1.0e-3), (1.0e-3, 1.0e-2), (1.0e-2, 1.0e-1))
            solver = SemialgebraicSetsHCSolver(;
                algorithm = TotalDegree(;
                    show_progress = false, excess_residual_tol = atol,
                ),
                real_tol = atol,
            )
            S = sort!(map(s -> round.(s; digits = 2), collect(_perturbed_set(x, y, ε, solver))))
            @test length(S) == 2
            @test isapprox(S[1], [0.0, 1.0]; atol = atol)
            @test isapprox(S[2], [1.0, 0.0]; atol = atol)
        end

        # Without the tolerance the same set has no solution that survives.
        strict = _perturbed_set(
            x, y, 1.0e-2, SemialgebraicSetsHCSolver(; real_tol = 1.0e-1),
        )
        @test isempty(collect(strict))

        # The knob lives on the algorithm, and the solver carries it unchanged.
        alg = TotalDegree(; show_progress = false, excess_residual_tol = 1.0e-2)
        @test excess_residual_tol(alg) == 1.0e-2
        @test excess_residual_tol(
            SemialgebraicSetsHCSolver(; algorithm = alg).algorithm,
        ) == 1.0e-2
    end

    # The knob lives on the algorithm, so it works without SemialgebraicSets too.
    @testset "excess_residual_tol on the algorithm" begin
        F = System(
            [
                -x - y + 1.01, -1.01 * x * y + 1.01 * y^2 - y,
                -1.01 * x^2 + y^2 - 2y + 1.01,
            ],
        )
        # Fixed seed: `F` is inconsistent, so how far the squared-up roots sit from
        # `V(F)` depends on the randomization. Seed 3 keeps the worst one at `1.3e-2`,
        # under the tolerance; other seeds reach `1.9e-1`, past it.
        for alg in (TotalDegree, Polyhedral)
            strict = hc_solve(
                F, alg(; show_progress = false, seed = UInt32(3)), Serial(),
            )
            @test nsolutions(strict) == 0
            @test nexcess_solutions(strict) > 0

            loose = hc_solve(
                F,
                alg(;
                    show_progress = false, seed = UInt32(3),
                    excess_residual_tol = 1.0e-1,
                ),
                Serial(),
            )
            @test nsolutions(loose) == 2
            @test nexcess_solutions(loose) < nexcess_solutions(strict)
        end
    end
end
