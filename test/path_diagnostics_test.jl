using Test
using LinearAlgebra: cond
using HomotopyContinuation

@testset "path diagnostics on a two-solution problem" begin
    @polyvar x y
    F = System([x^2 + y^2 - 1, x + y - 1])
    result = solve(F, TotalDegree(; seed = UInt32(0x1234), show_progress = false), Serial())
    paths = path_results(result)

    @test nfailed(result) == 0
    @test ntracked(result) == length(paths)
    @test nsolutions(result) == 2
    @test seed(result) == UInt32(0x1234)

    @testset "successful endpoints satisfy the mathematical system" begin
        for path in paths
            @test is_success(path)
            @test is_finite(path)
            @test !is_failed(path)
            @test !is_at_infinity(path)
            @test !is_excess_solution(path)
            @test residual(path) < 1.0e-6
            @test maximum(abs.(evaluate(F, solution(path)))) < 1.0e-6
            @test multiplicity(path) == 1
        end
    end

    @testset "diagnostics are internally consistent" begin
        for path in paths
            @test steps(path) == accepted_steps(path) + rejected_steps(path)
            @test steps(path) >= accepted_steps(path)
            @test accuracy(path) >= 0
            @test isfinite(accuracy(path))
            @test isfinite(condition_jacobian(path))
            @test cond(path) == condition_jacobian(path)

            point, t = last_path_point(path)
            @test length(point) == length(solution(path))
            @test isfinite(real(t))
            @test isfinite(imag(t))

            @test 1 <= path_number(path) <= ntracked(result)
            @test length(start_solution(path)) == length(solution(path))
            v = valuation(path)
            @test isempty(v) || length(v) == length(solution(path))
        end
        @test sort(path_number.(paths)) == collect(1:length(paths))
    end

    @testset "realness and result partitions agree" begin
        for path in paths
            @test isreal(path) == is_real(path)
            @test is_real(path, 1.0e-6) == is_real(path; tol = 1.0e-6)
        end
        @test isempty(failed(result))
        @test isempty(at_infinity(result))
        @test length(nonsingular(result)) == 2
        @test isempty(singular(result))
    end

    @testset "aggregate statistics agree with public result queries" begin
        stats = statistics(result)
        @test stats isa ResultStatistics
        @test stats.total == nsolutions(result)
        @test stats.nonsingular == nnonsingular(result)
        @test stats.singular == nsingular(result)
        @test stats.real == nreal(result)
        @test stats.at_infinity == nat_infinity(result)
        @test stats.failed == nfailed(result)
    end

    @testset "text display exposes useful diagnostics" begin
        rendered = sprint(show, MIME"text/plain"(), first(paths))
        @test occursin("PathResult", rendered)
        @test occursin("steps", rendered)
        @test occursin("residual", rendered)
    end
end
