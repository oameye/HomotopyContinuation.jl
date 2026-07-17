using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: solve, System, TotalDegree,
    solution, accuracy, residual, steps, accepted_steps, rejected_steps,
    winding_number, condition_jacobian, last_path_point, path_results,
    is_success, is_failed, is_finite, is_at_infinity, is_real,
    path_number, start_solution, valuation, multiplicity,
    seed, ntracked, failed, at_infinity, nonsingular, singular, nfailed,
    statistics, ResultStatistics
using DynamicPolynomials: @polyvar
using LinearAlgebra: cond

@testset "Path diagnostics accessors" begin
    @polyvar x y
    F = System([x^2 + y^2 - 1, x + y - 1])   # two finite solutions
    res = solve(F, TotalDegree(; seed = UInt32(0x1234)))
    prs = path_results(res)
    @test !isempty(prs)
    r = first(prs)

    @testset "value accessors mirror the stored fields" begin
        @test solution(r) === r.solution
        @test accuracy(r) == r.accuracy
        @test winding_number(r) == r.winding_number
        @test condition_jacobian(r) == r.condition_jacobian
        @test accepted_steps(r) == r.accepted_steps
        @test rejected_steps(r) == r.rejected_steps
    end

    @testset "steps is the total of accepted + rejected (v2 convention)" begin
        @test steps(r) == r.accepted_steps + r.rejected_steps
        @test steps(r) >= accepted_steps(r)
    end

    @testset "residual is finite and small for a genuine success" begin
        sr = first(filter(is_success, prs))
        @test residual(sr) >= 0
        @test isfinite(residual(sr))
        @test residual(sr) < 1.0e-6
    end

    @testset "last_path_point returns (point, t)" begin
        pt, t = last_path_point(r)
        @test pt == r.last_path_point
        @test t == r.last_path_t
    end

    @testset "show renders per-path diagnostics" begin
        str = sprint(show, MIME"text/plain"(), r)
        @test occursin("PathResult", str)
        @test occursin("steps", str)
    end

    @testset "v2-parity predicates partition the paths" begin
        # Every path is exactly one of: success, at-infinity, excess, or failed.
        for pr in prs
            n = is_success(pr) + is_at_infinity(pr) + HC.is_excess_solution(pr) + is_failed(pr)
            @test n == 1
        end
        # is_finite === is_success and Base.isfinite agrees.
        for pr in prs
            @test is_finite(pr) == is_success(pr)
            @test isfinite(pr) == is_finite(pr)
        end
    end

    @testset "cond is an alias for condition_jacobian (v2 parity)" begin
        @test cond(r) == condition_jacobian(r)
    end

    @testset "path_number / start_solution are recorded" begin
        # path_number is 1-based and unique across the successful paths.
        nums = sort(path_number.(prs))
        @test nums == collect(1:length(prs))
        # start_solution is the (nonempty) start point for each path.
        @test all(!isempty ∘ start_solution, prs)
        @test length(start_solution(r)) == length(solution(r))
    end

    @testset "valuation is empty or per-coordinate" begin
        for pr in prs
            v = valuation(pr)
            @test isempty(v) || length(v) == length(solution(pr))
        end
    end

    @testset "multiplicity is the cluster size (1 for these simple roots)" begin
        for pr in filter(is_success, prs)
            @test multiplicity(pr) == HC.multiplicity(res, path_number(pr))
            @test multiplicity(pr) >= 1
        end
    end

    @testset "is_real positional and Base.isreal overloads (v2 parity)" begin
        sr = first(filter(is_success, prs))
        @test is_real(sr, 1.0e-6) == is_real(sr; tol = 1.0e-6)
        @test isreal(sr) == is_real(sr)
        @test isreal(sr, 1.0e-6) == is_real(sr; tol = 1.0e-6)
    end

    @testset "Result-level v2-parity accessors" begin
        @test seed(res) == UInt32(0x1234)
        @test ntracked(res) == length(prs)
        @test failed(res) == filter(is_failed, prs)
        @test at_infinity(res) == filter(is_at_infinity, prs)
        @test nfailed(res) == count(is_failed, prs)
        @test nonsingular(res) == HC.results(res; only_nonsingular = true)
        @test singular(res) == HC.results(res; only_singular = true)
    end

    @testset "statistics summarizes the result (v2 ResultStatistics)" begin
        st = statistics(res)
        @test st isa ResultStatistics
        @test st.nonsingular == HC.nnonsingular(res)
        @test st.singular == HC.nsingular(res)
        @test st.total == st.nonsingular + st.singular
        @test st.real == st.real_nonsingular + st.real_singular
        @test st.at_infinity == HC.nat_infinity(res)
        @test st.failed == nfailed(res)
    end
end
