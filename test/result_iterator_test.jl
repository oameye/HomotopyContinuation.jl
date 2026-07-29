using Test
using HomotopyContinuationNext
using HomotopyContinuationNext: Serial, Result, PathResult, TotalDegree, Polyhedral,
    solution, is_success, is_real, seed, path_results, nsolutions, solutions,
    nexcess_solutions, path_number, start_solution
using CommonSolve: CommonSolve
using DynamicPolynomials: @polyvar
using LinearAlgebra: norm

@testset "ResultIterator" begin

    @polyvar x y

    @testset "total degree" begin
        F = System([x^2 + y - 1, x * y - 2]; variables = [x, y])
        alg = TotalDegree(; seed = UInt32(5))
        ri = result_iterator(F, alg)
        @test eltype(ri) === PathResult
        @test length(ri) == 4                     # Bezout bound
        @test Base.IteratorSize(typeof(ri)) === Base.HasLength()
        prs = collect(ri)
        @test prs isa Vector{PathResult}
        @test length(prs) == 4
        # tracking is lazy: one path at a time
        @test first(ri) isa PathResult
        eager = solve(F, alg, Serial(); show_progress = false)
        @test seed(ri) == seed(eager) == UInt32(5)
        @test nsolutions(Result(ri)) == nsolutions(eager)
    end

    @testset "polyhedral" begin
        F = System([x^2 + y - 1, x * y - 2]; variables = [x, y])
        ri = result_iterator(F, Polyhedral())
        prs = collect(ri)
        @test !isempty(prs)
        @test all(pr -> pr isa PathResult, prs)
        @test nsolutions(Result(ri)) ==
            nsolutions(solve(F, Polyhedral(), Serial(); show_progress = false))
    end

    @testset "sliced" begin
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L = rand_subspace(2; codim = 1)
        ri = result_iterator(F, L)
        w = collect(ri)
        @test length(w) == 2
        @test w isa Vector{PathResult}
        E = extrinsic(L)
        @test maximum(pr -> maximum(abs, E.A * solution(pr) - E.b), w) < 1.0e-10
    end

    @testset "subspace to subspace, and iterator as start solutions" begin
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        l₁ = rand_subspace(2; codim = 1)
        l₂ = rand_subspace(2; codim = 1)
        r₁ = result_iterator(F, l₁)
        r₂ = result_iterator(F, r₁, l₁, l₂)
        w₂ = collect(r₂)
        @test length(w₂) == 2
        E = extrinsic(l₂)
        @test maximum(pr -> maximum(abs, E.A * solution(pr) - E.b), w₂) < 1.0e-10
        # also usable as start solutions for an eager solve and a sweep
        @test nsolutions(solve(F, r₁, l₁, l₂; show_progress = false)) == 2
        sweep = solve_targets(F, r₁, l₁, [l₂], Serial(); show_progress = false)
        @test nsolutions(first(only(sweep))) == 2
    end

    @testset "parameter homotopy" begin
        @polyvar p
        F = System([x^2 + y^2 - p, x - y]; variables = [x, y], parameters = [p])
        G = System([x^2 + y^2 - 2.0, x - y]; variables = [x, y])
        S = solutions(solve(G, Serial(); show_progress = false))
        ri = result_iterator(F, S, [2.0], [8.0])
        @test length(collect(ri)) == length(S)
        for s in solutions(Result(ri))
            @test abs(s[1]^2 + s[2]^2 - 8) < 1.0e-10
        end
    end

    @testset "bitmask and bitmask_filter" begin
        F = System([x^2 - 1, y^2 - 4]; variables = [x, y])
        ri = result_iterator(F)
        @test length(ri) == 4
        @test bitmask(ri) == trues(4)
        bm = bitmask(is_real, ri)
        @test bm isa BitVector
        @test count(bm) == 4                      # all four solutions are real
        filtered = bitmask_filter(pr -> real(solution(pr)[1]) > 0, ri)
        @test length(filtered) == 2
        @test all(pr -> real(solution(pr)[1]) > 0, collect(filtered))
        @test count(bitmask(filtered)) == 2
        # `bitmask` hands out a copy, and `length(ri)` counts the mask.
        mutated = bitmask(ri)
        push!(mutated, true)
        @test length(bitmask(ri)) == 4
        @test length(ri) == 4
    end

    @testset "explicit mask selects a subset of the paths up front" begin
        F = System([x^2 - 1, y^2 - 4]; variables = [x, y])
        cache = CommonSolve.init(F, TotalDegree(; seed = UInt32(3)), Serial(); show_progress = false)
        mask = BitVector([true, false, false, true])
        ri = ResultIterator(cache, mask)
        @test length(ri) == 2
        @test bitmask(ri) == mask
        prs = collect(ri)
        @test length(prs) == 2
        # Only the selected start solutions were tracked.
        @test [path_number(pr) for pr in prs] == [1, 4]
        @test [start_solution(pr) for pr in prs] ==
            [cache.start_solutions[1], cache.start_solutions[4]]

        @test_throws ArgumentError ResultIterator(cache, BitVector([true, false]))
    end

    @testset "excess solutions are reclassified by Result" begin
        F = System([x^2 + y^2 - 5, x * y - 1, x - y + 1]; variables = [x, y])
        ri = result_iterator(F)
        res = Result(ri)
        @test nexcess_solutions(res) ==
            nexcess_solutions(solve(F, Serial(); show_progress = false))
    end

    @testset "show" begin
        F = System([x^2 + y - 1, x * y - 2]; variables = [x, y])
        io = IOBuffer()
        show(io, result_iterator(F))
        @test occursin("ResultIterator over 4 of 4", String(take!(io)))
    end
end
