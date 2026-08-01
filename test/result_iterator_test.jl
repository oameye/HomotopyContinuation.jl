using Test
using HomotopyContinuationNext
using HomotopyContinuationNext: Serial, Result, PathResult, TotalDegree, Polyhedral,
    solution, is_success, is_real, seed, path_results, nsolutions, solutions,
    nexcess_solutions, path_number, start_solution, total_degree_start_solutions,
    System, Continuation
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
        eager = solve(F, alg, Serial())
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
            nsolutions(solve(F, Polyhedral(; show_progress = false), Serial()))
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
        @test nsolutions(solve(F, r₁, l₁, l₂, Continuation(; show_progress = false))) == 2
        sweep = solve(F, r₁, l₁, [l₂], Sweep(; show_progress = false), Serial())
        @test nsolutions(first(only(sweep))) == 2
    end

    @testset "parameter homotopy" begin
        @polyvar p
        F = System([x^2 + y^2 - p, x - y]; variables = [x, y], parameters = [p])
        G = System([x^2 + y^2 - 2.0, x - y]; variables = [x, y])
        S = solutions(solve(G, TotalDegree(; show_progress = false), Serial()))
        ri = result_iterator(F, S, [2.0], [8.0])
        @test length(collect(ri)) == length(S)
        for s in solutions(Result(ri))
            @test abs(s[1]^2 + s[2]^2 - 8) < 1.0e-10
        end
    end

    @testset "selection and restrict" begin
        F = System([x^2 - 1, y^2 - 4]; variables = [x, y])
        ri = result_iterator(F)
        @test length(ri) == 4
        @test selection(ri) == trues(4)
        bm = selection(is_real, ri)
        @test bm isa BitVector
        @test count(bm) == 4                      # all four solutions are real
        filtered = restrict(ri, selection(pr -> real(solution(pr)[1]) > 0, ri))
        @test length(filtered) == 2
        @test all(pr -> real(solution(pr)[1]) > 0, collect(filtered))
        @test count(selection(filtered)) == 2
        # `selection` hands out a copy, and `length(ri)` counts the mask.
        mutated = selection(ri)
        push!(mutated, true)
        @test length(selection(ri)) == 4
        @test length(ri) == 4
    end

    @testset "filter keeps the results, getindex tracks one path" begin
        F = System([x^2 - 1, y^2 - 4]; variables = [x, y])
        ri = result_iterator(F)
        real_paths = filter(is_real, ri)
        @test real_paths isa Vector{PathResult}
        @test length(real_paths) == 4
        @test isempty(filter(pr -> !is_success(pr), ri))
        @test [solution(pr) for pr in Iterators.filter(is_real, ri)] ==
            [solution(pr) for pr in real_paths]

        @test start_solution(ri[2]) == start_solutions(ri)[2]
        @test path_number(ri[end]) == 4
        @test_throws BoundsError ri[5]
        # Indices count the selected paths, not the start solutions.
        kept = restrict(ri, BitVector([false, false, true, true]))
        @test path_number(kept[1]) == 3
        @test_throws BoundsError kept[3]
    end

    @testset "explicit mask selects a subset of the paths up front" begin
        F = System([x^2 - 1, y^2 - 4]; variables = [x, y])
        cache = CommonSolve.init(
            F, TotalDegree(; seed = UInt32(3), show_progress = false), Serial(),
        )
        mask = BitVector([true, false, false, true])
        ri = ResultIterator(cache, mask)
        @test length(ri) == 2
        @test selection(ri) == mask
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
            nexcess_solutions(solve(F, TotalDegree(; show_progress = false), Serial()))
    end

    @testset "show" begin
        F = System([x^2 + y - 1, x * y - 2]; variables = [x, y])
        io = IOBuffer()
        show(io, result_iterator(F))
        @test occursin("ResultIterator over 4 of 4", String(take!(io)))
    end

    @testset "restrict with a precomputed mask" begin
        F = System([x^2 + y - 1, x * y - 2]; variables = [x, y])
        ri = result_iterator(F)
        kept = restrict(ri, BitVector([true, false, true, false]))
        @test length(kept) == 2
        @test selection(kept) == BitVector([true, false, true, false])
        @test [start_solution(pr) for pr in kept] ==
            start_solutions(ri)[[1, 3]]

        # Masks compose, in the one index domain of the start solutions, and
        # never widen a restriction.
        again = restrict(kept, BitVector([true, true, true, false]))
        @test selection(again) == BitVector([true, false, true, false])
        @test selection(restrict(kept, BitVector([false, false, true, false]))) ==
            BitVector([false, false, true, false])
        @test_throws ArgumentError restrict(ri, BitVector([true]))
    end

    @testset "start to target route" begin
        G = System([x^2 - 1, y^2 - 1]; variables = [x, y])
        F = System([x^2 + 2 * y^2 - 3, x * y - 1]; variables = [x, y])
        ri = result_iterator(G, F, total_degree_start_solutions([2, 2]))
        @test length(ri) == 4
        @test all(is_success, collect(ri))
        @test nsolutions(Result(ri)) == 4
    end

    # Re-encode a solved system as its start system plus the mask of the start solutions
    # that reach one, so replaying tracks only those paths. Both tracks share one
    # homotopy, so the path correspondence holds.
    @testset "compression" begin
        include("test_systems.jl")
        polys, vars, _ = cyclic_system(5)
        F = System(polys; variables = vars)
        d = HomotopyContinuationNext.degrees(F)
        S = total_degree_start_solutions(d)
        @test length(S) == 120

        G = System([vi^di - 1 for (vi, di) in zip(vars, d)]; variables = vars)
        alg = Continuation(; seed = UInt32(1), show_progress = false)
        full = result_iterator(G, F, S, alg)
        @test length(full) == 120

        B = selection(is_success, full)
        @test count(B) < 120
        compressed = restrict(full, B)
        @test compressed isa ResultIterator
        @test length(compressed) == count(B)

        replayed = collect(compressed)
        @test all(is_success, replayed)
        @test nsolutions(Result(compressed)) == 70
        # Replaying the same seed retracks the same paths to the same endpoints.
        @test [start_solution(pr) for pr in replayed] == S[findall(B)]
        @test sort(round.(abs.(reduce(vcat, solutions(Result(compressed)))); digits = 8)) ==
            sort(round.(abs.(reduce(vcat, solutions(Result(full)))); digits = 8))
    end
end
