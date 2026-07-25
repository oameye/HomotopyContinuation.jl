using Test
using HomotopyContinuationNext
using HomotopyContinuationNext: Serial, Threaded, Result, PathResult, TotalDegree,
    solution, is_success, path_results, steps, nparameters, is_homogeneous,
    AmbientWorkerState, IntrinsicWorkerState, IntrinsicSubspaceHomotopy,
    ExtrinsicSubspaceHomotopy, AffineChartHomotopy, WorkerSolveCache, _to_ambient
using DynamicPolynomials: @polyvar
using CommonSolve: CommonSolve
using LinearAlgebra: norm

subspace_residual(L, x) = (E = extrinsic(L); isempty(E.b) ? 0.0 : maximum(abs, E.A * x - E.b))

@testset "Subspace to subspace solve" begin

    @testset "both regimes reach the target subspace" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L₁ = rand_subspace(2; codim = 1)
        L₂ = rand_subspace(2; codim = 1)
        S₁ = solutions(solve(F, L₁; show_progress = false))
        @test length(S₁) == 2
        for intrinsic in (nothing, true, false)
            res = solve(F, S₁, L₁, L₂; intrinsic = intrinsic, show_progress = false)
            @test nsolutions(res) == 2
            for s in solutions(res)
                @test subspace_residual(L₂, s) < 1.0e-10
                @test abs(s[1]^2 + s[2]^2 - 5) < 1.0e-10
            end
        end
    end

    @testset "regime selection and homotopy types" begin
        @polyvar x y z
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        # dim 1 == codim 1 → intrinsic by default
        L = rand_subspace(2; codim = 1)
        cache = CommonSolve.init(F, [[1.0 + 0im, 2.0 + 0im]], L, L, Serial())
        @test cache.worker isa IntrinsicWorkerState
        @test cache.worker.homotopy isa IntrinsicSubspaceHomotopy

        # dim 2 > codim 1 → extrinsic by default
        G = System([x^2 + y^2 - 5, x * y + 1]; variables = [x, y, z])
        K = rand_subspace(3; dim = 2)
        starts = [[1.0 + 0im, 2.0 + 0im, 3.0 + 0im]]
        cache2 = CommonSolve.init(G, starts, K, K, Serial())
        @test cache2.worker isa AmbientWorkerState{ExtrinsicSubspaceHomotopy}

        forced = CommonSolve.init(G, starts, K, K, Serial(); intrinsic = true)
        @test forced.worker isa IntrinsicWorkerState
    end

    @testset "two equations in three variables" begin
        @polyvar x y z
        F = System([x^2 + y^2 - 5, x * y + 1]; variables = [x, y, z])
        K₁ = rand_subspace(3; dim = 2)
        K₂ = rand_subspace(3; dim = 2)
        S = solutions(solve(F, K₁; show_progress = false))
        @test length(S) == 4
        for intrinsic in (nothing, true, false)
            res = solve(F, S, K₁, K₂; intrinsic = intrinsic, show_progress = false)
            @test nsolutions(res) == 4
            @test maximum(s -> subspace_residual(K₂, s), solutions(res)) < 1.0e-10
        end
    end

    @testset "projective" begin
        @polyvar x y z
        F = System([x^2 + y^2 - z^2]; variables = [x, y, z])
        @test is_homogeneous(F)
        L₁ = rand_subspace(3; codim = 1, affine = false)
        L₂ = rand_subspace(3; codim = 1, affine = false)
        S = solutions(solve(F, L₁; show_progress = false))
        for intrinsic in (nothing, true, false)
            res = solve(F, S, L₁, L₂; intrinsic = intrinsic, show_progress = false)
            @test nsolutions(res) == 2
            for s in solutions(res)
                scale = norm(s, Inf)
                @test abs(s[1]^2 + s[2]^2 - s[3]^2) < 1.0e-10 * scale^2
                @test norm(extrinsic(L₂).A * s, Inf) < 1.0e-10 * scale
            end
        end
        cache = CommonSolve.init(F, S, L₁, L₂, Serial(); intrinsic = false)
        @test cache.worker isa
            AmbientWorkerState{AffineChartHomotopy{ExtrinsicSubspaceHomotopy}}
    end

    @testset "parametric" begin
        @polyvar x y a
        F = System([x^2 + y^2 - a]; variables = [x, y], parameters = [a])
        L₁ = rand_subspace(2; codim = 1)
        L₂ = rand_subspace(2; codim = 1)
        S = solutions(solve(F, L₁; target_parameters = [5.0], show_progress = false))
        res = solve(F, S, L₁, L₂; target_parameters = [5.0], show_progress = false)
        @test nsolutions(res) == 2
        @test maximum(s -> abs(s[1]^2 + s[2]^2 - 5), solutions(res)) < 1.0e-10
        @test_throws ArgumentError solve(F, S, L₁, L₂; show_progress = false)
    end

    @testset "executors agree" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L₁ = rand_subspace(2; codim = 1)
        L₂ = rand_subspace(2; codim = 1)
        S = solutions(solve(F, L₁; show_progress = false))
        for intrinsic in (true, false)
            serial = solve(
                F, S, L₁, L₂, Serial();
                intrinsic = intrinsic, seed = UInt32(7), show_progress = false,
            )
            threaded = solve(
                F, S, L₁, L₂, Threaded();
                intrinsic = intrinsic, seed = UInt32(7), show_progress = false,
            )
            @test nsolutions(serial) == nsolutions(threaded) == 2
            sa = sort(solutions(serial); by = real ∘ first)
            sb = sort(solutions(threaded); by = real ∘ first)
            @test maximum(norm.(sa .- sb, Inf)) < 1.0e-8
        end
    end

    @testset "mismatched subspaces are rejected" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L = rand_subspace(2; codim = 1)
        starts = [[1.0 + 0im, 2.0 + 0im]]
        # different dimension: the geodesic connects one Grassmannian to itself
        full = HomotopyContinuationNext._full_subspace(2)
        for intrinsic in (true, false)
            @test_throws ArgumentError solve(
                F, starts, L, full; intrinsic = intrinsic, show_progress = false,
            )
        end
        # different ambient dimension
        @test_throws ArgumentError solve(
            F, starts, L, rand_subspace(3; codim = 2); show_progress = false,
        )
    end

    @testset "start solutions are not aliased into the results" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L₁ = rand_subspace(2; codim = 1)
        L₂ = rand_subspace(2; codim = 1)
        S = solutions(solve(F, L₁; show_progress = false))
        cache = CommonSolve.init(F, S, L₁, L₂, Serial(); show_progress = false)
        res = CommonSolve.solve!(cache)
        for (k, pr) in enumerate(path_results(res))
            @test start_solution(pr) == cache.start_solutions[k]
            @test start_solution(pr) !== cache.start_solutions[k]
        end
    end

    @testset "non-square subspace homotopy is rejected" begin
        @polyvar x y z
        # 1 equation, codim-1 subspace in 3-space: 2 equations in 3 unknowns
        F = System([x^2 + y^2 + z^2 - 1]; variables = [x, y, z])
        L = rand_subspace(3; codim = 1)
        starts = [[1.0 + 0im, 0.0 + 0im, 0.0 + 0im]]
        @test_throws ArgumentError solve(
            F, starts, L, L; intrinsic = false, show_progress = false,
        )
    end

    @testset "intrinsic results are ambient, diagnostics are not converted" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L₁ = rand_subspace(2; codim = 1)
        L₂ = rand_subspace(2; codim = 1)
        S = solutions(solve(F, L₁; show_progress = false))
        res = solve(F, S, L₁, L₂; intrinsic = true, show_progress = false)
        for pr in path_results(res)
            @test length(solution(pr)) == 2            # ambient
            @test length(start_solution(pr)) == 2      # the caller's ambient point
            @test length(last_path_point(pr)[1]) == 2  # ambient
            @test residual(pr) < 1.0e-8                # intrinsic residual, still small
        end
    end
end
