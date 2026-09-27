using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar
using LinearAlgebra: norm

subspace_residual(L, x) =
    (E = extrinsic(L); isempty(E.b) ? 0.0 : maximum(abs, E.A * x - E.b))

@testset "Subspace to subspace solve" begin
    @testset "both coordinate regimes reach the target subspace" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L₁ = rand_subspace(2; codim = 1)
        L₂ = rand_subspace(2; codim = 1)
        S₁ = solutions(solve(F, L₁, TotalDegree(; show_progress = false)))
        @test length(S₁) == 2

        for kw in (
                NamedTuple(),
                (; coords = SubspaceCoords.INTRINSIC),
                (; coords = SubspaceCoords.EXTRINSIC),
            )
            res = solve(F, S₁, L₁, L₂, Continuation(; kw..., show_progress = false))
            @test nsolutions(res) == 2
            for s in solutions(res)
                @test subspace_residual(L₂, s) < 1.0e-10
                @test abs(s[1]^2 + s[2]^2 - 5) < 1.0e-10
            end
        end
    end

    @testset "two equations in three variables" begin
        @polyvar x y z
        F = System([x^2 + y^2 - 5, x * y + 1]; variables = [x, y, z])
        K₁ = rand_subspace(3; dim = 2)
        K₂ = rand_subspace(3; dim = 2)
        S = solutions(solve(F, K₁, TotalDegree(; show_progress = false)))
        @test length(S) == 4

        for kw in (
                NamedTuple(),
                (; coords = SubspaceCoords.INTRINSIC),
                (; coords = SubspaceCoords.EXTRINSIC),
            )
            res = solve(F, S, K₁, K₂, Continuation(; kw..., show_progress = false))
            @test nsolutions(res) == 4
            @test maximum(s -> subspace_residual(K₂, s), solutions(res)) < 1.0e-10
        end
    end

    @testset "projective" begin
        @polyvar x y z
        F = System([x^2 + y^2 - z^2]; variables = [x, y, z])
        L₁ = rand_subspace(3; codim = 1, affine = false)
        L₂ = rand_subspace(3; codim = 1, affine = false)
        S = solutions(solve(F, L₁, TotalDegree(; show_progress = false)))

        for kw in (
                NamedTuple(),
                (; coords = SubspaceCoords.INTRINSIC),
                (; coords = SubspaceCoords.EXTRINSIC),
            )
            res = solve(F, S, L₁, L₂, Continuation(; kw..., show_progress = false))
            @test nsolutions(res) == 2
            for s in solutions(res)
                scale = norm(s, Inf)
                @test abs(s[1]^2 + s[2]^2 - s[3]^2) < 1.0e-10 * scale^2
                @test norm(extrinsic(L₂).A * s, Inf) < 1.0e-10 * scale
            end
        end
    end

    @testset "parametric" begin
        @polyvar x y a
        F = System([x^2 + y^2 - a]; variables = [x, y], parameters = [a])
        L₁ = rand_subspace(2; codim = 1)
        L₂ = rand_subspace(2; codim = 1)
        fixed = fix_parameters(F, [5.0])
        S = solutions(solve(fixed, L₁, TotalDegree(; show_progress = false)))
        res = solve(fixed, S, L₁, L₂, Continuation(; show_progress = false))
        @test nsolutions(res) == 2
        @test maximum(s -> abs(s[1]^2 + s[2]^2 - 5), solutions(res)) < 1.0e-10
        @test_throws ArgumentError solve(F, S, L₁, L₂, Continuation(; show_progress = false))
    end

    @testset "executors agree" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L₁ = rand_subspace(2; codim = 1)
        L₂ = rand_subspace(2; codim = 1)
        S = solutions(solve(F, L₁, TotalDegree(; show_progress = false)))

        for coords in (SubspaceCoords.INTRINSIC, SubspaceCoords.EXTRINSIC)
            serial = solve(
                F,
                S,
                L₁,
                L₂,
                Continuation(; coords, seed = UInt32(7), show_progress = false),
                Serial(),
            )
            threaded = solve(
                F,
                S,
                L₁,
                L₂,
                Continuation(; coords, seed = UInt32(7), show_progress = false),
                Threaded(),
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
        full = rand_subspace(2; dim = 2)

        for coords in (SubspaceCoords.INTRINSIC, SubspaceCoords.EXTRINSIC)
            @test_throws ArgumentError solve(
                F,
                starts,
                L,
                full,
                Continuation(; coords, show_progress = false),
            )
        end
        @test_throws ArgumentError solve(
            F,
            starts,
            L,
            rand_subspace(3; codim = 2),
            Continuation(; show_progress = false),
        )
    end

    @testset "start solutions are preserved and copied into diagnostics" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L₁ = rand_subspace(2; codim = 1)
        L₂ = rand_subspace(2; codim = 1)
        starts = solutions(solve(F, L₁, TotalDegree(; show_progress = false)))
        snapshot = deepcopy(starts)

        res = solve(F, starts, L₁, L₂, Continuation(; show_progress = false), Serial())
        @test starts == snapshot
        for pr in path_results(res)
            i = path_number(pr)
            @test start_solution(pr) == snapshot[i]
            @test start_solution(pr) !== starts[i]
        end
    end

    @testset "non-square subspace homotopy is rejected" begin
        @polyvar x y z
        F = System([x^2 + y^2 + z^2 - 1]; variables = [x, y, z])
        L = rand_subspace(3; codim = 1)
        starts = [[1.0 + 0im, 0.0 + 0im, 0.0 + 0im]]
        @test_throws ArgumentError solve(
            F,
            starts,
            L,
            L,
            Continuation(; coords = SubspaceCoords.EXTRINSIC, show_progress = false),
        )
    end

    @testset "intrinsic results and diagnostics are ambient" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L₁ = rand_subspace(2; codim = 1)
        L₂ = rand_subspace(2; codim = 1)
        S = solutions(solve(F, L₁, TotalDegree(; show_progress = false)))
        res = solve(
            F,
            S,
            L₁,
            L₂,
            Continuation(; coords = SubspaceCoords.INTRINSIC, show_progress = false),
        )
        for pr in path_results(res)
            @test length(solution(pr)) == 2
            @test length(start_solution(pr)) == 2
            @test length(last_path_point(pr)[1]) == 2
            @test residual(pr) < 1.0e-8
        end
    end
end
