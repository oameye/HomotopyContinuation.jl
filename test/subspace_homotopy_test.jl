using Test, Random
using LinearAlgebra: norm
using HomotopyContinuation
using DynamicPolynomials: @polyvar

@polyvar z[1:3]
const SUBSPACE_QUADRIC =
    System([z[1]^2 + 2z[2]^2 + 3z[3]^2 + z[1] * z[2] - 1]; variables = z)

function exact_quadric_slice_points(L)
    A, b = intrinsic(L).A, intrinsic(L).b
    f(u) = begin
        x = A .* u .+ b
        x[1]^2 + 2x[2]^2 + 3x[3]^2 + x[1] * x[2] - 1
    end
    γ = f(0.0 + 0im)
    α = (f(1.0 + 0im) + f(-1.0 + 0im)) / 2 - γ
    β = (f(1.0 + 0im) - f(-1.0 + 0im)) / 2
    Δ = sqrt(β^2 - 4α * γ)
    return [vec(A .* u .+ b) for u in ((-β + Δ) / (2α), (-β - Δ) / (2α))]
end

function same_solution_set(a, b; atol = 1.0e-8)
    length(a) == length(b) || return false
    return all(x -> any(y -> isapprox(x, y; atol), b), a)
end

@testset "Subspace homotopy through public continuation" begin
    @testset "intrinsic and extrinsic coordinates track exact witness points" begin
        Random.seed!(11)
        V = rand_subspace(3; dim = 1)
        W = rand_subspace(3; dim = 1)
        starts = exact_quadric_slice_points(V)
        targets = exact_quadric_slice_points(W)

        @test all(x -> norm(extrinsic(V).A * x - extrinsic(V).b) < 1.0e-12, starts)

        results = map((SubspaceCoords.INTRINSIC, SubspaceCoords.EXTRINSIC)) do coords
            solve(
                SUBSPACE_QUADRIC,
                starts,
                V,
                W,
                Continuation(; coords, seed = UInt32(11), show_progress = false),
                Serial(),
            )
        end

        for result in results
            @test nfailed(result) == 0
            @test nsolutions(result) == 2
            @test same_solution_set(solutions(result), targets)
            for x in solutions(result)
                @test abs(x[1]^2 + 2x[2]^2 + 3x[3]^2 + x[1] * x[2] - 1) < 1.0e-9
                @test norm(extrinsic(W).A * x - extrinsic(W).b) < 1.0e-9
            end
        end
        @test same_solution_set(solutions(results[1]), solutions(results[2]))
    end

    @testset "the same start slice can be continued to independent targets" begin
        Random.seed!(29)
        V = rand_subspace(3; dim = 1)
        starts = exact_quadric_slice_points(V)
        targets = [rand_subspace(3; dim = 1) for _ in 1:3]

        for (k, W) in enumerate(targets)
            result = solve(
                SUBSPACE_QUADRIC,
                starts,
                V,
                W,
                Continuation(; seed = UInt32(0x2900 + k), show_progress = false),
                Serial(),
            )
            expected = exact_quadric_slice_points(W)
            @test nfailed(result) == 0
            @test nsolutions(result) == 2
            @test same_solution_set(solutions(result), expected)
            @test all(x -> norm(extrinsic(W).A * x - extrinsic(W).b) < 1.0e-9, solutions(result))
        end
    end
end
