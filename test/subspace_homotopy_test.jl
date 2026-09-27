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

include("solution_sets.jl")

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

    @testset "perpendicular subspaces (geodesic angle π/2)" begin
        @polyvar x y w
        p = (x * y - x^2) + 1 - w
        q = x^4 + x^2 - y - 1
        f = [
            p * q * (x - 3) * (x - 5),
            p * q * (y - 3) * (y - 5),
            p * (w - 3) * (w - 5),
        ]
        # Subspace tracking needs a square system, so two fixed random
        # combinations of f stand in for it. Every fᵢ vanishes on the curve
        # C = {q = 0, w = 3}, so C remains a component of V(g).
        g = [f[1] + (0.3 + 0.7im) * f[3], f[2] - (1.1 - 0.2im) * f[3]]
        G = System(g; variables = [x, y, w])
        at(h, v) = h(x => v[1], y => v[2], w => v[3])

        L1 = LinearSubspace(reshape([1.0, 0.0, 0.0], 1, 3), [1.0])
        L2 = LinearSubspace(reshape([-1.0, 1.0, 0.0], 1, 3), [1.0])
        @test geodesic_distance(L1, L2) ≈ π / 2

        # (1, 1, 3) ∈ C ∩ {x = 1}, where p = -2 ≠ 0 and the Jacobian of g
        # restricted to {x = 1} is regular.
        start = ComplexF64[1, 1, 3]
        @test all(h -> at(h, start) == 0, f)

        for coords in (SubspaceCoords.INTRINSIC, SubspaceCoords.EXTRINSIC)
            result = solve(
                G, [start], L1, L2,
                Continuation(; coords, seed = UInt32(0x90), show_progress = false),
                Serial(),
            )
            @test nfailed(result) == 0
            @test nsolutions(result) == 1
            endpoint = only(solutions(result))
            @test maximum(h -> abs(at(h, endpoint)), f) < 1.0e-8
            @test norm(extrinsic(L2).A * endpoint - extrinsic(L2).b) < 1.0e-10
            # The path stays on C: C ∩ L2 is x⁴ + x² - x - 2 = 0, w = 3.
            @test abs(at(q, endpoint)) < 1.0e-8
            @test abs(endpoint[3] - 3) < 1.0e-8
        end
    end
end
