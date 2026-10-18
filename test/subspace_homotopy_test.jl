using Test, Random
using LinearAlgebra: I, norm
using HomotopyContinuation
using HomotopyContinuation: TaylorVector
using DynamicPolynomials: @polyvar, differentiate

@static if VERSION < v"1.11"
    buffer(A::AbstractArray) = collect(A)
else
    using FixedSizeArrays: FixedSizeArrayDefault
    buffer(A::AbstractArray) = FixedSizeArrayDefault(A)
end

include("cauchy_oracle.jl")

function hom_value(H, x::Vector{ComplexF64}, t::ComplexF64)
    u = buffer(zeros(ComplexF64, size(H, 1)))
    evaluate!(u, H, buffer(x), t)
    return Vector(u)
end

# Order-K coefficient in λ of H(x₀ + x₁λ + … + x_{K-1}λ^{K-1}, t₀ + λ) from
# taylor! and from the Cauchy integral over evaluate!. The order-K slot of the
# path is left zero, as the predictor leaves the unknown coefficient.
function hom_taylor(H, X::Matrix{ComplexF64}, t0::ComplexF64, K::Int)
    tx = TaylorVector{K + 1, ComplexF64}(size(X, 2))
    for i in axes(X, 2)
        tx[i] = (X[1:K, i]..., zero(ComplexF64))
    end
    u = buffer(zeros(ComplexF64, size(H, 1)))
    taylor!(u, Val(K), H, tx, t0)
    return Vector(u)
end
function cauchy_hom_taylor(H, X::Matrix{ComplexF64}, t0::ComplexF64, K::Int)
    return cauchy_coefficients(K) do λ
        hom_value(H, [sum(X[k + 1, i] * λ^k for k in 0:(K - 1)) for i in axes(X, 2)], t0 + λ)
    end[K + 1]
end

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

    @testset "intrinsic evaluate!, Jacobian and taylor! agree with evaluate! derivatives" begin
        Random.seed!(12)
        V = rand_subspace(3; dim = 1)
        W = rand_subspace(3; dim = 1)
        H = IntrinsicSubspaceHomotopy(SUBSPACE_QUADRIC, V, W)
        @test size(H) == (1, 1)
        v = randn(ComplexF64, 1)
        h = 1.0e-6

        for t in (complex(0.6), 0.3 + 0.2im)
            u = buffer(zeros(ComplexF64, 1))
            taylor!(u, Val(1), H, buffer(v), t)
            @test Vector(u) ≈ (hom_value(H, v, t + h) .- hom_value(H, v, t - h)) ./ (2h) atol = 1.0e-6

            U = buffer(zeros(ComplexF64, 1, 1))
            evaluate_and_jacobian!(u, U, H, buffer(v), t)
            @test Vector(u) ≈ hom_value(H, v, t) rtol = 1.0e-14
            @test U[1, 1] ≈ (hom_value(H, v .+ h, t)[1] - hom_value(H, v .- h, t)[1]) / (2h) rtol = 1.0e-7
        end

        X = randn(ComplexF64, 4, 1)
        for K in 2:3
            @test hom_taylor(H, X, complex(0.37), K) ≈
                cauchy_hom_taylor(H, X, complex(0.37), K) rtol = 1.0e-8
        end
    end

    @testset "extrinsic homotopy keeps the system and moves the slice from A to B" begin
        Random.seed!(41)
        @polyvar x[1:4]
        polys = [
            sum(randn(ComplexF64) * x[i] * x[j] for i in 1:4 for j in i:4) +
                sum(randn(ComplexF64, 4) .* x) + randn(ComplexF64)
                for _ in 1:2
        ]
        F = System(polys; variables = x)
        A = rand_subspace(4; codim = 2)
        B = rand_subspace(4; codim = 2)
        H = ExtrinsicSubspaceHomotopy(F, A, B; gamma = one(ComplexF64))
        @test size(H) == (4, 4)
        on(L) = intrinsic(L).A * randn(ComplexF64, 2) .+ intrinsic(L).b

        # The system rows are F itself for every t; the slice rows vanish on A at
        # t = 1 and on B at t = 0.
        for (t, L) in ((1.0, A), (0.0, B)), _ in 1:3
            xv = on(L)
            value = hom_value(H, xv, complex(t))
            @test value[1:2] ≈ [ComplexF64(f(x => xv)) for f in polys] rtol = 1.0e-12
            @test norm(value[3:4]) < 1.0e-12
        end
        xv = randn(ComplexF64, 4)
        for t in (complex(0.3), 0.8 - 0.4im)
            @test hom_value(H, xv, t)[1:2] ≈ [ComplexF64(f(x => xv)) for f in polys] rtol = 1.0e-12
        end

        # The slice rows are affine in x with orthonormal normals along the
        # Grassmannian geodesic.
        δ = randn(ComplexF64, 4)
        for t in (0.3, 0.7)
            u = buffer(zeros(ComplexF64, 4))
            U = buffer(zeros(ComplexF64, 4, 4))
            evaluate_and_jacobian!(u, U, H, buffer(xv), complex(t))
            J = Matrix(U)
            @test J[1:2, :] ≈ [ComplexF64(differentiate(f, xj)(x => xv)) for f in polys, xj in x] rtol = 1.0e-12
            @test J[3:4, :] * J[3:4, :]' ≈ I atol = 1.0e-12
            @test hom_value(H, xv .+ δ, complex(t))[3:4] - Vector(u)[3:4] ≈ J[3:4, :] * δ atol = 1.0e-12
        end

        t0 = complex(0.42)
        u = buffer(zeros(ComplexF64, 4))
        taylor!(u, Val(1), H, buffer(xv), t0)
        @test Vector(u) ≈ cauchy_coefficients(λ -> hom_value(H, xv, t0 + λ), 1)[2] rtol = 1.0e-9
        X = randn(ComplexF64, 4, 4)
        for K in 2:3
            @test hom_taylor(H, X, t0, K) ≈ cauchy_hom_taylor(H, X, t0, K) rtol = 1.0e-8
        end
    end
end
