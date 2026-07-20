using Test, Random
using LinearAlgebra
using HomotopyContinuationNext
using HomotopyContinuationNext: corank
using DynamicPolynomials: @polyvar
import MultivariatePolynomials as MP

# Test helper: dense random polynomial in `vars` of degree `d`.
# Mirrors v2's `rand_poly`. `homogeneous` restricts to monomials of degree exactly d.
function rand_poly(T, vars, d; homogeneous = false)
    monos = homogeneous ? MP.monomials(vars, d) : MP.monomials(vars, 0:d)
    return sum(randn(T) * m for m in monos)
end
rand_poly(vars, d; homogeneous = false) = rand_poly(Float64, vars, d; homogeneous = homogeneous)

@testset "Witness Sets" begin

    @testset "affine" begin
        @polyvar x y

        F = System([x^2 + y^2 - 5]; variables = [x, y])

        W = witness_set(F)

        @test dim(W) == 1
        @test codim(W) == 1
        @test degree(W) == 2
        @test solutions(W) isa Vector{Vector{ComplexF64}}

        L = LinearSubspace([1 1], [-1])

        W_L = witness_set(W, L)
        @test degree(W_L) == 2
        @test sort(real.(solutions(W_L))) ≈ [[-2, 1], [1, -2]]
        @test linear_subspace(W_L) == convert(typeof(linear_subspace(W_L)), L)

        @test trace_test(W) < 1.0e-8
        @test_throws MethodError trace_test(W; shwo_progress = false)

        W_seed₁ = witness_set(F; seed = 0x1234, threading = false)
        W_seed₂ = witness_set(F; seed = 0x1234, threading = false)
        @test linear_subspace(W_seed₁) == linear_subspace(W_seed₂)
        @test points(W_seed₁) == points(W_seed₂)
    end

    @testset "projective" begin
        @polyvar x y z

        F = System([x^2 + y^2 - 5z^2]; variables = [x, y, z])

        W = witness_set(F)

        @test dim(W) == 1
        @test codim(W) == 2
        @test degree(W) == 2
        @test solutions(W) isa Vector{Vector{ComplexF64}}
        @test trace_test(W) < 1.0e-8

        L = LinearSubspace([1 1 1])
        W_L = witness_set(W, L)
        @test degree(W_L) == 2

        L = rand_subspace(3; codim = 1, affine = false)
        W_L = witness_set(W, L)
        @test degree(W_L) == 2

        L = rand_subspace([x, y, z]; codim = 1, affine = false)
        W_L = witness_set(W, L)
        @test degree(W_L) == 2

        L = rand_subspace(3; codim = 1, affine = true)
        @test_throws ErrorException witness_set(W, L)
    end

    @testset "dim / codim" begin
        @polyvar x[1:6]
        homogeneous = true

        f = System(
            [
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
            ]
        )

        @test degree(witness_set(f; dim = 1)) == 16
        @test degree(witness_set(f; codim = 4)) == 16
        @test degree(witness_set(f)) == 16

        homogeneous = false
        f = System(
            [
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
            ]
        )
        @test degree(witness_set(f; dim = 2)) == 16
        @test degree(witness_set(f; codim = 4)) == 16
        @test degree(witness_set(f)) == 16
    end

    @polyvar x y z
    p = (x * y - x^2) + 1 - z
    q = x^4 + x^2 - y - 1
    F = [
        p * q * (x - 3) * (x - 5),
        p * q * (y - 3) * (y - 5),
        p * (z - 3) * (z - 5),
    ]

    @testset "membership" begin
        W = witness_set(System(F); codim = 2)

        pt = randn(3)
        q0 = solutions(W)[1]

        @test !membership(pt, W)
        @test membership(q0, W; show_progress = false)
        a = membership([pt, q0], W; show_progress = false)
        @test a == [false, true]
        @test membership([pt, q0], W; show_progress = true) == [false, true]
    end

    @testset "intersect" begin
        H = [witness_set(System([f])) for f in F]
        B = intersect(H[1], H[2])
        C = vcat([intersect(Hi, H[3]; show_progress = false) for Hi in B]...)
        @test sort(degree.(C)) == [2, 8, 8]
    end

end
