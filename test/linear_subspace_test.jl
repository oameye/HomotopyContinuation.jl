using Test, Random
using LinearAlgebra
using HomotopyContinuationNext
using HomotopyContinuationNext: LinearSubspace, rand_subspace, rand_subspace!, dim,
    codim, ambient_dim,
    intrinsic, extrinsic, coord_change, translate, geodesic, geodesic_distance,
    Intrinsic, Extrinsic, IntrinsicDescription, ExtrinsicDescription, is_linear

@testset "LinearSubspace round trips" begin
    Random.seed!(7)
    L = rand_subspace(5; dim = 2)
    @test dim(L) == 2
    @test codim(L) == 3
    @test ambient_dim(L) == 5

    # intrinsic -> ambient -> extrinsic residual
    u = randn(ComplexF64, 2)
    x = intrinsic(L).A * u + intrinsic(L).b
    @test norm(extrinsic(L).A * x - extrinsic(L).b) < 1.0e-13

    # coord_change round trip
    u2 = coord_change(L, Extrinsic, Intrinsic, x)
    x2 = coord_change(L, Intrinsic, Extrinsic, u2)
    @test x2 ≈ x atol = 1.0e-12
end

@testset "rand_subspace through a point" begin
    Random.seed!(8)
    x0 = randn(ComplexF64, 4)
    L = rand_subspace(x0; dim = 2)
    @test norm(extrinsic(L).A * x0 - extrinsic(L).b) < 1.0e-12
end

@testset "rand_subspace!" begin
    Random.seed!(21)
    A = zeros(ComplexF64, 3, 5)
    b = zeros(ComplexF64, 3)

    L = rand_subspace!(A, b)
    @test dim(L) == 2
    @test codim(L) == 3
    @test ambient_dim(L) == 5
    @test !is_linear(L)

    Llin = rand_subspace!(A, b; affine = false)
    @test is_linear(Llin)
    @test iszero(extrinsic(Llin).b)

    x0 = randn(ComplexF64, 5)
    Lx = rand_subspace!(A, b, x0)
    @test norm(extrinsic(Lx).A * x0 - extrinsic(Lx).b) < 1.0e-12

    # Linear through a point: the whole ray of `x0` lies in the subspace.
    Lray = rand_subspace!(A, b, x0; affine = false)
    @test is_linear(Lray)
    @test norm(extrinsic(Lray).A * x0) < 1.0e-12
    @test norm(extrinsic(Lray).A * (3.5 * x0)) < 1.0e-12

    # The returned subspace keeps copies, so redrawing does not disturb it.
    kept = copy(extrinsic(Lray).A)
    rand_subspace!(A, b)
    @test extrinsic(Lray).A == kept

    @test_throws ArgumentError rand_subspace!(A, zeros(ComplexF64, 2))
    @test_throws ArgumentError rand_subspace!(A, b, randn(ComplexF64, 4))
end

@testset "translate" begin
    Random.seed!(9)
    L = rand_subspace(4; dim = 1)
    v = randn(ComplexF64, 3)   # codim-sized translation
    Lt = translate(L, v)
    u = randn(ComplexF64, 1)
    x = intrinsic(Lt).A * u + intrinsic(Lt).b
    @test norm(extrinsic(Lt).A * x - extrinsic(Lt).b) < 1.0e-12
    @test extrinsic(Lt).b ≈ extrinsic(L).b + v atol = 1.0e-13
end

@testset "LinearSubspace API" begin
    Random.seed!(16)
    A = LinearSubspace([1 0 3; 2 1 3], [5, -2])
    @test dim(A) == dim(intrinsic(A)) == dim(extrinsic(A)) == 1
    @test codim(A) == codim(intrinsic(A)) == codim(extrinsic(A)) == 2
    @test ambient_dim(A) == 3
    @test startswith(sprint(show, A), "1-dim. affine linear subspace")
    @test identity.(A) == A

    @test intrinsic(A) isa IntrinsicDescription
    @test startswith(sprint(show, intrinsic(A)), "IntrinsicDescription")
    @test extrinsic(A) isa ExtrinsicDescription
    @test startswith(sprint(show, extrinsic(A)), "ExtrinsicDescription")

    @test identity.(extrinsic(A)) == extrinsic(A)
    @test identity.(intrinsic(A)) == intrinsic(A)
    @test identity.(Intrinsic) == Intrinsic

    u = rand(1)
    x = A(u, Intrinsic)
    @test coord_change(A, Extrinsic, Intrinsic, x) ≈ u rtol = 1.0e-12
    @test coord_change(A, Intrinsic, Extrinsic, u) ≈ x rtol = 1.0e-12
    @test norm(A(x, Extrinsic)) ≈ 0 atol = 1.0e-14
    @test norm(ExtrinsicDescription(intrinsic(A))(x)) ≈ 0.0 atol = 1.0e-12

    B = rand_subspace(3; codim = 2)
    @test B != A
    copy!(B, A)
    @test extrinsic(B) == extrinsic(A)
    @test intrinsic(B) == intrinsic(A)
    @test B == A
    @test copy(A) == A
    @test copy(A) !== A

    C = rand_subspace(3; dim = 1, real = true)
    @test geodesic_distance(A, C) > 0
    γ = geodesic(A, C)
    γ1 = γ(1)
    @test ambient_dim(γ1) == ambient_dim(A)
    @test dim(γ1) == dim(A)

    A2 = translate(A, [1, 1], Extrinsic)
    A3 = LinearSubspace(extrinsic(A).A, extrinsic(A).b + [1, 1])
    @test A2.intrinsic.X ≈ A3.intrinsic.X

    # linear (non-affine) subspace through a point
    x5 = randn(ComplexF64, 5)
    L2 = rand_subspace(x5; dim = 2, affine = false)
    @test is_linear(L2)
    @test norm(L2(x5)) ≈ 0 atol = 1.0e-8
end

@testset "intersect subspaces" begin
    Random.seed!(17)
    L₁ = rand_subspace(7; codim = 2)
    L₂ = rand_subspace(7; codim = 3)
    L₃ = L₁ ∩ L₂
    @test codim(L₃) == 5
    @test dim(L₃) == 2
    @test ambient_dim(L₃) == 7
    E₃ = extrinsic(L₃)
    @test norm(L₁(E₃.A \ E₃.b, Extrinsic)) ≈ 0 atol = 1.0e-12
    @test norm(L₂(E₃.A \ E₃.b, Extrinsic)) ≈ 0 atol = 1.0e-12
    @test norm(L₃(E₃.A \ E₃.b, Extrinsic)) ≈ 0 atol = 1.0e-12
end

@testset "geodesic: consistent endpoint convention" begin
    Random.seed!(10)
    A = rand_subspace(4; dim = 1)
    B = rand_subspace(4; dim = 1)
    γ = geodesic(A, B)
    # t=1 recovers A, t=0 recovers B: a point of the endpoint subspace satisfies
    # the geodesic subspace's extrinsic equations (up to representation).
    for (t, L) in ((1.0, A), (0.0, B))
        G = γ(t)
        u = randn(ComplexF64, 1)
        x = intrinsic(L).A * u + intrinsic(L).b
        @test norm(extrinsic(G).A * x - extrinsic(G).b) < 1.0e-8
    end
    @test geodesic_distance(A, B) ≈ geodesic_distance(B, A) atol = 1.0e-12
end
