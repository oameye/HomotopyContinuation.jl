using Test, Random
using HomotopyContinuation

@testset "GroupActions" begin
    action1(s) = [s[2], s[1]]
    G1 = GroupActions(action1)
    @test sort(G1([3, 9])) == sort([[3, 9], [9, 3]])

    action2(s) = [-s[1], -s[2]]
    G2 = GroupActions(action1, action2)
    orbit = Set(G2([3, 9]))
    @test orbit == Set([[3, 9], [9, 3], [-3, -9], [-9, -3]])
    @test length(orbit) == 4

    action3(s) = ([s[2], s[1]], [-s[1], -s[2]])
    G3 = GroupActions(action3)
    @test Set(G3([3, 9])) == Set([[3, 9], [9, 3], [-3, -9]])
end

@testset "SymmetricGroup" begin
    S3 = SymmetricGroup(3)
    perms = collect(S3)
    @test length(perms) == 6
    @test allunique(perms)
    @test all(p -> sort(p) == [1, 2, 3], perms)
end

@testset "UniquePoints with group actions" begin
    Random.seed!(42)
    sign_flip(s) = ([-s[1], -s[2]],)
    UP = UniquePoints(2; group_actions = sign_flip)
    a = [1.0 + 0im, 2.0 + 0im]
    @test add!(UP, a, 1; atol = 1.0e-10) |> last
    id, fresh = add!(UP, -a, 2; atol = 1.0e-10)
    @test !fresh && id == 1
    @test length(UP) == 1
end

@testset "multiplicities" begin
    Random.seed!(43)
    pts = [[1.0 + 0im, 2.0], [1.0 + 1.0e-12im, 2.0], [3.0 + 0im, 4.0]]
    m = multiplicities(pts; atol = 1.0e-8)
    @test length(m) == 1
    @test sort(only(m)) == [1, 2]

    base = [randn(ComplexF64, 3) for _ in 1:100]
    cloud = vcat(base, [-b for b in base], [b .+ 1.0e-13 .* randn(ComplexF64, 3) for b in base])
    classes = multiplicities(cloud; atol = 1.0e-8, group_actions = s -> (-s,))
    @test length(classes) == 100
    @test all(c -> length(c) == 3, classes)

    u = unique_points(pts; atol = 1.0e-8)
    @test length(u) == 2
end
