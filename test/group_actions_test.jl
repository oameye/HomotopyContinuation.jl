using Test, Random
using HomotopyContinuationNext
using HomotopyContinuationNext: GroupActions, SymmetricGroup, apply_actions

@testset "GroupActions" begin
    # Single swap action; the orbit is asserted as a SET since element order
    # is an implementation detail.
    action1(s) = [s[2], s[1]]
    G1 = GroupActions(action1)
    @test sort(G1([3, 9])) == sort([[3, 9], [9, 3]])

    # Chained actions: each image gets only the remaining actions.
    action2(s) = [-s[1], -s[2]]
    G2 = GroupActions(action1, action2)
    orbit = Set(G2([3, 9]))
    @test orbit == Set([[3, 9], [9, 3], [-3, -9], [-9, -3]])
    @test length(orbit) == 4

    # Multi-image action (tuple return). Each action is applied once (no group
    # closure), so a single action with two images yields a 3-element orbit.
    action3(s) = ([s[2], s[1]], [-s[1], -s[2]])
    G3 = GroupActions(action3)
    @test Set(G3([3, 9])) == Set([[3, 9], [9, 3], [-3, -9]])

    # Early-exit callback protocol
    hits = 0
    found = apply_actions(GroupActions(action1, action2), [3, 9]) do y
        hits += 1
        y == [9, 3]  # stop after finding this image
    end
    @test found
    @test hits <= 3
end

@testset "SymmetricGroup" begin
    S3 = SymmetricGroup(3)
    perms = collect(S3)
    @test length(perms) == 6
    @test allunique(perms)
    @test all(p -> sort(p) == [1, 2, 3], perms)
end

using HomotopyContinuationNext: UniquePoints, multiplicities, unique_points,
    search_in_radius, add!

@testset "UniquePoints with group actions" begin
    Random.seed!(42)
    sign_flip(s) = ([-s[1], -s[2]],)
    UP = UniquePoints(2; group_actions = sign_flip)
    a = [1.0 + 0im, 2.0 + 0im]
    @test add!(UP, a, 1; atol = 1.0e-10) |> last   # inserted
    # The negated point is in the same orbit: rejected as duplicate
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

    # 300-point sign-flip cloud collapses to 100 classes (prototype 8 case)
    base = [randn(ComplexF64, 3) for _ in 1:100]
    cloud = vcat(base, [-b for b in base], [b .+ 1.0e-13 .* randn(ComplexF64, 3) for b in base])
    classes = multiplicities(cloud; atol = 1.0e-8, group_actions = s -> (-s,))
    @test length(classes) == 100
    @test all(c -> length(c) == 3, classes)

    u = unique_points(pts; atol = 1.0e-8)
    @test length(u) == 2
end
