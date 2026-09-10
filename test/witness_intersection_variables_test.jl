using Test
using HomotopyContinuationNext
using DynamicPolynomials: @polyvar

@testset "Witness-set ambient coordinates" begin
    @polyvar x y z
    F = System([x]; variables = [x, y])
    G = System([y]; variables = [x, y])
    H = System([z]; variables = [x, z])
    R = System([x]; variables = [y, x])

    @test isnothing(HomotopyContinuationNext._check_same_ambient_variables(F, G))
    @test_throws ArgumentError HomotopyContinuationNext._check_same_ambient_variables(F, H)
    @test_throws ArgumentError HomotopyContinuationNext._check_same_ambient_variables(F, R)

    L = LinearSubspace(zeros(ComplexF64, 0, 2), ComplexF64[])
    empty_points = Vector{Vector{ComplexF64}}()
    W = WitnessSet(F, L, copy(empty_points))
    WH = WitnessSet(H, L, copy(empty_points))
    @test_throws ArgumentError intersect(
        W, WH, Intersection(; show_progress = false), Serial(),
    )
end
