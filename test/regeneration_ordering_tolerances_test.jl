using Test
using HomotopyContinuationNext
using HomotopyContinuationNext:
    MonodromyOptions, EquationSorting, _regeneration_sortperm,
    _regeneration_monodromy_options
using DynamicPolynomials: @polyvar

@testset "Regeneration ordering and tolerances" begin
    @testset "degree ordering policy" begin
        @polyvar x y
        L = LinearSubspace(zeros(ComplexF64, 0, 2), ComplexF64[])
        F = System([x]; variables = [x, y])
        H = [
            WitnessSet(F, L, [ComplexF64[0, 0] for _ in 1:n])
            for n in (3, 1, 2)
        ]
        @test _regeneration_sortperm(H, EquationSorting.UNSORTED) == [1, 2, 3]
        @test _regeneration_sortperm(H, EquationSorting.BY_DEGREE) == [2, 3, 1]
        @test _regeneration_sortperm(H, EquationSorting.RANDOMIZED) == [1, 3, 2]
    end

    @testset "outer algorithm owns witness-point identity" begin
        @polyvar u v
        G = System([u]; variables = [u, v])
        L = LinearSubspace(zeros(ComplexF64, 0, 2), ComplexF64[])
        W = WitnessSet(G, L, [ComplexF64[0, 0]])

        # Standalone monodromy keeps its concrete absolute default and adaptive
        # endpoint-relative default.
        default_m = MonodromyOptions()
        @test default_m.unique_points_atol == 1.0e-14
        @test default_m.unique_points_rtol === nothing

        inherited = _regeneration_monodromy_options(default_m, W, 2.0e-9, 3.0e-7)
        @test inherited.unique_points_atol == 2.0e-9
        @test inherited.unique_points_rtol == 3.0e-7

        # Once monodromy is used as an internal regeneration engine, the outer
        # algorithm's identity tolerance is authoritative.  Nested monodromy
        # uniqueness settings must not create a second notion of witness identity.
        explicit_m = MonodromyOptions(;
            unique_points_atol = 7.0e-12,
            unique_points_rtol = 8.0e-10,
            group_action = x -> -x,
            equivalence_classes = true,
        )
        explicit = _regeneration_monodromy_options(explicit_m, W, 2.0e-9, 3.0e-7)
        @test explicit.unique_points_atol == 2.0e-9
        @test explicit.unique_points_rtol == 3.0e-7
        @test explicit.distance === explicit_m.distance
        @test explicit.triangle_inequality == explicit_m.triangle_inequality
        @test !explicit.equivalence_classes
        @test explicit.group_actions !== nothing
    end
end
