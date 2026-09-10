using Test
using HomotopyContinuationNext
using DynamicPolynomials: @polyvar

@testset "NID preserves unresolved witness points" begin
    @polyvar x y
    F = System([x * y]; variables = [x, y])
    # Exact complete witness set of V(xy): x + y = 1 meets the two
    # irreducible lines at (1,0) and (0,1).
    L = LinearSubspace(ComplexF64[1 1], ComplexF64[1])
    W = WitnessSet(F, L, [ComplexF64[1, 0], ComplexF64[0, 1]])
    @test degree(W) == 2

    dec = solve(
        W,
        Decomposition(;
            max_iters = 0, warning = false,
            seed = UInt32(0x92), show_progress = false,
        ),
        Serial(),
    )
    @test !isempty(dec)
    @test sum(degree, dec; init = 0) == degree(W)
    @test all(Wi -> is_irreducible(Wi) == Irreducibility.UNKNOWN, dec)

    N = NumericalIrreducibleDecomposition(dec, UInt32(0x92))
    all_sets = witness_sets(N)
    @test sum(degree, Iterators.flatten(values(all_sets)); init = 0) == degree(W)
    @test isempty(irreducible_components(N))
    @test ncomponents(N) == 0
    @test isempty(degrees(N))
    @test sum(degree, Iterators.flatten(values(unresolved_witness_sets(N))); init = 0) == degree(W)
    @test unresolved_degree(N) == degree(W)
    @test occursin("unresolved", sprint(show, N))
end

@testset "Decomposition owns point-identity tolerances" begin
    nested = HomotopyContinuationNext.MonodromyOptions(;
        unique_points_atol = 7.0e-12,
        unique_points_rtol = 8.0e-10,
        group_action = x -> -x,
        equivalence_classes = true,
    )
    alg = Decomposition(;
        atol = 2.0e-9, rtol = 3.0e-7,
        monodromy = nested,
        show_progress = false,
    )
    @test alg.atol == 2.0e-9
    @test alg.rtol == 3.0e-7
    opts = HomotopyContinuationNext._decompose_monodromy_options(alg.monodromy, alg.atol, alg.rtol)
    @test opts.unique_points_atol == alg.atol
    @test opts.unique_points_rtol == alg.rtol
    @test !opts.equivalence_classes
end
