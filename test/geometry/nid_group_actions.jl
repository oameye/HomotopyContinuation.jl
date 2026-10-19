using Test, Random
using HomotopyContinuation
using HomotopyContinuation: TrackerOptions, MonodromyOptions, linear_subspace,
    is_linear, dim, EquationSorting
using DynamicPolynomials: @polyvar
import MultivariatePolynomials as MP

function rand_poly(T, vars, d; homogeneous = false)
    monos = homogeneous ? MP.monomials(vars, d) : MP.monomials(vars, 0:d)
    return sum(randn(T) * m for m in monos)
end
rand_poly(vars, d; homogeneous = false) = rand_poly(Float64, vars, d; homogeneous = homogeneous)


@testset "Regeneration preserves witness cardinality under monodromy group actions" begin
    @var a b c
    F = System([a^2 + b^2 + c^2 - 4, a + b + c]; variables = [a, b, c])

    seen = Ref{Union{Nothing, Vector{ComplexF64}}}(nothing)
    function collapse_to_first(p)
        seen[] === nothing && (seen[] = collect(ComplexF64, p))
        return (seen[]::Vector{ComplexF64},)
    end

    plain = MonodromyOptions(; trace_test = true)
    symmetric = MonodromyOptions(;
        trace_test = true,
        group_action = collapse_to_first,
        equivalence_classes = true,
    )

    expected = degree.(
        solve(
            F,
            Regeneration(;
                monodromy = plain, seed = UInt32(0x1234), show_progress = false,
            ),
        )
    )
    @test expected == [2]

    seen[] = nothing
    got = degree.(
        solve(
            F,
            Regeneration(;
                monodromy = symmetric, seed = UInt32(0x1234), show_progress = false,
            ),
        )
    )
    @test got == expected
end

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

@testset "Decomposition preserves component identity under nested monodromy group actions" begin
    @var a b c
    F = System([a * (a^2 + b^2 + c^2 - 1)])

    seen = Ref{Union{Nothing, Vector{ComplexF64}}}(nothing)
    function collapse_to_first(p)
        seen[] === nothing && (seen[] = collect(ComplexF64, p))
        return (seen[]::Vector{ComplexF64},)
    end

    plain = MonodromyOptions(; trace_test = true)
    symmetric = MonodromyOptions(;
        trace_test = true,
        group_action = collapse_to_first,
        equivalence_classes = true,
    )

    sorted_degrees(N) = Dict(d => sort(ds) for (d, ds) in degrees(N))

    N_plain = solve(
        F,
        Decomposition(;
            monodromy = plain, seed = UInt32(0x1234), show_progress = false,
        ),
    )
    @test ncomponents(N_plain) == 2
    @test sorted_degrees(N_plain) == Dict(2 => [1, 2])

    seen[] = nothing
    N_symmetric = solve(
        F,
        Decomposition(;
            monodromy = symmetric, seed = UInt32(0x1234), show_progress = false,
        ),
    )
    @test ncomponents(N_symmetric) == ncomponents(N_plain)
    @test sorted_degrees(N_symmetric) == sorted_degrees(N_plain)
end
