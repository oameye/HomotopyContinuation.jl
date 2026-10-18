using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar

@testset "NID result concreteness" begin
    @polyvar x y
    F = System([x * y]; variables = [x, y])
    seed = UInt32(0x1234)

    witness_superset = solve(
        F,
        Regeneration(; seed, show_progress = false),
        Serial(),
    )
    @test isconcretetype(eltype(witness_superset))

    decomposition = solve(
        witness_superset,
        Decomposition(; seed, show_progress = false),
        Serial(),
    )
    @test isconcretetype(eltype(decomposition))
    @test eltype(decomposition) === eltype(witness_superset)

    nid = solve(F, Decomposition(; seed, show_progress = false), Serial())
    @test isconcretetype(typeof(nid))
end
