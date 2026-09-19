using Test
using HomotopyContinuation

@testset "Monodromy compiler contracts" begin
    base = @inferred Monodromy(; show_progress = false)
    by_dim = @inferred Monodromy(; dim = 1, show_progress = false)
    by_codim = @inferred Monodromy(; codim = 1, show_progress = false)

    @test isconcretetype(typeof(base))
    @test isconcretetype(typeof(by_dim))
    @test isconcretetype(typeof(by_codim))
    @test isconcretetype(eltype(base.variables))
    @test isconcretetype(eltype(base.parameters))
end
