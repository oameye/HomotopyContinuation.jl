using Test
using Aqua
using HomotopyContinuation

@testset "Aqua.jl" begin
    Aqua.test_all(HomotopyContinuation)
end
