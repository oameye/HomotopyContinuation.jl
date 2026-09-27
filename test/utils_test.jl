using Test
using HomotopyContinuation
using Random: MersenneTwister

@testset "public solution and parameter I/O" begin
    dir = mktempdir()
    rng = MersenneTwister(1234)

    @testset "round trip is exact" begin
        S = [rand(rng, ComplexF64, 3) for _ in 1:2]
        path = joinpath(dir, "sols.txt")
        write_solutions(path, S)
        @test read_solutions(path) == S

        p = rand(rng, ComplexF64, 14)
        ppath = joinpath(dir, "params.txt")
        write_parameters(ppath, p)
        @test read_parameters(ppath) == p
    end

    @testset "integer input reads back as complex" begin
        path = joinpath(dir, "ints.txt")
        write_solutions(path, [[1, 1], [-1, 2]])
        @test read_solutions(path) == [[1.0 + 0im, 1.0 + 0im], [-1.0 + 0im, 2.0 + 0im]]
        @test read(path, String) == "2\n\n1 0\n1 0\n\n-1 0\n2 0\n"
    end

    @testset "empty and single" begin
        path = joinpath(dir, "empty.txt")
        write_solutions(path, Vector{ComplexF64}[])
        @test isempty(read_solutions(path))

        ppath = joinpath(dir, "one.txt")
        write_parameters(ppath, [2.0 - 3.0im])
        @test read_parameters(ppath) == [2.0 - 3.0im]
    end

    @testset "a missing imaginary part is zero" begin
        path = joinpath(dir, "real_only.txt")
        write(path, "2\n\n1.5\n-2.5\n")
        @test read_parameters(path) == [1.5 + 0im, -2.5 + 0im]
    end

    @testset "declared count is checked" begin
        path = joinpath(dir, "bad_count.txt")
        write(path, "3\n\n1.0 0.0\n2.0 0.0\n")
        @test_throws ArgumentError read_parameters(path)

        write(path, "")
        @test_throws ArgumentError read_solutions(path)

        write(path, "1\n\n1.0 0.0 2.0\n")
        @test_throws ArgumentError read_parameters(path)
    end
end
