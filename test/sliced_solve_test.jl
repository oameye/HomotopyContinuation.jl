using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar
using CommonSolve: CommonSolve
using LinearAlgebra: norm

# Largest residual of the linear equations of `L` at `x`.
function subspace_residual(L, x)
    E = extrinsic(L)
    return isempty(E.b) ? 0.0 : maximum(abs, E.A * x - E.b)
end

@testset "Sliced solve public behavior" begin
    @testset "slice shape, metadata, and validation" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L = rand_subspace(2; codim = 1)
        G = slice(F, L)
        @test size(G) == (2, 2)
        @test degrees(G) == [2, 1]
        @test collect(variables(G)) == collect(variables(F))

        @polyvar a
        P = System(
            [x^2 + y^2 - a]; variables = [x, y], parameters = [a],
            compile = CompileMode.COMPILED,
        )
        PG = slice(P, L)
        @test nparameters(PG) == 1
        @test collect(parameters(PG)) == collect(parameters(P))
        @test nsolutions(
            solve(fix_parameters(PG, [5.0]), TotalDegree(; show_progress = false)),
        ) == 2

        @polyvar z
        projective = System([x^2 + y^2 - z^2]; variables = [x, y, z])
        projective_slice = rand_subspace(3; codim = 1, affine = false)
        @test size(slice(projective, projective_slice)) == (2, 3)
        @test size(slice(projective, projective_slice; chart = randn(ComplexF64, 3))) == (3, 3)
        @test_throws ArgumentError slice(projective, rand_subspace(2; codim = 1))
    end

    @testset "affine line through a conic" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L = rand_subspace(2; codim = 1)
        result = solve(F, L, TotalDegree(; show_progress = false))
        @test nsolutions(result) == 2
        for s in solutions(result)
            @test length(s) == 2
            @test abs(s[1]^2 + s[2]^2 - 5) < 1.0e-10
            @test subspace_residual(L, s) < 1.0e-10
        end
    end

    @testset "two equations in three variables" begin
        @polyvar x y z
        F = System([x^2 + y^2 - 5, x * y + 1]; variables = [x, y, z])
        L = rand_subspace(3; codim = 1)
        result = solve(F, L, TotalDegree(; show_progress = false))
        @test nsolutions(result) == 4
        for s in solutions(result)
            @test subspace_residual(L, s) < 1.0e-10
        end
    end

    @testset "projective slicing" begin
        @polyvar x y z
        F = System([x^2 + y^2 - z^2]; variables = [x, y, z])
        L = rand_subspace(3; codim = 1, affine = false)
        result = solve(F, L, TotalDegree(; show_progress = false))
        @test nsolutions(result) == 2
        for s in solutions(result)
            @test abs(s[1]^2 + s[2]^2 - s[3]^2) < 1.0e-10 * norm(s, Inf)^2
            @test subspace_residual(L, s) < 1.0e-10 * norm(s, Inf)
        end
    end

    @testset "parameters" begin
        @polyvar x y a
        F = System([x^2 + y^2 - a]; variables = [x, y], parameters = [a])
        L = rand_subspace(2; codim = 1)
        result = solve(fix_parameters(F, [5.0]), L, TotalDegree(; show_progress = false))
        @test nsolutions(result) == 2
        for s in solutions(result)
            @test abs(s[1]^2 + s[2]^2 - 5) < 1.0e-10
            @test subspace_residual(L, s) < 1.0e-10
        end
        @test_throws ArgumentError solve(F, L, TotalDegree(; show_progress = false))

        parameter_free = System([x^2 + y^2 - 5]; variables = [x, y])
        @test_throws ArgumentError fix_parameters(parameter_free, [1.0])
    end

    @testset "overdetermined slicing preserves excess filtering" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5, x * y - 1]; variables = [x, y])
        result = solve(F, rand_subspace(2; codim = 1), TotalDegree(; show_progress = false))
        @test nsolutions(result) == 0
        @test nexcess_solutions(result) > 0
    end

    @testset "Polyhedral sliced solve" begin
        @polyvar x y z
        F = System([x^2 + y^2 - 5, x * y + 1]; variables = [x, y, z])
        L = rand_subspace(3; codim = 1)
        result = solve(F, L, Polyhedral(; show_progress = false))
        @test nsolutions(result) == 4
        for s in solutions(result)
            @test subspace_residual(L, s) < 1.0e-10
        end
    end

    @testset "executors and seed agree" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L = rand_subspace(2; codim = 1)
        alg = TotalDegree(; seed = UInt32(0x1234), show_progress = false)
        serial = solve(F, L, alg, Serial())
        threaded = solve(F, L, alg, Threaded())
        @test nsolutions(serial) == nsolutions(threaded) == 2
        @test seed(serial) == seed(threaded) == UInt32(0x1234)
        @test nsolutions(solve(F, L, TotalDegree(; show_progress = false), Serial())) == 2
    end

    @testset "CommonSolve init/solve!" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L = rand_subspace(2; codim = 1)
        cache = CommonSolve.init(F, L, TotalDegree(; show_progress = false), Serial())
        result = CommonSolve.solve!(cache)
        @test result isa Result
        @test nsolutions(result) == 2
    end
end
