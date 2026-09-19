using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar

@testset "System public behavior" begin
    @testset "metadata" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2]; variables = [x, y])

        @test size(F) == (2, 2)
        @test nvariables(F) == 2
        @test nparameters(F) == 0
        @test collect(variables(F)) == [x, y]
        @test isempty(parameters(F))
        @test degrees(F) == [2, 2]

        @polyvar a b
        P = System(
            [a^3 * x^2 + y, a * b * x * y - a^5, b^2 - x];
            variables = [x, y], parameters = [a, b],
        )
        @test size(P) == (3, 2)
        @test nvariables(P) == 2
        @test nparameters(P) == 2
        @test collect(variables(P)) == [x, y]
        @test collect(parameters(P)) == [a, b]
        @test degrees(P) == [2, 2, 1]
    end

    @testset "evaluation and Jacobian" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2]; variables = [x, y])
        point = [2.0, 3.0]

        @test evaluate(F, point) ≈ ComplexF64[6, 4]
        @test F(point) ≈ ComplexF64[6, 4]
        @test jacobian(F, point) ≈ ComplexF64[4 1; 3 2]

        @polyvar a b
        P = System(
            [x^2 + a * y, x * y - b];
            variables = [x, y], parameters = [a, b],
        )
        params = [1.0, 2.0]
        @test evaluate(P, point, params) ≈ ComplexF64[7, 4]
        @test P(point, params) ≈ ComplexF64[7, 4]
        @test jacobian(P, point, params) ≈ ComplexF64[4 1; 3 2]
    end

    @testset "public evaluation validates dimensions" begin
        @polyvar x y a b
        F = System(
            [x^2 + a * y, x * y - b];
            variables = [x, y], parameters = [a, b],
        )

        @test_throws ArgumentError evaluate(F, [1.0], [2.0, 3.0])
        @test_throws ArgumentError evaluate(F, [1.0, 2.0], [3.0])
        @test_throws ArgumentError jacobian(F, [1.0], [2.0, 3.0])
        @test_throws ArgumentError jacobian(F, [1.0, 2.0], [3.0])
    end
end
