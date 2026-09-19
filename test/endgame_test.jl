using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar

include("test_systems.jl")

@testset "Endgame behavior" begin
    @testset "multiple roots expose multiplicity and winding" begin
        @polyvar x

        quadratic = solve(
            System([(x - 10)^2]),
            TotalDegree(; seed = UInt32(0xabcd), show_progress = false),
        )
        @test nresults(quadratic) == 1
        @test nsingular(quadratic) == 1
        @test multiplicity(only(singular(quadratic))) == 2
        @test any(path -> winding_number(path) == 2, path_results(quadratic))

        sextic = solve(
            System([(x - 10)^6]),
            TotalDegree(; seed = UInt32(0x6006), show_progress = false),
        )
        @test count(path -> winding_number(path) == 6, path_results(sextic)) >= 4
    end

    @testset "paths at infinity are distinguished from finite solutions" begin
        @polyvar x y
        result = solve(
            System(
                [
                    2.3x^2 + 1.2y^2 + 3x - 2y + 3,
                    2.3x^2 + 1.2y^2 + 5x + 2y - 5,
                ],
            ),
            TotalDegree(; seed = UInt32(0x1f1f), show_progress = false),
        )
        @test count(is_success, path_results(result)) == 2
        @test nat_infinity(result) == 2
        @test nfailed(result) == 0
    end

    @testset "winding-number family remains trackable" begin
        for d in 2:2:6
            @testset "d=$d" begin
                @polyvar x y
                a = [0.257, -0.139, -1.73, -0.199, 1.79, -1.32]
                f1 = (a[1] * x^d + a[2] * y) * (a[3] * x + a[4] * y) + 1
                f2 = (a[1] * x^d + a[2] * y) * (a[5] * x + a[6] * y) + 1
                result = solve(
                    System([f1, f2]),
                    TotalDegree(;
                        seed = UInt32(0x1000 + d),
                        show_progress = false,
                    ),
                )
                @test count(is_success, path_results(result)) == d + 1
            end
        end
    end

    @testset "Hyperbolic 6,6 singular roots" begin
        @polyvar x z
        y = 1
        F = System(
            [
                0.75x^4 + 1.5x^2 * y^2 - 2.5x^2 * z^2 + 0.75y^4 -
                    2.5y^2 * z^2 + 0.75z^4,
                10x^2 * z + 10y^2 * z - 6z^3,
            ],
        )
        result = solve(F, TotalDegree(; seed = UInt32(1), show_progress = false))

        @test count(is_success, path_results(result)) == 12
        @test count(path -> winding_number(path) == 3, path_results(result)) == 12
        @test nresults(result) == 2
        @test nsingular(result) == 2
        @test all(path -> multiplicity(path) == 6, singular(result))
    end

    @testset "singular and nonsingular roots are partitioned correctly" begin
        @polyvar x y
        z = 1
        F = System(
            [
                x^2 + 2y^2 + 2im * y * z,
                (18 + 3im) * x * y + 7im * y^2 - (3 - 18im) * x * z -
                    14y * z - 7im * z^2,
            ],
        )
        result = solve(
            F,
            TotalDegree(; seed = UInt32(12345), show_progress = false),
        )
        @test nresults(result) == 2
        @test nsingular(result) == 1
        @test nnonsingular(result) == 1
    end

    @testset "cyclic 7 has 924 solutions" begin
        polys, vars, _ = cyclic_system(7)
        F = System(polys; variables = vars)

        td = solve(F, TotalDegree(; seed = UInt32(1), show_progress = false))
        @test ntracked(td) == 5040
        @test nsolutions(td) == 924

        @test mixed_volume(F) == 924
        ph = solve(F, Polyhedral(; seed = UInt32(1), show_progress = false))
        @test ntracked(ph) == 924
        @test nsolutions(ph) == 924
    end
end
