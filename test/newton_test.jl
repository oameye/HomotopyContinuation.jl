using Test
using HomotopyContinuation

@testset "standalone Newton scenarios" begin
    @polyvar x y
    F = System([x^2 + y^2 - 1, x - y])
    root = 1 / sqrt(2)

    @testset "two basins converge to the two circle-diagonal roots" begin
        positive = newton(F, [root + 0.05, root - 0.05])
        negative = newton(F, [-root - 0.05, -root + 0.05])

        @test is_success(positive)
        @test is_success(negative)
        @test positive.residual < 1.0e-12
        @test negative.residual < 1.0e-12
        @test solution(positive) ≈ ComplexF64[root, root] atol = 1.0e-8
        @test solution(negative) ≈ ComplexF64[-root, -root] atol = 1.0e-8
    end

    @testset "extended precision reaches a tighter residual" begin
        ordinary = newton(F, [root + 0.05, root - 0.05])
        extended = newton(F, [root + 0.05, root - 0.05]; extended_precision = true)
        @test is_success(extended)
        @test extended.residual < 1.0e-14
        @test extended.residual <= ordinary.residual
    end

    @testset "iteration and first-step guards report their public outcomes" begin
        limited = newton(F, [root + 0.05, root - 0.05]; max_iters = 1)
        @test !is_success(limited)
        @test limited.return_code == NewtonReturnCode.NEWTON_MAX_ITERS
        @test limited.iters == 1

        rejected = newton(
            F, [root + 0.05, root - 0.05]; max_abs_norm_first_update = 1.0e-6,
        )
        @test !is_success(rejected)
        @test rejected.return_code == NewtonReturnCode.NEWTON_REJECTED
    end

    @testset "consistent overdetermined problem converges by least squares" begin
        G = System([x^2 + y^2 - 1, x - y, x * y - 0.5])
        result = newton(G, [root + 0.02, root - 0.02])
        @test is_success(result)
        @test result.residual < 1.0e-10
        @test solution(result) ≈ ComplexF64[root, root] atol = 1.0e-8
    end

    @testset "underdetermined Newton lands on the solution variety" begin
        C = System([x^2 + y^2 - 1])
        result = newton(C, [1.1, 0.2])
        @test is_success(result)
        point = solution(result)
        @test abs(point[1]^2 + point[2]^2 - 1) < 1.0e-10

        extended = newton(C, [1.1, 0.2]; extended_precision = true)
        @test is_success(extended)
        @test extended.residual < 1.0e-14
    end

    @testset "reusable public NewtonCache preserves independent solves" begin
        cache = NewtonCache(F)
        positive = newton(F, [root + 0.05, root - 0.05]; cache = cache)
        negative = newton(F, [-root - 0.05, -root + 0.05]; cache = cache)
        @test is_success(positive)
        @test is_success(negative)
        @test solution(positive) ≈ ComplexF64[root, root] atol = 1.0e-8
        @test solution(negative) ≈ ComplexF64[-root, -root] atol = 1.0e-8
    end
end
