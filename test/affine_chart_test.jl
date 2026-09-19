using Test, Random
using HomotopyContinuation
using DynamicPolynomials: @polyvar

@testset "public affine-chart behavior" begin
    @testset "on_affine_chart preserves the system and appends one affine row" begin
        Random.seed!(0xaff1ce)
        @polyvar x[1:3] p
        F = System(
            [x[1]^2 + x[2]^2 - p * x[3]^2, x[2] - x[3]];
            variables = x, parameters = [p],
        )
        A = on_affine_chart(F)

        @test size(A) == (3, 3)
        @test nparameters(A) == 1

        point = ComplexF64[0.7 + 0.1im, -0.3 + 0.2im, 1.2 - 0.4im]
        params = ComplexF64[1.7]
        value = evaluate(A, point, params)
        J = jacobian(A, point, params)
        @test value[1:2] ≈ evaluate(F, point, params)
        @test J[1:2, :] ≈ jacobian(F, point, params)

        other = ComplexF64[-0.2 + 0.4im, 0.6 - 0.3im, 0.9 + 0.2im]
        Jother = jacobian(A, other, params)
        @test J[end, :] ≈ Jother[end, :] atol = 1.0e-14
        @test evaluate(A, other, params)[end] - value[end] ≈
            sum(J[end, :] .* (other .- point)) atol = 1.0e-13

        B = on_affine_chart(F)
        @test jacobian(B, point, params)[end, :] != J[end, :]
    end

    @testset "an affine-chart ParameterHomotopy tracks a known projective branch" begin
        @polyvar x[1:3] q
        F = System(
            [x[1]^2 - q * x[3]^2, x[2] - x[3]];
            variables = x, parameters = [q],
        )
        H = ParameterHomotopy(F, [1.0], [4.0])
        chart = ComplexF64[1, 2, 1]
        A = on_affine_chart(H, chart)

        start = ComplexF64[1, 1, 1]
        start ./= sum(chart .* start)
        result = solve(
            A,
            [start],
            Continuation(; seed = UInt32(0xaff1ce), show_progress = false),
            Serial(),
        )
        @test nsolutions(result) == 1
        target = only(solutions(result))
        @test target[1] / target[3] ≈ 2 atol = 1.0e-8
        @test target[2] / target[3] ≈ 1 atol = 1.0e-8
        @test sum(chart .* target) ≈ 1 atol = 1.0e-10
        @test maximum(abs, evaluate(F, target, [4.0])) < 1.0e-8
    end
end
