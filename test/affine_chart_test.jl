using Test, Random
using HomotopyContinuation
using HomotopyContinuation: TaylorVector, ComplexDF64
using DynamicPolynomials: @polyvar, differentiate
using MultivariatePolynomials: coefficient

@static if VERSION < v"1.11"
    buffer(A::AbstractArray) = collect(A)
else
    using FixedSizeArrays: FixedSizeArrayDefault
    buffer(A::AbstractArray) = FixedSizeArrayDefault(A)
end

include("cauchy_oracle.jl")

@testset "public affine-chart behavior" begin
    @testset "on_affine_chart preserves the system and appends one affine row" begin
        Random.seed!(0x00aff1ce)
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
            Continuation(; seed = UInt32(0x00aff1ce), show_progress = false),
            Serial(),
        )
        @test nsolutions(result) == 1
        target = only(solutions(result))
        @test target[1] / target[3] ≈ 2 atol = 1.0e-8
        @test target[2] / target[3] ≈ 1 atol = 1.0e-8
        @test sum(chart .* target) ≈ 1 atol = 1.0e-10
        @test maximum(abs, evaluate(F, target, [4.0])) < 1.0e-8
    end

    @testset "affine-chart system: evaluate!, Jacobian and taylor! against symbolic truth" begin
        Random.seed!(0x000c4a27)
        @polyvar w[1:3] a λ
        polys = [w[1]^2 + w[2]^2 - a * w[3]^2, w[1] * w[2] - w[3]^2]
        S = on_affine_chart(System(polys; variables = w, parameters = [a]))
        xv = randn(ComplexF64, 3)
        pv = ComplexF64[1.7 - 0.4im]
        # The chart row v'x - 1 is linear, so its gradient is the chart v.
        v = jacobian(S, randn(ComplexF64, 3), pv)[end, :]
        truth = ComplexF64[[f(w => xv, [a] => pv) for f in polys]..., sum(v .* xv) - 1]

        u = buffer(zeros(ComplexF64, 3))
        evaluate!(u, S, buffer(xv), buffer(pv))
        @test u ≈ truth rtol = 1.0e-12
        evaluate!(u, S, buffer(ComplexDF64.(xv)), buffer(pv))
        @test u ≈ truth rtol = 1.0e-12

        U = buffer(zeros(ComplexF64, 3, 3))
        evaluate_and_jacobian!(u, U, S, buffer(xv), buffer(pv))
        @test u ≈ truth rtol = 1.0e-12
        @test U[1:2, :] ≈ [ComplexF64(differentiate(f, wj)(w => xv, [a] => pv)) for f in polys, wj in w] rtol = 1.0e-12
        @test U[3, :] == v

        # The order-K coefficient of F(x₀ + x₁λ + … + x_Kλ^K), including the
        # chart row's v'x_K.
        X = randn(ComplexF64, 4, 3)
        for K in 1:3
            tx = TaylorVector{K + 1, ComplexF64}(3)
            for i in 1:3
                tx[i] = Tuple(X[1:(K + 1), i])
            end
            taylor!(u, Val(K), S, tx, buffer(pv))
            path = [sum(X[k + 1, i] * λ^k for k in 0:K) for i in 1:3]
            for i in 1:2
                @test u[i] ≈ ComplexF64(coefficient(polys[i](w => path, [a] => pv), λ^K)) rtol = 1.0e-10
            end
            @test u[3] ≈ sum(v .* X[K + 1, :]) rtol = 1.0e-10
        end
    end

    @testset "affine-chart homotopy: evaluate!, Jacobian and taylor! against its parts" begin
        Random.seed!(0x000c4a28)
        @polyvar w[1:3] q
        polys = [w[1]^2 + w[2]^2 - q * w[3]^2, w[1] * w[2] - 2q * w[3]^2]
        F = System(polys; variables = w, parameters = [q])
        pstart = ComplexF64[0.5 + 0.1im]
        ptarget = ComplexF64[1.3 - 0.7im]
        chart = randn(ComplexF64, 3)
        H = AffineChartHomotopy(ParameterHomotopy(F, pstart, ptarget), chart)
        @test size(H) == (3, 3)

        value(x, t) = (out = buffer(zeros(ComplexF64, 3)); evaluate!(out, H, buffer(x), t); Vector(out))
        truth(x, t) = ComplexF64[
            [f(w => x, [q] => t .* pstart .+ (1 - t) .* ptarget) for f in polys]...,
            sum(chart .* x) - 1,
        ]
        xv = randn(ComplexF64, 3)
        xv ./= sum(chart .* xv)
        t = 0.41 - 0.13im
        @test value(xv, t) ≈ truth(xv, t) rtol = 1.0e-12
        @test value(xv, t)[3] ≈ 0 atol = 1.0e-14

        u = buffer(zeros(ComplexF64, 3))
        U = buffer(zeros(ComplexF64, 3, 3))
        evaluate_and_jacobian!(u, U, H, buffer(xv), t)
        @test u ≈ truth(xv, t) rtol = 1.0e-12
        @test U[1:2, :] ≈ [
            ComplexF64(differentiate(f, wj)(w => xv, [q] => t .* pstart .+ (1 - t) .* ptarget))
                for f in polys, wj in w
        ] rtol = 1.0e-12
        @test U[3, :] == chart

        t0 = complex(0.37)
        taylor!(u, Val(1), H, buffer(xv), t0)
        @test Vector(u) ≈ cauchy_coefficients(λ -> value(xv, t0 + λ), 1)[2] rtol = 1.0e-9
        X = randn(ComplexF64, 3, 3)
        for K in 2:3
            tx = TaylorVector{K + 1, ComplexF64}(3)
            for i in 1:3
                tx[i] = (X[1:K, i]..., zero(ComplexF64))
            end
            taylor!(u, Val(K), H, tx, t0)
            expected = cauchy_coefficients(K) do λ
                value([sum(X[k + 1, i] * λ^k for k in 0:(K - 1)) for i in 1:3], t0 + λ)
            end[K + 1]
            @test Vector(u) ≈ expected rtol = 1.0e-8
        end
    end
end
