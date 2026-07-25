using Test, Random
using HomotopyContinuationNext: System, AffineChartSystem, AffineChartHomotopy,
    StraightLineHomotopy, HomotopyEvaluator, on_affine_chart, on_chart!,
    evaluate!, evaluate_and_jacobian!, taylor!,
    TaylorVector, ComplexDF64, FSVec, FSMat, nparameters, vectors
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: MultivariatePolynomials as MP
using LinearAlgebra: norm

# The chart row v'x - 1 is appended below the wrapped system's rows.
@testset "AffineChartSystem" begin
    Random.seed!(0x000c4a27)
    @polyvar w[1:3] a
    polys = [w[1]^2 + w[2]^2 - a * w[3]^2, w[1] * w[2] - w[3]^2]
    F = System(polys; variables = w, parameters = [a])
    chart = randn(ComplexF64, 3)
    S = AffineChartSystem(F.evaluator, chart)

    @test size(S) == (3, 3)
    @test nparameters(S) == 1

    xvals = randn(ComplexF64, 3)
    pvals = ComplexF64[1.7 - 0.4im]
    xf = FSVec{ComplexF64}(xvals)
    pf = FSVec{ComplexF64}(pvals)

    truth = ComplexF64[
        [f(w => xvals, [a] => pvals) for f in polys]...,
        sum(chart .* xvals) - 1,
    ]

    @testset "evaluate!" begin
        u = FSVec{ComplexF64}(zeros(ComplexF64, 3))
        evaluate!(u, S, xf, pf)
        @test u ≈ truth rtol = 1.0e-12

        u_df64 = FSVec{ComplexDF64}(zeros(ComplexDF64, 3))
        evaluate!(u_df64, S, FSVec{ComplexDF64}(ComplexDF64.(xvals)), pf)
        @test ComplexF64.(Vector(u_df64)) ≈ truth rtol = 1.0e-12
    end

    @testset "evaluate_and_jacobian!" begin
        u = FSVec{ComplexF64}(zeros(ComplexF64, 3))
        U = FSMat{ComplexF64}(zeros(ComplexF64, 3, 3))
        evaluate_and_jacobian!(u, U, S, xf, pf)
        @test u ≈ truth rtol = 1.0e-12
        for j in 1:3
            for i in 1:2
                dp = MP.differentiate(polys[i], w[j])
                @test U[i, j] ≈ dp(w => xvals, [a] => pvals) rtol = 1.0e-12
            end
            # The chart row is linear, so its gradient is the chart itself.
            @test U[3, j] == chart[j]
        end
    end

    @testset "taylor!: wrapped rows" begin
        @polyvar λ
        X = randn(ComplexF64, 4, 3)
        for K in 1:3
            tx = TaylorVector{K + 1, ComplexF64}(3)
            tx.data .= X[1:(K + 1), :]
            u = FSVec{ComplexF64}(zeros(ComplexF64, 3))
            taylor!(u, Val(K), S, tx, pf)

            series = [sum(X[k + 1, i] * λ^k for k in 0:K) for i in 1:3]
            for i in 1:2
                g = polys[i](w => series, [a] => pvals)
                @test u[i] ≈ ComplexF64(MP.coefficient(g, λ^K)) rtol = 1.0e-10
            end
        end
    end

    @testset "taylor!: chart row" begin
        X = randn(ComplexF64, 4, 3)
        for K in 1:3
            tx = TaylorVector{K + 1, ComplexF64}(3)
            tx.data .= X[1:(K + 1), :]
            u = FSVec{ComplexF64}(zeros(ComplexF64, 3))
            taylor!(u, Val(K), S, tx, pf)
            # The order-K coefficient of v'x(t) - 1 is v'x_K, which must be
            # reported even when the top row of `tx` is nonzero.
            @test u[3] ≈ sum(chart .* X[K + 1, :]) rtol = 1.0e-10
        end
        # A zeroed top row is the only case in which the chart row vanishes.
        tx = TaylorVector{4, ComplexF64}(3)
        tx.data .= X
        vectors(tx)[4] .= 0
        u = FSVec{ComplexF64}(zeros(ComplexF64, 3))
        taylor!(u, Val(3), S, tx, pf)
        @test u[3] == 0
    end

    @testset "on_affine_chart draws a random chart" begin
        G = System([w[1]^2 - w[2] * w[3]]; variables = w)
        S2 = on_affine_chart(G)
        @test S2 isa AffineChartSystem
        @test size(S2) == (2, 3)
        @test length(S2.chart) == 3
        @test S2.chart != on_affine_chart(G).chart
    end
end

@testset "AffineChartHomotopy" begin
    Random.seed!(0x000c4a28)
    @polyvar w[1:3]
    start = [w[1]^2 - w[3]^2, w[2]^2 - w[3]^2]
    target = [w[1]^2 + w[2]^2 - w[3]^2, w[1] * w[2] - 2 * w[3]^2]
    G = System(start; variables = w)
    F = System(target; variables = w)
    γ = cis(2π * 0.17)
    H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(γ))
    chart = randn(ComplexF64, 3)
    HC_chart = AffineChartHomotopy(H, chart)
    He = HomotopyEvaluator(HC_chart)

    @test size(HC_chart) == (3, 3)

    x = randn(ComplexF64, 3)
    on_chart!(x, HC_chart)
    @test sum(chart .* x) ≈ 1.0 + 0im atol = 1.0e-12

    t = 0.41 - 0.13im
    u = FSVec{ComplexF64}(zeros(ComplexF64, 3))
    evaluate!(u, He, FSVec{ComplexF64}(x), ComplexF64(t))
    expected = ComplexF64[
        (
            γ * t .* [f(w => x) for f in start] .+
                (1 - t) .* [f(w => x) for f in target]
        )...,
        sum(chart .* x) - 1,
    ]
    @test u ≈ expected rtol = 1.0e-12

    U = FSMat{ComplexF64}(zeros(ComplexF64, 3, 3))
    evaluate_and_jacobian!(u, U, He, FSVec{ComplexF64}(x), ComplexF64(t))
    @test u ≈ expected rtol = 1.0e-12
    for j in 1:3
        @test U[3, j] == chart[j]
    end

    # A point orthogonal to the chart normal cannot be normalized.
    orth = [chart[2], -chart[1], 0.0 + 0.0im]
    @test norm(sum(chart .* orth)) < 1.0e-15
    @test_throws ArgumentError on_chart!(orth, HC_chart)
end
