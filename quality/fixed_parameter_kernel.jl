using Test, Random
import HomotopyContinuation as HC
using HomotopyContinuation: System, fix_parameters, evaluate!, evaluate_and_jacobian!, taylor!,
    TaylorVector, ComplexDF64, FSVec, FSMat, nparameters
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: MultivariatePolynomials as MP

@testset "Fixed-parameter evaluator kernel" begin
    Random.seed!(0x1f3a55c1)
    @polyvar x y a b
    polys = [x^2 + a * y^2 - b, x * y^3 - a * b + 2]
    pvals = ComplexF64[1.7 - 0.4im, 2.3]
    F = System(polys; variables = [x, y], parameters = [a, b])
    G = fix_parameters(F, pvals)

    empty_p = FSVec{ComplexF64}(ComplexF64[])
    xvals = randn(ComplexF64, 2)
    xf = FSVec{ComplexF64}(xvals)
    truth = ComplexF64[f([x, y] => xvals, [a, b] => pvals) for f in polys]
    bound = HC._bound_evaluator(F.evaluator, pvals)

    @test nparameters(bound) == 0
    @test size(bound) == (2, 2)

    @testset "value, DF64 and Jacobian" begin
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u, bound, xf, empty_p)
        @test u ≈ truth rtol = 1.0e-12

        u2 = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u2, bound, FSVec{ComplexDF64}(ComplexDF64.(xvals)), empty_p)
        @test u2 ≈ truth rtol = 1.0e-12

        u3 = FSVec{ComplexDF64}(zeros(ComplexDF64, 2))
        evaluate!(u3, bound, FSVec{ComplexDF64}(ComplexDF64.(xvals)), empty_p)
        @test ComplexF64.(Vector(u3)) ≈ truth rtol = 1.0e-12

        U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        evaluate_and_jacobian!(u, U, bound, xf, empty_p)
        @test u ≈ truth rtol = 1.0e-12
        for i in 1:2, j in 1:2
            dp = MP.differentiate(polys[i], [x, y][j])
            @test U[i, j] ≈ dp([x, y] => xvals, [a, b] => pvals) rtol = 1.0e-12
        end
    end

    @testset "Taylor order $K, $(P === nothing ? "scalar" : "TaylorVector") parameters" for
        K in 1:3, P in (nothing, TaylorVector)
        tx = TaylorVector{K + 1, ComplexF64}(2)
        tx.data .= randn(ComplexF64, K + 1, 2)

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        if P === nothing
            taylor!(u, Val(K), bound, tx, empty_p)
        else
            taylor!(u, Val(K), bound, tx, TaylorVector{K + 1, ComplexF64}(0))
        end

        expected = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        taylor!(expected, Val(K), G.evaluator, tx, empty_p)
        @test u ≈ expected rtol = 1.0e-10
    end
end
