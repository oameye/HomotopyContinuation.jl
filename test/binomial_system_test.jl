using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: BinomialSystemSolver, _hnf!, _hnf_big!, solve_binomial!
using LinearAlgebra: det

@testset "BinomialSystemSolver" begin
    @testset "HNF: 2×2 identity" begin
        A = Int32[1 0; 0 1]
        H = zeros(Int64, 2, 2)
        U = zeros(Int64, 2, 2)
        _hnf!(H, U, A)

        # HNF of identity is identity
        @test H == [1 0; 0 1]
        # A * U = H
        @test Int64.(A) * U == H
    end

    @testset "HNF: 2×2 general" begin
        A = Int32[3 1; 2 1]
        H = zeros(Int64, 2, 2)
        U = zeros(Int64, 2, 2)
        _hnf!(H, U, A)

        # H should be lower triangular with positive diagonal
        @test H[1, 2] == 0
        @test H[1, 1] > 0
        @test H[2, 2] > 0
        # A * U = H
        @test Int64.(A) * U == H
        # det(H) = det(A) (up to sign)
        @test abs(H[1, 1] * H[2, 2]) == abs(Int64(A[1, 1]) * A[2, 2] - Int64(A[1, 2]) * A[2, 1])
    end

    @testset "HNF: 3×3" begin
        A = Int32[2 0 1; 1 3 0; 0 1 2]
        H = zeros(Int64, 3, 3)
        U = zeros(Int64, 3, 3)
        _hnf!(H, U, A)

        # Lower triangular
        @test H[1, 2] == 0 && H[1, 3] == 0 && H[2, 3] == 0
        # Positive diagonal
        @test all(H[i, i] > 0 for i in 1:3)
        # A * U = H
        @test Int64.(A) * U == H
        # |det U| = 1 (unimodular)
        @test abs(round(Int, det(Float64.(U)))) == 1
    end

    @testset "HNF: 1×1" begin
        A = Int32[-3;;]
        H = zeros(Int64, 1, 1)
        U = zeros(Int64, 1, 1)
        _hnf!(H, U, A)

        @test H[1, 1] == 3  # positive
        @test Int64.(A) * U == H
    end

    @testset "BinomialSystemSolver: construction" begin
        BSS = BinomialSystemSolver(3; max_d_hat = 6)
        @test size(BSS.A) == (3, 3)
        @test size(BSS.H) == (3, 3)
        @test size(BSS.unit_roots_table) == (3, 6)
    end

    # Unimodular (det = 1) matrix with entries up to ~6e6 whose Kannan-Bachem HNF
    # overflows Int64 intermediates. Found by composing random elementary matrices.
    A_overflow = Int32[
        105561  5908632   -545  -58    -4
        -35    -1959      0    0     0
        50     2800  -1149    0  -200
        -1820  -101872      9    1     0
        -42    -2352    966    0   169
    ]

    @testset "HNF: Int64 overflow throws, BigInt fallback succeeds" begin
        H = zeros(Int64, 5, 5)
        U = zeros(Int64, 5, 5)
        @test_throws OverflowError _hnf!(H, U, A_overflow)

        H_big = [big(0) for _ in 1:5, _ in 1:5]
        U_big = [big(0) for _ in 1:5, _ in 1:5]
        _hnf_big!(H_big, U_big, A_overflow)
        # A * U = H, lower triangular, positive diagonal
        @test big.(A_overflow) * U_big == H_big
        @test all(iszero(H_big[i, j]) for j in 2:5 for i in 1:(j - 1))
        @test all(H_big[i, i] > 0 for i in 1:5)
    end

    @testset "solve_binomial! direct entry: x^A = b" begin
        A = Int32[0 1 1 0 -1; 0 0 0 1 -1; -1 0 -1 0 -1; 0 -1 0 -1 -1; 1 0 0 0 -1]
        b = [
            0.9053223983046926 + 0.4247250347316951im,
            0.7000429487508004 - 0.7141007421255663im,
            0.018811552539476483 - 0.9998230470893611im,
            -0.9983533473373446 + 0.057363698105327716im,
            0.9999409355072137 - 0.010868555421874721im,
        ]
        d_hat = abs(round(Int, det(Float64.(A))))
        BSS = BinomialSystemSolver(5; max_d_hat = d_hat)
        X = zeros(ComplexF64, 5, d_hat)
        d = solve_binomial!(X, BSS, A, b)
        @test d == d_hat
        for k in 1:d
            x = X[:, k]
            @test maximum(
                abs.([prod(x .^ a) for a in eachcol(A)] - b) ./ abs.(b)
            ) < 1.0e-12
        end
    end

    @testset "solve_binomial! survives Int64 HNF overflow (BigInt fallback)" begin
        # det(A_overflow) == 1, so the binomial system has exactly one solution.
        # Exactly unit-modulus rhs so the exponents up to ~6e6 keep |x| = 1
        # and the residual check stays well-conditioned.
        b = [
            -0.8047980327963311 - 0.5935487565542998im,
            0.9375177495224977 - 0.34793745031294243im,
            -0.20150673758952237 - 0.9794871284024244im,
            0.05675209803822075 + 0.998388300897131im,
            -0.9438232385317978 - 0.3304507442983735im,
        ]
        BSS = BinomialSystemSolver(5; max_d_hat = 1)
        X = zeros(ComplexF64, 5, 1)
        d = solve_binomial!(X, BSS, A_overflow, b)
        @test d == 1
        x = X[:, 1]
        @test maximum(
            abs.([prod(big.(x) .^ a) for a in eachcol(A_overflow)] - b) ./ abs.(b)
        ) < 1.0e-8
    end

    @testset "BinomialSystemSolver: end-to-end via polyhedral solve" begin
        # The best test for solve_binomial! is that polyhedral solve finds all solutions.
        # This is already tested in solve_test.jl (polyhedral katsura systems).
        # Here we just verify the solver can be constructed for various sizes.
        for n in [2, 3, 4, 5]
            BSS = BinomialSystemSolver(n; max_d_hat = 100)
            @test size(BSS.A) == (n, n)
        end
    end
end
