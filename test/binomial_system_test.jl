using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: BinomialSystemSolver, _hnf!
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
