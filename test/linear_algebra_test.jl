using Test
using LinearAlgebra: LinearAlgebra
using FixedSizeArrays: FixedSizeVector, FixedSizeMatrix
using HomotopyContinuationNext:
    MatrixWorkspace,
    updated!,
    factorize!,
    skeel_row_scaling!,
    apply_row_scaling!,
    mixed_precision_iterative_refinement!,
    fixed_precision_iterative_refinement!,
    residual!,
    inverse_inf_norm_est

const LA = LinearAlgebra
const FSVec{T} = FixedSizeVector{T}
const FSMat{T} = FixedSizeMatrix{T}

@testset "MatrixWorkspace" begin
    @testset "construction" begin
        WS = MatrixWorkspace(4, 4)
        @test size(WS) == (4, 4)

        WS2 = MatrixWorkspace(6, 4)
        @test size(WS2) == (6, 4)

        @test_throws ArgumentError MatrixWorkspace(2, 5)
    end

    @testset "AbstractMatrix interface" begin
        WS = MatrixWorkspace(3, 3)
        WS[1, 1] = 1.0 + 0.0im
        @test WS[1, 1] == 1.0 + 0.0im
        @test WS[1] == 1.0 + 0.0im

        WS[2] = 2.0 + 0.0im
        @test WS[2] == 2.0 + 0.0im
    end

    @testset "LU solve (square)" begin
        n = 4
        A_data = rand(ComplexF64, n, n) + 5.0 * LA.I
        b_data = rand(ComplexF64, n)

        WS = MatrixWorkspace(n, n)
        copyto!(WS.A, A_data)
        updated!(WS)

        x = FSVec{ComplexF64}(zeros(ComplexF64, n))
        b = FSVec{ComplexF64}(b_data)
        LA.ldiv!(x, WS, b)

        @test Matrix(WS.A) * Vector(x) ≈ b_data atol = 1.0e-10
    end

    @testset "1x1 system" begin
        WS = MatrixWorkspace(1, 1)
        WS[1, 1] = 3.0 + 1.0im
        updated!(WS)
        x = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        b = FSVec{ComplexF64}([2.0 + 0.5im])
        LA.ldiv!(x, WS, b)
        @test x[1] ≈ (2.0 + 0.5im) / (3.0 + 1.0im)
    end

    @testset "repeated solves" begin
        n = 3
        WS = MatrixWorkspace(n, n)

        for _ in 1:5
            A_data = rand(ComplexF64, n, n) + 3.0 * LA.I
            b_data = rand(ComplexF64, n)
            copyto!(WS.A, A_data)
            updated!(WS)
            x = FSVec{ComplexF64}(zeros(ComplexF64, n))
            LA.ldiv!(x, WS, FSVec{ComplexF64}(b_data))
            @test Matrix(A_data) * Vector(x) ≈ b_data atol = 1.0e-10
        end
    end

    @testset "overdetermined (QR)" begin
        m, n = 6, 4
        A_data = rand(ComplexF64, m, n)
        x_true = rand(ComplexF64, n)
        b_data = A_data * x_true  # consistent system

        WS = MatrixWorkspace(m, n)
        copyto!(WS.A, A_data)
        updated!(WS)

        x = FSVec{ComplexF64}(zeros(ComplexF64, n))
        LA.ldiv!(x, WS, FSVec{ComplexF64}(b_data))

        @test LA.norm(A_data * Vector(x) - b_data) < 1.0e-10
    end

    @testset "factorize! called automatically" begin
        n = 3
        WS = MatrixWorkspace(n, n)
        A_data = rand(ComplexF64, n, n) + 3.0 * LA.I
        copyto!(WS.A, A_data)
        updated!(WS)
        @test WS.factorized == false

        x = FSVec{ComplexF64}(zeros(ComplexF64, n))
        b = FSVec{ComplexF64}(rand(ComplexF64, n))
        LA.ldiv!(x, WS, b)
        @test WS.factorized == true
    end

    @testset "explicit factorize!" begin
        n = 3
        WS = MatrixWorkspace(n, n)
        A_data = rand(ComplexF64, n, n) + 3.0 * LA.I
        copyto!(WS.A, A_data)
        updated!(WS)
        factorize!(WS)
        @test WS.factorized == true
    end
end

@testset "Row scaling" begin
    n = 4
    WS = MatrixWorkspace(n, n)
    A_data = rand(ComplexF64, n, n) + 5.0 * LA.I
    copyto!(WS.A, A_data)
    updated!(WS)
    c = FSVec{Float64}(ones(n))
    skeel_row_scaling!(WS, c)
    @test all(WS.row_scaling .> 0.0)
    @test all(isfinite.(WS.row_scaling))
end

@testset "Residual computation" begin
    n = 3
    A = FSMat{ComplexF64}(rand(ComplexF64, n, n))
    x = FSVec{ComplexF64}(rand(ComplexF64, n))
    b = FSVec{ComplexF64}(Matrix(A) * Vector(x))
    r = FSVec{ComplexF64}(zeros(ComplexF64, n))
    residual!(r, A, x, b)
    @test LA.norm(r) < 1.0e-12
end

@testset "Iterative refinement" begin
    n = 4
    A_data = rand(ComplexF64, n, n) + 5.0 * LA.I
    x_true = rand(ComplexF64, n)
    b_data = A_data * x_true

    WS = MatrixWorkspace(n, n)
    copyto!(WS.A, A_data)
    updated!(WS)

    x = FSVec{ComplexF64}(zeros(ComplexF64, n))
    LA.ldiv!(x, WS, FSVec{ComplexF64}(copy(b_data)))
    err_before = LA.norm(Vector(x) - x_true)

    mixed_precision_iterative_refinement!(x, WS, FSVec{ComplexF64}(b_data))
    err_after = LA.norm(Vector(x) - x_true)
    @test err_after <= err_before + 1.0e-14
end

@testset "Condition number estimation" begin
    n = 4
    A_data = rand(ComplexF64, n, n) + 5.0 * LA.I
    WS = MatrixWorkspace(n, n)
    copyto!(WS.A, A_data)
    updated!(WS)
    factorize!(WS)

    inv_norm = inverse_inf_norm_est(WS)
    @test inv_norm > 0.0
    @test isfinite(inv_norm)

    cond = LA.cond(WS)
    @test 1.0 ≤ cond < 1.0e6
end
