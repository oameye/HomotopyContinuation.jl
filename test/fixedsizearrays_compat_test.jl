using Test
using LinearAlgebra: LinearAlgebra
using FixedSizeArrays: FixedSizeVector, FixedSizeMatrix

const LA = LinearAlgebra
const FSVec{T} = FixedSizeVector{T}
const FSMat{T} = FixedSizeMatrix{T}

@testset "FixedSizeArrays LAPACK compatibility" begin
    n = 4

    @testset "strides" begin
        A = FSMat{ComplexF64}(rand(ComplexF64, n, n))
        @test strides(A) == (1, n)
        @test stride(A, 1) == 1
        @test stride(A, 2) == n
    end

    @testset "lu! on FSMat" begin
        A = FSMat{ComplexF64}(rand(ComplexF64, n, n) + 5.0 * LA.I)
        A_orig = copy(A)
        F = LA.lu!(A)
        @test F isa LA.LU
        # Factors stored in FSMat, ipiv in FSVec{Int64}
        @test F.factors isa FSMat{ComplexF64}
        @test eltype(F.ipiv) == Int64
        b = rand(ComplexF64, n)
        x = F \ b
        @test LA.norm(Matrix(A_orig) * x - b) < 1.0e-12
    end

    @testset "ldiv! with FSVec" begin
        A = FSMat{ComplexF64}(rand(ComplexF64, n, n) + 5.0 * LA.I)
        A_orig = copy(A)
        F = LA.lu!(A)
        b = FSVec{ComplexF64}(rand(ComplexF64, n))
        x = FSVec{ComplexF64}(zeros(ComplexF64, n))
        LA.ldiv!(x, F, b)
        @test LA.norm(Matrix(A_orig) * Vector(x) - Vector(b)) < 1.0e-12
    end

    @testset "mul! with FSMat and FSVec" begin
        A = FSMat{ComplexF64}(rand(ComplexF64, n, n))
        x = FSVec{ComplexF64}(rand(ComplexF64, n))
        y = FSVec{ComplexF64}(zeros(ComplexF64, n))
        LA.mul!(y, A, x)
        @test Vector(y) ≈ Matrix(A) * Vector(x)
    end

    @testset "qr! on FSMat returns QRCompactWY (not QR)" begin
        # qr! on FSMat goes through LAPACK and returns QRCompactWY, not QR.
        # Our custom QR (v2-style qrfactUnblocked!) operates on Matrix{ComplexF64}.
        # For overdetermined systems, we copy FSMat→Matrix for QR factorization.
        m, k = 6, 4
        A = FSMat{ComplexF64}(rand(ComplexF64, m, k))
        A_copy = copy(A)
        F = LA.qr!(A_copy)
        @test F isa LA.QRCompactWY  # NOT LA.QR

        # Unblocked QR on Matrix works and gives LA.QR
        A_mat = Matrix{ComplexF64}(A)
        F2 = LA.qrfactUnblocked!(A_mat)
        @test F2 isa LA.QR
    end

    @testset "copyto! between FSMat and Matrix" begin
        A = FSMat{ComplexF64}(rand(ComplexF64, n, n))
        B = Matrix{ComplexF64}(undef, n, n)
        copyto!(B, A)
        @test B == Matrix(A)
        C = FSMat{ComplexF64}(zeros(ComplexF64, n, n))
        copyto!(C, B)
        @test Matrix(C) == B
    end
end
