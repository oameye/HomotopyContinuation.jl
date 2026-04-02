using Test
using LinearAlgebra: LinearAlgebra, diagm, opnorm
using FixedSizeArrays: FixedSizeArray
using HomotopyContinuationNext:
    MatrixWorkspace, updated!, factorize!,
    skeel_row_scaling!, apply_row_scaling!,
    mixed_precision_iterative_refinement!,
    residual!, inverse_inf_norm_est,
    Jacobian, WeightedNorm, init!

const LA = LinearAlgebra
const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

@testset "MatrixWorkspace" begin
    @testset "construction" begin
        @test size(MatrixWorkspace(4, 4)) == (4, 4)
        @test size(MatrixWorkspace(6, 4)) == (6, 4)
        @test_throws ArgumentError MatrixWorkspace(2, 5)
    end

    @testset "1x1 system" begin
        WS = MatrixWorkspace(1, 1)
        WS[1, 1] = 3.0 + 1.0im
        updated!(WS)
        x = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        LA.ldiv!(x, WS, FSVec{ComplexF64}([2.0 + 0.5im]))
        @test x[1] ≈ (2.0 + 0.5im) / (3.0 + 1.0im)
    end

    @testset "LU solve (square, repeated)" begin
        n = 3
        WS = MatrixWorkspace(n, n)
        for _ in 1:2
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
        b_data = A_data * x_true

        WS = MatrixWorkspace(m, n)
        copyto!(WS.A, A_data)
        updated!(WS)
        x = FSVec{ComplexF64}(zeros(ComplexF64, n))
        LA.ldiv!(x, WS, FSVec{ComplexF64}(b_data))
        @test LA.norm(A_data * Vector(x) - b_data) < 1.0e-10
    end
end

@testset "Row scaling" begin
    n = 4
    WS = MatrixWorkspace(n, n)
    copyto!(WS.A, rand(ComplexF64, n, n) + 5.0 * LA.I)
    updated!(WS)
    skeel_row_scaling!(WS, FSVec{Float64}(ones(n)))
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

@testset "Mixed precision iterative refinement" begin
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
    @test LA.norm(Vector(x) - x_true) <= err_before + 1.0e-14
end

@testset "Jacobian wrapper" begin
    n = 4
    A_data = rand(ComplexF64, n, n) + 5.0 * LA.I
    x_true = rand(ComplexF64, n)
    b_data = A_data * x_true

    J = Jacobian(MatrixWorkspace(n, n))
    @test J.factorizations[] == 0
    @test J.ldivs[] == 0

    copyto!(J.workspace.A, A_data)
    updated!(J)

    # Plain ldiv!
    x = FSVec{ComplexF64}(zeros(ComplexF64, n))
    LA.ldiv!(x, J, FSVec{ComplexF64}(b_data))
    @test LA.norm(Vector(x) - x_true) < 1.0e-10
    @test J.factorizations[] >= 1
    @test J.ldivs[] >= 1

    # ldiv! with WeightedNorm (Skeel scaling path)
    init!(J)
    copyto!(J.workspace.A, A_data)
    updated!(J)
    w = WeightedNorm(n)
    init!(w, FSVec{ComplexF64}(x_true))
    x = FSVec{ComplexF64}(zeros(ComplexF64, n))
    LA.ldiv!(x, J, FSVec{ComplexF64}(b_data), w)
    @test LA.norm(Vector(x) - x_true) < 1.0e-10

    # Condition number delegation
    @test 1.0 ≤ LA.cond(J) < 1.0e6

    @testset "repeated weighted solves reuse scaled factorization correctly" begin
        A_bad = ComplexF64[
            1.0e8 1.0 0.0 0.0
            1.0 1.0e-8 1.0 0.0
            0.0 1.0 1.0e6 1.0
            1.0 0.0 1.0 1.0e-6
        ]
        x_true_bad = ComplexF64[1.0e-3, -2.0, 5.0e1, -3.0e-2]
        b_bad = A_bad * x_true_bad

        init!(J)
        copyto!(J.workspace.A, A_bad)
        updated!(J)

        w_bad = WeightedNorm(n)
        @inbounds for (i, wi) in enumerate((1.0e-3, 1.0, 1.0e2, 1.0e-1))
            w_bad.weights[i] = wi
        end

        x1 = FSVec{ComplexF64}(zeros(ComplexF64, n))
        x2 = FSVec{ComplexF64}(zeros(ComplexF64, n))
        LA.ldiv!(x1, J, FSVec{ComplexF64}(copy(b_bad)), w_bad)
        LA.ldiv!(x2, J, FSVec{ComplexF64}(copy(b_bad)), w_bad)
        x_dense = A_bad \ b_bad

        @test LA.norm(Vector(x1) - x_dense) < 1.0e-7
        @test LA.norm(Vector(x2) - x_dense) < 1.0e-7
        @test LA.norm(Vector(x1) - Vector(x2)) < 1.0e-12
    end
end

@testset "Zero allocations (ldiv!)" begin
    for n in [3, 13]
        A = rand(ComplexF64, n, n) + 5.0 * LA.I
        b = FSVec{ComplexF64}(rand(ComplexF64, n))
        x = FSVec{ComplexF64}(zeros(ComplexF64, n))
        WS = MatrixWorkspace(n, n)
        copyto!(WS.A, A)
        updated!(WS)
        LA.ldiv!(x, WS, b)  # warmup
        WS.factorized = false
        LA.ldiv!(x, WS, b)  # warmup
        WS.factorized = false
        @test (@allocated LA.ldiv!(x, WS, b)) == 0
    end
end

@testset "Condition estimator vs opnorm" begin
    d_r = rand() * exp10.(range(-6; stop = 6, length = 6))
    D_R = diagm(d_r)
    d_l = rand() * exp10.(range(6; stop = -6, length = 6))
    A = randn(6, 6)

    # Right-scaled
    B = A * inv(D_R)
    WB = MatrixWorkspace(6, 6)
    copyto!(WB.A, ComplexF64.(B))
    updated!(WB)
    @test 0.1 ≤ opnorm(inv(B), Inf) / inverse_inf_norm_est(WB) ≤ 10
    @test 0.1 ≤ LA.cond(ComplexF64.(B), Inf) / LA.cond(WB) ≤ 10

    # Left-scaled
    C = inv(diagm(d_l)) * A
    WC = MatrixWorkspace(6, 6)
    copyto!(WC.A, ComplexF64.(C))
    updated!(WC)
    @test 0.1 ≤ opnorm(inv(C), Inf) / inverse_inf_norm_est(WC) ≤ 10
    @test 0.1 ≤ LA.cond(ComplexF64.(C), Inf) / LA.cond(WC) ≤ 10
end
