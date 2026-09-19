using Test
using LinearAlgebra: LinearAlgebra, diagm
using HomotopyContinuation:
    MatrixWorkspace,
    WeightedNorm,
    FSVec,
    FSMat,
    updated!,
    factorize!,
    skeel_row_scaling!,
    apply_row_scaling!,
    mixed_precision_iterative_refinement!,
    _scaled_cond

const LA = LinearAlgebra

@testset "linear algebra kernel invariants" begin
    @testset "row scaling is equivariant under global matrix scale" begin
        A = ComplexF64[1 2 3; 4 5 6; 7 8 10]
        weights = FSVec{Float64}(ones(3))
        reference = FSVec{Float64}(zeros(3))
        skeel_row_scaling!(reference, FSMat{ComplexF64}(A), weights)

        for σ in (exp2(-40.0), exp2(20.0), exp2(60.0))
            scaled = FSVec{Float64}(zeros(3))
            skeel_row_scaling!(scaled, FSMat{ComplexF64}(σ .* A), weights)
            @test collect(scaled) ≈ collect(reference) ./ σ rtol = 8eps(Float64)
        end
    end

    @testset "scaled condition estimate tracks dense condition number" begin
        A = ComplexF64[
            1.0e6 2 0
            3 4.0e-4 5
            0 6 7.0e2
        ]
        col = FSVec{Float64}([0.5, 1.0, 2.0])
        workspace = MatrixWorkspace(3, 3)
        copyto!(workspace.A, A)
        updated!(workspace)
        row = FSVec{Float64}(zeros(3))
        skeel_row_scaling!(row, workspace.A, col)
        factorize!(workspace)

        estimate = _scaled_cond(workspace, row, col)
        exact = LA.cond(diagm(collect(row)) * A * diagm(collect(col)), Inf)
        @test 1.0 <= estimate <= exact * (1 + 1.0e-8)
        @test estimate >= exact / 10

        scaled_workspace = MatrixWorkspace(3, 3)
        copyto!(scaled_workspace.A, A)
        updated!(scaled_workspace)
        skeel_row_scaling!(scaled_workspace, col)
        apply_row_scaling!(scaled_workspace)
        factorize!(scaled_workspace)
        @test _scaled_cond(scaled_workspace, row, col) ≈ estimate rtol = 1.0e-10
    end

    @testset "iterative refinement does not degrade a badly scaled solve" begin
        A = ComplexF64[
            1.0e8 1.0 0.0 0.0
            1.0 1.0e-8 1.0 0.0
            0.0 1.0 1.0e6 1.0
            1.0 0.0 1.0 1.0e-6
        ]
        x_true = ComplexF64[1.0e-3, -2.0, 5.0e1, -3.0e-2]
        b = A * x_true
        workspace = MatrixWorkspace(4, 4)
        copyto!(workspace.A, A)
        updated!(workspace)
        x = FSVec{ComplexF64}(zeros(ComplexF64, 4))
        LA.ldiv!(x, workspace, FSVec{ComplexF64}(copy(b)))
        before = LA.norm(Vector(x) - x_true)

        norm = WeightedNorm(FSVec{Float64}([1.0e-3, 1.0, 1.0e2, 1.0e-1]))
        mixed_precision_iterative_refinement!(x, workspace, FSVec{ComplexF64}(b), norm)
        after = LA.norm(Vector(x) - x_true)

        @test after <= before + 1.0e-12
        @test LA.norm(A * Vector(x) - b) <= 1.0e-6
    end
end
