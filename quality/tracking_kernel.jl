using Test
import HomotopyContinuation as HC
using LinearAlgebra: I

@testset "tracking numerical kernel" begin
    @testset "start Jacobian corank classification" begin
        n = 3
        A = HC.FSMat{ComplexF64}(Matrix{ComplexF64}(I, n, n))
        @test HC._start_jacobian_corank(A) == 0

        Z = HC.FSMat{ComplexF64}(zeros(ComplexF64, n, n))
        @test HC._start_jacobian_corank(Z) == n

        R1 = HC.FSMat{ComplexF64}(ComplexF64[1 2 3; 2 4 6; 3 6 9])
        @test HC._start_jacobian_corank(R1) == 2

        N = HC.FSMat{ComplexF64}(fill(ComplexF64(NaN), n, n))
        @test HC._start_jacobian_corank(N) == 0
    end

    @testset "rejected-step update respects beta_a" begin
        seg = HC.SegmentStepper(ComplexF64(1.0), ComplexF64(0.0))
        HC.propose_step!(seg, 0.8)

        state = HC.TrackerState(1, 1, seg)
        pred = HC.Predictor(1, 1)
        opts = HC.TrackerOptions(; a = 0.125, β_a = 1.6)
        consts = HC.TrackerConstants(opts)
        result = HC.NewtonCorrectorResult(
            HC.NewtonCode.NEWT_TERMINATED,
            1.0,
            3,
            1.0,
            0.04,
            NaN,
            0.0,
        )

        h(a) = 2a * (sqrt(4a^2 + 1) - 2a)
        p = pred.order
        Θ_j = sqrt(result.θ)
        expected = (
            (sqrt(1 + 2 * h(0.5 * opts.β_a * opts.a)) - 1) /
                (sqrt(1 + 2 * h(Θ_j)) - 1)
        )^(1 / p) * 0.8
        old_expected = (
            (sqrt(1 + 2 * h(0.5 * opts.a)) - 1) /
                (sqrt(1 + 2 * h(Θ_j)) - 1)
        )^(1 / p) * 0.8

        HC._update_stepsize!(state, result, pred, opts, consts)

        @test abs(state.segment.Δs) ≈ expected rtol = 1.0e-12
        @test !isapprox(abs(state.segment.Δs), old_expected; rtol = 1.0e-6)
    end

    @testset "predictor trust region stays finite through zero Taylor data" begin
        pred = HC.Predictor(1, 1)
        fill!(pred.tx3.data, 0)
        pred.tx3.data[2, 1] = 1
        pred.tx_norm = (0.0, 1.0, 0.0, 0.0)
        pred.local_error = NaN

        HC._compute_trust_region!(pred)

        @test pred.trust_region == 1.0
        @test pred.local_error == 1.0
        @test isfinite(pred.trust_region)
        @test pred.trust_region > 0

        pred2 = HC.Predictor(1, 1)
        fill!(pred2.tx3.data, 0)
        pred2.tx3.data[2, 1] = 1
        pred2.tx3.data[3, 1] = 2
        pred2.tx3.data[4, 1] = 4
        pred2.tx_norm = (0.0, 1.0, 2.0, 4.0)
        pred2.local_error = NaN

        HC._compute_trust_region!(pred2)

        @test pred2.trust_region ≈ 0.5
        @test isfinite(pred2.local_error)
    end
end
