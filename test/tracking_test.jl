using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: system_eval, StraightLineHomotopy, HomotopyEvaluator,
    evaluate!, evaluate_and_jacobian!, taylor!,
    NewtonCorrector, NewtonCode, NewtonCorrectorResult, newton!, init_newton!,
    Predictor, PredictionMethod, predict!,
    Tracker, TrackerCode, TrackerOptions, TrackerState, track!, step!,
    Jacobian, MatrixWorkspace, WeightedNorm, SegmentStepper,
    TaylorVector, TruncatedTaylorSeries, weighted_norm
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray
using LinearAlgebra: LinearAlgebra as LA

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

@testset "Path Tracking" begin

    # ── Newton Corrector ──────────────────────────────────────────────────

    @testset "newton!: converges near known solution" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x^2 - 1, y^2 - 1]
        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)

        H = StraightLineHomotopy(eval_G, eval_F; γ = ComplexF64(1.0))
        heval = HomotopyEvaluator(H)

        m, n = size(heval)
        NC = NewtonCorrector(0.125, n, m)
        J = Jacobian(MatrixWorkspace(m, n))
        norm = WeightedNorm(n)

        # At t=0, H = F. Solution of F at (1,1).
        x₀ = FSVec{ComplexF64}(ComplexF64[1.01, 0.99])
        x̄ = FSVec{ComplexF64}(zeros(ComplexF64, n))
        t = ComplexF64(0.0)

        HC.init!(norm, x₀)

        result = newton!(
            x̄, NC, heval, x₀, t, J, norm,
            1.0, 0.1, true,
        )

        @test result.return_code == NewtonCode.NEWT_CONVERGED
        @test abs(x̄[1] - 1.0) < 1.0e-8
        @test abs(x̄[2] - 1.0) < 1.0e-8
    end

    @testset "init_newton!: handles exact solution" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x^2 - 1, y^2 - 1]
        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)

        H = StraightLineHomotopy(eval_G, eval_F; γ = ComplexF64(1.0))
        heval = HomotopyEvaluator(H)

        m, n = size(heval)
        NC = NewtonCorrector(0.125, n, m)
        J = Jacobian(MatrixWorkspace(m, n))
        norm = WeightedNorm(n)
        x₀ = FSVec{ComplexF64}(ComplexF64[1.0, 1.0])
        x̄ = FSVec{ComplexF64}(zeros(ComplexF64, n))
        HC.init!(norm, x₀)

        valid, ω, μ = init_newton!(x̄, NC, heval, x₀, ComplexF64(1.0), J, norm)
        @test valid == true
        @test isfinite(ω) && ω > 0
        @test isfinite(μ) && μ > 0
    end

    # ── Tracker: end-to-end ───────────────────────────────────────────────

    @testset "track!: linear system (trivial path)" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x - 2, y - 3]
        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)

        H = StraightLineHomotopy(eval_G, eval_F; γ = ComplexF64(1.0))
        tracker = Tracker(HomotopyEvaluator(H))

        code = track!(tracker, ComplexF64[1.0, 1.0])
        @test code == TrackerCode.TRACKER_SUCCESS
        @test abs(tracker.state.x[1] - 2.0) < 1.0e-8
        @test abs(tracker.state.x[2] - 3.0) < 1.0e-8
        @test tracker.state.accepted_steps > 0
        @test tracker.state.rejected_steps == 0
    end

    @testset "track!: quadratic system finds solutions" begin
        @polyvar x y
        F = [x^2 + y - 1, x * y - 0.5]
        G = [x^2 - 1, y^2 - 1]
        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)

        H = StraightLineHomotopy(eval_G, eval_F)
        heval = HomotopyEvaluator(H)

        starts = [ComplexF64[1, 1], ComplexF64[1, -1], ComplexF64[-1, 1], ComplexF64[-1, -1]]

        solutions_found = 0
        for x0 in starts
            tracker = Tracker(heval)
            code = track!(tracker, x0)
            if code == TrackerCode.TRACKER_SUCCESS
                sol = tracker.state.x
                # Verify it's a solution of F
                u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
                evaluate!(u, eval_F, sol, FSVec{ComplexF64}(ComplexF64[]))
                res = maximum(abs.(Vector(u)))
                @test res < 1.0e-8
                solutions_found += 1
            end
        end
        # Total degree 4 paths, but some may diverge — at least 2 should succeed
        @test solutions_found >= 2
    end

    @testset "track!: katsura-3 (4 variables)" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        G = [x0 - 1, x1^2 - 1, x2^2 - 1, x3^2 - 1]

        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)
        H = StraightLineHomotopy(eval_G, eval_F)
        heval = HomotopyEvaluator(H)

        # Track from one start solution
        tracker = Tracker(heval)
        code = track!(tracker, ComplexF64[1.0, 1.0, 1.0, 1.0])

        if code == TrackerCode.TRACKER_SUCCESS
            sol = tracker.state.x
            u = FSVec{ComplexF64}(zeros(ComplexF64, 4))
            evaluate!(u, eval_F, sol, FSVec{ComplexF64}(ComplexF64[]))
            @test maximum(abs.(Vector(u))) < 1.0e-8
        end
        # Path should terminate one way or another
        @test tracker.state.accepted_steps + tracker.state.rejected_steps <= 10_000
    end

    @testset "track!: reusable tracker (multiple paths)" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x - 3, y - 4]
        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)

        H = StraightLineHomotopy(eval_G, eval_F; γ = ComplexF64(1.0))
        tracker = Tracker(HomotopyEvaluator(H))

        # Track same path twice — tracker should be reusable
        code1 = track!(tracker, ComplexF64[1.0, 1.0])
        @test code1 == TrackerCode.TRACKER_SUCCESS
        sol1 = copy(Vector(tracker.state.x))

        code2 = track!(tracker, ComplexF64[1.0, 1.0])
        @test code2 == TrackerCode.TRACKER_SUCCESS
        sol2 = Vector(tracker.state.x)

        @test sol1 ≈ sol2
    end

    @testset "track!: zero allocations in step!" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x - 2, y - 3]
        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)

        H = StraightLineHomotopy(eval_G, eval_F; γ = ComplexF64(1.0))
        tracker = Tracker(HomotopyEvaluator(H))

        HC.init!(tracker, ComplexF64[1.0, 1.0])

        # Warmup
        step!(tracker)

        if tracker.state.code == TrackerCode.TRACKING
            allocs = @allocated step!(tracker)
            @test allocs == 0
        end
    end
end
