using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: System, StraightLineHomotopy, HomotopyEvaluator,
    evaluate!, evaluate_and_jacobian!, taylor!,
    NewtonCorrector, NewtonCode, NewtonCorrectorResult, newton!, init_newton!,
    Predictor, PredictionMethod, predict!, update!, compute_local_error!,
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
        eval_G = System(G)
        eval_F = System(F)

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = ComplexF64(1.0))
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
        eval_G = System(G)
        eval_F = System(F)

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = ComplexF64(1.0))
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

    @testset "track!: rejects invalid singular start value" begin
        @polyvar x y
        G = [x^2 - 1, y^2 - 1]
        eval_G = System(G)

        H = StraightLineHomotopy(eval_G.evaluator, eval_G.evaluator; γ = ComplexF64(1.0))
        tracker = Tracker(HomotopyEvaluator(H))

        # At the origin the Jacobian diag(2x, 2y) vanishes: the invalid start
        # is classified as singular-Jacobian.
        code = track!(tracker, ComplexF64[0.0, 0.0])
        @test code == TrackerCode.TERMINATED_INVALID_STARTVALUE_SINGULAR_JACOBIAN
    end

    @testset "tracker option presets" begin
        d = HC.DEFAULT_TRACKER_OPTIONS
        f = HC.FAST_TRACKER_OPTIONS
        c = HC.CONSERVATIVE_TRACKER_OPTIONS
        @test d isa TrackerOptions && f isa TrackerOptions && c isa TrackerOptions
        @test d == TrackerOptions()
        @test f.β_τ == 0.75 && f.β_ω == 2.0
        @test f.strict_β_τ == min(0.75 * f.β_τ, 0.4)
        @test c.β_τ == 0.25 && c.β_ω == 4.0
        @test c.strict_β_τ == min(0.75 * c.β_τ, 0.4)

        # Both presets still track a simple path end to end.
        @polyvar x y
        G = [x^2 - 1, y^2 - 1]
        F = [x^2 - 2, y^2 - 3]
        eval_G, eval_F = System(G), System(F)
        for opts in (f, c)
            H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = cis(0.9))
            tracker = Tracker(HomotopyEvaluator(H); options = opts)
            code = track!(tracker, ComplexF64[1.0, 1.0])
            @test code == TrackerCode.TRACKER_SUCCESS
        end
    end

    @testset "_start_jacobian_corank classifies the start Jacobian" begin
        n = 3
        A = FSMat{ComplexF64}(Matrix{ComplexF64}(LA.I, n, n))
        @test HC._start_jacobian_corank(A) == 0
        Z = FSMat{ComplexF64}(zeros(ComplexF64, n, n))
        @test HC._start_jacobian_corank(Z) == n
        R1 = FSMat{ComplexF64}(ComplexF64[1 2 3; 2 4 6; 3 6 9])  # rank 1
        @test HC._start_jacobian_corank(R1) == 2
        # Non-finite entries cannot be classified: fall back to corank 0 (generic)
        N = FSMat{ComplexF64}(fill(ComplexF64(NaN), n, n))
        @test HC._start_jacobian_corank(N) == 0
    end

    # ── Tracker: end-to-end ───────────────────────────────────────────────

    @testset "track!: linear system (trivial path)" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x - 2, y - 3]
        eval_G = System(G)
        eval_F = System(F)

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = ComplexF64(1.0))
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
        eval_G = System(G)
        eval_F = System(F)

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator)
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
                evaluate!(u, eval_F.evaluator, sol, FSVec{ComplexF64}(ComplexF64[]))
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

        eval_G = System(G)
        eval_F = System(F)
        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator)
        heval = HomotopyEvaluator(H)

        # Track from one start solution
        tracker = Tracker(heval)
        code = track!(tracker, ComplexF64[1.0, 1.0, 1.0, 1.0])

        if code == TrackerCode.TRACKER_SUCCESS
            sol = tracker.state.x
            u = FSVec{ComplexF64}(zeros(ComplexF64, 4))
            evaluate!(u, eval_F.evaluator, sol, FSVec{ComplexF64}(ComplexF64[]))
            @test maximum(abs.(Vector(u))) < 1.0e-8
        end
        # Path should terminate one way or another
        @test tracker.state.accepted_steps + tracker.state.rejected_steps <= 10_000
    end

    @testset "track!: reusable tracker (multiple paths)" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x - 3, y - 4]
        eval_G = System(G)
        eval_F = System(F)

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = ComplexF64(1.0))
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

    # ── Predictor ─────────────────────────────────────────────────────────

    @testset "predict!: Pade (2,1) prediction via tracker init" begin
        # Setup: use tracker's init! which properly initializes Jacobian + predictor
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x - 2, y - 3]
        eval_G = System(G)
        eval_F = System(F)

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = ComplexF64(1.0))
        heval = HomotopyEvaluator(H)
        tracker = Tracker(heval)

        # init! sets up Jacobian factorization + predictor Taylor coefficients
        HC.init!(tracker, ComplexF64[1.0, 1.0])

        pred = tracker.predictor
        @test pred.trust_region > 0
        @test all(isfinite, pred.tx_norm)
        @test isfinite(pred.local_error)
        @test pred.local_error >= 0

        # predict! uses Taylor coefficients to predict next point
        n = size(heval)[2]
        x̂ = FSVec{ComplexF64}(zeros(ComplexF64, n))
        dt = ComplexF64(-0.01)
        predict!(x̂, pred, dt)
        @test all(isfinite, Vector(x̂))
        # Prediction should be close to x₀ for small dt
        @test maximum(abs.(Vector(x̂) .- Vector(tracker.state.x))) < 0.1
    end

    @testset "predict!: zero allocations" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x - 2, y - 3]
        eval_G = System(G)
        eval_F = System(F)

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = ComplexF64(1.0))
        tracker = Tracker(HomotopyEvaluator(H))
        HC.init!(tracker, ComplexF64[1.0, 1.0])

        pred = tracker.predictor
        n = size(tracker.homotopy)[2]
        x̂ = FSVec{ComplexF64}(zeros(ComplexF64, n))

        # Warmup
        predict!(x̂, pred, ComplexF64(-0.01))

        allocs = @allocated predict!(x̂, pred, ComplexF64(-0.02))
        @test allocs == 0
    end

    # ── Tracker: options and step control ────────────────────────────────

    @testset "track!: max_step_size is enforced" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x - 2, y - 3]
        eval_G = System(G)
        eval_F = System(F)

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = ComplexF64(1.0))
        opts = TrackerOptions(; max_step_size = 0.01)
        tracker = Tracker(HomotopyEvaluator(H); options = opts)

        code = track!(tracker, ComplexF64[1.0, 1.0])
        @test code == TrackerCode.TRACKER_SUCCESS
        # With max_step_size=0.01, should take many more steps than default
        @test tracker.state.accepted_steps >= 50
    end

    @testset "_update_stepsize!: rejection branch respects β_a" begin
        seg = SegmentStepper(ComplexF64(1.0), ComplexF64(0.0))
        HC.propose_step!(seg, 0.8)

        state = TrackerState(1, 1, seg)
        pred = Predictor(1, 1)
        opts = TrackerOptions(; a = 0.125, β_a = 1.6)
        consts = HC.TrackerConstants(opts)
        result = NewtonCorrectorResult(
            NewtonCode.NEWT_TERMINATED,
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
            (
                sqrt(1 + 2 * h(0.5 * opts.β_a * opts.a)) - 1
            ) / (
                sqrt(1 + 2 * h(Θ_j)) - 1
            )
        )^(1 / p) * 0.8
        old_expected = (
            (
                sqrt(1 + 2 * h(0.5 * opts.a)) - 1
            ) / (
                sqrt(1 + 2 * h(Θ_j)) - 1
            )
        )^(1 / p) * 0.8

        HC._update_stepsize!(state, result, pred, opts, consts)

        @test abs(state.segment.Δs) ≈ expected rtol = 1.0e-12
        @test !isapprox(abs(state.segment.Δs), old_expected; rtol = 1.0e-6)
    end

    @testset "track!: step count sanity (katsura-3)" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        G = [x0 - 1, x1^2 - 1, x2^2 - 1, x3^2 - 1]

        eval_G = System(G)
        eval_F = System(F)
        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator)
        tracker = Tracker(HomotopyEvaluator(H))

        code = track!(tracker, ComplexF64[1.0, 1.0, 1.0, 1.0])
        @test code == TrackerCode.TRACKER_SUCCESS
        # Should complete in fewer than 1000 steps
        @test tracker.state.accepted_steps < 1000
        @test tracker.state.rejected_steps < 50
    end

    # ── Tracker: allocation tests ────────────────────────────────────────

    @testset "track!: zero allocations in step!" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x - 2, y - 3]
        eval_G = System(G)
        eval_F = System(F)

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = ComplexF64(1.0))
        tracker = Tracker(HomotopyEvaluator(H))

        HC.init!(tracker, ComplexF64[1.0, 1.0])

        # Warmup
        step!(tracker)

        if tracker.state.code == TrackerCode.TRACKING
            allocs = @allocated step!(tracker)
            @test allocs == 0
        end
    end

    @testset "TrackerState: ext step counters" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x - 2, y - 3]
        eval_G = System(G)
        eval_F = System(F)
        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = ComplexF64(1.0))
        tracker = Tracker(HomotopyEvaluator(H))
        code = track!(tracker, ComplexF64[1.0, 1.0])
        @test code == TrackerCode.TRACKER_SUCCESS
        # ext counters exist and are non-negative
        @test tracker.state.ext_accepted_steps >= 0
        @test tracker.state.ext_rejected_steps >= 0
        # ext steps <= total steps
        @test tracker.state.ext_accepted_steps <= tracker.state.accepted_steps
        @test tracker.state.ext_rejected_steps <= tracker.state.rejected_steps
    end
end
