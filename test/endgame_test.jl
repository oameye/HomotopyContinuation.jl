using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: System, StraightLineHomotopy, HomotopyEvaluator,
    Tracker, TrackerCode, TrackerOptions, track!, step!,
    TaylorVector, WeightedNorm, Jacobian, MatrixWorkspace,
    evaluate!, evaluate_and_jacobian!, weighted_norm, inf_norm, inf_distance
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

@testset "Endgame Tracker" begin

    @testset "Valuation" begin
        using HomotopyContinuationNext: Valuation, estimate_winding_number

        @testset "Valuation: construction" begin
            val = Valuation(3)
            @test length(val.val_x) == 3
            @test length(val.val_tẋ) == 3
            @test length(val.Δval_x) == 3
            @test val.samples == 0
            @test isnan(val.logt_1)
            @test isnan(val.logt_2)
        end

        @testset "Valuation: init! resets state" begin
            val = Valuation(2)
            val.samples = 5
            val.logt_1 = 1.0
            val.val_x[1] = 42.0
            HC.init!(val)
            @test val.samples == 0
            @test isnan(val.logt_1)
            @test isnan(val.val_x[1])
        end

        @testset "estimate_winding_number: m=1 regular" begin
            val = Valuation(2)
            val.val_tẋ[1] = 1.02
            val.val_tẋ[2] = 0.98
            m, err = estimate_winding_number(val, 2, 6)
            @test m == 1
            @test err < 0.05
        end

        @testset "estimate_winding_number: m=2 singular" begin
            val = Valuation(2)
            val.val_tẋ[1] = 0.51
            val.val_tẋ[2] = 0.49
            m, err = estimate_winding_number(val, 2, 6)
            @test m == 2
            @test err < 0.05
        end

        @testset "estimate_winding_number: m=3" begin
            val = Valuation(2)
            val.val_tẋ[1] = 0.34
            val.val_tẋ[2] = 0.33
            m, err = estimate_winding_number(val, 2, 6)
            @test m == 3
            @test err < 0.05
        end

        @testset "update!: zero allocations" begin
            val = Valuation(2)
            pred = HC.Predictor(2, 2)
            pred.tx3.data[1, 1] = 1.0 + 0.5im
            pred.tx3.data[1, 2] = 0.5 + 1.0im
            pred.tx3.data[2, 1] = 0.1 + 0.1im
            pred.tx3.data[2, 2] = 0.2 - 0.1im
            pred.tx3.data[3, 1] = 0.01 + 0.01im
            pred.tx3.data[3, 2] = 0.02 - 0.01im
            pred.tx3.data[4, 1] = 0.001 + 0.0im
            pred.tx3.data[4, 2] = 0.002 + 0.0im
            pred.winding_number = 1

            HC.update!(val, pred, 0.5)
            HC.update!(val, pred, 0.4)
            HC.update!(val, pred, 0.3)

            allocs = @allocated HC.update!(val, pred, 0.2)
            @test allocs == 0
        end

        @testset "direct Taylor derivative matches finite difference" begin
            x_of_t(t) = ComplexF64(1 + t + t^2)
            ẋ_of_t(t) = ComplexF64(1 + 2t)
            t = 0.5
            h = 1.0e-7

            ν, dν_dt = HC._val_dval(x_of_t(t), ẋ_of_t(t), ComplexF64(2.0), t)
            ν_fd = (
                HC._val(x_of_t(t + h), ẋ_of_t(t + h), t + h) -
                HC._val(x_of_t(t - h), ẋ_of_t(t - h), t - h)
            ) / (2h)

            @test ν ≈ HC._val(x_of_t(t), ẋ_of_t(t), t) atol = 1.0e-12
            @test dν_dt ≈ ν_fd rtol = 1.0e-6 atol = 1.0e-8
        end
    end

    @testset "EndgameTracker construction" begin
        using HomotopyContinuationNext: EndgameTracker, EndgameOptions, EndgameCode, EndgameState

        @polyvar x y
        G = System([x - 1, y - 1])
        F = System([x^2 - 1, y^2 - 1])
        H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
        heval = HomotopyEvaluator(H)
        tracker = Tracker(heval)
        opts = EndgameOptions()

        eg = EndgameTracker(tracker, opts)
        @test eg.state.code == EndgameCode.TRACKING
        @test eg.state.winding_number == 0
        @test eg.state.in_endgame == false
        @test eg.state.in_singular_endgame == false
        @test eg.state.singular == false
        @test isnan(eg.state.accuracy)
        @test eg.val.samples == 0
        @test opts.endgame_start == 0.1
        @test opts.max_winding_number == 6
        @test opts.lambda == 0.25
    end

    @testset "EndgameTracker: init! and track! (regular paths)" begin
        using HomotopyContinuationNext: EndgameTracker, EndgameOptions, EndgameCode

        @testset "init!: valid start" begin
            @polyvar x y
            G = System([x - 1, y - 1])
            F = System([x - 2, y - 3])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            tracker = Tracker(HomotopyEvaluator(H))
            eg = EndgameTracker(tracker)
            code = HC.init!(eg, ComplexF64[1.0, 1.0])
            @test code == EndgameCode.TRACKING
        end

        @testset "init!: invalid start propagates" begin
            @polyvar x y
            G = System([x^2 - 1, y^2 - 1])
            H = StraightLineHomotopy(G.evaluator, G.evaluator; γ = ComplexF64(1.0))
            tracker = Tracker(HomotopyEvaluator(H))
            eg = EndgameTracker(tracker)
            code = HC.init!(eg, ComplexF64[0.0, 0.0])
            @test code == EndgameCode.TERMINATED_INVALID_STARTVALUE
        end

        @testset "track!: linear system succeeds" begin
            @polyvar x y
            G = System([x - 1, y - 1])
            F = System([x - 2, y - 3])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            tracker = Tracker(HomotopyEvaluator(H))
            eg = EndgameTracker(tracker)
            code = HC.track!(eg, ComplexF64[1.0, 1.0])
            @test code == EndgameCode.SUCCESS
            @test abs(eg.state.solution[1] - 2.0) < 1.0e-8
            @test abs(eg.state.solution[2] - 3.0) < 1.0e-8
        end

        @testset "track!: reusable (multiple paths)" begin
            @polyvar x y
            G = System([x - 1, y - 1])
            F = System([x - 2, y - 3])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            tracker = Tracker(HomotopyEvaluator(H))
            eg = EndgameTracker(tracker)

            code1 = HC.track!(eg, ComplexF64[1.0, 1.0])
            sol1 = copy(Vector(eg.state.solution))
            code2 = HC.track!(eg, ComplexF64[1.0, 1.0])
            sol2 = Vector(eg.state.solution)

            @test code1 == EndgameCode.SUCCESS
            @test code2 == EndgameCode.SUCCESS
            @test sol1 ≈ sol2
        end
    end

    @testset "check_finite!" begin
        using HomotopyContinuationNext: EndgameTracker, EndgameOptions, EndgameCode, Valuation

        @testset "returns false for regular path (m=1, all valuations near zero)" begin
            @polyvar x y
            G = System([x - 1, y - 1])
            F = System([x - 2, y - 3])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            eg = EndgameTracker(Tracker(HomotopyEvaluator(H)))
            eg.val.val_x[1] = 0.001
            eg.val.val_x[2] = -0.002
            eg.val.Δval_x[1] = 0.001
            eg.val.Δval_x[2] = -0.001
            eg.val.val_tẋ[1] = 1.01
            eg.val.val_tẋ[2] = 0.99
            eg.val.samples = 3
            eg.state.in_endgame = true

            result = HC.check_finite!(eg)
            @test result == false
            @test eg.state.in_singular_endgame == false
        end

        @testset "returns true and switches to singular for m=2" begin
            @polyvar x y
            G = System([x - 1, y - 1])
            F = System([x - 2, y - 3])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            eg = EndgameTracker(Tracker(HomotopyEvaluator(H)))
            HC.init!(eg, ComplexF64[1.0, 1.0])
            eg.val.val_x[1] = 0.5
            eg.val.val_x[2] = 0.5
            eg.val.Δval_x[1] = 0.001
            eg.val.Δval_x[2] = -0.001
            eg.val.val_tẋ[1] = 0.51
            eg.val.val_tẋ[2] = 0.49
            eg.val.samples = 3
            eg.state.in_endgame = true

            result = HC.check_finite!(eg)
            @test result == true
            @test eg.state.in_singular_endgame == true
            @test eg.state.winding_number == 2
            @test eg.tracker.predictor.winding_number == 2
            @test eg.tracker.state.keep_extended_prec == true
        end
    end

    @testset "check_at_infinity!" begin
        using HomotopyContinuationNext: EndgameTracker, EndgameOptions, EndgameCode

        @testset "returns false for finite path" begin
            @polyvar x y
            G = System([x - 1, y - 1])
            F = System([x - 2, y - 3])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            eg = EndgameTracker(Tracker(HomotopyEvaluator(H)))
            HC.init!(eg, ComplexF64[1.0, 1.0])
            eg.val.val_x[1] = 0.01
            eg.val.val_x[2] = -0.01
            eg.val.val_tẋ[1] = 1.0
            eg.val.val_tẋ[2] = 1.0
            eg.val.Δval_x[1] = 0.5
            eg.val.Δval_x[2] = -0.5
            eg.val.Δval_tẋ[1] = 0.5
            eg.val.Δval_tẋ[2] = -0.5
            eg.val.samples = 3
            eg.state.in_endgame = true

            result = HC.check_at_infinity!(eg)
            @test result == false
        end

        @testset "returns true for diverging coordinate" begin
            @polyvar x y
            G = System([x - 1, y - 1])
            F = System([x - 2, y - 3])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            eg = EndgameTracker(Tracker(HomotopyEvaluator(H)))
            HC.init!(eg, ComplexF64[1.0, 1.0])
            eg.state.in_endgame = true

            # Coordinate 1 already marked as divergence candidate
            eg.state.at_inf_active[1] = true
            eg.state.at_inf_starts[1] = 0.05
            eg.state.at_inf_abs_coords[1] = 1.0
            eg.state.at_inf_conds[1] = 1.0e-12

            eg.val.val_x[1] = -1.0
            eg.val.val_x[2] = 0.0
            eg.val.val_tẋ[1] = -1.0
            eg.val.val_tẋ[2] = 1.0
            eg.val.Δval_x[1] = 0.0
            eg.val.Δval_x[2] = 0.0
            eg.val.Δval_tẋ[1] = 0.0
            eg.val.Δval_tẋ[2] = 0.0
            eg.val.samples = 3

            # Mock: large coordinate and condition (κ must exceed max(1e8, min_cond))
            eg.tracker.state.x[1] = 200.0 + 0im
            eg.tracker.predictor.cond_H_x = 1.0e9

            result = HC.check_at_infinity!(eg)
            @test result == true
            @test eg.state.code == EndgameCode.AT_INFINITY
        end

        @testset "clears stale divergence candidate" begin
            @polyvar x y
            G = System([x - 1, y - 1])
            F = System([x - 2, y - 3])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            eg = EndgameTracker(Tracker(HomotopyEvaluator(H)))
            HC.init!(eg, ComplexF64[1.0, 1.0])
            eg.state.in_endgame = true
            eg.state.at_inf_active[1] = true
            eg.state.at_inf_starts[1] = 0.05
            eg.state.at_inf_abs_coords[1] = 10.0
            eg.state.at_inf_conds[1] = 1.0

            eg.val.val_x[1] = -1.0
            eg.val.val_tẋ[1] = -1.0
            eg.val.Δval_x[1] = 10.0
            eg.val.Δval_tẋ[1] = 0.0

            result = HC.check_at_infinity!(eg)
            @test result == false
            @test !eg.state.at_inf_active[1]
            @test isnan(eg.state.at_inf_starts[1])
            @test isnan(eg.state.at_inf_abs_coords[1])
            @test isnan(eg.state.at_inf_conds[1])
        end
    end

    @testset "Singular endgame helpers" begin
        @testset "cubic_hermite!: linear function" begin
            n = 1
            ty0 = TaylorVector{2, ComplexF64}(n)
            ty1 = TaylorVector{2, ComplexF64}(n)
            # y(s) = 2 + 3s → y(0) = 2
            ty0.data[1, 1] = 3.5 + 0im   # at s0=0.5
            ty0.data[2, 1] = 3.0 + 0im   # dy/ds=3
            ty1.data[1, 1] = 2.75 + 0im  # at s1=0.25
            ty1.data[2, 1] = 3.0 + 0im   # dy/ds=3

            x_hat = FSVec{ComplexF64}(zeros(ComplexF64, n))
            HC.cubic_hermite!(x_hat, ty0, 0.5, ty1, 0.25, 0.0)
            @test abs(x_hat[1] - 2.0) < 1.0e-10
        end

        @testset "add_sample! stores scaled sample condition" begin
            @polyvar x
            G = System([x^2 - 1])
            F = System([x^2])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            eg = EndgameTracker(Tracker(HomotopyEvaluator(H)))
            HC.init!(eg, ComplexF64[1.0])
            eg.state.winding_number = 2

            HC.switch_to_singular!(eg, real(eg.tracker.state.segment.t))

            expected = HC._scaled_cond(
                eg.tracker.state.jacobian.workspace,
                eg.state.row_scaling,
                eg.state.col_scaling,
            )
            @test eg.state.sample_conds[1] ≈ expected
        end

        @testset "first accurate singular prediction is retained" begin
            @polyvar x
            G = System([x^2 - 1])
            F = System([x^2])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            eg = EndgameTracker(Tracker(HomotopyEvaluator(H)))

            code = HC.track!(eg, ComplexF64[1.0])

            @test code == EndgameCode.SUCCESS
            @test eg.state.singular
            @test eg.state.winding_number == 2
            @test eg.state.accuracy < eg.options.singular_min_accuracy
            @test abs(eg.state.solution[1]) < 1.0e-8
        end
    end

    @testset "PathResult from EndgameTracker" begin
        using HomotopyContinuationNext: EndgameTracker, EndgameCode, PathResult,
            PathResultCode

        @testset "regular success" begin
            @polyvar x y
            G = System([x - 1, y - 1])
            F = System([x - 2, y - 3])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            eg = EndgameTracker(Tracker(HomotopyEvaluator(H)))
            HC.track!(eg, ComplexF64[1.0, 1.0])

            pr = PathResult(eg)
            @test pr.return_code == PathResultCode.PATH_SUCCESS
            @test abs(pr.solution[1] - 2.0) < 1.0e-8
            @test abs(pr.solution[2] - 3.0) < 1.0e-8
            @test pr.t == 0.0
            @test pr.singular == false
            @test pr.winding_number == 0
            @test pr.accepted_steps > 0
            @test pr.steps_eg >= 0
            @test pr.extended_precision_used isa Bool
            @test length(pr.last_path_point) == 2
        end

        @testset "singular success reports extrapolated endpoint" begin
            @polyvar x
            G = System([x^2 - 1])
            F = System([x^2])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            eg = EndgameTracker(Tracker(HomotopyEvaluator(H)))
            HC.track!(eg, ComplexF64[1.0])

            pr = PathResult(eg)
            @test pr.return_code == PathResultCode.PATH_SUCCESS
            @test pr.singular
            @test pr.t == 0.0
            @test abs(pr.solution[1]) < 1.0e-8
            @test pr.last_path_t > 0.0
            @test pr.last_path_point ≈ Vector{ComplexF64}(eg.tracker.state.x)
        end

        @testset "invalid start reports tracker point" begin
            @polyvar x y
            F = System([x^2 - 1, y^2 - 1])
            H = StraightLineHomotopy(F.evaluator, F.evaluator; γ = ComplexF64(1.0))
            eg = EndgameTracker(Tracker(HomotopyEvaluator(H)))
            code = HC.track!(eg, ComplexF64[2.0, 0.0])

            pr = PathResult(eg)
            @test code == EndgameCode.TERMINATED_INVALID_STARTVALUE
            @test pr.return_code == PathResultCode.PATH_TERMINATED_INVALID_START
            @test pr.solution == ComplexF64[2.0, 0.0]
            @test pr.last_path_point == ComplexF64[2.0, 0.0]
            @test pr.t == 1.0
            @test pr.last_path_t == 1.0
        end
    end

    @testset "EndgameTracker: zero allocations in step!" begin
        using HomotopyContinuationNext: EndgameTracker, EndgameCode

        # Measure allocations inside a function barrier to avoid
        # global-scope type inference artifacts in @allocated
        function _measure_endgame_step_allocs()
            @polyvar x y
            G = System([x - 1, y - 1])
            F = System([x - 2, y - 3])
            H = StraightLineHomotopy(G.evaluator, F.evaluator; γ = ComplexF64(1.0))
            eg = EndgameTracker(Tracker(HomotopyEvaluator(H)))
            HC.init!(eg, ComplexF64[1.0, 1.0])
            for _ in 1:3
                eg.state.code == EndgameCode.TRACKING || return -1
                HC.step!(eg)
            end
            eg.state.code == EndgameCode.TRACKING || return -1
            return @allocated HC.step!(eg)
        end
        _measure_endgame_step_allocs()  # warmup
        allocs = _measure_endgame_step_allocs()
        @test allocs == 0
    end

end # top-level testset
