using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: System, StraightLineHomotopy, HomotopyEvaluator,
    Tracker, TrackerCode, TrackerOptions, track!, Valuation,
    CONSERVATIVE_TRACKER_OPTIONS, _total_degree_startevaluator,
    total_degree_start_solutions
using DynamicPolynomials: @polyvar

# Track every total-degree path to `t_end` and read off the valuation there.
# `val_x[i]` estimates the exponent of the leading term of xᵢ(t), and `val_tẋ[i]`
# the exponent of t·ẋᵢ(t); together they classify finite, diverging and singular
# endpoints before the endgame commits to one.
function path_valuations(polys, vars, γ, t_end::Float64, opts = TrackerOptions())
    F = System(polys; variables = vars)
    degrees = F.degrees
    H = StraightLineHomotopy(
        _total_degree_startevaluator(degrees), F.evaluator; γ = ComplexF64(γ),
    )
    tracker = Tracker(HomotopyEvaluator(H); options = opts)
    val = Valuation(length(vars))
    return map(total_degree_start_solutions(degrees)) do s
        code = track!(tracker, s; t₁ = complex(1.0), t₀ = complex(t_end))
        HC.init!(val)
        HC.update!(val, tracker.predictor, real(tracker.state.segment.t))
        (code, copy(val.val_x), copy(val.val_tẋ))
    end
end

@testset "Valuation" begin

    @testset "(x-10)^5: finite endpoint of winding number 5" begin
        @polyvar x
        t_end = 1.0e-13
        atol = 10 * t_end^(1 / 5)
        vals = path_valuations([(x - 10)^5], [x], cis(2π * 0.1), t_end)

        @test length(vals) == 5
        @test all(v -> v[1] == TrackerCode.TRACKER_SUCCESS, vals)
        for (_, val_x, val_tẋ) in vals
            # The limit is finite and nonzero, so x has valuation 0 ...
            @test val_x[1] ≈ 0 atol = atol
            # ... while t·ẋ/x → 1/m with m = 5 the winding number.
            @test val_tẋ[1] ≈ 1 / 5 atol = atol
        end
    end

    @testset "two finite and two diverging paths" begin
        @polyvar x y
        f = [
            2.3 * x^2 + 1.2 * y^2 + 3x - 2y + 3,
            2.3 * x^2 + 1.2 * y^2 + 5x + 2y - 5,
        ]
        t_end = 1.0e-10
        atol = 10 * sqrt(t_end)
        vals = path_valuations(
            f, [x, y], 1.3im + 0.4, t_end, CONSERVATIVE_TRACKER_OPTIONS,
        )

        @test length(vals) == 4
        @test all(v -> v[1] == TrackerCode.TRACKER_SUCCESS, vals)

        finite = filter(v -> all(isapprox(0; atol = atol), v[2]), vals)
        diverging = filter(v -> all(isapprox(-1; atol = atol), v[2]), vals)
        @test length(finite) == 2
        @test length(diverging) == 2
        for (_, _, val_tẋ) in finite
            @test val_tẋ ≈ [1, 1] atol = atol
        end
        for (_, _, val_tẋ) in diverging
            @test val_tẋ ≈ [-1, -1] atol = atol
        end
    end

    @testset "winding number family: fractional valuations" begin
        a = [0.257, -0.139, -1.73, -0.199, 1.79, -1.32]
        @polyvar x y
        f1 = (a[1] * x^2 + a[2] * y) * (a[3] * x + a[4] * y) + 1
        f2 = (a[1] * x^2 + a[2] * y) * (a[5] * x + a[6] * y) + 1
        t_end = 1.0e-10
        atol = t_end^(1 / 6)
        vals = path_valuations(
            [f1, f2], [x, y], 1.3im + 0.4, t_end, CONSERVATIVE_TRACKER_OPTIONS,
        )

        @test length(vals) == 9
        @test all(v -> v[1] == TrackerCode.TRACKER_SUCCESS, vals)

        # Three paths end at the three solutions; the other six diverge along a
        # branch of winding number 6, with valuations -1/6 and -2/6.
        finite = filter(v -> all(isapprox(0; atol = atol), v[2]), vals)
        fractional = filter(v -> isapprox(v[2], [-1 / 6, -2 / 6]; atol = atol), vals)
        @test length(finite) == 3
        @test length(fractional) == 6
        for (_, _, val_tẋ) in fractional
            @test val_tẋ ≈ [-1 / 6, -2 / 6] atol = atol
        end
    end

    @testset "show" begin
        @test !isempty(sprint(show, Valuation(2)))
    end
end
