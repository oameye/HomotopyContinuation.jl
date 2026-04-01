# Phase 4: Path tracking benchmarks
# Measures single path tracking time and Newton step cost.

using BenchmarkTools
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray
using HomotopyContinuationNext:
    System, StraightLineHomotopy, HomotopyEvaluator,
    Tracker, TrackerOptions, TrackerCode, track!, step!,
    Predictor, predict!, update!,
    NewtonCorrector, newton!, init_newton!,
    Jacobian, MatrixWorkspace, WeightedNorm

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}

function benchmark_tracking!(SUITE::BenchmarkGroup)
    SUITE["tracking"] = BenchmarkGroup()

    # ── Katsura-3: 4 equations, 4 variables ──────────────────────────────
    @polyvar x0 x1 x2 x3
    F_k3 = [
        x0 + 2x1 + 2x2 + 2x3 - 1,
        x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
        2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
        x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
    ]
    G_k3 = [x0 - 1, x1^2 - 1, x2^2 - 1, x3^2 - 1]
    sys_G = System(G_k3)
    sys_F = System(F_k3)

    H = StraightLineHomotopy(sys_G.evaluator, sys_F.evaluator)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval)

    x₀ = ComplexF64[1.0, 1.0, 1.0, 1.0]

    # Warmup
    track!(tracker, x₀)

    SUITE["tracking"]["track_one_path_katsura3"] =
        @benchmarkable track!($tracker, $x₀)

    # ── Newton corrector micro-benchmark ─────────────────────────────────
    m, n = size(heval)
    NC = NewtonCorrector(0.125, n, m)
    J = Jacobian(MatrixWorkspace(m, n))
    norm = WeightedNorm(n)
    x_nc = FSVec{ComplexF64}(ComplexF64[1.01, 0.99, 1.01, 0.99])
    x̄_nc = FSVec{ComplexF64}(zeros(ComplexF64, n))
    HomotopyContinuationNext.init!(norm, x_nc)
    newton!(x̄_nc, NC, heval, x_nc, ComplexF64(0.5), J, norm, 1.0, 0.1, true)

    SUITE["tracking"]["newton_katsura3"] =
        @benchmarkable newton!($x̄_nc, $NC, $heval, $x_nc, $(ComplexF64(0.5)), $J, $norm, 1.0, 0.1, true)

    # ── Predictor micro-benchmark ────────────────────────────────────────
    pred = Predictor(m, n)
    x_pred = FSVec{ComplexF64}(ComplexF64[1.0, 1.0, 1.0, 1.0])
    HomotopyContinuationNext.init!(norm, x_pred)
    update!(pred, heval, x_pred, ComplexF64(1.0), J, norm)
    x̂_pred = FSVec{ComplexF64}(zeros(ComplexF64, n))
    predict!(x̂_pred, pred, ComplexF64(-0.01))

    SUITE["tracking"]["predict_katsura3"] =
        @benchmarkable predict!($x̂_pred, $pred, $(ComplexF64(-0.01)))

    # ── Single step! micro-benchmark ─────────────────────────────────────
    tracker_step = Tracker(heval)
    HomotopyContinuationNext.init!(tracker_step, x₀)
    step!(tracker_step)  # warmup

    SUITE["tracking"]["step_katsura3"] =
        @benchmarkable step!($tracker_step) setup = (HomotopyContinuationNext.init!($tracker_step, $x₀))

    return SUITE
end
