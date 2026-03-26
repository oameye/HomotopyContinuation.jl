# Phase 4: Path tracking benchmarks
# Measures single path tracking time and Newton step cost.

using BenchmarkTools
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray
using HomotopyContinuationNext:
    System, StraightLineHomotopy, HomotopyEvaluator,
    Tracker, TrackerOptions, TrackerCode, track!

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

    return SUITE
end
