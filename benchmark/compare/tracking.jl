# Compare path tracking: per-path time for katsura systems.
# Standalone: julia --project=benchmark benchmark/compare/tracking.jl

if !@isdefined(print_header)
    include(joinpath(@__DIR__, "common.jl"))
end

using HomotopyContinuationNext: system_eval, StraightLineHomotopy, HomotopyEvaluator,
    Tracker, TrackerCode, TrackerOptions, track!

print_header("Path Tracking: Next vs HC v2")

function _next_track_all(F_polys, G_polys, starts)
    _, eval_G = system_eval(G_polys)
    _, eval_F = system_eval(F_polys)
    H = StraightLineHomotopy(eval_G, eval_F)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval)
    n_success = 0
    for x0 in starts
        code = track!(tracker, ComplexF64.(x0))
        if code == TrackerCode.TRACKER_SUCCESS
            n_success += 1
        end
    end
    return n_success
end

println("\n── Katsura systems (total-degree start) ──")
for n in [3, 4]
    @polyvar kv[1:(n + 1)]
    F = _katsura_polys(kv, n)
    nvars = n + 1
    td_degrees = [1; fill(2, n)]
    G = [kv[1] - 1; [kv[i]^td_degrees[i] - 1 for i in 2:nvars]]
    starts = _td_starts(td_degrees)
    npaths = length(starts)

    # Warmup + run Next
    _next_track_all(F, G, starts)
    t_next = @belapsed _next_track_all($F, $G, $starts)

    # HC v2: full solve
    @var hkv[1:(n + 1)]
    hc_F = _katsura_polys(hkv, n)
    HC.solve(hc_F)
    t_hc = @belapsed HC.solve($hc_F)

    print_row("katsura$n ($npaths paths)", t_next / npaths, t_hc / npaths)
    println("    Next: $(round(t_next * 1.0e3, digits = 2))ms total  HC: $(round(t_hc * 1.0e3, digits = 2))ms total")
end

println("\n  Note: HC v2 includes endgame, solution filtering, polyhedral start system.")
println("  Next tracks total-degree paths only (no endgame, no filtering).")
println("  Per-path comparison is most meaningful for raw tracking speed.")
