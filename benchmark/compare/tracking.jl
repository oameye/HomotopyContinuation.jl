# Compare path tracking: end-to-end solve() for katsura systems.
# Standalone: julia --project=benchmark benchmark/compare/tracking.jl
#
# Fair comparison rules:
# - System construction is done OUTSIDE the timed region for both sides.
#   Next builds System() (compiles interpreter/RGF), v2 builds via SymEngine.
#   Neither pays compilation cost in the timed region.
# - Only katsura systems are compared because total-degree == mixed volume,
#   so both sides track the same number of paths.
# - Both sides use their full solve() pipeline with pre-built systems.
# - v3 uses CompileMode.COMPILED (best available) for fair comparison.
# - Fixed seed for reproducible step counts.

if !@isdefined(print_header)
    include(joinpath(@__DIR__, "common.jl"))
end

using HomotopyContinuationNext: CompileMode

const TRACKING_SEED = UInt32(0x4567)

print_header("Solve: Next vs HC v2 (end-to-end)")

println("\n── Katsura systems ──")
println("  (total-degree = mixed volume → same number of paths)")
println("  v3 uses CompileMode.COMPILED, fixed seed=$(TRACKING_SEED)")

for n in [3, 4, 5]
    @polyvar kv[1:(n + 1)]
    next_F = _katsura_polys(kv, n)

    @var hkv[1:(n + 1)]
    hc_F = _katsura_polys(hkv, n)

    # Pre-build systems outside timed region
    next_sys = Next.System(next_F; compile = CompileMode.COMPILED)
    hc_sys = HC.ModelKit.System(hc_F)

    # Warmup (unseeded, just for JIT)
    Next.solve(next_sys)
    HC.solve(hc_sys)

    t_next = @belapsed Next.solve($next_sys, Next.TotalDegree(; seed = $TRACKING_SEED))
    t_hc = @belapsed HC.solve($hc_sys; seed = $TRACKING_SEED)

    print_row("katsura$n", t_next, t_hc)

    # Step counts from fixed-seed run
    r_next = Next.solve(next_sys, Next.TotalDegree(; seed = TRACKING_SEED))
    r_hc = HC.solve(hc_sys; seed = TRACKING_SEED)

    np = r_next.tracked_paths
    ta_next = sum(p.accepted_steps for p in r_next.path_results)
    tr_next = sum(p.rejected_steps for p in r_next.path_results)
    total_next = ta_next + tr_next

    np_hc = HC.ntracked(r_hc)
    ta_hc = sum(p.accepted_steps for p in HC.results(r_hc))
    tr_hc = sum(p.rejected_steps for p in HC.results(r_hc))
    total_hc = ta_hc + tr_hc

    println("    Next: $(Next.nsolutions(r_next)) sol, $(round(total_next / np; digits = 1)) total steps/path ($ta_next acc, $tr_next rej)")
    println("    HC:   $(length(HC.solutions(r_hc))) sol, $(round(total_hc / np_hc; digits = 1)) total steps/path ($ta_hc acc, $tr_hc rej)")
end

println()
println("  ratio > 1.0 means Next is faster.")
println("  Both v3 and v2 include endgame.")
