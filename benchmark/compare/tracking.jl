# Compare path tracking: end-to-end solve() for katsura systems.
# Standalone: julia --project=benchmark benchmark/compare/tracking.jl
#
# Fair comparison rules:
# - System construction is done OUTSIDE the timed region for both sides.
#   Next builds System() (compiles interpreter), v2 builds InterpretedSystem
#   (compiles via SymEngine). Neither pays compilation cost in the timed region.
# - Only katsura systems are compared because total-degree == mixed volume,
#   so both sides track the same number of paths.
# - Both sides use their full solve() pipeline with pre-built systems.

if !@isdefined(print_header)
    include(joinpath(@__DIR__, "common.jl"))
end

print_header("Solve: Next vs HC v2 (end-to-end)")

println("\n── Katsura systems ──")
println("  (total-degree = mixed volume → same number of paths)")

for n in [3, 4, 5]
    @polyvar kv[1:(n + 1)]
    next_F = _katsura_polys(kv, n)

    @var hkv[1:(n + 1)]
    hc_F = _katsura_polys(hkv, n)

    # Pre-build systems outside timed region
    next_sys = Next.System(next_F)
    hc_sys = HC.ModelKit.System(hc_F)

    # Warmup
    Next.solve(next_sys)
    HC.solve(hc_sys)

    t_next = @belapsed Next.solve($next_sys)
    t_hc = @belapsed HC.solve($hc_sys)

    print_row("katsura$n", t_next, t_hc)

    nsol_next = Next.nsolutions(Next.solve(next_sys))
    nsol_hc = length(HC.solve(hc_sys))
    println("    Next: $(nsol_next) solutions, HC: $(nsol_hc) solutions")
end

println()
println("  ratio > 1.0 means Next is faster.")
println("  Note: HC v2 includes endgame; Next does not yet.")
