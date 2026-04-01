# Compare TTFX: time-to-first-solve for v3 vs v2.
# Standalone: julia --project=benchmark benchmark/compare/ttfx.jl
#
# This script must be run in a FRESH Julia session (no prior compilation).
# It measures wall-clock time of the first solve() call for both packages.

println("="^72)
println("  TTFX Comparison: first solve() wall-clock time")
println("="^72)
println()

# ── v3 ──
println("── v3 (HomotopyContinuationNext) ──")
t_next_load = @elapsed using HomotopyContinuationNext
println("  Package load: $(round(t_next_load; digits=2))s")

using DynamicPolynomials: @polyvar
@polyvar x y
t_next_solve = @elapsed begin
    F_next = System([x^2 + y - 1, x * y - 2])
    result_next = solve(F_next)
end
t_next_total = t_next_load + t_next_solve
println("  First solve(): $(round(t_next_solve; digits=2))s")
println("  Total (load + solve): $(round(t_next_total; digits=2))s")
println("  Solutions found: $(nsolutions(result_next))")
println()

# ── v2 with default :mixed ──
println("── v2 (HomotopyContinuation, default :mixed) ──")
t_hc_load = @elapsed using HomotopyContinuation
println("  Package load: $(round(t_hc_load; digits=2))s")

using HomotopyContinuation.ModelKit: @var
@var hx hy
t_hc_mixed = @elapsed begin
    result_hc = HomotopyContinuation.solve([hx^2 + hy - 1, hx * hy - 2])
end
t_hc_mixed_total = t_hc_load + t_hc_mixed
println("  First solve() [:mixed]: $(round(t_hc_mixed; digits=2))s")
println("  Total (load + solve): $(round(t_hc_mixed_total; digits=2))s")
println("  Solutions found: $(length(HomotopyContinuation.solutions(result_hc)))")
println()

# Second call with a DIFFERENT system (tests recompilation cost)
@var hz hw
t_hc_second = @elapsed begin
    result_hc2 = HomotopyContinuation.solve([hz^3 + hw^2 - 1, hz * hw - hz])
end
println("  Second solve() (different system): $(round(t_hc_second; digits=2))s")
println()

# ── v2 with :none ──
println("── v2 (HomotopyContinuation, compile=Val(:none)) ──")
@var nx ny
t_hc_none = @elapsed begin
    result_none = HomotopyContinuation.solve(
        [nx^2 + ny - 1, nx * ny - 2]; compile = :none,
    )
end
println("  First solve() [:none]: $(round(t_hc_none; digits=2))s")
println("  Solutions found: $(length(HomotopyContinuation.solutions(result_none)))")
println()

# ── Summary ──
println("── Summary ──")
println("  v3 total:                $(round(t_next_total; digits=2))s")
println("  v2 total [:mixed]:       $(round(t_hc_mixed_total; digits=2))s")
println("  v2 solve only [:mixed]:  $(round(t_hc_mixed; digits=2))s")
println("  v2 solve only [:none]:   $(round(t_hc_none; digits=2))s")
println("  v2 2nd system [:mixed]:  $(round(t_hc_second; digits=2))s")
println()
println("  The key metric: v2[:mixed] pays a large first-solve cost because")
println("  CompiledSystem{ID} creates a unique type that recompiles the full")
println("  tracker pipeline. v3 uses FunctionWrapper (monomorphic tracker).")
