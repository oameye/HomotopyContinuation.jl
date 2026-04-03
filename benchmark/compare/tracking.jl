# Compare path tracking: end-to-end solve() for various system families.
# Standalone: julia --project=benchmark benchmark/compare/tracking.jl
#
# Fair comparison rules:
# - System construction is done OUTSIDE the timed region for both sides.
#   Next builds System() (compiles interpreter/RGF), v2 builds via SymEngine.
#   Neither pays compilation cost in the timed region.
# - Both sides use their full solve() pipeline with pre-built systems.
# - v3 uses CompileMode.COMPILED (best available) for fair comparison.
# - Fixed seed for reproducible step counts.
# - For total-degree systems (katsura, chain), both sides track the same
#   number of paths. For cyclic/random-sparse, polyhedral is used.

if !@isdefined(print_header)
    include(joinpath(@__DIR__, "common.jl"))
end

using HomotopyContinuationNext: CompileMode

const TRACKING_SEED = UInt32(0x4567)

## ── Helper: benchmark + print step counts for a system pair ─────────────────

function _bench_solve(
        name::String,
        next_sys, hc_sys,
        next_alg, hc_kwargs::NamedTuple,
    )
    # Warmup (unseeded, just for JIT)
    Next.solve(next_sys, next_alg)
    HC.solve(hc_sys; hc_kwargs...)

    t_next = @belapsed Next.solve($next_sys, $next_alg)
    t_hc = @belapsed HC.solve($hc_sys; $(hc_kwargs)...)

    print_row(name, t_next, t_hc)

    # Step counts from fixed-seed run
    r_next = Next.solve(next_sys, next_alg)
    r_hc = HC.solve(hc_sys; hc_kwargs...)

    np = r_next.tracked_paths
    ta_next = sum(p.accepted_steps for p in r_next.path_results)
    tr_next = sum(p.rejected_steps for p in r_next.path_results)
    total_next = ta_next + tr_next

    np_hc = HC.ntracked(r_hc)
    # Use r_hc.path_results (all tracked paths), not HC.results() which filters
    ta_hc = sum(p.accepted_steps for p in r_hc.path_results)
    tr_hc = sum(p.rejected_steps for p in r_hc.path_results)
    total_hc = ta_hc + tr_hc

    println("    Next: $(Next.nsolutions(r_next)) sol, $np paths, $(round(total_next / np; digits = 1)) steps/path ($ta_next acc, $tr_next rej)")
    return println("    HC:   $(length(HC.solutions(r_hc))) sol, $np_hc paths, $(round(total_hc / np_hc; digits = 1)) steps/path ($ta_hc acc, $tr_hc rej)")
end

## ── Main benchmark ──────────────────────────────────────────────────────────

print_header("Solve: Next vs HC v2 (end-to-end)")

# ── 1. Katsura (total-degree) ────────────────────────────────────────────────

println("\n── Katsura systems (total-degree) ──")
println("  total-degree = mixed volume → same number of paths")

for n in [3, 4, 5]
    @polyvar kv[1:(n + 1)]
    next_F = _katsura_polys(kv, n)

    @var hkv[1:(n + 1)]
    hc_F = _katsura_polys(hkv, n)

    next_sys = Next.System(next_F; compile = CompileMode.COMPILED)
    hc_sys = HC.ModelKit.System(hc_F)

    _bench_solve(
        "katsura$n", next_sys, hc_sys,
        Next.TotalDegree(; seed = TRACKING_SEED),
        (; seed = TRACKING_SEED),
    )
end

# ── 2. Cyclic (polyhedral — mixed volume < total degree) ─────────────────────

println("\n── Cyclic systems (polyhedral) ──")
println("  mixed volume < total degree → polyhedral is more efficient")

for n in [4, 5]
    specs = _cyclic_specs(n)

    @polyvar cv[1:n]
    next_F = [_poly(cv, s) for s in specs]

    @var hcv[1:n]
    hc_F = [_poly(hcv, s) for s in specs]

    next_sys = Next.System(next_F; compile = CompileMode.COMPILED)
    hc_sys = HC.ModelKit.System(hc_F)

    _bench_solve(
        "cyclic$n", next_sys, hc_sys,
        Next.Polyhedral(; seed = TRACKING_SEED),
        (; start_system = :polyhedral, seed = TRACKING_SEED),
    )
end

# ── 3. Chain systems (total-degree) ──────────────────────────────────────────

println("\n── Chain systems (total-degree) ──")
println("  quadratic system with nearest-neighbour coupling")

for n in [3, 4, 5]
    specs = _chain_specs(n)

    @polyvar chv[1:n]
    next_F = [_poly(chv, s) for s in specs]

    @var hchv[1:n]
    hc_F = [_poly(hchv, s) for s in specs]

    next_sys = Next.System(next_F; compile = CompileMode.COMPILED)
    hc_sys = HC.ModelKit.System(hc_F)

    _bench_solve(
        "chain$n", next_sys, hc_sys,
        Next.TotalDegree(; seed = TRACKING_SEED),
        (; seed = TRACKING_SEED),
    )
end

# ── 4. Random sparse systems (polyhedral) ────────────────────────────────────

println("\n── Random sparse systems (polyhedral) ──")
println("  10 terms/poly, random coefficients, polyhedral start system")

rng_bench = MersenneTwister(0x1234)
for n in [3, 4, 5]
    specs = _random_sparse_specs(rng_bench, n, n)

    @polyvar rsv[1:n]
    next_F = [_poly(rsv, s) for s in specs]

    @var hrsv[1:n]
    hc_F = [_poly(hrsv, s) for s in specs]

    next_sys = Next.System(next_F; compile = CompileMode.COMPILED)
    hc_sys = HC.ModelKit.System(hc_F)

    _bench_solve(
        "sparse$(n)x$n", next_sys, hc_sys,
        Next.Polyhedral(; seed = TRACKING_SEED),
        (; start_system = :polyhedral, seed = TRACKING_SEED),
    )
end

println()
println("  ratio > 1.0 means Next is faster.")
println("  Both v3 and v2 include endgame.")
