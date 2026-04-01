#!/usr/bin/env julia
# Trace raw tracker behavior on a fixed Katsura path for Next vs HC v2.
#
# Usage:
#   julia --project=benchmark benchmark/compare/trace_tracker.jl [n] [start_index]
#
# If `start_index` is omitted, the script first scans all total-degree starts and
# chooses the path with the largest absolute step-count gap.

if !@isdefined(print_header)
    include(joinpath(@__DIR__, "common.jl"))
end

using HomotopyContinuationNext: CompileMode
using Printf: @sprintf

const DEFAULT_GAMMA = ComplexF64(1.0)

function _katsura_start_polys_next(vars, n)
    return [vars[1] - 1; [vars[i]^2 - 1 for i in 2:(n + 1)]]
end

function _katsura_start_polys_hc(vars, n)
    return [vars[1] - 1; [vars[i]^2 - 1 for i in 2:(n + 1)]]
end

function _next_tracker_bundle(n::Int; gamma::ComplexF64 = DEFAULT_GAMMA)
    @polyvar kv[1:(n + 1)]
    F = _katsura_polys(kv, n)
    G = _katsura_start_polys_next(kv, n)
    sys_F = Next.System(F; compile = CompileMode.COMPILED)
    sys_G = Next.System(G; compile = CompileMode.COMPILED)
    H = Next.StraightLineHomotopy(sys_G.evaluator, sys_F.evaluator; γ = gamma)
    tracker = Next.Tracker(Next.HomotopyEvaluator(H))
    return (; tracker, starts = _td_starts([1; fill(2, n)]))
end

function _hc_tracker_bundle(n::Int; gamma::ComplexF64 = DEFAULT_GAMMA)
    @var hv[1:(n + 1)]
    F = _katsura_polys(hv, n)
    G = _katsura_start_polys_hc(hv, n)
    sys_F = HC.ModelKit.System(F)
    sys_G = HC.ModelKit.System(G)
    H = HC.StraightLineHomotopy(sys_G, sys_F; gamma = gamma)
    tracker = HC.Tracker(H)
    return (; tracker, starts = _td_starts([1; fill(2, n)]))
end

@inline _h(a::Float64) = 2a * (sqrt(4a^2 + 1) - 2a)

function _next_step_bounds(tracker::Next.Tracker)
    state = tracker.state
    pred = tracker.predictor
    opts = tracker.options
    p = pred.order
    step_a = opts.β_a * opts.a
    h_step = sqrt(1 + 2 * _h(step_a)) - 1
    ω = state.ω
    e = pred.local_error
    τ = state.τ
    β_τ = if state.use_strict_β_τ || Next.dist_to_target(state.segment) < opts.β_τ * τ
        opts.strict_β_τ
    else
        opts.β_τ
    end
    ds_err = if isfinite(e) && e > 0 && isfinite(ω)
        Next.nthroot(h_step / (ω * e), p) / opts.β_ω
    else
        Inf
    end
    ds_tau = β_τ * τ
    limiter = ds_err <= ds_tau ? "error" : "tau"
    return ds_err, ds_tau, limiter
end

function _hc_step_bounds(tracker::HC.Tracker)
    state = HC.state(tracker)
    pred = tracker.predictor
    opts = tracker.options
    params = opts.parameters
    p = pred.order
    a = params.β_a * params.a
    h_step = sqrt(1 + 2 * _h(a)) - 1
    ω = state.ω
    e = pred.local_error
    τ = state.τ
    β_τ = if state.use_strict_β_τ || HC.dist_to_target(state.segment_stepper) < params.β_τ * τ
        params.strict_β_τ
    else
        params.β_τ
    end
    ds_err = if isfinite(e) && e > 0 && isfinite(ω)
        HC.nthroot(h_step / (ω * e), p) / params.β_ω_p
    else
        Inf
    end
    ds_tau = β_τ * τ
    limiter = ds_err <= ds_tau ? "error" : "tau"
    return ds_err, ds_tau, limiter
end

function _next_snapshot(tracker::Next.Tracker)
    state = tracker.state
    pred = tracker.predictor
    ds_err, ds_tau, limiter = _next_step_bounds(tracker)
    return (
        t = state.segment.t,
        ds = abs(state.segment.Δs),
        dist = Next.dist_to_target(state.segment),
        tau = state.τ,
        local_error = pred.local_error,
        omega = state.ω,
        mu = state.μ,
        accuracy = state.accuracy,
        cond = pred.cond_H_x,
        ds_err = ds_err,
        ds_tau = ds_tau,
        limiter = limiter,
        accepted = state.accepted_steps,
        rejected = state.rejected_steps,
    )
end

function _hc_snapshot(tracker::HC.Tracker)
    state = HC.state(tracker)
    pred = tracker.predictor
    ds_err, ds_tau, limiter = _hc_step_bounds(tracker)
    return (
        t = state.segment_stepper.t,
        ds = abs(state.segment_stepper.Δs),
        dist = HC.dist_to_target(state.segment_stepper),
        tau = state.τ,
        local_error = pred.local_error,
        omega = state.ω,
        mu = state.μ,
        accuracy = state.accuracy,
        cond = pred.cond_H_ẋ,
        ds_err = ds_err,
        ds_tau = ds_tau,
        limiter = limiter,
        accepted = state.accepted_steps,
        rejected = state.rejected_steps,
    )
end

function _trace_next!(tracker::Next.Tracker, x0::Vector{ComplexF64})
    code = Next.init!(tracker, x0)
    rows = NamedTuple[]
    if code != Next.TrackerCode.TRACKING
        return rows, code
    end
    k = 0
    while tracker.state.code == Next.TrackerCode.TRACKING
        pre = _next_snapshot(tracker)
        accepted = Next.step!(tracker)
        post = _next_snapshot(tracker)
        k += 1
        push!(rows, (
            step = k,
            accepted = accepted,
            t = pre.t,
            ds = pre.ds,
            tau = pre.tau,
            local_error = pre.local_error,
            omega = pre.omega,
            mu = pre.mu,
            accuracy = pre.accuracy,
            cond = pre.cond,
            ds_err = pre.ds_err,
            ds_tau = pre.ds_tau,
            limiter = pre.limiter,
            next_ds = post.ds,
            next_t = post.t,
            next_tau = post.tau,
            next_error = post.local_error,
        ))
    end
    return rows, tracker.state.code
end

function _trace_hc!(tracker::HC.Tracker, x0::Vector{ComplexF64})
    ok = HC.init!(tracker, x0)
    rows = NamedTuple[]
    if !ok
        return rows, HC.status(tracker)
    end
    k = 0
    while HC.is_tracking(HC.status(tracker))
        pre = _hc_snapshot(tracker)
        accepted = HC.step!(tracker)
        post = _hc_snapshot(tracker)
        k += 1
        push!(rows, (
            step = k,
            accepted = accepted,
            t = pre.t,
            ds = pre.ds,
            tau = pre.tau,
            local_error = pre.local_error,
            omega = pre.omega,
            mu = pre.mu,
            accuracy = pre.accuracy,
            cond = pre.cond,
            ds_err = pre.ds_err,
            ds_tau = pre.ds_tau,
            limiter = pre.limiter,
            next_ds = post.ds,
            next_t = post.t,
            next_tau = post.tau,
            next_error = post.local_error,
        ))
    end
    return rows, HC.status(tracker)
end

function _count_steps_next!(tracker::Next.Tracker, x0::Vector{ComplexF64})
    code = Next.track!(tracker, x0)
    state = tracker.state
    return (
        code = code,
        accepted = state.accepted_steps,
        rejected = state.rejected_steps,
        total = state.accepted_steps + state.rejected_steps,
    )
end

function _count_steps_hc!(tracker::HC.Tracker, x0::Vector{ComplexF64})
    HC.track!(tracker, x0)
    state = HC.state(tracker)
    return (
        code = HC.status(tracker),
        accepted = state.accepted_steps,
        rejected = state.rejected_steps,
        total = state.accepted_steps + state.rejected_steps,
    )
end

function _print_summary(rows)
    println("idx   start[2:end]          Next(acc/rej/tot)    HC(acc/rej/tot)   gap")
    for row in rows
        start_tail = join((real(s) > 0 ? "+1" : "-1" for s in row.start[2:end]), " ")
        println(
            lpad(row.idx, 3), "   ",
            rpad(start_tail, 18), "   ",
            lpad("$(row.next.accepted)/$(row.next.rejected)/$(row.next.total)", 18), "   ",
            lpad("$(row.hc.accepted)/$(row.hc.rejected)/$(row.hc.total)", 16), "   ",
            row.next.total - row.hc.total,
        )
    end
end

function _print_trace_side_by_side(next_rows, hc_rows; max_rows::Int = 40)
    println()
    println("step  Next(acc ds tau err lim -> next_ds)          HC(acc ds tau err lim -> next_ds)")
    n = min(max(length(next_rows), length(hc_rows)), max_rows)
    for i in 1:n
        nr = i <= length(next_rows) ? next_rows[i] : nothing
        hr = i <= length(hc_rows) ? hc_rows[i] : nothing
        next_txt = isnothing(nr) ? "" : @sprintf(
            "%s %7.3g %7.3g %7.3g %-5s -> %7.3g",
            nr.accepted ? "A" : "R", nr.ds, nr.tau, nr.local_error, nr.limiter, nr.next_ds,
        )
        hc_txt = isnothing(hr) ? "" : @sprintf(
            "%s %7.3g %7.3g %7.3g %-5s -> %7.3g",
            hr.accepted ? "A" : "R", hr.ds, hr.tau, hr.local_error, hr.limiter, hr.next_ds,
        )
        println(lpad(i, 4), "  ", rpad(next_txt, 44), "  ", hc_txt)
    end
end

function main()
    n = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 3
    forced_idx = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 0

    print_header("Tracker Trace: Next vs HC v2")
    println("  Katsura-$n, raw Tracker vs raw Tracker, gamma = $(DEFAULT_GAMMA)")

    next_bundle = _next_tracker_bundle(n)
    hc_bundle = _hc_tracker_bundle(n)

    summaries = NamedTuple[]
    for (idx, x0) in enumerate(next_bundle.starts)
        next_stats = _count_steps_next!(next_bundle.tracker, x0)
        hc_stats = _count_steps_hc!(hc_bundle.tracker, x0)
        push!(summaries, (idx = idx, start = x0, next = next_stats, hc = hc_stats))
    end

    println()
    println("── Per-path step counts ──")
    _print_summary(summaries)

    chosen = if forced_idx > 0
        forced_idx
    else
        findmax([s.next.total - s.hc.total for s in summaries])[2]
    end
    x0 = next_bundle.starts[chosen]
    println()
    println("Tracing start index $chosen: ", x0)

    next_rows, next_code = _trace_next!(next_bundle.tracker, x0)
    hc_rows, hc_code = _trace_hc!(hc_bundle.tracker, x0)

    println("  Next final code: $next_code, steps=$(length(next_rows))")
    println("  HC final code:   $hc_code, steps=$(length(hc_rows))")

    _print_trace_side_by_side(next_rows, hc_rows)
end

main()
