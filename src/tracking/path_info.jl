## Per-step introspection of a single path: `iterator` yields the accepted
## points, `path_info` records the full step table.

# ── Path iterator ───────────────────────────────────────────────────────────

"""
    PathIterator{T}

Stateful iterator over the accepted steps of one path, yielding `(x, t)` with
`t::T`. Returned by [`iterator`](@ref).
"""
struct PathIterator{T <: Union{Float64, ComplexF64}}
    tracker::Tracker
end

Base.IteratorSize(::Type{<:PathIterator}) = Base.SizeUnknown()
Base.eltype(::Type{PathIterator{T}}) where {T} = Tuple{Vector{ComplexF64}, T}

"""
    iterator(tracker::Tracker, x₀, t₁ = 1.0, t₀ = 0.0) -> PathIterator

Prepare `tracker` to track `x₀` from `t₁` to `t₀` one accepted step at a time,
yielding the tuple `(x, t)` in each iteration. `t` is a `Float64` when `t₁` and
`t₀` are both real and a `ComplexF64` otherwise, so `eltype` is concrete.

The first iteration yields the start point, every later one an accepted step.
Iteration ends at the target, or as soon as the path fails.

The iterator is stateful and shares `tracker`'s state, so the tracker can be
inspected between iterations.

```julia
for (x, t) in iterator(tracker, x₀, 1.0, 0.25)
    println("x at t = ", t, ": ", x)
end
```
"""
function iterator(
        tracker::Tracker, x₀::AbstractVector{<:Number},
        t₁::Real = 1.0, t₀::Real = 0.0,
    )::PathIterator{Float64}
    init!(tracker, x₀, ComplexF64(t₁), ComplexF64(t₀))
    return PathIterator{Float64}(tracker)
end

function iterator(
        tracker::Tracker, x₀::AbstractVector{<:Number},
        t₁::Number, t₀::Number = 0.0,
    )::PathIterator{ComplexF64}
    init!(tracker, x₀, ComplexF64(t₁), ComplexF64(t₀))
    return PathIterator{ComplexF64}(tracker)
end

_t_value(t::ComplexF64, ::PathIterator{Float64})::Float64 = real(t)
_t_value(t::ComplexF64, ::PathIterator{ComplexF64})::ComplexF64 = t

_current_x_t(iter::PathIterator) = (
    Vector{ComplexF64}(iter.tracker.state.x),
    _t_value(iter.tracker.state.segment.t, iter),
)

function Base.iterate(iter::PathIterator, i::Int = 0)
    state = iter.tracker.state
    if i > 0
        state.code == TrackerCode.TRACKING || return nothing
        # Rejected steps retry from the same point, so only accepted ones are yielded.
        accepted = false
        while !accepted && state.code == TrackerCode.TRACKING
            accepted = step!(iter.tracker)
        end
        accepted || return nothing
    end
    return _current_x_t(iter), i + 1
end

# ── Path info ───────────────────────────────────────────────────────────────

"""
    PathStep

One attempted step of a tracked path, an element of [`PathInfo`](@ref).

The step starts at arc length `s` and spans `Δs`. `ω` and `μ` are the Newton
contraction and accuracy certificates it was taken with, `accuracy` the achieved
accuracy, `τ` the trust region and `cond` the condition estimate of the Jacobian
at the point it starts from. `Δx₀` is the first Newton update, `Δx_t` the
predictor's local error, `Δx̂x` the distance from the predicted to the corrected
point and `norm_x` the sup norm of the point the step ended at. `accepted` says
whether the step was taken, `extended_prec` whether it ran in extended precision.
"""
struct PathStep
    s::Float64
    Δs::Float64
    ω::Float64
    μ::Float64
    accuracy::Float64
    τ::Float64
    cond::Float64
    Δx₀::Float64
    Δx_t::Float64
    Δx̂x::Float64
    norm_x::Float64
    accepted::Bool
    extended_prec::Bool
end

"""
    PathInfo

Per-step record of one tracked path, built by [`path_info`](@ref). Behaves as a
vector of [`PathStep`](@ref), one entry per attempted step, and additionally
carries the path's `return_code`, `n_factorizations` and `n_ldivs`.

```julia
info = path_info(tracker, x₀)
filter(step -> !step.accepted, info)   # the rejected steps
map(step -> step.Δs, info)             # the step sizes
```
"""
struct PathInfo <: AbstractVector{PathStep}
    steps::Vector{PathStep}
    return_code::TrackerCode.T
    n_factorizations::Int
    n_ldivs::Int
end

Base.size(info::PathInfo) = size(info.steps)
Base.getindex(info::PathInfo, i::Int)::PathStep = info.steps[i]
Base.IndexStyle(::Type{PathInfo}) = Base.IndexLinear()

"""
    accepted_steps(info::PathInfo)

Number of accepted steps in `info`.
"""
accepted_steps(info::PathInfo)::Int = count(step -> step.accepted, info.steps)

"""
    rejected_steps(info::PathInfo)

Number of rejected steps in `info`.
"""
rejected_steps(info::PathInfo)::Int = length(info.steps) - accepted_steps(info)

"""
    steps(info::PathInfo)

Number of attempted steps in `info` (accepted + rejected), the same as
`length(info)`.
"""
steps(info::PathInfo)::Int = length(info.steps)

"""
    path_info(tracker::Tracker, x₀, t₁ = 1.0, t₀ = 0.0) -> PathInfo

Track `x₀` from `t₁` to `t₀` and record every attempted step as a
[`PathStep`](@ref). Displaying the result prints the step table, which
[`path_table`](@ref) prints on its own.
"""
function path_info(
        tracker::Tracker, x₀::AbstractVector{<:Number},
        t₁::Number = 1.0, t₀::Number = 0.0,
    )::PathInfo
    state = tracker.state
    pred = tracker.predictor
    β_τ = tracker.options.β_τ
    steps = PathStep[]

    init!(tracker, x₀, ComplexF64(t₁), ComplexF64(t₀))
    while state.code == TrackerCode.TRACKING
        s = real(state.segment.t)
        Δs = real(state.segment.Δt)
        Δx_t = pred.local_error * abs(state.segment.Δt)^pred.order
        τ = β_τ * pred.trust_region
        ω = state.ω
        μ = state.μ
        acc = state.accuracy
        # Not `state.cond_J_ẋ`: that lags a step behind and is not rewritten on a reject.
        cond = pred.cond_H_x
        extended_prec = state.extended_prec

        accepted = step!(tracker)

        push!(
            steps,
            PathStep(
                s, Δs, ω, μ, acc, τ, cond, state.norm_Δx₀, Δx_t,
                weighted_distance(state.x, state.x̂, state.norm),
                maximum(fast_abs, state.x), accepted, extended_prec,
            ),
        )
    end

    return PathInfo(
        steps, state.code,
        state.jacobian.factorizations[], state.jacobian.ldivs[],
    )
end

# ── Step table ──────────────────────────────────────────────────────────────

const _PATH_INFO_HEADER = (
    "", "s", "Δs", "ω", "|Δx₀|", "h₀", "acc", "μ", "τ", "Δx_t", "Δpred", "cond", "|x|",
)
const _PATH_INFO_NCOLS = length(_PATH_INFO_HEADER)
const _PathRow = NTuple{_PATH_INFO_NCOLS, String}
const _PATH_INFO_ELISION = ntuple(_ -> "⋮", Val(_PATH_INFO_NCOLS))

_path_info_cell(v::Float64)::String = Printf.@sprintf("%.2g", v)

function _path_info_row(step::PathStep)::_PathRow
    return (
        step.accepted ? "✓" : "✗",
        _path_info_cell(step.s),
        _path_info_cell(step.Δs),
        _path_info_cell(step.ω),
        _path_info_cell(step.Δx₀),
        _path_info_cell(step.ω * step.Δx₀),
        _path_info_cell(step.accuracy),
        step.extended_prec ? string(_path_info_cell(step.μ), "*") :
            _path_info_cell(step.μ),
        _path_info_cell(step.τ),
        _path_info_cell(step.Δx_t),
        _path_info_cell(step.Δx̂x),
        _path_info_cell(step.cond),
        _path_info_cell(step.norm_x),
    )
end

# The 12 lines left free carry the summary, the three rules, the header and the
# elision row.
function _path_table_rows(io::IO, n::Int)::Tuple{UnitRange{Int}, UnitRange{Int}}
    get(io, :limit, false) || return (1:n, 1:0)
    k = max(displaysize(io)[1] - 12, 10)
    n <= k && return (1:n, 1:0)
    head = cld(k, 2)
    return (1:head, (n - (k - head) + 1):n)
end

function _print_rule(io::IO, widths::Vector{Int}, l::Char, m::Char, r::Char)::Nothing
    print(io, l)
    for j in eachindex(widths)
        j > 1 && print(io, m)
        print(io, "─"^(widths[j] + 2))
    end
    println(io, r)
    return nothing
end

# The mark column is left-aligned, the numbers right-aligned.
function _print_row(io::IO, widths::Vector{Int}, row::_PathRow)::Nothing
    print(io, "│")
    for j in eachindex(widths)
        cell = j == 1 ? Base.rpad(row[j], widths[j]) : Base.lpad(row[j], widths[j])
        print(io, " ", cell, " │")
    end
    println(io)
    return nothing
end

"""
    path_table([io::IO = stdout], info::PathInfo)

Print the step table of `info`: one row per attempted step, `✓`/`✗` marking
whether it was accepted and a starred `μ` a step taken in extended precision.

Every step is printed unless `io` limits its height, as the display of `info`
itself does.
"""
function path_table(io::IO, info::PathInfo)::Nothing
    head, tail = _path_table_rows(io, length(info))
    rows = Vector{_PathRow}(undef, length(head) + length(tail))
    k = 0
    for part in (head, tail), i in part
        k += 1
        rows[k] = _path_info_row(info[i])
    end

    widths = collect(map(length, _PATH_INFO_HEADER))
    for row in rows, j in eachindex(widths)
        widths[j] = max(widths[j], length(row[j]))
    end

    _print_rule(io, widths, '┌', '┬', '┐')
    _print_row(io, widths, _PATH_INFO_HEADER)
    _print_rule(io, widths, '├', '┼', '┤')
    for (i, row) in enumerate(rows)
        i == length(head) + 1 && _print_row(io, widths, _PATH_INFO_ELISION)
        _print_row(io, widths, row)
    end
    _print_rule(io, widths, '└', '┴', '┘')
    return nothing
end

path_table(info::PathInfo)::Nothing = path_table(stdout, info)

function Base.show(io::IO, info::PathInfo)
    print(
        io, "PathInfo(", length(info), " steps, ", accepted_steps(info), " ✓ / ",
        rejected_steps(info), " ✗, ", info.return_code, ")",
    )
    return
end

function Base.show(io::IO, ::MIME"text/plain", info::PathInfo)
    println(io, "PathInfo:")
    println(io, " • return code → ", info.return_code)
    println(
        io, " • steps (✓/✗) → ", length(info), " (", accepted_steps(info), " / ",
        rejected_steps(info), ")",
    )
    println(io, " • factorizations → ", info.n_factorizations)
    println(io, " • ldivs → ", info.n_ldivs)
    any(step -> step.extended_prec, info.steps) &&
        println(io, " • a starred μ marks a step taken in extended precision")
    path_table(io, info)
    return
end
