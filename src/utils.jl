fast_abs(z::Complex) = sqrt(abs2(z))
fast_abs(x::Real) = abs(x)

const DEFAULT_REAL_TOL = 1.0e-6

nanmin(a, b) = isnan(a) ? b : (isnan(b) ? a : min(a, b))
nanmax(a, b) = isnan(a) ? b : (isnan(b) ? a : max(a, b))

function _stable_sort!(values::Vector{T}, lt::F)::Vector{T} where {T, F}
    for i in 2:length(values)
        value = values[i]
        j = i - 1
        while j >= 1 && lt(value, values[j])
            values[j + 1] = values[j]
            j -= 1
        end
        values[j + 1] = value
    end
    return values
end

function _stable_sort_by!(values::Vector{T}, by::F)::Vector{T} where {T, F}
    for i in 2:length(values)
        value = values[i]
        value_key = by(value)
        j = i - 1
        while j >= 1 && isless(value_key, by(values[j]))
            values[j + 1] = values[j]
            j -= 1
        end
        values[j + 1] = value
    end
    return values
end

function nthroot(x::Real, N::Integer)
    return if N == 4
        sqrt(sqrt(x))
    elseif N == 2
        sqrt(x)
    elseif N == 3
        cbrt(x)
    elseif N == 1
        x
    elseif N == 0
        one(x)
    else
        x^(1 / N)
    end
end

# SegmentStepper: arc-length parametrization of a path segment from start to target.
# s ∈ [0, abs_Δ] is the current arc-length parameter.
# forward == true  → s increases from 0 to abs_Δ (abs(start) < abs(target))
# forward == false → s decreases from abs_Δ to 0
#
# Mutable justification: s and s′ are advanced every tracker step.
# start, target, abs_Δ, forward are reset via reinit! for each new path segment.
mutable struct SegmentStepper
    start::ComplexF64
    target::ComplexF64
    abs_Δ::Float64
    forward::Bool
    s::Float64
    s′::Float64
end

function SegmentStepper(start::ComplexF64, target::ComplexF64)
    abs_Δ = abs(target - start)
    forward = abs(start) < abs(target)
    s = forward ? 0.0 : abs_Δ
    return SegmentStepper(start, target, abs_Δ, forward, s, s)
end
SegmentStepper(start::Number, target::Number) =
    SegmentStepper(ComplexF64(start), ComplexF64(target))

function reinit!(S::SegmentStepper, start::ComplexF64, target::ComplexF64)::Nothing
    S.start = start
    S.target = target
    S.abs_Δ = abs(target - start)
    S.forward = abs(start) < abs(target)
    S.s = S.forward ? 0.0 : S.abs_Δ
    S.s′ = S.s
    return nothing
end
reinit!(S::SegmentStepper, start::Number, target::Number) =
    reinit!(S, ComplexF64(start), ComplexF64(target))

is_done(S::SegmentStepper)::Bool = S.forward ? S.s == S.abs_Δ : S.s == 0.0

step_success!(S::SegmentStepper) = (S.s = S.s′; S)

function propose_step!(S::SegmentStepper, Δs::Real)
    if S.forward
        S.s′ = min(S.s + Δs, S.abs_Δ)
    else
        S.s′ = max(S.s - Δs, 0.0)
    end
    return S
end

dist_to_target(S::SegmentStepper)::Float64 = S.forward ? S.abs_Δ - S.s : S.s

function _t_helper(
        start::ComplexF64,
        target::ComplexF64,
        s::Float64,
        Δ::Float64,
        forward::Bool,
    )::ComplexF64
    return if forward
        if s == 0.0
            return start
        elseif s == Δ
            return target
        else
            return start + (s / Δ) * (target - start)
        end
    else
        if s == Δ
            return start
        elseif s == 0.0
            return target
        else
            return target + (s / Δ) * (start - target)
        end
    end
end

function Base.getproperty(S::SegmentStepper, sym::Symbol)
    if sym === :Δs
        s = getfield(S, :s)
        s′ = getfield(S, :s′)
        forward = getfield(S, :forward)
        return forward ? s′ - s : s - s′
    elseif sym === :t
        return _t_helper(
            getfield(S, :start),
            getfield(S, :target),
            getfield(S, :s),
            getfield(S, :abs_Δ),
            getfield(S, :forward),
        )
    elseif sym === :t′
        return _t_helper(
            getfield(S, :start),
            getfield(S, :target),
            getfield(S, :s′),
            getfield(S, :abs_Δ),
            getfield(S, :forward),
        )
    elseif sym === :Δt
        s = getfield(S, :s)
        s′ = getfield(S, :s′)
        abs_Δ = getfield(S, :abs_Δ)
        start = getfield(S, :start)
        target = getfield(S, :target)
        forward = getfield(S, :forward)
        return if forward
            ((s′ - s) / abs_Δ) * (target - start)
        else
            ((s - s′) / abs_Δ) * (target - start)
        end
    else
        return getfield(S, sym)
    end
end

function Base.show(io::IO, S::SegmentStepper)
    print(io, "SegmentStepper:")
    for field in (:start, :target, :t, :Δt)
        print(io, "\n • ", field, " → ", getproperty(S, field))
    end
    return
end
