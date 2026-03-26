## ToricHomotopy: H(x,t) = F(x; p(t)) where p_j(t) = c_j * t^{w_j}
#
# Part of polyhedral homotopy — reverses toric degeneration from t=0 to t=1.
# At t=0: only monomials with w=0 survive (the binomial face).
# At t=1: all coefficients are active (full system).

struct ToricHomotopy <: AbstractHomotopy
    system::SystemEvaluator
    system_coeffs::FSVec{ComplexF64}     # base coefficients c_j
    weights::FSVec{Float64}              # w_j per coefficient
    t_weights::FSVec{Float64}            # cached t^{w_j} for real t
    coeffs::FSVec{ComplexF64}            # c_j * t^{w_j} -- current values (mutated)
    dt_coeffs::FSVec{ComplexF64}         # w_j * c_j * t^{w_j-1} (mutated)
    t_cache::Base.RefValue{ComplexF64}
    dt_cache::Base.RefValue{ComplexF64}  # cached t for dt_coeffs
    # Scratch
    u_cache::FSVec{ComplexF64}
    U_cache::FSMat{ComplexF64}
end

function ToricHomotopy(
        system::SystemEvaluator,
        system_coeffs::AbstractVector{<:AbstractVector{ComplexF64}},
    )
    nparams = sum(length, system_coeffs)
    @assert nparams == nparameters(system)

    flat_coeffs = ComplexF64[]
    for c in system_coeffs
        append!(flat_coeffs, c)
    end

    m, n = size(system)
    return ToricHomotopy(
        system,
        FSVec{ComplexF64}(flat_coeffs),
        FSVec{Float64}(zeros(nparams)),                  # weights
        FSVec{Float64}(zeros(nparams)),                  # t_weights
        FSVec{ComplexF64}(zeros(ComplexF64, nparams)),   # coeffs
        FSVec{ComplexF64}(zeros(ComplexF64, nparams)),   # dt_coeffs
        Ref(complex(NaN)),
        Ref(complex(NaN)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSMat{ComplexF64}(zeros(ComplexF64, m, n)),
    )
end

Base.size(H::ToricHomotopy) = size(H.system)

## ── update_weights! ──────────────────────────────────────────────────────────

function update_weights!(
        H::ToricHomotopy,
        support::Vector{Matrix{Int32}},
        lifting::Vector{Vector{Int32}},
        cell::MixedSubdivisions.MixedCell;
        min_weight::Union{Nothing, Float64} = nothing,
        max_weight::Union{Nothing, Float64} = nothing,
    )::Tuple{Float64, Float64}
    l = 1
    s_max = 0.0
    s_min = Inf
    n = length(cell.normal)

    for (i, Ai) in enumerate(support)
        wi = lifting[i]
        betai = cell.β[i]
        mi = size(Ai, 2)
        ai, bi = cell.indices[i]
        for j in 1:mi
            if j == ai || j == bi
                H.weights[l] = 0.0
            else
                sij = Float64(wi[j]) - betai
                @inbounds for k in 1:n
                    sij += Float64(Ai[k, j]) * cell.normal[k]
                end
                H.weights[l] = sij
                s_max = max(s_max, sij)
                s_min = min(s_min, sij)
            end
            l += 1
        end
    end

    # Normalize weights (skip if all weights are zero — s_min stays Inf)
    if min_weight !== nothing && isfinite(s_min)
        lambda = s_min / min_weight
        @inbounds for i in eachindex(H.weights)
            H.weights[i] /= lambda
        end
        s_min, s_max = min_weight, s_max / lambda
    elseif max_weight !== nothing && s_max > 0.0
        lambda = s_max / max_weight
        @inbounds for i in eachindex(H.weights)
            H.weights[i] /= lambda
        end
        s_min, s_max = s_min / lambda, max_weight
    end

    H.t_cache[] = complex(NaN)
    H.dt_cache[] = complex(NaN)

    return s_min, s_max
end

## ── Coefficient computation ──────────────────────────────────────────────────

function _update_toric_coeffs!(H::ToricHomotopy, t::ComplexF64)::Nothing
    if H.t_cache[] == t
        return nothing
    end
    tr = real(t)
    if tr > 0 && isreal(t)
        @inbounds for i in eachindex(H.coeffs)
            tw = tr^H.weights[i]
            H.t_weights[i] = tw
            H.coeffs[i] = H.system_coeffs[i] * tw
        end
    elseif tr == 0.0 && isreal(t)
        @inbounds for i in eachindex(H.coeffs)
            H.coeffs[i] = iszero(H.weights[i]) ? H.system_coeffs[i] : zero(ComplexF64)
        end
    else
        # Complex t or negative real -- use t^w = exp(w * log(t))
        log_t = log(t)
        @inbounds for i in eachindex(H.coeffs)
            H.coeffs[i] = H.system_coeffs[i] * exp(H.weights[i] * log_t)
        end
    end
    H.t_cache[] = t
    return nothing
end

function _update_toric_dt_coeffs!(H::ToricHomotopy, t::ComplexF64)::Nothing
    if H.dt_cache[] == t
        return nothing
    end
    _update_toric_coeffs!(H, t)
    tr = real(t)
    if tr > 0 && isreal(t)
        @fastmath t_inv = inv(tr)
        @inbounds for i in eachindex(H.dt_coeffs)
            w = H.weights[i]
            if iszero(w)
                H.dt_coeffs[i] = zero(ComplexF64)
            else
                H.dt_coeffs[i] = w * H.coeffs[i] * t_inv
            end
        end
    elseif tr == 0.0 && isreal(t)
        # At t=0, only w=1 terms contribute: d/dt(c * t^w) = w * c * t^{w-1}
        # For w=1: derivative = c. For w!=1 and w>0: derivative = 0 (since t^{w-1} -> 0 for w>1).
        @inbounds for i in eachindex(H.dt_coeffs)
            w = H.weights[i]
            H.dt_coeffs[i] = isone(w) ? H.system_coeffs[i] : zero(ComplexF64)
        end
    else
        # Complex t
        @fastmath t_inv = inv(t)
        @inbounds for i in eachindex(H.dt_coeffs)
            w = H.weights[i]
            if iszero(w)
                H.dt_coeffs[i] = zero(ComplexF64)
            else
                H.dt_coeffs[i] = w * H.coeffs[i] * t_inv
            end
        end
    end
    H.dt_cache[] = t
    return nothing
end

## ── Interface methods ────────────────────────────────────────────────────────

function evaluate!(
        u::FSVec{ComplexF64}, H::ToricHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    _update_toric_coeffs!(H, t)
    evaluate!(u, H.system, x, H.coeffs)
    return nothing
end

## evaluate! DF64 variant

function evaluate!(
        u::FSVec{ComplexF64}, H::ToricHomotopy,
        x::FSVec{ComplexDF64}, t::ComplexF64,
    )::Nothing
    _update_toric_coeffs!(H, t)
    evaluate!(u, H.system, x, H.coeffs)
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64},
        H::ToricHomotopy, x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    _update_toric_coeffs!(H, t)
    evaluate_and_jacobian!(u, U, H.system, x, H.coeffs)
    return nothing
end

## taylor! order 1: dH/dt at fixed x
# Since F is linear in parameters: dH/dt = F(x; dp/dt)

function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, H::ToricHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    _update_toric_dt_coeffs!(H, t)
    evaluate!(u, H.system, x, H.dt_coeffs)
    return nothing
end

## taylor! order 2

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, H::ToricHomotopy,
        tx::TaylorVector{3, ComplexF64}, t::ComplexF64;
        incremental::Bool = false,
    )::Nothing
    _update_toric_coeffs!(H, t)
    taylor!(u, Val(2), H.system, tx, H.coeffs)
    return nothing
end

## taylor! order 3

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, H::ToricHomotopy,
        tx::TaylorVector{4, ComplexF64}, t::ComplexF64;
        incremental::Bool = false,
    )::Nothing
    _update_toric_coeffs!(H, t)
    taylor!(u, Val(3), H.system, tx, H.coeffs)
    return nothing
end

## ── set_solution! / get_solution! ────────────────────────────────────────────

function set_solution!(
        x::FSVec{ComplexF64}, ::ToricHomotopy,
        y::FSVec{ComplexF64}, ::ComplexF64,
    )::Nothing
    copyto!(x, y)
    return nothing
end

function get_solution!(
        out::FSVec{ComplexF64}, ::ToricHomotopy,
        x::FSVec{ComplexF64}, ::ComplexF64,
    )::Nothing
    copyto!(out, x)
    return nothing
end

## ── start_parameters! / target_parameters! ───────────────────────────────────

start_parameters!(::ToricHomotopy, ::FSVec{ComplexF64})::Nothing = nothing
target_parameters!(::ToricHomotopy, ::FSVec{ComplexF64})::Nothing = nothing
