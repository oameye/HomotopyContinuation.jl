# ---------------------------------------------------------------------------
# Endgame linear algebra helpers
# ---------------------------------------------------------------------------

@inline function scaled_inf_norm_matrix(
        ws::MatrixWorkspace,
        row_scaling::FSVec{Float64},
        col_scaling::FSVec{Float64},
    )::Float64
    A = ws.A
    m, n = size(A)
    norm_val = -Inf
    @inbounds for i in 1:m
        row_sum = 0.0
        for j in 1:n
            row_sum += fast_abs(A[i, j]) * col_scaling[j]
        end
        norm_val = @fastmath max(norm_val, row_sum * row_scaling[i])
    end
    return norm_val
end

# |H_i| against the size the terms of row i reach at this point, so the same
# threshold works for a system whose Jacobian row sums are O(1) and one where
# they are O(10^40). A zero row scale leaves the residual unscaled.
@inline function max_relative_residual(
        r::FSVec{ComplexF64},
        A::FSMat{ComplexF64},
        col_scaling::FSVec{Float64},
    )::Float64
    m, n = size(A)
    worst = 0.0
    @inbounds for i in 1:m
        scale = 0.0
        for j in 1:n
            scale += fast_abs(A[i, j]) * col_scaling[j]
        end
        rᵢ = fast_abs(r[i])
        worst = @fastmath max(worst, scale > 0.0 ? rᵢ / scale : rᵢ)
    end
    return worst
end

# Row-scaled-only inf norm for J₀ (no col_scaling — used for singular endgame acceptance)
@inline function row_scaled_inf_norm_matrix(
        ws::MatrixWorkspace,
        row_scaling::FSVec{Float64},
    )::Float64
    A = ws.A
    m, n = size(A)
    norm_val = -Inf
    @inbounds for i in 1:m
        row_sum = 0.0
        for j in 1:n
            row_sum += fast_abs(A[i, j])
        end
        norm_val = @fastmath max(norm_val, row_sum * row_scaling[i])
    end
    return norm_val
end

function scaled_cond(
        ws::MatrixWorkspace,
        row_scaling::FSVec{Float64},
        col_scaling::FSVec{Float64},
    )::Float64
    m, n = size(ws)
    if m == n == 1
        return inv(row_scaling[1] * fast_abs(ws.A[1, 1]) * col_scaling[1])
    elseif m > n
        ws.factorized || factorize!(ws)
        rmax = -Inf
        rmin = Inf
        @inbounds for i in 1:n
            ri = fast_abs(ws.qr.factors[i, i]) * col_scaling[i]
            rmax = max(rmax, ri)
            rmin = min(rmin, ri)
        end
        return rmax / rmin
    else
        ws.factorized || factorize!(ws)
        return scaled_inf_norm_matrix(ws, row_scaling, col_scaling) *
            _inverse_inf_norm_est(
            ws.lu, row_scaling, col_scaling, ws.row_scaling, ws.scaled,
            ws.inf_norm_est_work, ws.inf_norm_est_rwork,
        )
    end
end

# Raw tolerance: used by is_finite in valuation.jl
@inline function at_infinity_tol(
        val_x::Float64,
        val_tẋ::Float64,
        Δval_x::Float64,
        Δval_tẋ::Float64,
    )::Float64
    if abs(val_x) < 1.0e-30 || abs(val_tẋ) < 1.0e-30
        return Inf
    end
    ε∞ = max(
        abs(1.0 - val_tẋ / val_x),
        abs(Δval_x / val_x),
        abs(Δval_tẋ / val_tẋ),
    )
    return isfinite(ε∞) ? ε∞ : Inf
end

# Gated version for check_at_infinity!: only returns finite ε∞ when the valuation
# actually indicates divergence (val_x < 0 → ∞) or convergence to zero (val_x > 0 → 0).
# Without this gate, regular coordinates with small ε∞ get spuriously marked.
@inline function at_infinity_tol_gated(
        val_x::Float64,
        val_tẋ::Float64,
        Δval_x::Float64,
        Δval_tẋ::Float64,
        finite_tol::Float64,
        zero_is_at_infinity::Bool,
    )::Float64
    ε∞ = at_infinity_tol(val_x, val_tẋ, Δval_x, Δval_tẋ)
    if !isfinite(ε∞)
        return Inf
    end
    if val_x + ε∞ < -finite_tol
        return ε∞
    elseif zero_is_at_infinity && val_x - ε∞ > finite_tol
        return ε∞
    else
        return Inf
    end
end

# A path that exhausts its step budget still holds a valuation. If a coordinate
# is a standing at-infinity candidate there, its divergence is what stopped the
# path, and the only thing missing is the coordinate growth the step budget ran
# out before reaching. How far a path gets before the budget runs out varies with
# the seed, so without this the same divergent path is reported at-infinity on
# one seed and out-of-steps on another.
function check_at_infinity_at_giveup!(eg::EndgameTracker)::Bool
    (eg.options.at_infinity_check && eg.val.samples >= 2) || return false
    return check_at_infinity!(eg, true)
end

@inline function clear_at_infinity_candidate!(state::EndgameState, i::Int)::Nothing
    state.at_inf_active[i] = false
    state.at_inf_starts[i] = NaN
    state.at_inf_abs_coords[i] = NaN
    state.at_inf_conds[i] = NaN
    return nothing
end

function ensure_endgame_scaling!(state::EndgameState, tracker::Tracker)::Nothing
    if all(iszero, state.col_scaling)
        @inbounds for i in eachindex(state.col_scaling)
            state.col_scaling[i] = tracker.state.norm.weights[i]
        end
    end
    return nothing
end
