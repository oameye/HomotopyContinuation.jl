# Valuation — Puiseux series valuation tracking for endgame detection.
# Monitors per-coordinate valuations to detect singular endpoints and
# at-infinity paths during the endgame phase.

"""
    Valuation

Tracks per-coordinate Puiseux series valuations along a homotopy path.
Uses two-path update: direct Taylor formula for m==1, finite differences for m>1.

**Mutable justification:** `logt_1`, `logt_2`, `samples` update every endgame step.
All FSVec buffer fields are `const` — pre-allocated, never reassigned.
"""
mutable struct Valuation
    # Current valuations
    const val_x::FSVec{Float64}
    const val_tẋ::FSVec{Float64}
    const Δval_x::FSVec{Float64}
    const Δval_tẋ::FSVec{Float64}
    # 2-point history for finite differences (m > 1 path)
    # Slot 1 = older, slot 2 = newer; rotated each update
    const val_x_1::FSVec{Float64}
    const val_x_2::FSVec{Float64}
    const val_ẋ_1::FSVec{Float64}
    const val_ẋ_2::FSVec{Float64}
    const logx_1::FSVec{Float64}
    const logx_2::FSVec{Float64}
    const logẋ_1::FSVec{Float64}
    const logẋ_2::FSVec{Float64}
    logt_1::Float64
    logt_2::Float64
    samples::Int
end

function Valuation(n::Int)
    _nanvec() = FSVec{Float64}(fill(NaN, n))
    return Valuation(
        _nanvec(), _nanvec(), _nanvec(), _nanvec(),  # val_x, val_tẋ, Δval_x, Δval_tẋ
        _nanvec(), _nanvec(),                          # val_x_1, val_x_2
        _nanvec(), _nanvec(),                          # val_ẋ_1, val_ẋ_2
        _nanvec(), _nanvec(),                          # logx_1, logx_2
        _nanvec(), _nanvec(),                          # logẋ_1, logẋ_2
        NaN, NaN,                                       # logt_1, logt_2
        0,                                              # samples
    )
end

function init!(val::Valuation)::Nothing
    for v in (
            val.val_x, val.val_tẋ, val.Δval_x, val.Δval_tẋ,
            val.val_x_1, val.val_x_2, val.val_ẋ_1, val.val_ẋ_2,
            val.logx_1, val.logx_2, val.logẋ_1, val.logẋ_2,
        )
        fill!(v, NaN)
    end
    val.logt_1 = NaN
    val.logt_2 = NaN
    val.samples = 0
    return nothing
end

# Non-uniform 2-point finite difference (v2 parity: valuation.jl:138)
@inline function _finite_diff(
        f3::Float64, s3::Float64, f2::Float64, s2::Float64, f1::Float64, s1::Float64,
    )::Float64
    Δ1 = s3 - s1
    Δ2 = s3 - s2
    Δ12 = s1 - s2
    return (Δ2 * f1) / (Δ12 * Δ1) - ((Δ12 + Δ2) * f2) / (Δ12 * Δ2) - (Δ12 * f3) / (Δ1 * Δ2)
end

# Valuation: ν(x, ẋ, t) = t * Re(x̄·ẋ) / |x|²
@inline function _val(x::ComplexF64, ẋ::ComplexF64, t::Float64)::Float64
    return t * (real(x) * real(ẋ) + imag(x) * imag(ẋ)) / abs2(x)
end

# Direct Taylor path: ν and dν/dt from (x, ẋ, x̃², t)
# where x̃² = 2*x² (the second-order Taylor coefficient scaled by 2)
#
# ν(t) = t * l(t)  where  l = Re(x̄·ẋ) / |x|²
# dν/dt = l + t * dl/dt
# dl/dt = [Re(ẋ·ẋ) + Re(x̄·x'') - 2*Re(x̄·ẋ)² / |x|²] / |x|²
#       = [(|ẋ|² + Re(x̄·x̃²)) - 2*(xẋ)²/r2] / r2
@inline function _val_dval(
        x::ComplexF64, ẋ::ComplexF64, x̃2::ComplexF64, t::Float64,
    )::Tuple{Float64, Float64}
    r2 = abs2(x)
    r2 == 0.0 && return (0.0, 0.0)
    xẋ = real(x) * real(ẋ) + imag(x) * imag(ẋ)
    l = xẋ / r2
    ν = t * l
    # x̃² is the second derivative x'' (twice the quadratic Taylor coefficient).
    xx̃2 = real(x) * real(x̃2) + imag(x) * imag(x̃2)
    ẋẋ = abs2(ẋ)
    dl_dt = (ẋẋ + xx̃2) / r2 - 2.0 * xẋ^2 / (r2 * r2)
    dν_dt = l + t * dl_dt
    return (ν, dν_dt)
end

"""
    estimate_winding_number(val, n, max_m) -> (m::Int, err::Float64)

Test winding numbers m=1..max_m by checking if `m * val_tẋ[i]` is near-integer
for all coordinates. Returns the best-fit m and its max error.
"""
function estimate_winding_number(val::Valuation, n::Int, max_m::Int)::Tuple{Int, Float64}
    best_m = 1
    best_err = Inf
    @inbounds for m in 1:max_m
        err = check_winding_number(val, m, n)
        if err < best_err
            best_m = m
            best_err = err
        end
    end
    return (best_m, best_err)
end

function check_winding_number(val::Valuation, m::Int, n::Int)::Float64
    err = 0.0
    @inbounds for i in 1:n
        mv = m * val.val_tẋ[i]
        err = max(err, abs(round(mv) - mv))
    end
    return err
end

function is_finite(
        val::Valuation;
        finite_tol::Float64,
        zero_is_finite::Bool,
        max_winding_number::Int,
    )::Bool
    δ = inv(max_winding_number)
    n = length(val.val_x)
    @inbounds for i in 1:n
        val_xi = val.val_x[i]
        val_tẋi = val.val_tẋ[i]

        if abs(val_xi) < finite_tol
            if !(abs(val.Δval_x[i]) < finite_tol) || val_tẋi < 0.5 * δ
                return false
            end
        elseif zero_is_finite && val_xi > (δ - finite_tol)
            ε∞ = _at_infinity_tol(val_xi, val_tẋi, val.Δval_x[i], val.Δval_tẋ[i])
            if !(ε∞ < finite_tol)
                return false
            end
        else
            return false
        end
    end
    return true
end

"""
    update!(val, pred, t)

Update valuations from predictor Taylor coefficients at homotopy parameter `t`.
- m==1 (or insufficient history): direct Taylor formula via `_val_dval`
- m>1 with history: nested finite differences
Always rotates history buffers.
"""
function update!(val::Valuation, pred::Predictor, t::Float64)::Nothing
    n = size(pred.tx3.data, 2)
    logt = log(t)
    use_fd = pred.winding_number > 1 && !isnan(val.logt_2)

    @inbounds for i in 1:n
        xi = pred.tx3.data[1, i]
        ẋi = pred.tx3.data[2, i]
        x2i = pred.tx3.data[3, i]
        x3i = pred.tx3.data[4, i]

        logxi = log(fast_abs(xi))
        logẋi = log(fast_abs(ẋi))

        if use_fd
            # Finite-difference path (m > 1)
            νi = _val(xi, ẋi, t)
            Δνi = _finite_diff(
                νi, logt, val.val_x_2[i], val.logt_2,
                val.val_x_1[i], val.logt_1
            )
            val.val_x[i] = νi
            val.Δval_x[i] = Δνi

            val_ẋi = _finite_diff(
                logẋi, logt, val.logẋ_2[i], val.logt_2,
                val.logẋ_1[i], val.logt_1
            )
            Δval_ẋi = _finite_diff(
                val_ẋi, logt, val.val_ẋ_2[i], val.logt_2,
                val.val_ẋ_1[i], val.logt_1
            )
            val.val_tẋ[i] = val_ẋi + 1.0
            val.Δval_tẋ[i] = Δval_ẋi
        else
            # Direct Taylor path (m == 1 or insufficient history)
            νi, ν1i = _val_dval(xi, ẋi, 2.0 * x2i, t)
            val.val_x[i] = νi
            val.Δval_x[i] = t * ν1i

            val_ẋi, ν1_ẋi = _val_dval(ẋi, 2.0 * x2i, 6.0 * x3i, t)
            val.val_tẋ[i] = val_ẋi + 1.0
            val.Δval_tẋ[i] = t * ν1_ẋi
        end

        # Always rotate history
        val.val_x_1[i] = val.val_x_2[i]
        val.val_x_2[i] = val.val_x[i]
        val.val_ẋ_1[i] = val.val_ẋ_2[i]
        val.val_ẋ_2[i] = val.val_tẋ[i] - 1.0
        val.logx_1[i] = val.logx_2[i]
        val.logx_2[i] = logxi
        val.logẋ_1[i] = val.logẋ_2[i]
        val.logẋ_2[i] = logẋi
    end

    val.logt_1 = val.logt_2
    val.logt_2 = logt
    val.samples += 1
    return nothing
end
