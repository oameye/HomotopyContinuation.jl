# Predictor for homotopy continuation path tracking.
# Computes Taylor coefficients of the solution path and performs Pade (2,1) prediction.

@enumx PredictionMethod::Int8 begin
    PADE21
    HERMITE
end

"""
    Predictor

Stores Taylor coefficient data and scratch buffers for predicting the next point
along a homotopy continuation path. Uses implicit differentiation of H(x(t),t)=0
to compute Taylor coefficients up to order 3, then applies Pade (2,1) approximation.

**Mutable justification:** `method`, `use_hermite`, `trust_region`, `local_error`,
`cond_H_x`, `t`, `tx_norm`, `prev_t`, and `winding_number` are updated at every
predictor step. Buffer fields (`tx3`, `tv2`, `xtemp`, `u`, `prev_tx1`) are `const`.
"""
mutable struct Predictor
    method::PredictionMethod.T
    const order::Int
    use_hermite::Bool
    trust_region::Float64
    local_error::Float64
    cond_H_x::Float64
    const tx3::TaylorVector{4, ComplexF64}        # Master storage: rows 1-4 = x0,x1,x2,x3
    const tv2::TaylorVector{3, ComplexF64}        # Scratch for taylor!(Val(2)) calls
    t::ComplexF64
    tx_norm::NTuple{4, Float64}                   # Norms of x0,x1,x2,x3
    const xtemp::FSVec{ComplexF64}                # Scratch for ldiv result
    const u::FSVec{ComplexF64}                     # Scratch for homotopy eval output
    const prev_tx1::TaylorVector{2, ComplexF64}   # Previous [x, dx] for Hermite
    prev_t::ComplexF64
    winding_number::Int
end

function Predictor(m::Int, n::Int)
    return Predictor(
        PredictionMethod.PADE21,
        4,                                             # order
        true,                                          # use_hermite
        Inf,                                           # trust_region
        NaN,                                           # local_error
        NaN,                                           # cond_H_x
        TaylorVector{4, ComplexF64}(n),                # tx3
        TaylorVector{3, ComplexF64}(n),                # tv2
        complex(NaN),                                  # t
        (NaN, NaN, NaN, NaN),                          # tx_norm
        FSVec{ComplexF64}(zeros(ComplexF64, n)),        # xtemp
        FSVec{ComplexF64}(zeros(ComplexF64, m)),        # u
        TaylorVector{2, ComplexF64}(n),                # prev_tx1
        complex(NaN),                                  # prev_t
        1,                                             # winding_number
    )
end

"""
    update!(pred, H, x, t, J, norm)

Compute Taylor coefficients of the solution path at `(x, t)` by implicit
differentiation of H(x(t),t) = 0. Stores orders 0-3 in `pred.tx3` and
computes the trust region radius.
"""
function update!(
        pred::Predictor,
        H::HomotopyEvaluator,
        x::FSVec{ComplexF64},
        t::ComplexF64,
        J::Jacobian,
        norm::WeightedNorm,
    )::Nothing
    u = pred.u
    xtemp = pred.xtemp
    n = length(x)

    # Save previous Taylor data for Hermite predictor
    if !isnan(pred.t)
        @inbounds for i in 1:n
            pred.prev_tx1.data[1, i] = pred.tx3.data[1, i]
            pred.prev_tx1.data[2, i] = pred.tx3.data[2, i]
        end
        pred.prev_t = pred.t
    end
    pred.t = t

    # -- Order 0: store x --
    @inbounds for i in 1:n
        pred.tx3.data[1, i] = x[i]
    end
    n0 = weighted_norm(x, norm)

    # -- Order 1: dx = -J^{-1} * taylor!(u, Val(1), H, x, t) --
    taylor!(u, Val(1), H, x, t)
    @inbounds for i in eachindex(u)
        u[i] = -u[i]
    end
    LA.ldiv!(xtemp, J, u)
    @inbounds for i in 1:n
        pred.tx3.data[2, i] = xtemp[i]
    end
    n1 = weighted_norm(xtemp, norm)

    pred.cond_H_x = LA.cond(J)

    # -- Order 2: x2 = -J^{-1} * taylor!(u, Val(2), H, tv2, t) --
    # Populate tv2 with [x0, x1, 0]
    @inbounds for i in 1:n
        pred.tv2.data[1, i] = pred.tx3.data[1, i]
        pred.tv2.data[2, i] = pred.tx3.data[2, i]
        pred.tv2.data[3, i] = zero(ComplexF64)
    end
    taylor!(u, Val(2), H, pred.tv2, t)
    @inbounds for i in eachindex(u)
        u[i] = -u[i]
    end
    LA.ldiv!(xtemp, J, u)
    @inbounds for i in 1:n
        pred.tx3.data[3, i] = xtemp[i]
    end
    n2 = weighted_norm(xtemp, norm)

    # -- Order 3: x3 = -J^{-1} * taylor!(u, Val(3), H, tx3, t) --
    # tx3 already has [x0, x1, x2, ?] -- set row 4 to zero first
    @inbounds for i in 1:n
        pred.tx3.data[4, i] = zero(ComplexF64)
    end
    taylor!(u, Val(3), H, pred.tx3, t)
    @inbounds for i in eachindex(u)
        u[i] = -u[i]
    end
    LA.ldiv!(xtemp, J, u)
    @inbounds for i in 1:n
        pred.tx3.data[4, i] = xtemp[i]
    end
    n3 = weighted_norm(xtemp, norm)

    pred.tx_norm = (n0, n1, n2, n3)

    _compute_trust_region!(pred)
    return nothing
end

"""
    _compute_trust_region!(pred)

Estimate the trust region radius from the Taylor coefficient magnitudes.
Uses component-wise ratio of successive coefficients.
"""
function _compute_trust_region!(pred::Predictor)::Nothing
    n = size(pred.tx3.data, 2)
    tau = Inf
    tol = 1.0e-14

    @inbounds for i in 1:n
        c1 = fast_abs(pred.tx3.data[2, i])  # |x1|
        c2 = fast_abs(pred.tx3.data[3, i])  # |x2|
        c3 = fast_abs(pred.tx3.data[4, i])  # |x3|

        lambda = max(1.0e-6, c1)
        c1n = c1 / lambda
        c2n = c2 / lambda^2
        c3n = c3 / lambda^3

        thresh = tol * max(c1n, c2n, c3n)
        if c1n <= thresh && c2n <= thresh && c3n <= thresh
            continue
        end
        if c2n <= thresh
            continue
        end
        if c3n > thresh
            tau_i = c2n / c3n / lambda
            tau = min(tau, tau_i)
        end
    end

    if !isfinite(tau)
        if pred.tx_norm[3] > 0 && pred.tx_norm[4] > 0
            tau = pred.tx_norm[3] / pred.tx_norm[4]
        else
            tau = pred.tx_norm[1] / max(pred.tx_norm[1], pred.tx_norm[2], pred.tx_norm[3], pred.tx_norm[4])
        end
    end

    pred.trust_region = isfinite(tau) ? tau : 1.0
    return nothing
end

"""
    compute_local_error!(pred, x_hat, x, norm, ds)

Compute the local error estimate from the difference between predicted (`x_hat`)
and corrected (`x`) points, normalized by the step size raised to the predictor order.
"""
function compute_local_error!(
        pred::Predictor,
        x_hat::FSVec{ComplexF64},
        x::FSVec{ComplexF64},
        norm::WeightedNorm,
        ds::Float64,
    )::Nothing
    if isnan(ds) || ds == 0.0
        pred.local_error = NaN
        return nothing
    end
    err = weighted_distance(x, x_hat, norm)
    pred.local_error = err / ds^pred.order
    return nothing
end

"""
    predict!(x_hat, pred, dt)

Compute the Pade (2,1) prediction for the next point along the path.
Falls back to quadratic Taylor when the Pade denominator is degenerate.
"""
function predict!(
        x_hat::FSVec{ComplexF64},
        pred::Predictor,
        dt::ComplexF64,
    )::Nothing
    n = length(x_hat)
    data = pred.tx3.data
    tol2 = 1.0e-28  # tol^2 for abs2 comparisons

    @inbounds for i in 1:n
        x0 = data[1, i]
        x1 = data[2, i]
        x2 = data[3, i]
        x3 = data[4, i]

        c2sq = abs2(x2)
        c3sq = abs2(x3)

        if c3sq < tol2 || c2sq < tol2
            # Quadratic Taylor fallback
            x_hat[i] = x0 + dt * (x1 + dt * x2)
        else
            # Pade (2,1): x_hat = x0 + dt*(x1 + dt*x2/(1 - dt*x3/x2))
            delta = 1 - dt * x3 / x2
            if abs2(delta) < tol2
                x_hat[i] = x0 + dt * (x1 + dt * x2)
            else
                x_hat[i] = x0 + dt * (x1 + dt * x2 / delta)
            end
        end
    end

    return nothing
end
