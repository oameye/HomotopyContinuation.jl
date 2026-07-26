# MatrixWorkspace — efficient repeated solution of square or overdetermined
# linear systems Ax = b, with custom LU/QR factorizations.

const LA = LinearAlgebra

const QRFact = LA.QR{ComplexF64, Matrix{ComplexF64}, Vector{ComplexF64}}
const QRFactorizeFW = FunctionWrapper{Nothing, Tuple{QRFact}}
const QRSolveFW =
    FunctionWrapper{Nothing, Tuple{FSVec{ComplexF64}, QRFact, FSVec{ComplexF64}}}

# A square workspace never reaches the QR branch. Both entry points are erased
# behind a shape-chosen `FunctionWrapper`, and the tall pair is built through a
# dynamic call so a square-only session never compiles `qr!`.
_unreachable_qr_factorize(::QRFact)::Nothing = nothing
_unreachable_qr_solve(
    ::FSVec{ComplexF64}, ::QRFact, ::FSVec{ComplexF64},
)::Nothing = nothing

_square_qr_ops()::Tuple{QRFactorizeFW, QRSolveFW} =
    (QRFactorizeFW(_unreachable_qr_factorize), QRSolveFW(_unreachable_qr_solve))
_tall_qr_ops()::Tuple{QRFactorizeFW, QRSolveFW} =
    (QRFactorizeFW(qr!), QRSolveFW(qr_ldiv!))

"""
    MatrixWorkspace

Data structure for the efficient repeated solution of a square or
overdetermined linear system `Ax = b`.

**Mutable justification:** `factorized` and `scaled` flags toggle every
Jacobian update. `lu` and `qr` must be reassigned after factorization.
All buffer fields are `const`.
"""
mutable struct MatrixWorkspace <: AbstractMatrix{ComplexF64}
    const A::FSMat{ComplexF64}
    factorized::Bool
    lu::LA.LU{ComplexF64, FSMat{ComplexF64}, Vector{Int64}}
    qr::QRFact
    const qr_factorize::QRFactorizeFW
    const qr_solve::QRSolveFW
    const qr_x::FSVec{ComplexF64}        # QR solution buffer
    const row_scaling::FSVec{Float64}
    scaled::Bool
    const x̄::FSVec{ComplexDF64}          # extended precision workspace
    const r::FSVec{ComplexF64}            # residual
    const r̄::FSVec{ComplexDF64}          # extended precision residual
    const δx::FSVec{ComplexF64}           # correction
    const inf_norm_est_work::FSVec{ComplexF64}
    const inf_norm_est_rwork::FSVec{Float64}
end

function MatrixWorkspace(m::Integer, n::Integer)
    A = FSMat{ComplexF64}(zeros(ComplexF64, m, n))
    return _make_matrix_workspace(A, m, n)
end

function _make_matrix_workspace(A::FSMat{ComplexF64}, m::Int, n::Int)
    m >= n || throw(ArgumentError("Expected m >= n, got m=$m, n=$n"))

    row_scaling = FSVec{Float64}(ones(m))

    # LU: use copy of A for factors, ipiv as Vector{Int64}
    ipiv = zeros(Int64, m)
    lu = LA.LU{ComplexF64, FSMat{ComplexF64}, Vector{Int64}}(copy(A), ipiv, 0)

    # A plain Matrix, since `qr!` on an FSMat returns a QRCompactWY. The factors
    # copy `A` so `factorize!` is defined before the first `updated!`.
    qr_factors = m == n ? Matrix{ComplexF64}(undef, 0, 0) :
        copyto!(Matrix{ComplexF64}(undef, m, n), A)
    qr = LA.QR(qr_factors, Vector{ComplexF64}(undef, size(qr_factors, 2)))
    qr_x = FSVec{ComplexF64}(Vector{ComplexF64}(undef, m == n ? 0 : n))
    qr_factorize, qr_solve = m == n ? _square_qr_ops() :
        Base.inferencebarrier(_tall_qr_ops)()::Tuple{QRFactorizeFW, QRSolveFW}

    r = FSVec{ComplexF64}(zeros(ComplexF64, m))
    r̄ = FSVec{ComplexDF64}(zeros(ComplexDF64, m))
    x̄ = FSVec{ComplexDF64}(zeros(ComplexDF64, n))
    δx = FSVec{ComplexF64}(zeros(ComplexF64, n))
    inf_norm_est_work = FSVec{ComplexF64}(Vector{ComplexF64}(undef, n))
    inf_norm_est_rwork = FSVec{Float64}(Vector{Float64}(undef, n))

    return MatrixWorkspace(
        A, false, lu, qr, qr_factorize, qr_solve, qr_x, row_scaling, false,
        x̄, r, r̄, δx, inf_norm_est_work, inf_norm_est_rwork,
    )
end

# ---------------------------------------------------------------------------
# AbstractMatrix interface
# ---------------------------------------------------------------------------
Base.size(MW::MatrixWorkspace) = size(MW.A)

Base.@propagate_inbounds Base.getindex(MW::MatrixWorkspace, i::Integer) =
    getindex(MW.A, i)
Base.@propagate_inbounds Base.getindex(MW::MatrixWorkspace, i::Integer, j::Integer) =
    getindex(MW.A, i, j)
Base.@propagate_inbounds Base.setindex!(MW::MatrixWorkspace, x, i::Integer) =
    setindex!(MW.A, x, i)
Base.@propagate_inbounds Base.setindex!(MW::MatrixWorkspace, x, i::Integer, j::Integer) =
    setindex!(MW.A, x, i, j)
# Note: copyto! not overloaded for MatrixWorkspace to avoid SparseArrays ambiguities.
# Use copyto!(WS.A, data) directly.

# ---------------------------------------------------------------------------
# updated! / factorize!
# ---------------------------------------------------------------------------

"""
    updated!(MW::MatrixWorkspace)

Indicate that the matrix `MW` got updated. Copies A into the appropriate
factorization buffer and resets flags.
"""
function updated!(MW::MatrixWorkspace)
    MW.factorized = false
    MW.scaled = false
    m, n = size(MW)
    if m == n
        @inbounds copyto!(MW.lu.factors, MW.A)
    else
        # Explicit element copy to avoid unaliascopy in Matrix←FSMat copyto!
        @inbounds for j in 1:n, i in 1:m
            MW.qr.factors[i, j] = MW.A[i, j]
        end
    end
    return MW
end

"""
    factorize!(WS::MatrixWorkspace)

Compute the LU (square) or QR (overdetermined) factorization in-place.
"""
function factorize!(WS::MatrixWorkspace)
    m, n = size(WS)
    if m == n
        lu!(WS.lu.factors, WS.lu.ipiv)
    else
        WS.qr_factorize(WS.qr)
    end
    WS.factorized = true
    return WS
end

# ---------------------------------------------------------------------------
# Custom LU factorization
# ---------------------------------------------------------------------------
# Adapted from LA.generic_lufact! with three changes:
# 1) abs2 instead of abs for pivot selection (avoids sqrt)
# 2) @fastmath naive division (no robust complex division needed)
# 3) Stores ipiv for row permutation

function lu!(A::AbstractMatrix{ComplexF64}, ipiv::Vector{Int64})
    m, n = size(A)
    minmn = min(m, n)
    @inbounds begin
        for k in 1:minmn
            # find pivot: index of max abs2 in column k, rows k:m
            kp = k
            amax = abs2(A[k, k])
            for i in (k + 1):m
                absi = abs2(A[i, k])
                kp, amax = ifelse(absi > amax, (i, absi), (kp, amax))
            end
            ipiv[k] = kp

            if !iszero(amax)
                # interchange rows k and kp
                if k != kp
                    for i in 1:n
                        tmp = A[k, i]
                        A[k, i] = A[kp, i]
                        A[kp, i] = tmp
                    end
                end
                # scale first column
                Akk = A[k, k]
                for i in (k + 1):m
                    @fastmath A[i, k] = A[i, k] / Akk
                end
            end
            # update the rest
            for j in (k + 1):n
                A_kj = A[k, j]
                for i in (k + 1):m
                    A[i, j] -= A[i, k] * A_kj
                end
            end
        end
    end
    return A
end

# ---------------------------------------------------------------------------
# Custom QR factorization
# ---------------------------------------------------------------------------
# Adapted from base implementation with @fastmath for division (~50% speedup)

@inline function reflector!(x::AbstractVector{ComplexF64})
    n = length(x)
    @inbounds begin
        ξ1 = x[1]
        normu = abs2(ξ1)
        for i in 2:n
            normu += abs2(x[i])
        end
        if iszero(normu)
            return zero(ComplexF64)
        end
        normu = sqrt(normu)
        ν = copysign(normu, real(ξ1))
        ξ1 += ν
        x[1] = -ν
        for i in 2:n
            @fastmath x[i] = x[i] / ξ1
        end
    end
    return ξ1 / ν
end

@inline function reflectorApply!(
        x::AbstractVector{ComplexF64}, τ::ComplexF64, A::StridedMatrix{ComplexF64},
    )
    m, n = size(A)
    @inbounds begin
        for j in 1:n
            # dot
            vAj = A[1, j]
            for i in 2:m
                vAj += x[i]' * A[i, j]
            end

            vAj = conj(τ) * vAj

            # ger
            A[1, j] -= vAj
            for i in 2:m
                A[i, j] -= x[i] * vAj
            end
        end
    end
    return A
end

function qr!(qr::LA.QR{ComplexF64})
    A = qr.factors
    τ = qr.τ
    m, n = size(A)
    for k in 1:min(m, n)
        x = view(A, k:m, k)
        τk = reflector!(x)
        τ[k] = τk
        reflectorApply!(x, τk, view(A, k:m, (k + 1):n))
    end
    return qr
end

# ---------------------------------------------------------------------------
# ldiv! helpers
# ---------------------------------------------------------------------------

@inline function _ipiv!(LU::LA.LU, b::AbstractVector)
    return _apply_ipiv!(b, 1:length(LU.ipiv), LU.ipiv)
end

@inline function _inverse_ipiv!(LU::LA.LU, b::AbstractVector)
    return _apply_ipiv!(b, length(LU.ipiv):-1:1, LU.ipiv)
end

@inline function _apply_ipiv!(b::AbstractVector, range::OrdinalRange, ipiv::Vector{Int64})
    @inbounds for i in range
        if i != ipiv[i]
            b[i], b[ipiv[i]] = b[ipiv[i]], b[i]
        end
    end
    return b
end

@inline function ldiv_upper!(
        A::AbstractMatrix{ComplexF64}, b::AbstractVector{ComplexF64},
        x::AbstractVector{ComplexF64} = b,
    )
    n = size(A, 2)
    @inbounds for j in n:-1:1
        xj = x[j] = (@fastmath A[j, j] \ b[j])
        for i in 1:(j - 1)
            b[i] -= A[i, j] * xj
        end
    end
    return b
end

@inline function ldiv_unit_lower!(
        A::AbstractMatrix{ComplexF64}, b::AbstractVector{ComplexF64},
        x::AbstractVector{ComplexF64} = b,
    )
    n = size(A, 2)
    @inbounds for j in 1:n
        xj = x[j] = b[j]
        for i in (j + 1):n
            b[i] -= A[i, j] * xj
        end
    end
    return x
end

function lu_ldiv!(
        x::AbstractVector{ComplexF64}, LU::LA.LU, b::AbstractVector{ComplexF64},
    )
    x === b || copyto!(x, b)
    _ipiv!(LU, x)
    ldiv_unit_lower!(LU.factors, x)
    ldiv_upper!(LU.factors, x)
    return x
end

# ---------------------------------------------------------------------------
# QR ldiv!
# ---------------------------------------------------------------------------

function lmul_Q_adj!(A::LA.QR{ComplexF64}, b::AbstractVector{ComplexF64})
    mA, nA = size(A.factors)
    mB = length(b)
    Afactors = A.factors
    @inbounds begin
        for k in 1:min(mA, nA)
            vBj = b[k]
            for i in (k + 1):mB
                vBj += conj(Afactors[i, k]) * b[i]
            end
            vBj = conj(A.τ[k]) * vBj
            b[k] -= vBj
            for i in (k + 1):mB
                b[i] -= Afactors[i, k] * vBj
            end
        end
    end
    return b
end

function qr_ldiv!(
        x::AbstractVector{ComplexF64}, QR::LA.QR{ComplexF64}, b::AbstractVector{ComplexF64},
    )
    # overwrites b; assumes QR is a tall matrix
    lmul_Q_adj!(QR, b)
    @inbounds for i in 1:length(x)
        x[i] = b[i]
    end
    ldiv_upper!(QR.factors, x)
    return x
end

# ---------------------------------------------------------------------------
# Main ldiv! dispatch
# ---------------------------------------------------------------------------

function LA.ldiv!(
        x::AbstractVector{ComplexF64}, WS::MatrixWorkspace, b::AbstractVector{ComplexF64},
    )
    m, n = size(WS)
    if (m, n) == (1, 1)
        @inbounds x[1] = b[1] / WS[1, 1]
        return x
    end
    WS.factorized || factorize!(WS)
    if m == n
        if WS.scaled
            @inbounds for i in eachindex(x, b)
                x[i] = WS.row_scaling[i] * b[i]
            end
            lu_ldiv!(x, WS.lu, x)
        else
            lu_ldiv!(x, WS.lu, b)
        end
    else
        copyto!(WS.r, b)
        WS.qr_solve(WS.qr_x, WS.qr, WS.r)
        @inbounds for i in eachindex(x, WS.qr_x)
            x[i] = WS.qr_x[i]
        end
    end
    return x
end

# ---------------------------------------------------------------------------
# Adjoint LU solve helpers (needed for condition number estimation)
# ---------------------------------------------------------------------------

@inline function ldiv_adj_unit_lower!(
        A::AbstractMatrix{ComplexF64}, b::AbstractVector{ComplexF64},
        x::AbstractVector{ComplexF64} = b,
    )
    n = size(A, 1)
    @inbounds for j in n:-1:1
        z = b[j]
        for i in n:-1:(j + 1)
            z -= conj(A[i, j]) * x[i]
        end
        x[j] = z
    end
    return x
end

@inline function ldiv_adj_upper!(
        A::AbstractMatrix{ComplexF64}, b::AbstractVector{ComplexF64},
        x::AbstractVector{ComplexF64} = b,
    )
    n = size(A, 1)
    @inbounds for j in 1:n
        z = b[j]
        for i in 1:(j - 1)
            z -= conj(A[i, j]) * x[i]
        end
        @fastmath x[j] = conj(A[j, j]) \ z
    end
    return x
end

function lu_ldiv_adj!(
        x::AbstractVector{ComplexF64}, LU::LA.LU, b::AbstractVector{ComplexF64},
    )
    x === b || copyto!(x, b)
    ldiv_adj_upper!(LU.factors, x)
    ldiv_adj_unit_lower!(LU.factors, x)
    _inverse_ipiv!(LU, x)
    return x
end

# ---------------------------------------------------------------------------
# Skeel Row Scaling
# ---------------------------------------------------------------------------

"""
    skeel_row_scaling!(d, A, c; scaling_threshold = -30.0)

Compute optimal scaling factors `d` for the matrix `A` following Skeel (1979)
if `c` is approximately of the order of the solution of the linear system
of interest. The scaling factors are rounded to powers of 2.
Row scaling is only applied to rows where the log-scale condition
`e - m ≥ scaling_threshold` holds, to prevent scaling near-zero rows.
"""
function skeel_row_scaling!(
        d::AbstractVector{Float64},
        A::AbstractMatrix{ComplexF64},
        c::AbstractVector{Float64};
        scaling_threshold::Float64 = -30.0,
    )
    n = length(c)
    @inbounds d .= zero(Float64)
    @inbounds for j in 1:n
        cj = c[j]
        for i in 1:n
            d[i] += fast_abs(A[i, j]) * cj
        end
    end

    m = maximum(d)
    s = scaling_threshold + m
    @inbounds for i in 1:n
        e = last(frexp(d[i]))
        if e < s
            d[i] = 1.0
        else
            d[i] = exp2(-e)
        end
    end

    return d
end

"""
    skeel_row_scaling!(W::MatrixWorkspace, c)

Convenience wrapper: compute Skeel row scaling for `W.A` and store in `W.row_scaling`.
"""
function skeel_row_scaling!(W::MatrixWorkspace, c::AbstractVector{Float64})
    skeel_row_scaling!(W.row_scaling, W.A, c)
    return W
end

"""
    apply_row_scaling!(W::MatrixWorkspace)

Apply the computed row scaling to `W.lu.factors` and set `W.scaled = true`.
"""
function apply_row_scaling!(W::MatrixWorkspace)
    A, d = W.lu.factors, W.row_scaling
    m, n = size(A)
    @inbounds for j in 1:n, i in 1:m
        A[i, j] = A[i, j] * d[i]
    end
    W.scaled = true
    return W
end

# ---------------------------------------------------------------------------
# Residual Computation
# ---------------------------------------------------------------------------

"""
    residual!(r, A, x, b)

Compute the residual `r = Ax - b` in-place using column-major order.
"""
function residual!(
        r::AbstractVector{T},
        A::AbstractMatrix{T},
        x::AbstractVector{T},
        b::AbstractVector{T},
    ) where {T}
    m, n = size(A)
    r .= zero(T)
    @inbounds for j in 1:n
        xj = x[j]
        for i in 1:m
            r[i] += A[i, j] * xj
        end
    end
    @inbounds for i in 1:m
        r[i] -= b[i]
    end
    return r
end

# Overload for mixed-precision: r is ComplexDF64, A is ComplexF64, x is ComplexDF64, b is ComplexF64
function residual!(
        r::AbstractVector{ComplexDF64},
        A::AbstractMatrix{ComplexF64},
        x::AbstractVector{ComplexDF64},
        b::AbstractVector{ComplexF64},
    )
    m, n = size(A)
    r .= zero(ComplexDF64)
    @inbounds for j in 1:n
        xj = x[j]
        for i in 1:m
            r[i] += ComplexDF64(A[i, j]) * xj
        end
    end
    @inbounds for i in 1:m
        r[i] -= ComplexDF64(b[i])
    end
    return r
end

# ---------------------------------------------------------------------------
# Mixed Precision Iterative Refinement
# ---------------------------------------------------------------------------

"""
    mixed_precision_iterative_refinement!(x, M, b, norm)

Perform one step of mixed-precision iterative refinement using extended
precision (ComplexDF64) residual computation.

Returns the normwise relative error `weighted_norm(δx, norm) / weighted_norm(x, norm)`.
"""
function mixed_precision_iterative_refinement!(
        x::AbstractVector{ComplexF64},
        M::MatrixWorkspace,
        b::AbstractVector{ComplexF64},
        norm::WeightedNorm,
    )
    @inbounds for i in eachindex(M.x̄, x)
        M.x̄[i] = x[i]
    end
    residual!(M.r̄, M.A, M.x̄, b)
    @inbounds for i in eachindex(M.r, M.r̄)
        M.r[i] = M.r̄[i]
    end
    LA.ldiv!(M.δx, M, M.r)
    @inbounds for i in eachindex(x)
        x[i] -= M.δx[i]
    end
    return weighted_norm(M.δx, norm) / weighted_norm(x, norm)
end

# ---------------------------------------------------------------------------
# Fixed Precision Iterative Refinement
# ---------------------------------------------------------------------------

"""
    fixed_precision_iterative_refinement!(x, M, b, norm)

Perform one step of iterative refinement using Float64 (single) precision
residual computation.

Returns the normwise relative error `weighted_norm(δx, norm) / weighted_norm(x, norm)`.
"""
function fixed_precision_iterative_refinement!(
        x::AbstractVector{ComplexF64},
        M::MatrixWorkspace,
        b::AbstractVector{ComplexF64},
        norm::WeightedNorm,
    )
    residual!(M.r, M.A, x, b)
    LA.ldiv!(M.δx, M, M.r)
    @inbounds for i in eachindex(x)
        x[i] -= M.δx[i]
    end
    return weighted_norm(M.δx, norm) / weighted_norm(x, norm)
end

# ---------------------------------------------------------------------------
# Multi-round Mixed-Precision Iterative Refinement
# ---------------------------------------------------------------------------

"""
    iterative_refinement!(x, M, b, norm, tol, max_iters)
    iterative_refinement!(x, M, b, norm; tol, max_iters)

Perform multiple rounds of mixed-precision iterative refinement until the
relative correction reaches `tol` or convergence stalls.

Uses weighted-norm refinement with the given `norm`.
"""
function iterative_refinement!(
        x::AbstractVector{ComplexF64},
        M::MatrixWorkspace,
        b::AbstractVector{ComplexF64},
        norm::WeightedNorm,
        tol::Float64,
        max_iters::Int,
    )
    refine!() = mixed_precision_iterative_refinement!(x, M, b, norm)
    return _iterative_refinement_loop!(refine!, tol, max_iters)
end

function iterative_refinement!(
        x::AbstractVector{ComplexF64},
        M::MatrixWorkspace,
        b::AbstractVector{ComplexF64},
        norm::WeightedNorm;
        tol::Float64 = sqrt(eps()),
        max_iters::Int = 3,
    )
    return iterative_refinement!(x, M, b, norm, tol, max_iters)
end

"""
    iterative_refinement!(x, M, b, tol, max_iters)
    iterative_refinement!(x, M, b; tol, max_iters)

Perform multiple rounds of mixed-precision iterative refinement until the
relative correction reaches `tol` or convergence stalls.

Uses inf-norm refinement (appropriate for dx/dt coefficients).
"""
function iterative_refinement!(
        x::AbstractVector{ComplexF64},
        M::MatrixWorkspace,
        b::AbstractVector{ComplexF64},
        tol::Float64,
        max_iters::Int,
    )
    refine!() = _mixed_precision_refinement_infnorm!(x, M, b)
    return _iterative_refinement_loop!(refine!, tol, max_iters)
end

function iterative_refinement!(
        x::AbstractVector{ComplexF64},
        M::MatrixWorkspace,
        b::AbstractVector{ComplexF64};
        tol::Float64 = sqrt(eps()),
        max_iters::Int = 3,
    )
    return iterative_refinement!(x, M, b, tol, max_iters)
end

# Shared refinement loop — Julia specializes on the concrete type of `refine!`,
# so this has zero dispatch overhead.
@inline function _iterative_refinement_loop!(refine!::F, tol::Float64, max_iters::Int) where {F}
    δ = refine!()
    δ < tol && return (accuracy = δ, diverged = false)
    for _ in 2:max_iters
        δ′ = refine!()
        if δ′ < tol
            return (accuracy = δ′, diverged = false)
        elseif δ′ > 0.5 * δ
            return (accuracy = δ′, diverged = true)
        end
        δ = δ′
    end
    return (accuracy = δ, diverged = false)
end

function _mixed_precision_refinement_infnorm!(
        x::AbstractVector{ComplexF64},
        M::MatrixWorkspace,
        b::AbstractVector{ComplexF64},
    )::Float64
    @inbounds for i in eachindex(M.x̄, x)
        M.x̄[i] = x[i]
    end
    residual!(M.r̄, M.A, M.x̄, b)
    @inbounds for i in eachindex(M.r, M.r̄)
        M.r[i] = M.r̄[i]
    end
    LA.ldiv!(M.δx, M, M.r)
    norm_δx = 0.0
    norm_x = 0.0
    @inbounds for i in eachindex(x)
        x[i] -= M.δx[i]
        norm_δx = max(norm_δx, fast_abs(M.δx[i]))
        norm_x = max(norm_x, fast_abs(x[i]))
    end
    return norm_x > 0 ? norm_δx / norm_x : norm_δx
end

# ---------------------------------------------------------------------------
# Condition Number Estimation (Higham 1988)
# ---------------------------------------------------------------------------

@inline function _norm_inf_real(x::AbstractVector{Float64})::Float64
    isempty(x) && return 0.0
    @inbounds m = abs(x[1])
    @inbounds for i in 2:length(x)
        xi = abs(x[i])
        xi > m && (m = xi)
    end
    return m
end

"""
    inverse_inf_norm_est(WS::MatrixWorkspace)

Estimate the infinity norm of `A⁻¹` using Higham's 1-norm condition estimator (1988).
When `WS.scaled`, incorporates the row scaling.
"""
function inverse_inf_norm_est(WS::MatrixWorkspace)::Float64
    m, n = size(WS)
    m == n || return Inf
    WS.factorized || factorize!(WS)
    work = WS.inf_norm_est_work
    rwork = WS.inf_norm_est_rwork
    return if WS.scaled
        _inverse_inf_norm_est_row(WS.lu, WS.row_scaling, work, rwork)
    else
        _inverse_inf_norm_est_noscale(WS.lu, work, rwork)
    end
end

# Higham (1988) inverse infinity-norm estimator — no scaling.
function _inverse_inf_norm_est_noscale(
        lu::LA.LU{ComplexF64, FSMat{ComplexF64}, Vector{Int64}},
        work::FSVec{ComplexF64},
        rwork::FSVec{Float64},
    )::Float64
    n = size(lu.factors, 1)
    y = work
    z = work
    ξ = work
    x = rwork

    @inbounds for i in 1:n
        x[i] = inv(n)
    end

    @inbounds for i in 1:n
        y[i] = x[i]
    end
    lu_ldiv_adj!(y, lu, y)

    γ = sum(fast_abs, y)

    @inbounds for i in 1:n
        ay = fast_abs(y[i])
        ξ[i] = iszero(ay) ? one(ComplexF64) : y[i] / ay
    end

    lu_ldiv!(z, lu, ξ)

    @inbounds for i in 1:n
        x[i] = real(z[i])
    end

    k = 2
    while true
        j = 1
        @inbounds max_xi = abs(x[1])
        @inbounds for i in 2:n
            abs_xi = abs(x[i])
            if abs_xi > max_xi
                j = i
                max_xi = abs_xi
            end
        end

        @inbounds for i in 1:n
            x[i] = zero(Float64)
        end
        @inbounds x[j] = one(Float64)

        @inbounds for i in 1:n
            y[i] = x[i]
        end
        lu_ldiv_adj!(y, lu, y)

        γ̄ = γ
        γ = sum(fast_abs, y)

        if γ ≤ γ̄
            γ = γ̄
            break
        end

        @inbounds for i in 1:n
            ay = fast_abs(y[i])
            ξ[i] = iszero(ay) ? one(ComplexF64) : y[i] / ay
        end

        lu_ldiv!(z, lu, ξ)

        @inbounds for i in 1:n
            x[i] = real(z[i])
        end

        k += 1
        if @inbounds(x[j] == _norm_inf_real(x)) || k > 2
            break
        end
    end

    return nanmin(γ, Inf)
end

# Higham (1988) inverse infinity-norm estimator — row scaling only.
function _inverse_inf_norm_est_row(
        lu::LA.LU{ComplexF64, FSMat{ComplexF64}, Vector{Int64}},
        row_scaling::FSVec{Float64},
        work::FSVec{ComplexF64},
        rwork::FSVec{Float64},
    )::Float64
    n = size(lu.factors, 1)
    y = work
    z = work
    ξ = work
    x = rwork

    @inbounds for i in 1:n
        x[i] = inv(n)
    end

    @inbounds for i in 1:n
        y[i] = x[i]
    end
    lu_ldiv_adj!(y, lu, y)

    @inbounds for i in 1:n
        y[i] *= row_scaling[i]
    end

    γ = sum(fast_abs, y)

    @inbounds for i in 1:n
        ay = fast_abs(y[i])
        ξ[i] = iszero(ay) ? one(ComplexF64) : y[i] / ay
    end
    @inbounds for i in 1:n
        ξ[i] /= row_scaling[i]
    end

    lu_ldiv!(z, lu, ξ)

    @inbounds for i in 1:n
        x[i] = real(z[i])
    end

    k = 2
    while true
        j = 1
        @inbounds max_xi = abs(x[1])
        @inbounds for i in 2:n
            abs_xi = abs(x[i])
            if abs_xi > max_xi
                j = i
                max_xi = abs_xi
            end
        end

        @inbounds for i in 1:n
            x[i] = zero(Float64)
        end
        @inbounds x[j] = one(Float64)

        @inbounds for i in 1:n
            y[i] = x[i]
        end
        lu_ldiv_adj!(y, lu, y)

        @inbounds for i in 1:n
            y[i] *= row_scaling[i]
        end

        γ̄ = γ
        γ = sum(fast_abs, y)

        if γ ≤ γ̄
            γ = γ̄
            break
        end

        @inbounds for i in 1:n
            ay = fast_abs(y[i])
            ξ[i] = iszero(ay) ? one(ComplexF64) : y[i] / ay
        end
        @inbounds for i in 1:n
            ξ[i] /= row_scaling[i]
        end

        lu_ldiv!(z, lu, ξ)

        @inbounds for i in 1:n
            x[i] = real(z[i])
        end

        k += 1
        if @inbounds(x[j] == _norm_inf_real(x)) || k > 2
            break
        end
    end

    return nanmin(γ, Inf)
end

# Higham (1988) inverse infinity-norm estimator — row and column scaling.
function _inverse_inf_norm_est(
        lu::LA.LU{ComplexF64, FSMat{ComplexF64}, Vector{Int64}},
        row_scaling::FSVec{Float64},
        col_scaling::FSVec{Float64},
        work::FSVec{ComplexF64},
        rwork::FSVec{Float64},
    )::Float64
    n = size(lu.factors, 1)
    y = work
    z = work
    ξ = work
    x = rwork

    @inbounds for i in 1:n
        x[i] = inv(n) / col_scaling[i]
    end

    @inbounds for i in 1:n
        y[i] = x[i]
    end
    lu_ldiv_adj!(y, lu, y)

    @inbounds for i in 1:n
        y[i] *= row_scaling[i]
    end

    γ = sum(fast_abs, y)

    @inbounds for i in 1:n
        ay = fast_abs(y[i])
        ξ[i] = iszero(ay) ? one(ComplexF64) : y[i] / ay
    end
    @inbounds for i in 1:n
        ξ[i] /= row_scaling[i]
    end

    lu_ldiv!(z, lu, ξ)

    @inbounds for i in 1:n
        x[i] = real(z[i]) / col_scaling[i]
    end

    k = 2
    while true
        j = 1
        @inbounds max_xi = abs(x[1])
        @inbounds for i in 2:n
            abs_xi = abs(x[i])
            if abs_xi > max_xi
                j = i
                max_xi = abs_xi
            end
        end

        @inbounds for i in 1:n
            x[i] = zero(Float64)
        end
        @inbounds x[j] = inv(col_scaling[j])

        @inbounds for i in 1:n
            y[i] = x[i]
        end
        lu_ldiv_adj!(y, lu, y)

        @inbounds for i in 1:n
            y[i] *= row_scaling[i]
        end

        γ̄ = γ
        γ = sum(fast_abs, y)

        if γ ≤ γ̄
            γ = γ̄
            break
        end

        @inbounds for i in 1:n
            ay = fast_abs(y[i])
            ξ[i] = iszero(ay) ? one(ComplexF64) : y[i] / ay
        end
        @inbounds for i in 1:n
            ξ[i] /= row_scaling[i]
        end

        lu_ldiv!(z, lu, ξ)

        @inbounds for i in 1:n
            x[i] = real(z[i]) / col_scaling[i]
        end

        k += 1
        if @inbounds(x[j] == _norm_inf_real(x)) || k > 2
            break
        end
    end

    return nanmin(γ, Inf)
end

"""
    inf_norm_matrix(WS::MatrixWorkspace)

Compute the infinity norm (maximum absolute row sum) of `WS.A`.
"""
function inf_norm_matrix(WS::MatrixWorkspace)::Float64
    A = WS.A
    m, n = size(A)
    norm_val = -Inf
    @inbounds for i in 1:m
        row_sum = 0.0
        for j in 1:n
            row_sum += fast_abs(A[i, j])
        end
        norm_val = @fastmath max(norm_val, row_sum)
    end
    return norm_val
end

"""
    LA.cond(WS::MatrixWorkspace)

Estimate the condition number of `WS.A` w.r.t. the infinity norm.
"""
function LA.cond(WS::MatrixWorkspace)
    m, n = size(WS)
    if m == n == 1
        return inv(fast_abs(WS.A[1, 1]))
    end
    return inf_norm_matrix(WS) * inverse_inf_norm_est(WS)
end

# ---------------------------------------------------------------------------
# Jacobian wrapper
# ---------------------------------------------------------------------------

"""
    Jacobian

Wraps a `MatrixWorkspace` with factorization and solve counters.
Used by the Newton corrector and tracker to track solver statistics.

**Mutable justification:** `factorizations` and `ldivs` are counters that
are incremented on each solve. They use `Base.RefValue{Int}` fields so that `Jacobian`
itself remains an immutable struct.
"""
struct Jacobian
    workspace::MatrixWorkspace
    factorizations::Base.RefValue{Int}
    ldivs::Base.RefValue{Int}
end

Jacobian(workspace::MatrixWorkspace) = Jacobian(workspace, Ref(0), Ref(0))

"""
    updated!(J::Jacobian)

Forward the `updated!` call to the underlying `MatrixWorkspace`.
"""
function updated!(J::Jacobian)
    updated!(J.workspace)
    return J
end

"""
    init!(J::Jacobian)

Reset the factorization and solve counters to zero.
"""
function init!(J::Jacobian)
    J.factorizations[] = 0
    J.ldivs[] = 0
    return J
end

Base.size(J::Jacobian) = size(J.workspace)

"""
    LA.ldiv!(x, J, b)

Solve `J x = b` using the underlying `MatrixWorkspace` and increment counters.
"""
function LA.ldiv!(
        x::AbstractVector{ComplexF64}, J::Jacobian, b::AbstractVector{ComplexF64},
    )
    J.factorizations[] += Int(!J.workspace.factorized)
    LA.ldiv!(x, J.workspace, b)
    J.ldivs[] += 1
    return x
end

"""
    LA.ldiv!(x, J, b, w)

Solve `J x = b` with automatic Skeel row scaling based on the weights of
`w::WeightedNorm`. For square systems, applies `skeel_row_scaling!` and
`apply_row_scaling!` before solving.
"""
function LA.ldiv!(
        x::AbstractVector{ComplexF64},
        J::Jacobian,
        b::AbstractVector{ComplexF64},
        w::WeightedNorm,
    )
    m, n = size(J.workspace)
    if m == n && !J.workspace.factorized
        skeel_row_scaling!(J.workspace, w.weights)
        apply_row_scaling!(J.workspace)
    end
    J.factorizations[] += Int(!J.workspace.factorized)
    LA.ldiv!(x, J.workspace, b)
    J.ldivs[] += 1
    return x
end

"""
    LA.cond(J::Jacobian)

Delegate condition number estimation to the underlying `MatrixWorkspace`.
"""
LA.cond(J::Jacobian) = LA.cond(J.workspace)
