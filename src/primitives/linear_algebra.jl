# MatrixWorkspace — efficient repeated solution of square or overdetermined
# linear systems Ax = b, with custom LU/QR factorizations.
# Ported from HomotopyContinuation.jl v2 (src/linear_algebra.jl, lines 1-410).

const LA = LinearAlgebra

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
    qr::LA.QR{ComplexF64, Matrix{ComplexF64}, Vector{ComplexF64}}
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

    # QR: use Matrix{ComplexF64} (qr! on FSMat returns QRCompactWY, not QR)
    qr = LA.qrfactUnblocked!(Matrix{ComplexF64}(copy(A)))

    r = FSVec{ComplexF64}(zeros(ComplexF64, m))
    r̄ = FSVec{ComplexDF64}(zeros(ComplexDF64, m))
    x̄ = FSVec{ComplexDF64}(zeros(ComplexDF64, n))
    δx = FSVec{ComplexF64}(zeros(ComplexF64, n))
    inf_norm_est_work = FSVec{ComplexF64}(Vector{ComplexF64}(undef, n))
    inf_norm_est_rwork = FSVec{Float64}(Vector{Float64}(undef, n))

    return MatrixWorkspace(
        A, false, lu, qr, row_scaling, false,
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
Base.@propagate_inbounds Base.copyto!(MW::MatrixWorkspace, A::AbstractArray) =
    copyto!(MW.A, A)

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
        @inbounds copyto!(MW.qr.factors, MW.A)
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
        qr!(WS.qr)
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
            x .= WS.row_scaling .* b
            lu_ldiv!(x, WS.lu, x)
        else
            lu_ldiv!(x, WS.lu, b)
        end
    else
        WS.r .= b
        qr_ldiv!(x, WS.qr, WS.r)
    end
    return x
end
