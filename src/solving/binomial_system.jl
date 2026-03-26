# BinomialSystemSolver — solve binomial systems x^A = b arising from mixed cells.
# Ported from HomotopyContinuation.jl v2 (src/binomial_system.jl).
#
# Given a mixed cell from MixedSubdivisions, extract and solve the binomial system
# x^A = b where A is an n×n integer matrix of exponent differences and b is a complex
# vector of coefficient ratios.
#
# Algorithm:
# 1. Compute Hermite Normal Form: A·U = H (H lower triangular)
# 2. Angular part: solve for phases using roots of unity + triangular solve in DoubleF64
# 3. Magnitude: solve A^T · log|x| = log|b| via LU

struct BinomialSystemSolver
    A::Matrix{Int32}
    b::Vector{ComplexF64}
    H::Matrix{Int64}
    U::Matrix{Int64}
    γ::Vector{Float64}           # angles of b
    μ::Vector{Float64}           # log magnitudes workspace
    Aᵀ::Matrix{Float64}
    # DoubleF64 workspace for angular computation
    μ_df64::Vector{DoubleF64}
    αs::Vector{DoubleF64}
    # unit roots combination table (n × d_hat_max)
    unit_roots_table::Matrix{Int32}
end

function BinomialSystemSolver(n::Int; max_d_hat::Int = 1)
    A = zeros(Int32, n, n)
    b = zeros(ComplexF64, n)
    H = zeros(Int64, n, n)
    U = zeros(Int64, n, n)
    γ = zeros(Float64, n)
    μ = zeros(Float64, n)
    Aᵀ = zeros(Float64, n, n)
    μ_df64 = zeros(DoubleF64, n)
    αs = zeros(DoubleF64, n)
    unit_roots_table = zeros(Int32, n, max_d_hat)
    return BinomialSystemSolver(A, b, H, U, γ, μ, Aᵀ, μ_df64, αs, unit_roots_table)
end

"""
    solve_binomial!(X, BSS, support, coeffs, cell) -> Int

Solve the binomial system defined by a mixed cell. Returns the number of solutions `d_hat`.
Solutions are written into columns `1:d_hat` of `X`.

`X` must be pre-allocated to at least `n × d_hat` where `d_hat = |det(A)|`.
If `X` is too small, it will not be resized — caller must ensure sufficient space.
"""
function solve_binomial!(
        X::Matrix{ComplexF64},
        BSS::BinomialSystemSolver,
        support::Vector{Matrix{Int32}},
        coeffs::Vector{Vector{ComplexF64}},
        cell::MixedSubdivisions.MixedCell,
    )::Int
    n = size(BSS.A, 1)

    # 1. Extract A, b from the mixed cell
    _init_binomial!(BSS, support, coeffs, cell)

    # 2. Compute Hermite Normal Form: A·U = H
    _hnf!(BSS.H, BSS.U, BSS.A)

    # 3. Compute d_hat = product of diagonal of H
    d_hat = 1
    @inbounds for i in 1:n
        d_hat *= Int(BSS.H[i, i])
    end

    # 4. Fill unit roots combinations table
    _fill_unit_roots_table!(BSS, d_hat)

    # 5. Compute angular part (phases)
    _compute_angular_part!(X, BSS, d_hat)

    # 6. Compute magnitude part
    _compute_magnitude!(X, BSS, d_hat)

    return d_hat
end

# --------------------------------------------------------------------------
# init! — extract A, b from mixed cell
# --------------------------------------------------------------------------

function _init_binomial!(
        BSS::BinomialSystemSolver,
        support::Vector{Matrix{Int32}},
        coeffs::Vector{Vector{ComplexF64}},
        cell::MixedSubdivisions.MixedCell,
    )::Nothing
    n = size(BSS.A, 1)
    @inbounds for (i, (aᵢ, bᵢ)) in enumerate(cell.indices)
        for j in 1:n
            BSS.A[j, i] = support[i][j, aᵢ] - support[i][j, bᵢ]
        end
        BSS.b[i] = -coeffs[i][bᵢ] / coeffs[i][aᵢ]
    end
    return nothing
end

# --------------------------------------------------------------------------
# Hermite Normal Form (Kannan-Bachem 1979)
# --------------------------------------------------------------------------

# Checked arithmetic operators for overflow detection
_checked_mul(x::Int64, y::Int64)::Int64 = Base.checked_mul(x, y)
_checked_add(x::Int64, y::Int64)::Int64 = Base.checked_add(x, y)

"""
    _hnf!(H, U, A)

Compute the Hermite Normal Form of `A` such that `A·U = H` where `H` is lower
triangular with positive diagonal entries. Uses checked arithmetic for overflow
detection.
"""
function _hnf!(H::Matrix{Int64}, U::Matrix{Int64}, A::Matrix{Int32})::Nothing
    n = size(A, 1)
    copyto!(H, A)

    # Initialize U to identity
    fill!(U, 0)
    @inbounds for i in 1:n
        U[i, i] = one(Int64)
    end

    @inbounds for i in 1:(n - 1)
        ii = i + 1
        for j in 1:i
            if !iszero(H[j, j]) || !iszero(H[j, ii])
                # Step 4.1: Extended GCD
                r, p, q = gcdx(H[j, j], H[j, ii])
                # Step 4.2: Column transformation
                d_j = -(H[j, ii] ÷ r)
                d_ii = H[j, j] ÷ r
                for k in 1:n
                    h_kj, h_kii = H[k, j], H[k, ii]
                    H[k, j] = _checked_add(_checked_mul(h_kj, p), _checked_mul(h_kii, q))
                    H[k, ii] =
                        _checked_add(_checked_mul(h_kj, d_j), _checked_mul(h_kii, d_ii))

                    u_kj, u_kii = U[k, j], U[k, ii]
                    U[k, j] = _checked_add(_checked_mul(u_kj, p), _checked_mul(u_kii, q))
                    U[k, ii] =
                        _checked_add(_checked_mul(u_kj, d_j), _checked_mul(u_kii, d_ii))
                end
            end
            # Step 4.3
            if j > 1
                _reduce_off_diagonal!(H, U, j)
            end
        end
        # Step 5
        _reduce_off_diagonal!(H, U, ii)
    end

    # Special case: 1×1 matrix — ensure positive diagonal
    if n == 1
        if H[1, 1] < 0
            H[1, 1] = -H[1, 1]
            U[1, 1] = -U[1, 1]
        end
    end

    return nothing
end

"""
    _cdiv(x, y)

Ceiling division: smallest integer d such that d * y >= x (for positive y).
"""
_cdiv(x::Int64, y::Int64)::Int64 = cld(x, y)

@inline function _reduce_off_diagonal!(H::Matrix{Int64}, U::Matrix{Int64}, k::Int)::Nothing
    n = size(H, 1)
    @inbounds if H[k, k] < 0
        for i in 1:n
            H[i, k] = -H[i, k]
            U[i, k] = -U[i, k]
        end
    end
    @inbounds for z in 1:(k - 1)
        if !iszero(H[z, z])
            r = -_cdiv(H[k, z], H[z, z])
            for i in 1:n
                U[i, z] = _checked_add(U[i, z], _checked_mul(r, U[i, k]))
                H[i, z] = _checked_add(H[i, z], _checked_mul(r, H[i, k]))
            end
        end
    end
    return nothing
end

# --------------------------------------------------------------------------
# Fill unit roots combinations table
# --------------------------------------------------------------------------

function _fill_unit_roots_table!(BSS::BinomialSystemSolver, d_hat::Int)::Nothing
    H = BSS.H
    n = size(H, 1)

    # Resize table if needed (reallocate — struct fields are not const)
    if size(BSS.unit_roots_table, 2) < d_hat
        # We need to create a new table; but since the struct is immutable
        # we work with the existing allocation if possible.
        # The caller should pre-allocate large enough via max_d_hat.
        throw(
            DimensionMismatch(
                "unit_roots_table has $(size(BSS.unit_roots_table, 2)) columns but needs $d_hat",
            )
        )
    end

    d = d_hat
    e = 1
    @inbounds for i in 1:n
        dᵢ = Int(H[i, i])
        d = d ÷ dᵢ
        k = 1
        for _ in 1:e, j in 0:(dᵢ - 1), _ in 1:d
            BSS.unit_roots_table[i, k] = Int32(j)
            k += 1
        end
        e *= dᵢ
    end
    return nothing
end

# --------------------------------------------------------------------------
# Angular part — roots of unity + triangular solve in DoubleF64 precision
# --------------------------------------------------------------------------

function _compute_angular_part!(
        X::Matrix{ComplexF64},
        BSS::BinomialSystemSolver,
        d_hat::Int,
    )::Nothing
    H = BSS.H
    U = BSS.U
    b = BSS.b
    γ = BSS.γ
    μ_df64 = BSS.μ_df64
    αs = BSS.αs
    unit_roots_table = BSS.unit_roots_table
    n = size(H, 1)

    # Compute angles: γ[i] = angle(b[i]) / 2π
    inv_2π = 1.0 / (2.0 * π)
    @inbounds for i in 1:n
        γ[i] = angle(b[i]) * inv_2π
    end

    # Apply coordinate change: μ_df64 = U^T · γ (in DoubleF64 precision)
    @inbounds for j in 1:n
        μ_df64[j] = DoubleF64(0.0)
    end
    @inbounds for j in 1:n
        for i in 1:n
            μij = DoubleF64(U[i, j]) * DoubleF64(γ[i])
            # Reduce to [-1, 1] range for numerical stability
            μij = μij - 2 * round(DoubleF64(0.5) * μij, RoundNearest)
            μ_df64[j] = μ_df64[j] + μij
        end
        μ_df64[j] = rem(μ_df64[j], DoubleF64(2.0), RoundNearest)
    end

    # Solve triangular system for each solution
    @inbounds for i in 1:d_hat, j in n:-1:1
        α = (μ_df64[j] + DoubleF64(Int(unit_roots_table[j, i]))) / DoubleF64(Int(H[j, j]))
        α = α - 2 * round(DoubleF64(0.5) * α, RoundNearest)
        for k in n:-1:(j + 1)
            αk = (αs[k] * DoubleF64(Int(H[k, j]))) / DoubleF64(Int(H[j, j]))
            αk = αk - 2 * round(DoubleF64(0.5) * αk, RoundNearest)
            α = α - αk
        end
        α = rem(α, DoubleF64(2.0), RoundNearest)
        # X[j, i] = cis(2π * α) using sinpi/cospi for accuracy
        α_f64 = Float64(α)
        X[j, i] = complex(cospi(2.0 * α_f64), sinpi(2.0 * α_f64))
        αs[j] = α
    end

    return nothing
end

# --------------------------------------------------------------------------
# Magnitude part — solve A^T · log|x| = log|b| via LU
# --------------------------------------------------------------------------

function _compute_magnitude!(
        X::Matrix{ComplexF64},
        BSS::BinomialSystemSolver,
        d_hat::Int,
    )::Nothing
    A = BSS.A
    Aᵀ = BSS.Aᵀ
    b = BSS.b
    μ = BSS.μ
    n = size(A, 1)

    # μ = log(|b|)
    @inbounds for i in 1:n
        μ[i] = log(fast_abs(b[i]))
    end

    # Aᵀ = transpose(A)
    @inbounds for j in 1:n, i in 1:n
        Aᵀ[i, j] = Float64(A[j, i])
    end

    # Solve Aᵀ · x = μ in-place
    LA.ldiv!(LA.lu!(Aᵀ), μ)

    # μ = exp(μ) — these are the magnitudes |x_i|
    @inbounds for i in 1:n
        μ[i] = exp(μ[i])
    end

    # Apply magnitudes to all solutions
    @inbounds for j in 1:d_hat, i in 1:n
        X[i, j] *= μ[i]
    end

    return nothing
end
