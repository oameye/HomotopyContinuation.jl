## support.jl — extract support (exponent matrices) and coefficients from expressions.

# `exponents_coefficients` expands an `Expression` into one term per monomial, in the
# column order `td_order_perm` produces.
"""
    support_coefficients(polys, variables) → (supports, coefficients)

Extract the support (exponent matrix per polynomial) and coefficient vectors.
Returns `(Vector{Matrix{Int32}}, Vector{Vector{ComplexF64}})`.

Each `supports[i]` is an `n × mᵢ` matrix where column j is the exponent vector
of the j-th term of polynomial i. `coefficients[i]` is the corresponding coefficient vector.
"""
function support_coefficients(
        polys::AbstractVector{Expression},
        variables::AbstractVector{Expression},
    )::Tuple{Vector{Matrix{Int32}}, Vector{Vector{ComplexF64}}}
    supports = Vector{Matrix{Int32}}(undef, length(polys))
    coeffs = Vector{Vector{ComplexF64}}(undef, length(polys))
    for (k, p) in enumerate(polys)
        supports[k], coeffs[k] = exponents_coefficients(p, variables)
    end
    return supports, coeffs
end

"""
    td_order_perm(S::Matrix{Int32}) → Vector{Int}

Compute permutation that sorts support columns by descending total degree,
with ties broken by descending lexicographic order of exponent vectors.
"""
function td_order_perm(S::Matrix{Int32})::Vector{Int}
    ncols = size(S, 2)
    nrows = size(S, 1)
    return stable_sort!(
        collect(1:ncols), (a, b) -> begin
            sa = zero(Int32)
            sb = zero(Int32)
            @inbounds for i in 1:nrows
                sa += S[i, a]
                sb += S[i, b]
            end
            if sa != sb
                return sa > sb
            end
            @inbounds for i in 1:nrows
                S[i, a] != S[i, b] && return S[i, a] > S[i, b]
            end
            return false
        end,
    )
end

"""
    has_zero_column(A::Matrix{Int32}) → Bool

Check if the matrix has a column that is all zeros (constant term in support).
"""
function has_zero_column(A::Matrix{Int32})::Bool
    @inbounds for j in 1:size(A, 2)
        all_zero = true
        for i in 1:size(A, 1)
            if !iszero(A[i, j])
                all_zero = false
                break
            end
        end
        all_zero && return true
    end
    return false
end
