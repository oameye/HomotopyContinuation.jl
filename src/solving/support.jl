## support.jl — extract support (exponent matrices) and coefficients from MP polynomials.

"""
    support_coefficients(polys, variables) → (supports, coefficients)

Extract the support (exponent matrix per polynomial) and coefficient vectors.
Returns `(Vector{Matrix{Int32}}, Vector{Vector{ComplexF64}})`.

Each `supports[i]` is an `n × mᵢ` matrix where column j is the exponent vector
of the j-th term of polynomial i. `coefficients[i]` is the corresponding coefficient vector.
"""
function support_coefficients(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
    )::Tuple{Vector{Matrix{Int32}}, Vector{Vector{ComplexF64}}}
    Base.@nospecialize polys variables
    n = length(variables)
    var_to_idx = Dict{Symbol, Int}(Symbol(v) => i for (i, v) in enumerate(variables))

    supports = Vector{Matrix{Int32}}(undef, length(polys))
    coeffs = Vector{Vector{ComplexF64}}(undef, length(polys))

    for (k, p) in enumerate(polys)
        ts = MP.terms(p)
        m = length(ts)
        S = zeros(Int32, n, m)
        c = zeros(ComplexF64, m)
        for (j, t) in enumerate(ts)
            c[j] = ComplexF64(MP.coefficient(t))
            mono = MP.monomial(t)
            for (var, exp) in zip(MP.variables(mono), MP.exponents(mono))
                idx = get(var_to_idx, Symbol(var), nothing)
                if idx !== nothing
                    S[idx, j] = Int32(exp)
                end
            end
        end
        # Sort columns by descending total degree, then descending lexicographic
        # (v2 parity: matches HC ModelKit's td_order for identical RNG paths)
        perm = _td_order_perm(S)
        supports[k] = S[:, perm]
        coeffs[k] = c[perm]
    end

    return supports, coeffs
end

"""
    _td_order_perm(S::Matrix{Int32}) → Vector{Int}

Compute permutation that sorts support columns by descending total degree,
with ties broken by descending lexicographic order of exponent vectors.
Matches v2's `td_order` for identical monomial ordering.
"""
function _td_order_perm(S::Matrix{Int32})::Vector{Int}
    ncols = size(S, 2)
    nrows = size(S, 1)
    return sortperm(
        1:ncols; lt = (a, b) -> begin
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
        end
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
