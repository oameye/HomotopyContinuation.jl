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
        supports[k] = S
        coeffs[k] = c
    end

    return supports, coeffs
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
