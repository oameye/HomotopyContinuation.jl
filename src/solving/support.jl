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
        # Sort columns by descending total degree, then descending lexicographic.
        # This canonical ordering ensures coefficient indices match the support.
        perm = _td_order_perm(S)
        supports[k] = S[:, perm]
        coeffs[k] = c[perm]
    end

    return supports, coeffs
end

const _MonomialTerms = Dict{Vector{Int32}, ComplexF64}

function _add_terms!(acc::_MonomialTerms, other::_MonomialTerms)::Nothing
    for (m, c) in other
        acc[m] = get(acc, m, zero(ComplexF64)) + c
    end
    return nothing
end

function _multiply_terms(a::_MonomialTerms, b::_MonomialTerms)::_MonomialTerms
    out = _MonomialTerms()
    for (ma, ca) in a, (mb, cb) in b
        m = ma .+ mb
        out[m] = get(out, m, zero(ComplexF64)) + ca * cb
    end
    return out
end

"""
Expand a polynomial `Expression`, which stores products and powers unexpanded,
into `exponent vector => coefficient`.
"""
function _monomial_terms(
        e::Expression, var_to_idx::Dict{Symbol, Int}, n::Int,
    )::_MonomialTerms
    storage = expr_storage(e)
    if storage isa ENumStorage
        return _MonomialTerms(zeros(Int32, n) => storage.val)
    elseif storage isa EVarStorage
        idx = get(var_to_idx, storage.name, 0)
        idx == 0 && throw(
            ArgumentError(
                "`$(storage.name)` is neither a variable nor a parameter of the system",
            ),
        )
        m = zeros(Int32, n)
        m[idx] = Int32(1)
        return _MonomialTerms(m => one(ComplexF64))
    elseif storage isa EAddStorage
        acc = _MonomialTerms()
        for a in storage.args
            _add_terms!(acc, _monomial_terms(a, var_to_idx, n))
        end
        return acc
    elseif storage isa EMulStorage
        acc = _MonomialTerms(zeros(Int32, n) => one(ComplexF64))
        for a in storage.args
            acc = _multiply_terms(acc, _monomial_terms(a, var_to_idx, n))
        end
        return acc
    elseif storage isa EPowStorage
        storage.exp >= 0 || throw(
            ArgumentError("the expression is not polynomial: it has a negative power"),
        )
        base = _monomial_terms(storage.base, var_to_idx, n)
        acc = _MonomialTerms(zeros(Int32, n) => one(ComplexF64))
        for _ in 1:(storage.exp)
            acc = _multiply_terms(acc, base)
        end
        return acc
    else # EFnStorage
        throw(
            ArgumentError(
                "the expression is not polynomial: it applies `$(storage.kind)` to a variable",
            ),
        )
    end
end

function support_coefficients(
        polys::AbstractVector{Expression},
        variables::AbstractVector{Expression},
    )::Tuple{Vector{Matrix{Int32}}, Vector{Vector{ComplexF64}}}
    n = length(variables)
    var_to_idx = Dict{Symbol, Int}(Symbol(v) => i for (i, v) in enumerate(variables))

    supports = Vector{Matrix{Int32}}(undef, length(polys))
    coeffs = Vector{Vector{ComplexF64}}(undef, length(polys))

    for (k, p) in enumerate(polys)
        terms = _monomial_terms(p, var_to_idx, n)
        # Expanding cancels terms the tree kept apart, so drop the zeros.
        monomials = [m for (m, c) in terms if !iszero(c)]
        S = zeros(Int32, n, length(monomials))
        c = Vector{ComplexF64}(undef, length(monomials))
        for (j, m) in enumerate(monomials)
            @inbounds S[:, j] .= m
            @inbounds c[j] = terms[m]
        end
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
