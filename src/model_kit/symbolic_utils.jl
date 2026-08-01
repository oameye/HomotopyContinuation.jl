## Polynomial-oriented utilities on `Expression`: expansion, monomial
## enumeration, coefficient extraction, Horner rewriting and numeric evaluation.
#
# `Expression` keeps products and powers unexpanded, so anything needing one term
# per monomial goes through `_expr_terms`, which expands into
# `exponent vector in vars => coefficient expression`; variables outside `vars`
# land in the coefficient.

## ── Expansion ───────────────────────────────────────────────────────────────

_summands(e::Expression)::Vector{Expression} =
let storage = expr_storage(e)
    storage isa EAddStorage ? storage_args(storage) : Expression[e]
end

function _distribute(
        terms::Vector{Expression}, factor::Vector{Expression},
    )::Vector{Expression}
    next = Vector{Expression}(undef, length(terms) * length(factor))
    k = 0
    for t in terms, f in factor
        next[k += 1] = _emul(Expression[t, f])
    end
    return next
end

# Collecting after every stage keeps the working set at the number of distinct monomials;
# distributing all stages first is exponential in the number of factors.
_distribute_collected(terms::Vector{Expression}, factor::Vector{Expression}) =
    _summands(_eadd(_distribute(terms, factor)))

function _expand_product(args::Vector{Expression})::Expression
    terms = Expression[one(Expression)]
    for a in args
        terms = _distribute_collected(terms, _summands(expand(a)))
    end
    return _eadd(terms)
end

# Splits the summands once, not once per exponent as `_expand_product` would.
function _expand_power(base::Expression, k::Int)::Expression
    k > 0 && expr_storage(base) isa EAddStorage || return _epow(base, k)
    factor = _summands(base)
    terms = factor
    for _ in 2:k
        terms = _distribute_collected(terms, factor)
    end
    return _eadd(terms)
end

"""
    expand(e::Expression) -> Expression

Distribute products over sums, so that a polynomial becomes a sum of monomials.
A negative power, or a sum under one, is left alone.

```julia
@var x y
expand((x + y)^2)   # 2*x*y + x^2 + y^2
```
"""
function expand(e::Expression)::Expression
    storage = expr_storage(e)
    if storage isa ENumStorage
        return e
    elseif storage isa EVarStorage
        return e
    elseif storage isa EAddStorage
        return _eadd(Expression[expand(a) for a in storage_args(storage)])
    elseif storage isa EMulStorage
        return _expand_product(storage_args(storage))
    elseif storage isa EPowStorage
        return _expand_power(expand(storage_base(storage)), storage.exp)
    else
        return _efn(storage.kind, expand(storage_arg(storage)))
    end
end

expand(exprs::AbstractArray{Expression}) = map(expand, exprs)

## ── Expansion into monomial => coefficient ──────────────────────────────────

const _ExprTerms = Dict{Vector{Int}, Expression}

_var_index(vars::AbstractVector{Expression})::Dict{Symbol, Int} =
    Dict{Symbol, Int}(Symbol(v) => i for (i, v) in enumerate(vars))

function _depends_on(e::Expression, var_to_idx::Dict{Symbol, Int})::Bool
    storage = expr_storage(e)
    if storage isa EVarStorage
        return haskey(var_to_idx, storage.name)
    elseif storage isa EAddStorage
        for a in storage_args(storage)
            _depends_on(a, var_to_idx) && return true
        end
        return false
    elseif storage isa EMulStorage
        for a in storage_args(storage)
            _depends_on(a, var_to_idx) && return true
        end
        return false
    elseif storage isa EPowStorage
        return _depends_on(storage_base(storage), var_to_idx)
    elseif storage isa EFnStorage
        return _depends_on(storage_arg(storage), var_to_idx)
    end
    return false
end

@inline function _accumulate_term!(acc::_ExprTerms, m::Vector{Int}, c::Expression)::Nothing
    old = get(acc, m, nothing)
    acc[m] = old === nothing ? c : old + c
    return nothing
end

function _add_expr_terms!(acc::_ExprTerms, other::_ExprTerms)::Nothing
    for (m, c) in other
        _accumulate_term!(acc, m, c)
    end
    return nothing
end

function _multiply_expr_terms(a::_ExprTerms, b::_ExprTerms)::_ExprTerms
    out = _ExprTerms()
    for (ma, ca) in a, (mb, cb) in b
        _accumulate_term!(out, ma .+ mb, ca * cb)
    end
    return out
end

# `nothing` when `e` is not polynomial in the indexed variables.
function _expr_terms(
        e::Expression, var_to_idx::Dict{Symbol, Int}, n::Int,
    )::Union{Nothing, _ExprTerms}
    storage = expr_storage(e)
    if storage isa ENumStorage
        return _ExprTerms(zeros(Int, n) => e)
    elseif storage isa EVarStorage
        idx = get(var_to_idx, storage.name, 0)
        idx == 0 && return _ExprTerms(zeros(Int, n) => e)
        m = zeros(Int, n)
        m[idx] = 1
        return _ExprTerms(m => one(Expression))
    elseif storage isa EAddStorage
        acc = _ExprTerms()
        for a in storage_args(storage)
            terms = _expr_terms(a, var_to_idx, n)
            terms === nothing && return nothing
            _add_expr_terms!(acc, terms)
        end
        return acc
    elseif storage isa EMulStorage
        acc = _ExprTerms(zeros(Int, n) => one(Expression))
        for a in storage_args(storage)
            terms = _expr_terms(a, var_to_idx, n)
            terms === nothing && return nothing
            acc = _multiply_expr_terms(acc, terms)
        end
        return acc
    elseif storage isa EPowStorage
        _depends_on(e, var_to_idx) || return _ExprTerms(zeros(Int, n) => e)
        storage.exp > 0 || return nothing
        base = _expr_terms(storage_base(storage), var_to_idx, n)
        base === nothing && return nothing
        acc = _ExprTerms(zeros(Int, n) => one(Expression))
        for _ in 1:(storage.exp)
            acc = _multiply_expr_terms(acc, base)
        end
        return acc
    else
        _depends_on(storage_arg(storage), var_to_idx) && return nothing
        return _ExprTerms(zeros(Int, n) => e)
    end
end

# Coefficients accumulate as sums of products, so a cancellation shows only after
# expanding.
function _clean_terms!(terms::_ExprTerms)::_ExprTerms
    for m in collect(keys(terms))
        terms[m] = expand(terms[m])
    end
    filter!(entry -> !iszero(entry[2]), terms)
    return terms
end

function _terms_or_throw(f::Expression, vars::AbstractVector{Expression})::_ExprTerms
    terms = _expr_terms(f, _var_index(vars), length(vars))
    terms === nothing && throw(
        ArgumentError("`" * string(f) * "` is not polynomial in the given variables"),
    )
    return _clean_terms!(terms)
end

"""
    to_dict(f::Expression, vars) -> Dict{Vector{Int}, Expression}

Coefficients of `f` with respect to `vars`, keyed by exponent vector. Variables
outside `vars` become part of the coefficient. Throws an `ArgumentError` if `f`
is not polynomial in `vars`.

```julia
@var x y a
to_dict(x^2 + y * x * a + y * x, [x, y])   # Dict([2, 0] => 1, [1, 1] => a + 1)
```
"""
to_dict(f::Expression, vars::AbstractVector{Expression})::Dict{Vector{Int}, Expression} =
    _terms_or_throw(f, vars)

# Descending total degree, ties broken by descending lexicographic order. Shared, so
# `monomials`, `coefficients` and `coeffs_as_dense_poly` agree column for column. The
# degree is cached per vector rather than resummed on every comparison.
function _sorted_td(exponents::Vector{Vector{Int}})::Vector{Vector{Int}}
    keyed = Tuple{Int, Vector{Int}}[(sum(e; init = 0), e) for e in exponents]
    sort!(keyed; lt = (a, b) -> a[1] == b[1] ? a[2] > b[2] : a[1] > b[1])
    return Vector{Int}[e for (_, e) in keyed]
end

_sorted_terms(terms::_ExprTerms)::Tuple{Vector{Vector{Int}}, Vector{Expression}} =
let exponents = _sorted_td(collect(keys(terms)))
    (exponents, Expression[terms[e] for e in exponents])
end

_sorted_terms(
    f::Expression, vars::AbstractVector{Expression},
)::Tuple{Vector{Vector{Int}}, Vector{Expression}} =
    _sorted_terms(_terms_or_throw(f, vars))

function _exponent_matrix(
        ::Type{T}, exponents::Vector{Vector{Int}}, n::Int,
    )::Matrix{T} where {T <: Integer}
    M = zeros(T, n, length(exponents))
    for (j, e) in enumerate(exponents)
        @inbounds for i in 1:n
            M[i, j] = e[i]
        end
    end
    return M
end

function _numeric_coefficients(c::Vector{Expression})::Vector{ComplexF64}
    out = Vector{ComplexF64}(undef, length(c))
    for (i, e) in enumerate(c)
        v = expr_number(e)
        v === nothing && throw(
            ArgumentError(
                "the coefficient `" * string(e) *
                    "` is not a number; use `to_dict` for symbolic coefficients",
            ),
        )
        out[i] = v
    end
    return out
end

"""
    exponents_coefficients(f::Expression, vars) -> (Matrix{Int32}, Vector{ComplexF64})

Exponent matrix of the terms of `f` in `vars`, one term per column, and the
corresponding coefficients. Columns are ordered by descending total degree, ties
broken by descending lexicographic order.

Every coefficient must be a number; use [`to_dict`](@ref) for symbolic ones.
[`poly_from_exponents_coefficients`](@ref) is the inverse.
"""
function exponents_coefficients(
        f::Expression, vars::AbstractVector{Expression},
    )::Tuple{Matrix{Int32}, Vector{ComplexF64}}
    exponents, c = _sorted_terms(f, vars)
    return _exponent_matrix(Int32, exponents, length(vars)), _numeric_coefficients(c)
end

exponents_coefficients(
    f::Expression, var::Expression,
)::Tuple{Matrix{Int32}, Vector{ComplexF64}} =
    exponents_coefficients(f, Expression[var])

"""
    coefficients(f::Expression, vars) -> Vector{ComplexF64}

Coefficients of the terms of `f` in `vars`, in the order
[`exponents_coefficients`](@ref) uses. Every coefficient must be a number; use
[`to_dict`](@ref) for symbolic ones.
"""
coefficients(f::Expression, vars::AbstractVector{Expression})::Vector{ComplexF64} =
    _numeric_coefficients(last(_sorted_terms(f, vars)))

coefficients(f::Expression, var::Expression)::Vector{ComplexF64} =
    coefficients(f, Expression[var])

"""
    poly_from_exponents_coefficients(M, c, vars) -> Expression

Polynomial with exponent matrix `M` (one term per column) and coefficients `c`.
Inverse of [`exponents_coefficients`](@ref).
"""
function poly_from_exponents_coefficients(
        M::AbstractMatrix{<:Integer}, c::AbstractVector{<:Number},
        vars::AbstractVector{Expression},
    )::Expression
    n, m = size(M)
    length(c) == m || throw(
        ArgumentError(
            string(
                "the exponent matrix has ", m,
                " columns but the coefficient vector has length ", length(c),
            ),
        ),
    )
    length(vars) == n || throw(
        ArgumentError(
            string(
                "the exponent matrix has ", n,
                " rows but the variable vector has length ", length(vars),
            ),
        ),
    )
    terms = Vector{Expression}(undef, m)
    for j in 1:m
        terms[j] = _term(Expression(c[j]), vars, @view M[:, j])
    end
    return _eadd(terms)
end

## ── Monomials and dense polynomials ─────────────────────────────────────────

function _monomial_exponents!(
        E::Vector{Vector{Int}}, e::Vector{Int}, i::Int, rest::Int, affine::Bool,
    )::Nothing
    if i > length(e)
        (affine || rest == 0) && push!(E, copy(e))
        return nothing
    end
    # Only the last coordinate absorbing everything left is homogeneous, so that walk
    # never descends the branches whose leaves it would discard.
    if i == length(e) && !affine
        e[i] = rest
        push!(E, copy(e))
        e[i] = 0
        return nothing
    end
    for k in 0:rest
        e[i] = k
        _monomial_exponents!(E, e, i + 1, rest - k, affine)
    end
    e[i] = 0
    return nothing
end

function _monomial_exponents(n::Int, d::Int; affine::Bool)::Vector{Vector{Int}}
    E = Vector{Int}[]
    _monomial_exponents!(E, zeros(Int, n), 1, d, affine)
    return _sorted_td(E)
end

function _monomial(
        vars::AbstractVector{Expression}, e::AbstractVector{<:Integer},
    )::Expression
    factors = Expression[]
    for i in eachindex(e)
        iszero(e[i]) && continue
        push!(factors, _epow(vars[i], Int(e[i])))
    end
    return _emul(factors)
end

_term(
    c::Expression, vars::AbstractVector{Expression}, e::AbstractVector{<:Integer},
)::Expression = _emul(Expression[c, _monomial(vars, e)])

_linear_combination(c::Vector{Expression}, M::Vector{Expression})::Expression =
    _eadd(Expression[_emul(Expression[c[i], M[i]]) for i in eachindex(M)])

"""
    monomials(vars, d::Integer; homogeneous = false, affine = !homogeneous)
    monomials(vars, degrees::AbstractVector{<:Integer})

All monomials in `vars` of degree at most `d`, or of degree exactly `d` when
`affine = false`. `homogeneous` is the same switch the other way round; passing
both, `affine` is the one that decides. Given a vector of degrees, the
homogeneous monomials of each, highest degree first.

```julia
@var x y
monomials([x, y], 2; affine = false)   # [x^2, x*y, y^2]
```
"""
function monomials(
        vars::AbstractVector{Expression}, d::Integer;
        homogeneous::Bool = false, affine::Bool = !homogeneous,
    )::Vector{Expression}
    exponents = _monomial_exponents(length(vars), Int(d); affine = affine)
    return Expression[_monomial(vars, e) for e in exponents]
end

function monomials(
        vars::AbstractVector{Expression}, degrees::AbstractVector{<:Integer},
    )::Vector{Expression}
    out = Expression[]
    for d in sort(degrees; rev = true)
        append!(out, monomials(vars, d; homogeneous = true))
    end
    return out
end

"""
    dense_poly(vars, d::Integer; homogeneous = false, coeff_name = gensym(:c))

The dense polynomial of degree `d` in `vars` whose coefficients are fresh
variables, and those coefficients. [`coeffs_as_dense_poly`](@ref) produces the
values that specialize it to a given polynomial.

```julia
@var x y
f, c = dense_poly([x, y], 2; coeff_name = :q)
```
"""
function dense_poly(
        vars::AbstractVector{Expression}, d::Integer;
        homogeneous::Bool = false, coeff_name::Symbol = gensym(:c),
    )::Tuple{Expression, Vector{Expression}}
    M = monomials(vars, d; homogeneous = homogeneous)
    c = Expression[variable(coeff_name, i) for i in eachindex(M)]
    return _linear_combination(c, M), c
end

"""
    coeffs_as_dense_poly(f::Expression, vars, d::Integer; homogeneous = false)

Coefficients `c` of `f` laid out so that substituting them into
`dense_poly(vars, d; homogeneous)` returns `f`. Monomials of `f` that the dense
polynomial does not carry make it an error; missing ones give a zero.

```julia
@var x[1:3]
f, c = dense_poly(x, 3; coeff_name = :c)
g = x[1]^3 + x[2]^3 + x[3]^3 - 1
subs(f, c => coeffs_as_dense_poly(g, x, 3)) == g
```
"""
function coeffs_as_dense_poly(
        f::Expression, vars::AbstractVector{Expression}, d::Integer;
        homogeneous::Bool = false,
    )::Vector{ComplexF64}
    exponents = _monomial_exponents(length(vars), Int(d); affine = !homogeneous)
    terms = _terms_or_throw(f, vars)
    out = zeros(ComplexF64, length(exponents))
    for (i, e) in enumerate(exponents)
        c = get(terms, e, nothing)
        c === nothing && continue
        v = expr_number(c)
        v === nothing && throw(
            ArgumentError("the coefficient `" * string(c) * "` is not a number"),
        )
        out[i] = v
        delete!(terms, e)
    end
    isempty(terms) || throw(
        ArgumentError(
            string(
                "the polynomial has ", length(terms),
                " term(s) the dense polynomial of degree ", d, " does not carry",
            ),
        ),
    )
    return out
end

"""
    rand_poly([rng], [T = ComplexF64], vars, d::Integer; homogeneous = false)

Dense polynomial of degree `d` in `vars` with coefficients drawn from
`randn(rng, T)`.
"""
function rand_poly(
        rng::Random.AbstractRNG, ::Type{T}, vars::AbstractVector{Expression}, d::Integer;
        homogeneous::Bool = false,
    )::Expression where {T <: Number}
    M = monomials(vars, d; homogeneous = homogeneous)
    c = Expression[Expression(z) for z in randn(rng, T, length(M))]
    return _linear_combination(c, M)
end

function rand_poly(
        ::Type{T}, vars::AbstractVector{Expression}, d::Integer;
        homogeneous::Bool = false,
    )::Expression where {T <: Number}
    return rand_poly(Random.default_rng(), T, vars, d; homogeneous = homogeneous)
end

rand_poly(
    rng::Random.AbstractRNG, vars::AbstractVector{Expression}, d::Integer;
    homogeneous::Bool = false,
)::Expression = rand_poly(rng, ComplexF64, vars, d; homogeneous = homogeneous)

rand_poly(
    vars::AbstractVector{Expression}, d::Integer; homogeneous::Bool = false,
)::Expression =
    rand_poly(Random.default_rng(), ComplexF64, vars, d; homogeneous = homogeneous)

## ── Horner ──────────────────────────────────────────────────────────────────

function _univariate_horner(c::Vector{Expression}, var::Expression)::Expression
    h = c[end]
    for k in (length(c) - 1):-1:1
        h = _eadd(Expression[_emul(Expression[h, var]), c[k]])
    end
    return h
end

# Factor out the variable occurring in the most terms, recurse on the rest.
function _multivariate_horner(
        M::Matrix{Int}, c::Vector{Expression}, vars::AbstractVector{Expression},
    )::Expression
    n, m = size(M)
    m == 1 && return _term(c[1], vars, @view M[:, 1])

    counts = zeros(Int, n)
    for j in 1:m, i in 1:n
        counts[i] += M[i, j] > 0
    end
    pivot = argmax(counts)
    d = maximum(@view M[pivot, :])

    groups = [Int[] for _ in 0:d]
    for j in 1:m
        push!(groups[M[pivot, j] + 1], j)
    end
    keep = [i for i in 1:n if i != pivot]
    reduced_vars = vars[keep]

    var_coeffs = Vector{Expression}(undef, d + 1)
    for k in 0:d
        js = groups[k + 1]
        var_coeffs[k + 1] = if isempty(js)
            zero(Expression)
        else
            _multivariate_horner(M[keep, js], c[js], reduced_vars)
        end
    end
    return _univariate_horner(var_coeffs, vars[pivot])
end

"""
    horner(f::Expression, vars = variables(f)) -> Expression

Rewrite `f` in a multivariate Horner scheme, which evaluates in fewer
operations. Returns `f` unchanged if it is not polynomial in `vars`.

```julia
@var u v c[1:3]
f = c[1] + c[2] * v + c[3] * u^2 * v^2 + c[3] * u^3 * v
horner(f)   # c₁ + v*(c₂ + u^3*c₃ + u^2*v*c₃)
```
"""
function horner(
        f::Expression, vars::AbstractVector{Expression} = variables(f),
    )::Expression
    raw = _expr_terms(f, _var_index(vars), length(vars))
    raw === nothing && return f
    terms = _clean_terms!(raw)
    isempty(terms) && return zero(Expression)
    exponents, c = _sorted_terms(terms)
    return _multivariate_horner(_exponent_matrix(Int, exponents, length(vars)), c, vars)
end

horner(f::Expression, var::Expression)::Expression = horner(f, Expression[var])

## ── Numeric evaluation ──────────────────────────────────────────────────────

"""
    to_number(e::Expression) -> ComplexF64

The value of `e`, which must carry no variables.
"""
function to_number(e::Expression)::ComplexF64
    v = expr_number(e)
    v === nothing &&
        throw(ArgumentError("`" * string(e) * "` is not a number"))
    return v
end

# `Expression <: Number`, so `convert(Number, e)` must stay the identity `Base.convert`
# guarantees; only a target `e` does not satisfy asks for the numeric value.
Base.convert(::Type{T}, e::Expression) where {T <: Number} =
    e isa T ? e : convert(T, to_number(e))

# All-real input evaluates to `Float64`, not to complex with vanishing imaginary parts.
_narrow(z::ComplexF64)::Union{Float64, ComplexF64} = iszero(imag(z)) ? real(z) : z

function _narrow(u::AbstractArray{ComplexF64, N}) where {N}
    all(z -> iszero(imag(z)), u) && return real.(u)
    return u
end

"""
    evaluate(f, subs...)

Substitute into the expression or array of expressions `f` and return the
resulting numbers. Every variable must be given a value, either as
`variables => values` pairs or as one dictionary.

```julia
@var x y
evaluate([x^2, x * y], [x, y] => [2, 3])   # [4.0, 6.0]
```
"""
evaluate(f::Expression, pairs::Pair...) = _narrow(to_number(subs(f, pairs...)))
evaluate(f::Expression, pairs::AbstractDict) = _narrow(to_number(subs(f, pairs)))

evaluate(f::AbstractArray{Expression}, pairs::Pair...) =
    _narrow(map(to_number, subs(f, pairs...)))
evaluate(f::AbstractArray{Expression}, pairs::AbstractDict) =
    _narrow(map(to_number, subs(f, pairs)))

(f::Expression)(pairs::Pair...) = evaluate(f, pairs...)
(f::Expression)(pairs::AbstractDict) = evaluate(f, pairs)
