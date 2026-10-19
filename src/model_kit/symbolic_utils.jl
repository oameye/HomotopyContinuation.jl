## Polynomial-oriented utilities on `Expression`: expansion, monomial
## enumeration, coefficient extraction, Horner rewriting and numeric evaluation.
#
# `Expression` keeps products and powers unexpanded, so anything needing one term
# per monomial goes through `expr_terms`, which expands into
# `exponent vector in vars => coefficient expression`; variables outside `vars`
# land in the coefficient.

## ── Expansion ───────────────────────────────────────────────────────────────

expression_summands(e::Expression)::Vector{Expression} = @match e begin
    SymExpr.EAdd(args) => args::Vector{Expression}
    _ => Expression[e]
end

function distribute(
        terms::Vector{Expression}, factor::Vector{Expression},
    )::Vector{Expression}
    next = Vector{Expression}(undef, length(terms) * length(factor))
    k = 0
    for t in terms, f in factor
        next[k += 1] = emul(Expression[t, f])
    end
    return next
end

# Collecting after every stage keeps the working set at the number of distinct monomials;
# distributing all stages first is exponential in the number of factors.
distribute_collected(terms::Vector{Expression}, factor::Vector{Expression}) =
    expression_summands(eadd(distribute(terms, factor)))

function expand_product(args::Vector{Expression})::Expression
    terms = Expression[one(Expression)]
    for a in args
        terms = distribute_collected(terms, expression_summands(expand(a)))
    end
    return eadd(terms)
end

# Splits the summands once, not once per exponent as `expand_product` would.
function expand_power(base::Expression, k::Int)::Expression
    k > 0 && isa_variant(base, SymExpr.EAdd) || return epow(base, k)
    factor = expression_summands(base)
    terms = factor
    for _ in 2:k
        terms = distribute_collected(terms, factor)
    end
    return eadd(terms)
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
    return @match e begin
        SymExpr.ENum(_) => e
        SymExpr.EVar(_) => e
        SymExpr.EAdd(args) =>
            eadd(Expression[expand(a) for a in args])
        SymExpr.EMul(args) => expand_product(args)
        SymExpr.EPow(base, exp) => expand_power(expand(base), exp)
        SymExpr.ERPow(base, exp) => erpow(expand(base), exp)
        SymExpr.EFn(kind, arg) => efn(kind, expand(arg))
    end
end

expand(exprs::AbstractArray{Expression}) = map(expand, exprs)

## ── Expansion into monomial => coefficient ──────────────────────────────────

const ExprTermDict = Dict{Vector{Int}, Expression}

var_index(vars::AbstractVector{Expression})::Dict{Symbol, Int} =
    Dict{Symbol, Int}(Symbol(v) => i for (i, v) in enumerate(vars))

function depends_on(e::Expression, var_to_idx::Dict{Symbol, Int})::Bool
    return @match e begin
        SymExpr.EVar(name) => haskey(var_to_idx, name)
        SymExpr.EAdd(args) => any_depends_on(args, var_to_idx)
        SymExpr.EMul(args) => any_depends_on(args, var_to_idx)
        SymExpr.EPow(base, _) => depends_on(base, var_to_idx)
        SymExpr.ERPow(base, _) => depends_on(base, var_to_idx)
        SymExpr.EFn(_, arg) => depends_on(arg, var_to_idx)
        _ => false
    end
end

function any_depends_on(args::Vector{Expression}, var_to_idx::Dict{Symbol, Int})::Bool
    for a in args
        depends_on(a, var_to_idx) && return true
    end
    return false
end

@inline function accumulate_term!(acc::ExprTermDict, m::Vector{Int}, c::Expression)::Nothing
    old = get(acc, m, nothing)
    acc[m] = old === nothing ? c : old + c
    return nothing
end

function add_expr_terms!(acc::ExprTermDict, other::ExprTermDict)::Nothing
    for (m, c) in other
        accumulate_term!(acc, m, c)
    end
    return nothing
end

function multiply_expr_terms(a::ExprTermDict, b::ExprTermDict)::ExprTermDict
    out = ExprTermDict()
    for (ma, ca) in a, (mb, cb) in b
        accumulate_term!(out, ma .+ mb, ca * cb)
    end
    return out
end

# `found` and not an empty dict: cancellation can legitimately leave no terms at
# all, since `clean_terms!` drops zero coefficients, so emptiness cannot double
# as the failure marker.
struct ExprTerms
    found::Bool
    terms::ExprTermDict
end

const NOT_POLYNOMIAL_TERMS = ExprTerms(false, ExprTermDict())
polynomial_terms(t::ExprTermDict)::ExprTerms = ExprTerms(true, t)

# `found` is `false` when `e` is not polynomial in the indexed variables.
function expr_terms(
        e::Expression, var_to_idx::Dict{Symbol, Int}, n::Int,
    )::ExprTerms
    constant() = polynomial_terms(ExprTermDict(zeros(Int, n) => e))
    return @match e begin
        SymExpr.ENum(_) => constant()
        SymExpr.EVar(name) => begin
            idx = get(var_to_idx, name, 0)
            if idx == 0
                constant()
            else
                m = zeros(Int, n)
                m[idx] = 1
                polynomial_terms(ExprTermDict(m => one(Expression)))
            end
        end
        SymExpr.EAdd(args) => begin
            acc = ExprTermDict()
            for a in args
                terms = expr_terms(a, var_to_idx, n)
                terms.found || return NOT_POLYNOMIAL_TERMS
                add_expr_terms!(acc, terms.terms)
            end
            polynomial_terms(acc)
        end
        SymExpr.EMul(args) => begin
            acc = ExprTermDict(zeros(Int, n) => one(Expression))
            for a in args
                terms = expr_terms(a, var_to_idx, n)
                terms.found || return NOT_POLYNOMIAL_TERMS
                acc = multiply_expr_terms(acc, terms.terms)
            end
            polynomial_terms(acc)
        end
        SymExpr.EPow(base, exp) => begin
            depends_on(e, var_to_idx) || return constant()
            exp > 0 || return NOT_POLYNOMIAL_TERMS
            b = expr_terms(base, var_to_idx, n)
            b.found || return NOT_POLYNOMIAL_TERMS
            acc = ExprTermDict(zeros(Int, n) => one(Expression))
            for _ in 1:exp
                acc = multiply_expr_terms(acc, b.terms)
            end
            polynomial_terms(acc)
        end
        SymExpr.ERPow(base, _) => begin
            depends_on(base, var_to_idx) && return NOT_POLYNOMIAL_TERMS
            constant()
        end
        SymExpr.EFn(_, arg) => begin
            depends_on(arg, var_to_idx) && return NOT_POLYNOMIAL_TERMS
            constant()
        end
    end
end

# Coefficients accumulate as sums of products, so a cancellation shows only after
# expanding.
function clean_terms!(terms::ExprTermDict)::ExprTermDict
    for m in collect(keys(terms))
        terms[m] = expand(terms[m])
    end
    filter!(entry -> !iszero(entry[2]), terms)
    return terms
end

function terms_or_throw(f::Expression, vars::AbstractVector{Expression})::ExprTermDict
    terms = expr_terms(f, var_index(vars), length(vars))
    terms.found || throw(
        ArgumentError("`" * string(f) * "` is not polynomial in the given variables"),
    )
    return clean_terms!(terms.terms)
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
    terms_or_throw(f, vars)

# Descending total degree, ties broken by descending lexicographic order. Shared, so
# `monomials`, `coefficients` and `coeffs_as_dense_poly` agree column for column. The
# degree is cached per vector rather than resummed on every comparison.
function sorted_td(exponents::Vector{Vector{Int}})::Vector{Vector{Int}}
    keyed = Tuple{Int, Vector{Int}}[(sum(e; init = 0), e) for e in exponents]
    sort!(keyed; lt = (a, b) -> a[1] == b[1] ? a[2] > b[2] : a[1] > b[1])
    return Vector{Int}[e for (_, e) in keyed]
end

sorted_terms(terms::ExprTermDict)::Tuple{Vector{Vector{Int}}, Vector{Expression}} =
let exponents = sorted_td(collect(keys(terms)))
    (exponents, Expression[terms[e] for e in exponents])
end

sorted_terms(
    f::Expression, vars::AbstractVector{Expression},
)::Tuple{Vector{Vector{Int}}, Vector{Expression}} =
    sorted_terms(terms_or_throw(f, vars))

function exponent_matrix(
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

function numeric_coefficients(c::Vector{Expression})::Vector{ComplexF64}
    out = Vector{ComplexF64}(undef, length(c))
    for (i, e) in enumerate(c)
        v = expr_number(e)
        v.found || throw(
            ArgumentError(
                "the coefficient `" * string(e) *
                    "` is not a number; use `to_dict` for symbolic coefficients",
            ),
        )
        out[i] = v.val
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
    exponents, c = sorted_terms(f, vars)
    return exponent_matrix(Int32, exponents, length(vars)), numeric_coefficients(c)
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
    numeric_coefficients(last(sorted_terms(f, vars)))

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
        terms[j] = expression_term(Expression(c[j]), vars, @view M[:, j])
    end
    return eadd(terms)
end

## ── Monomials and dense polynomials ─────────────────────────────────────────

function monomial_exponents!(
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
        monomial_exponents!(E, e, i + 1, rest - k, affine)
    end
    e[i] = 0
    return nothing
end

function monomial_exponents(n::Int, d::Int; affine::Bool)::Vector{Vector{Int}}
    E = Vector{Int}[]
    monomial_exponents!(E, zeros(Int, n), 1, d, affine)
    return sorted_td(E)
end

function expression_monomial(
        vars::AbstractVector{Expression}, e::AbstractVector{<:Integer},
    )::Expression
    factors = Expression[]
    for i in eachindex(e)
        iszero(e[i]) && continue
        push!(factors, epow(vars[i], Int(e[i])))
    end
    return emul(factors)
end

expression_term(
    c::Expression, vars::AbstractVector{Expression}, e::AbstractVector{<:Integer},
)::Expression = emul(Expression[c, expression_monomial(vars, e)])

linear_combination(c::Vector{Expression}, M::Vector{Expression})::Expression =
    eadd(Expression[emul(Expression[c[i], M[i]]) for i in eachindex(M)])

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
    exponents = monomial_exponents(length(vars), Int(d); affine = affine)
    return Expression[expression_monomial(vars, e) for e in exponents]
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
    return linear_combination(c, M), c
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
    exponents = monomial_exponents(length(vars), Int(d); affine = !homogeneous)
    terms = terms_or_throw(f, vars)
    out = zeros(ComplexF64, length(exponents))
    for (i, e) in enumerate(exponents)
        c = get(terms, e, nothing)
        c === nothing && continue
        v = expr_number(c)
        v.found || throw(
            ArgumentError("the coefficient `" * string(c) * "` is not a number"),
        )
        out[i] = v.val
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
    return linear_combination(c, M)
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

function univariate_horner(c::Vector{Expression}, var::Expression)::Expression
    h = c[end]
    for k in (length(c) - 1):-1:1
        h = eadd(Expression[emul(Expression[h, var]), c[k]])
    end
    return h
end

# Factor out the variable occurring in the most terms, recurse on the rest.
function multivariate_horner(
        M::Matrix{Int}, c::Vector{Expression}, vars::AbstractVector{Expression},
    )::Expression
    n, m = size(M)
    m == 1 && return expression_term(c[1], vars, @view M[:, 1])

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
            multivariate_horner(M[keep, js], c[js], reduced_vars)
        end
    end
    return univariate_horner(var_coeffs, vars[pivot])
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
    raw = expr_terms(f, var_index(vars), length(vars))
    raw.found || return f
    terms = clean_terms!(raw.terms)
    isempty(terms) && return zero(Expression)
    exponents, c = sorted_terms(terms)
    return multivariate_horner(exponent_matrix(Int, exponents, length(vars)), c, vars)
end

horner(f::Expression, var::Expression)::Expression = horner(f, Expression[var])

## ── Numeric evaluation ──────────────────────────────────────────────────────

"""
    to_number(e::Expression) -> ComplexF64

The value of `e`, which must carry no variables.
"""
function to_number(e::Expression)::ComplexF64
    v = expr_number(e)
    v.found || throw(ArgumentError("`" * string(e) * "` is not a number"))
    return v.val
end

# `Expression <: Number`, so `convert(Number, e)` must stay the identity `Base.convert`
# guarantees; only a target `e` does not satisfy asks for the numeric value.
Base.convert(::Type{T}, e::Expression) where {T <: Number} =
    e isa T ? e : convert(T, to_number(e))

"""
    evaluate(f, subs...)

Substitute into the expression or array of expressions `f` and return the
resulting numbers. Every variable must be given a value, either as
`variables => values` pairs or as one dictionary.

Always complex, including when every value comes out real: narrowing to `Float64`
would make the return type depend on the values rather than on the argument types.
Call `real.` on the result when a real answer is wanted.

```julia
@var x y
evaluate([x^2, x * y], [x, y] => [2, 3])   # ComplexF64[4.0 + 0.0im, 6.0 + 0.0im]
```
"""
evaluate(f::Expression, pairs::Pair...)::ComplexF64 = to_number(subs(f, pairs...))
evaluate(f::Expression, pairs::AbstractDict)::ComplexF64 = to_number(subs(f, pairs))

evaluate(f::AbstractArray{Expression}, pairs::Pair...) =
    map(to_number, subs(f, pairs...))
evaluate(f::AbstractArray{Expression}, pairs::AbstractDict) =
    map(to_number, subs(f, pairs))

(f::Expression)(pairs::Pair...) = evaluate(f, pairs...)
(f::Expression)(pairs::AbstractDict) = evaluate(f, pairs)
