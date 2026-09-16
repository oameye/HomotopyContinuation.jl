## Expression: symbolic frontend for non-polynomial input.
#
# `Expression` is a `Number` subtype, so ordinary Julia arithmetic, `sum`,
# broadcasting and matrix products build expression trees. It covers everything
# the tape can execute that `MultivariatePolynomials` cannot express: division,
# powers with a negative or non-integer exponent, and the unary functions
# (`sqrt`, `exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `sinh`, `cosh`, `tanh`).
#
# The tree is canonicalized on construction (flattening, constant folding, like
# term collection, power collection) so that structurally equal expressions are
# `==` and hash equal. Lowering to `SExpr` therefore feeds CSE the same shape it
# gets from the polynomial frontend.

@data SymExpr <: Number begin
    """Numeric literal."""
    struct ENum
        val::ComplexF64
    end

    """Named variable or parameter."""
    struct EVar
        name::Symbol
    end

    """Sum of args (n-ary, n >= 2)."""
    struct EAdd
        args::Vector{SymExpr}
    end

    """Product of args (n-ary, n >= 2)."""
    struct EMul
        args::Vector{SymExpr}
    end

    """Integer power, possibly negative. `exp` is never 0 or 1."""
    struct EPow
        base::SymExpr
        exp::Int
    end

    """Power with a numeric non-integer exponent."""
    struct ERPow
        base::SymExpr
        exp::ComplexF64
    end

    """Unary function application."""
    struct EFn
        kind::SUnaryKind.T
        arg::SymExpr
    end
end

@derive SymExpr[Eq]

const Expression = typeof(SymExpr.ENum(zero(ComplexF64)))

# Compound nodes are hashed structurally and used as Dict keys, so they must own their
# child storage rather than alias a caller vector: these take ownership of `args`, and
# the caller must not touch it again.
@inline _eadd_owned(args::Vector{Expression})::Expression =
    invoke(SymExpr.EAdd, Tuple{Any}, args)
@inline _emul_owned(args::Vector{Expression})::Expression =
    invoke(SymExpr.EMul, Tuple{Any}, args)

## ── Hashing and ordering ────────────────────────────────────────────────────

function _fold_expr_hash(seed, args)::UInt
    h = hash(seed, zero(UInt))
    for a in args
        h = hash(a, h)
    end
    return h
end

# `-0.0 == 0.0` but `hash(-0.0) != hash(0.0)`, and arithmetic on coefficients
# produces either sign of zero: `-1 * (0.0)` is `-0.0`. Literals are compared
# with `==`, so both zeros have to hash alike or two equal expressions get two
# hashes, which breaks every `Dict{Expression}` keyed on them.
@inline _canonical_zeros(v::ComplexF64)::ComplexF64 =
    complex(_canonical_zero(real(v)), _canonical_zero(imag(v)))
@inline _canonical_zero(x::Float64)::Float64 = iszero(x) ? zero(Float64) : x

function Base.hash(e::Expression, h::UInt)::UInt
    return @match e begin
        SymExpr.ENum(val) => hash(_canonical_zeros(val), hash(:ENum, h))
        SymExpr.EVar(name) => hash(name, hash(:EVar, h))
        SymExpr.EAdd(args) =>
            hash(_fold_expr_hash(:EAdd, args::Vector{Expression}), h)
        SymExpr.EMul(args) =>
            hash(_fold_expr_hash(:EMul, args::Vector{Expression}), h)
        SymExpr.EPow(base, exp) => hash(exp, hash(base::Expression, hash(:EPow, h)))
        SymExpr.ERPow(base, exp) => hash(exp, hash(base::Expression, hash(:ERPow, h)))
        SymExpr.EFn(kind, arg) => hash(arg::Expression, hash(kind, hash(:EFn, h)))
    end
end

_expr_lt(a::Expression, b::Expression)::Bool = hash(a) < hash(b)

## ── Predicates and accessors ────────────────────────────────────────────────

# `found` and not a sentinel value: any `ComplexF64`, `NaN` included, is a legal
# literal, so there is nothing left over to mean "not a literal".
struct ExprNumber
    found::Bool
    val::ComplexF64
end

"""Numeric value of `e`; `found` is `false` when `e` is not a literal."""
@inline function expr_number(e::Expression)::ExprNumber
    return @match e begin
        SymExpr.ENum(val) => ExprNumber(true, val)
        _ => ExprNumber(false, zero(ComplexF64))
    end
end

is_number(e::Expression)::Bool = isa_variant(e, SymExpr.ENum)
is_variable(e::Expression)::Bool = isa_variant(e, SymExpr.EVar)

function Base.Symbol(e::Expression)::Symbol
    return @match e begin
        SymExpr.EVar(name) => name
        _ => throw(ArgumentError("expression $(e) is not a variable"))
    end
end

Base.iszero(e::Expression)::Bool = @match e begin
    SymExpr.ENum(val) => iszero(val)
    _ => false
end
Base.isone(e::Expression)::Bool = @match e begin
    SymExpr.ENum(val) => isone(val)
    _ => false
end

Base.zero(::Type{Expression})::Expression = SymExpr.ENum(zero(ComplexF64))
Base.one(::Type{Expression})::Expression = SymExpr.ENum(one(ComplexF64))
Base.zero(::Expression)::Expression = zero(Expression)
Base.one(::Expression)::Expression = one(Expression)

# Numeric literals are conjugated; variables stand for real symbols, since the
# tape has no conjugation instruction.
function Base.conj(e::Expression)::Expression
    return @match e begin
        SymExpr.ENum(val) => SymExpr.ENum(conj(val))
        SymExpr.EVar(_) => e
        SymExpr.EAdd(args) => _eadd(Expression[conj(a) for a in args::Vector{Expression}])
        SymExpr.EMul(args) => _emul(Expression[conj(a) for a in args::Vector{Expression}])
        SymExpr.EPow(base, exp) => _epow(conj(base::Expression), exp)
        SymExpr.ERPow(base, exp) => _erpow(conj(base::Expression), conj(exp))
        SymExpr.EFn(kind, arg) => _efn(kind, conj(arg::Expression))
    end
end

Base.adjoint(e::Expression)::Expression = conj(e)
Base.transpose(e::Expression)::Expression = e

Base.promote_rule(::Type{Expression}, ::Type{<:Real}) = Expression
Base.promote_rule(::Type{Expression}, ::Type{<:Complex}) = Expression
Base.convert(::Type{Expression}, x::Expression)::Expression = x
Base.convert(::Type{Expression}, x::Number)::Expression =
    SymExpr.ENum(ComplexF64(x))
Expression(x::Number)::Expression = SymExpr.ENum(ComplexF64(x))

# `Expression <: Number` puts Moshi's untyped inner constructor in conflict with
# Base's `(::Type{T})(::AbstractChar)` and `(::Type{T})(::TwicePrecision)`.
Expression(x::AbstractChar)::Expression = SymExpr.ENum(ComplexF64(x))
Expression(x::Base.TwicePrecision)::Expression = SymExpr.ENum(ComplexF64(x))

## ── Canonical constructors ──────────────────────────────────────────────────

# Beyond this many distinct keys a linear scan stops beating a hash lookup.
const _LINEAR_SCAN_MAX = 8

@inline function _find_expr(keys::Vector{Expression}, e::Expression)::Int
    for i in eachindex(keys)
        @inbounds keys[i] == e && return i
    end
    return 0
end

function _flatten_eadd!(
        terms::Vector{Expression}, const_sum::ComplexF64, e::Expression,
    )::ComplexF64
    @match e begin
        SymExpr.EAdd(args) => for child in args::Vector{Expression}
            const_sum = _flatten_eadd!(terms, const_sum, child)
        end
        SymExpr.ENum(val) => (const_sum += val)
        _ => push!(terms, e)
    end
    return const_sum
end

# `split` maps a term to the key it is tallied under and the value added to it.
function _tally!(
        split::F, bases::Vector{Expression}, vals::Vector{T},
        terms::Vector{Expression},
    )::Nothing where {F, T}
    index = length(terms) > _LINEAR_SCAN_MAX ? Dict{Expression, Int}() : nothing
    for t in terms
        (base, v) = split(t)
        k = index === nothing ? _find_expr(bases, base) : get(index, base, 0)
        if k == 0
            push!(bases, base)
            push!(vals, v)
            index === nothing || (index[base] = length(bases))
        else
            @inbounds vals[k] += v
        end
    end
    return nothing
end

"""Split a product into the rest of it and its leading numeric coefficient."""
function _split_coefficient(e::Expression)::Tuple{Expression, ComplexF64}
    @match e begin
        SymExpr.EMul(args) => begin
            xs = args::Vector{Expression}
            @match xs[1] begin
                SymExpr.ENum(val) => begin
                    rest = xs[2:end]
                    return (length(rest) == 1 ? rest[1] : _emul_owned(rest)), val
                end
                _ => nothing
            end
        end
        _ => nothing
    end
    return e, one(ComplexF64)
end

function _eadd(args::Vector{Expression})::Expression
    terms = Expression[]
    const_sum = zero(ComplexF64)
    for a in args
        const_sum = _flatten_eadd!(terms, const_sum, a)
    end

    bases = Expression[]
    coeffs = ComplexF64[]
    _tally!(_split_coefficient, bases, coeffs, terms)

    collected = Expression[]
    nested = false
    for i in eachindex(bases)
        @inbounds (base, c) = (bases[i], coeffs[i])
        iszero(c) && continue
        if isone(c)
            # `base` can be a sum (`2 * (x + y)` splits to `x + y`), which has to
            # be re-flattened rather than nested.
            nested |= isa_variant(base, SymExpr.EAdd)
            push!(collected, base)
        else
            push!(collected, _emul(Expression[SymExpr.ENum(c), base]))
        end
    end
    if nested
        !iszero(const_sum) && push!(collected, SymExpr.ENum(const_sum))
        return _eadd(collected)
    end

    sort!(collected; lt = _expr_lt)
    !iszero(const_sum) && pushfirst!(collected, SymExpr.ENum(const_sum))
    isempty(collected) && return zero(Expression)
    length(collected) == 1 && return @inbounds collected[1]
    return _eadd_owned(collected)
end

function _flatten_emul!(
        factors::Vector{Expression}, coeff::ComplexF64, e::Expression,
    )::ComplexF64
    @match e begin
        SymExpr.EMul(args) => for child in args::Vector{Expression}
            coeff = _flatten_emul!(factors, coeff, child)
        end
        SymExpr.ENum(val) => (coeff *= val)
        _ => push!(factors, e)
    end
    return coeff
end

"""Split a factor into its base and its exponent."""
@inline function _split_power(e::Expression)::Tuple{Expression, ComplexF64}
    return @match e begin
        SymExpr.EPow(base, exp) => (base::Expression, ComplexF64(exp))
        SymExpr.ERPow(base, exp) => (base::Expression, exp)
        _ => (e, one(ComplexF64))
    end
end

function _emul(args::Vector{Expression})::Expression
    factors = Expression[]
    coeff = one(ComplexF64)
    for a in args
        coeff = _flatten_emul!(factors, coeff, a)
    end
    iszero(coeff) && return zero(Expression)

    bases = Expression[]
    exps = ComplexF64[]
    _tally!(_split_power, bases, exps, factors)

    # `_erpow` can fold a base to a literal (e.g. (1/2)^-1), so re-absorb them, and
    # returns an `EPow` for an integer sum, which keeps `x * x` an integer power.
    kept = Expression[]
    for i in eachindex(bases)
        @inbounds k = exps[i]
        iszero(k) && continue
        @inbounds p = _erpow(bases[i], k)
        v = expr_number(p)
        if !v.found
            push!(kept, p)
        else
            coeff *= v.val
        end
    end
    iszero(coeff) && return zero(Expression)

    sort!(kept; lt = _expr_lt)
    isone(coeff) || pushfirst!(kept, SymExpr.ENum(coeff))
    isempty(kept) && return one(Expression)
    length(kept) == 1 && return @inbounds kept[1]
    return _emul_owned(kept)
end

function _epow(base::Expression, k::Int)::Expression
    k == 0 && return one(Expression)
    k == 1 && return base
    return @match base begin
        SymExpr.ENum(val) => SymExpr.ENum(op_pow_int(val, k))
        SymExpr.EPow(b, exp) => _epow(b::Expression, exp * k)
        # (a*b)^k = a^k * b^k keeps powers next to their base for CSE.
        SymExpr.EMul(args) =>
            _emul(Expression[_epow(a, k) for a in args::Vector{Expression}])
        _ => SymExpr.EPow(base, k)
    end
end

function _erpow(base::Expression, r::ComplexF64)::Expression
    # An integer-valued exponent belongs in `EPow`, which the degree, numerator
    # and power-collection paths all read.
    iszero(imag(r)) && isinteger(real(r)) && return _epow(base, Int(real(r)))
    v = expr_number(base)
    v.found && return SymExpr.ENum(v.val^r)
    return SymExpr.ERPow(base, r)
end

function _unary_name(kind::SUnaryKind.T)::String
    kind == SUnaryKind.UNARY_SQRT && return "sqrt"
    kind == SUnaryKind.UNARY_SIN && return "sin"
    kind == SUnaryKind.UNARY_COS && return "cos"
    kind == SUnaryKind.UNARY_EXP && return "exp"
    kind == SUnaryKind.UNARY_LOG && return "log"
    kind == SUnaryKind.UNARY_TAN && return "tan"
    kind == SUnaryKind.UNARY_ASIN && return "asin"
    kind == SUnaryKind.UNARY_ACOS && return "acos"
    kind == SUnaryKind.UNARY_SINH && return "sinh"
    kind == SUnaryKind.UNARY_COSH && return "cosh"
    return "tanh"
end

function _efn(kind::SUnaryKind.T, arg::Expression)::Expression
    v = expr_number(arg)
    v.found && return SymExpr.ENum(apply_unary(kind, v.val))
    return SymExpr.EFn(kind, arg)
end

## ── Arithmetic ──────────────────────────────────────────────────────────────

Base.:+(a::Expression, b::Expression)::Expression = _eadd(Expression[a, b])
Base.:*(a::Expression, b::Expression)::Expression = _emul(Expression[a, b])
Base.:-(a::Expression)::Expression = _emul(Expression[SymExpr.ENum(-one(ComplexF64)), a])
Base.:-(a::Expression, b::Expression)::Expression = _eadd(Expression[a, -b])
Base.:/(a::Expression, b::Expression)::Expression = _emul(Expression[a, _epow(b, -1)])
Base.inv(a::Expression)::Expression = _epow(a, -1)
Base.:^(a::Expression, k::Integer)::Expression = _epow(a, Int(k))
Base.literal_pow(::typeof(^), a::Expression, ::Val{K}) where {K} = _epow(a, Int(K))
Base.sqrt(a::Expression)::Expression = _efn(SUnaryKind.UNARY_SQRT, a)
Base.sin(a::Expression)::Expression = _efn(SUnaryKind.UNARY_SIN, a)
Base.cos(a::Expression)::Expression = _efn(SUnaryKind.UNARY_COS, a)
# The coupled sin/cos Taylor recurrence asks for both at once, and Base's generic
# `sincos` would route a symbolic argument through `float`.
Base.sincos(a::Expression)::Tuple{Expression, Expression} = (sin(a), cos(a))
Base.exp(a::Expression)::Expression = _efn(SUnaryKind.UNARY_EXP, a)
Base.log(a::Expression)::Expression = _efn(SUnaryKind.UNARY_LOG, a)
Base.tan(a::Expression)::Expression = _efn(SUnaryKind.UNARY_TAN, a)
Base.asin(a::Expression)::Expression = _efn(SUnaryKind.UNARY_ASIN, a)
Base.acos(a::Expression)::Expression = _efn(SUnaryKind.UNARY_ACOS, a)
Base.sinh(a::Expression)::Expression = _efn(SUnaryKind.UNARY_SINH, a)
Base.cosh(a::Expression)::Expression = _efn(SUnaryKind.UNARY_COSH, a)
Base.tanh(a::Expression)::Expression = _efn(SUnaryKind.UNARY_TANH, a)

Base.:^(a::Expression, r::Real)::Expression = _erpow(a, ComplexF64(r))
Base.:^(a::Expression, r::Complex)::Expression = _erpow(a, ComplexF64(r))
# Disambiguates against `^(::Number, ::Rational)`.
Base.:^(a::Expression, r::Rational)::Expression = _erpow(a, ComplexF64(r))

# The exponent has to be numeric: the tape has no instruction for a symbolic one.
function Base.:^(a::Expression, b::Expression)::Expression
    v = expr_number(b)
    v.found || throw(ArgumentError("only numeric exponents are supported, got $(b)"))
    return _erpow(a, v.val)
end

## ── Linear algebra ──────────────────────────────────────────────────────────

# Laplace expansion along the first row: the generic `det` needs `abs` to pick a
# pivot, which a symbolic entry has no method for.
function LinearAlgebra.det(A::AbstractMatrix{Expression})::Expression
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix is not square: dimensions are $(size(A))"))
    n == 0 && return one(Expression)
    n == 1 && return @inbounds A[1, 1]
    if n == 2
        @inbounds return A[1, 1] * A[2, 2] - A[2, 1] * A[1, 2]
    end
    terms = Expression[]
    rows = 2:n
    for j in 1:n
        @inbounds a = A[1, j]
        iszero(a) && continue
        cols = [k for k in 1:n if k != j]
        minor = LinearAlgebra.det(@view A[rows, cols])
        push!(terms, isodd(j) ? a * minor : -(a * minor))
    end
    isempty(terms) && return zero(Expression)
    return _eadd(terms)
end

## ── Variable construction ───────────────────────────────────────────────────

const SUBSCRIPTS = ('₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉')
const SUBSCRIPT_MAP = Dict{Char, Char}(first(string(i)) => SUBSCRIPTS[i + 1] for i in 0:9)
const SUBSCRIPT_TO_INT = Dict{Char, Int}(SUBSCRIPTS[i + 1] => i for i in 0:9)

map_subscripts(index)::String = join(SUBSCRIPT_MAP[c] for c in string(index))

"""
    variable(name) -> Expression
    variable(name, indices...) -> Expression

Create a single variable. Indices are appended as unicode subscripts joined by
`₋`, so `variable(:y, 1, 2)` is named `y₁₋₂`.
"""
variable(name::Union{Symbol, AbstractString})::Expression = SymExpr.EVar(Symbol(name))
function variable(name::Union{Symbol, AbstractString}, indices::Integer...)::Expression
    return SymExpr.EVar(Symbol(name, join(map_subscripts.(indices), "₋")))
end

"""
    variable_array(prefix, indices...) -> Array{Expression}

Create an array of variables with the given `prefix`, one per index tuple.
`@var x[1:3, 1:2]` is equivalent to `x = variable_array(:x, 1:3, 1:2)`.
"""
function variable_array(prefix::Union{Symbol, AbstractString}, indices...)
    return map(i -> variable(prefix, i...), Iterators.product(indices...))
end

function _build_var(var, unique::Bool)::Tuple{Symbol, Expr}
    if var isa Symbol
        name = unique ? Symbol(String(gensym(var))[3:end]) : var
        return var, :($(esc(var)) = $(variable)($(QuoteNode(name))))
    end
    var isa Base.Expr ||
        throw(ArgumentError("expected $(var) to be a variable name"))
    Base.Meta.isexpr(var, :ref) ||
        throw(ArgumentError("expected $(var) to be of the form varname[idxset]"))
    length(var.args) >= 2 ||
        throw(ArgumentError("expected $(var) to have at least one index set"))
    varname = var.args[1]
    varname isa Symbol ||
        throw(ArgumentError("expected $(varname) to be a variable name"))
    prefix = unique ? String(gensym(varname))[3:end] : string(varname)
    return varname,
        :($(esc(varname)) = $(variable_array)($prefix, $(esc.(var.args[2:end])...)))
end

function _build_vars(args, unique::Bool)
    names = Symbol[]
    exprs = Any[]
    for arg in args
        subargs = (arg isa Base.Expr && arg.head === :tuple) ? arg.args : (arg,)
        for sub in subargs
            (name, ex) = _build_var(sub, unique)
            push!(names, name)
            push!(exprs, ex)
        end
    end
    return names, exprs
end

"""
    @var variable1 variable2 ...

Declare symbolic variables and bind them to the given names. Indexing notation
creates arrays of variables.

```julia-repl
julia> @var a x[1:2];

julia> a * x[1] + x[2]^2
a*x₁ + x₂^2
```
"""
macro var(args...)
    names, exprs = _build_vars(args, false)
    return Base.Expr(:block, exprs..., Base.Expr(:tuple, esc.(names)...))
end

"""
    @unique_var variable1 variable2 ...

Like [`@var`](@ref), but the generated names are made unique so they cannot
collide with user variables. The Julia bindings still use the declared names.
"""
macro unique_var(args...)
    names, exprs = _build_vars(args, true)
    return Base.Expr(:block, exprs..., Base.Expr(:tuple, esc.(names)...))
end

"""
    unique_variable(name, variables, parameters) -> Expression

Create a variable named `name` that clashes with neither `variables` nor
`parameters`, appending `##k` until the name is free.
"""
function unique_variable(
        name::Union{Symbol, AbstractString},
        vars::AbstractVector{Expression},
        params::AbstractVector{Expression},
    )::Expression
    v = variable(name)
    k = 0
    while v in vars || v in params
        v = variable(string(name, "##", k))
        k += 1
    end
    return v
end

## ── Variable ordering and discovery ─────────────────────────────────────────

# Order by base name first, then numerically by index, so `x₂` precedes `x₁₀`.
function _variable_sort_key(e::Expression)::Tuple{String, Vector{Int}}
    name = string(Symbol(e))
    sub_start = findfirst(c -> c in SUBSCRIPTS, name)
    sub_start === nothing && return (name, Int[])
    base = name[1:prevind(name, sub_start)]
    indices = Int[]
    for part in split(name[sub_start:end], "₋")
        value = 0
        for c in part
            digit = get(SUBSCRIPT_TO_INT, c, -1)
            digit < 0 && return (name, Int[])
            value = 10 * value + digit
        end
        push!(indices, value)
    end
    return (base, indices)
end

_variable_lt(a::Expression, b::Expression)::Bool =
    isless(_variable_sort_key(a), _variable_sort_key(b))

function _collect_expr_variables!(
        acc::Vector{Expression}, seen::Set{Symbol}, e::Expression,
    )::Nothing
    @match e begin
        SymExpr.EVar(name) => if !(name in seen)
            push!(seen, name)
            push!(acc, e)
        end
        SymExpr.EAdd(args) => for a in args::Vector{Expression}
            _collect_expr_variables!(acc, seen, a)
        end
        SymExpr.EMul(args) => for a in args::Vector{Expression}
            _collect_expr_variables!(acc, seen, a)
        end
        SymExpr.EPow(base, _) => _collect_expr_variables!(acc, seen, base::Expression)
        SymExpr.ERPow(base, _) => _collect_expr_variables!(acc, seen, base::Expression)
        SymExpr.EFn(_, arg) => _collect_expr_variables!(acc, seen, arg::Expression)
        _ => nothing
    end
    return nothing
end

"""
    variables(expr; parameters = Expression[]) -> Vector{Expression}
    variables(exprs::AbstractVector; parameters = Expression[]) -> Vector{Expression}

All variables occurring in `expr`, sorted by name and index, with `parameters`
removed.
"""
function variables(
        e::Expression; parameters::AbstractVector{Expression} = Expression[],
    )::Vector{Expression}
    return variables(Expression[e]; parameters = parameters)
end

function variables(
        exprs::AbstractVector{Expression};
        parameters::AbstractVector{Expression} = Expression[],
    )::Vector{Expression}
    acc = Expression[]
    seen = Set{Symbol}()
    for e in exprs
        _collect_expr_variables!(acc, seen, e)
    end
    if !isempty(parameters)
        param_names = Set{Symbol}(Symbol(p) for p in parameters)
        filter!(v -> !(Symbol(v) in param_names), acc)
    end
    sort!(acc; lt = _variable_lt)
    return acc
end

# Named wrapper so `System` can fill in its `variables` keyword, which shadows
# the `variables` function.
_default_variables(
    exprs::AbstractVector{Expression}, params::Vector{Expression},
)::Vector{Expression} = variables(exprs; parameters = params)

## ── Differentiation ─────────────────────────────────────────────────────────

function _differentiate(e::Expression, v::Symbol)::Expression
    return @match e begin
        SymExpr.ENum(_) => zero(Expression)
        SymExpr.EVar(name) => name === v ? one(Expression) : zero(Expression)
        SymExpr.EAdd(args) =>
            _eadd(Expression[_differentiate(a, v) for a in args::Vector{Expression}])
        SymExpr.EMul(args) => begin
            xs = args::Vector{Expression}
            terms = Expression[]
            for i in eachindex(xs)
                da = _differentiate(xs[i], v)
                iszero(da) && continue
                factors = Expression[da]
                for j in eachindex(xs)
                    j == i && continue
                    push!(factors, xs[j])
                end
                push!(terms, _emul(factors))
            end
            _eadd(terms)
        end
        SymExpr.EPow(base, exp) => begin
            b = base::Expression
            db = _differentiate(b, v)
            iszero(db) && return zero(Expression)
            _emul(Expression[SymExpr.ENum(ComplexF64(exp)), _epow(b, exp - 1), db])
        end
        SymExpr.ERPow(base, exp) => begin
            b = base::Expression
            db = _differentiate(b, v)
            iszero(db) && return zero(Expression)
            _emul(
                Expression[
                    SymExpr.ENum(exp), _erpow(b, exp - one(ComplexF64)), db,
                ],
            )
        end
        SymExpr.EFn(kind, arg) => begin
            a = arg::Expression
            da = _differentiate(a, v)
            iszero(da) && return zero(Expression)
            _emul(Expression[_unary_derivative(kind, a), da])
        end
    end
end

# The outer factor of the chain rule: d/dx f(a) = f'(a) · a'.
function _unary_derivative(kind::SUnaryKind.T, a::Expression)::Expression
    if kind == SUnaryKind.UNARY_SQRT
        return _emul(
            Expression[
                SymExpr.ENum(ComplexF64(0.5)), _epow(_efn(SUnaryKind.UNARY_SQRT, a), -1),
            ],
        )
    elseif kind == SUnaryKind.UNARY_SIN
        return _efn(SUnaryKind.UNARY_COS, a)
    elseif kind == SUnaryKind.UNARY_COS
        return _emul(Expression[-one(Expression), _efn(SUnaryKind.UNARY_SIN, a)])
    elseif kind == SUnaryKind.UNARY_EXP
        return _efn(SUnaryKind.UNARY_EXP, a)
    elseif kind == SUnaryKind.UNARY_LOG
        return _epow(a, -1)
    elseif kind == SUnaryKind.UNARY_TAN
        return _eadd(
            Expression[one(Expression), _epow(_efn(SUnaryKind.UNARY_TAN, a), 2)],
        )
    elseif kind == SUnaryKind.UNARY_ASIN
        return _epow(_efn(SUnaryKind.UNARY_SQRT, _one_minus_square(a)), -1)
    elseif kind == SUnaryKind.UNARY_ACOS
        return _emul(
            Expression[
                -one(Expression),
                _epow(_efn(SUnaryKind.UNARY_SQRT, _one_minus_square(a)), -1),
            ],
        )
    elseif kind == SUnaryKind.UNARY_SINH
        return _efn(SUnaryKind.UNARY_COSH, a)
    elseif kind == SUnaryKind.UNARY_COSH
        return _efn(SUnaryKind.UNARY_SINH, a)
    end
    return _eadd(
        Expression[
            one(Expression),
            _emul(Expression[-one(Expression), _epow(_efn(SUnaryKind.UNARY_TANH, a), 2)]),
        ],
    )
end

_one_minus_square(a::Expression)::Expression =
    _eadd(Expression[one(Expression), _emul(Expression[-one(Expression), _epow(a, 2)])])

"""
    differentiate(f::Expression, v::Expression) -> Expression
    differentiate(f::Expression, v::AbstractVector{Expression}) -> Vector{Expression}
    differentiate(f::AbstractVector{Expression}, v::AbstractVector{Expression}) -> Matrix{Expression}

Symbolic derivative of `f` with respect to the variable(s) `v`.
"""
differentiate(f::Expression, v::Expression)::Expression = _differentiate(f, Symbol(v))

differentiate(f::Expression, v::AbstractVector{Expression})::Vector{Expression} =
    Expression[_differentiate(f, Symbol(vi)) for vi in v]

function differentiate(
        f::AbstractVector{Expression}, v::Expression,
    )::Vector{Expression}
    return Expression[_differentiate(fi, Symbol(v)) for fi in f]
end

function differentiate(
        f::AbstractVector{Expression}, v::AbstractVector{Expression},
    )::Matrix{Expression}
    return Expression[_differentiate(f[i], Symbol(v[j])) for i in eachindex(f), j in eachindex(v)]
end

## ── Substitution ────────────────────────────────────────────────────────────

function _subs(e::Expression, map::Dict{Symbol, Expression})::Expression
    return @match e begin
        SymExpr.ENum(_) => e
        SymExpr.EVar(name) => get(map, name, e)
        SymExpr.EAdd(args) =>
            _eadd(Expression[_subs(a, map) for a in args::Vector{Expression}])
        SymExpr.EMul(args) =>
            _emul(Expression[_subs(a, map) for a in args::Vector{Expression}])
        SymExpr.EPow(base, exp) => _epow(_subs(base::Expression, map), exp)
        SymExpr.ERPow(base, exp) => _erpow(_subs(base::Expression, map), exp)
        SymExpr.EFn(kind, arg) => _efn(kind, _subs(arg::Expression, map))
    end
end

function _substitution_map(pairs)::Dict{Symbol, Expression}
    map = Dict{Symbol, Expression}()
    for pair in pairs
        lhs = first(pair)
        rhs = last(pair)
        if lhs isa Expression
            map[Symbol(lhs)] = convert(Expression, rhs)
        else
            length(lhs) == length(rhs) ||
                throw(ArgumentError("substitution lhs and rhs must have equal length"))
            for (l, r) in zip(lhs, rhs)
                map[Symbol(l)] = convert(Expression, r)
            end
        end
    end
    return map
end

"""
    subs(f, pairs...)

Substitute variables in `f`. Each pair maps either a single variable or a vector
of variables to replacement expressions or numbers.

A dictionary of the same pairs is accepted in place of the pair arguments.

```julia
@var x y
subs(x^2 + y, x => y + 1)
subs([x / y], [x, y] => [1, 2])
subs(x^2 + y, Dict(x => 2, y => 3))
```
"""
subs(f::Expression, pairs::Pair...)::Expression = _subs(f, _substitution_map(pairs))

function subs(f::AbstractArray{Expression}, pairs::Pair...)
    map = _substitution_map(pairs)
    return Base.map(fi -> _subs(fi, map), f)
end

subs(f::Expression, pairs::AbstractDict)::Expression = _subs(f, _substitution_map(pairs))

function subs(f::AbstractArray{Expression}, pairs::AbstractDict)
    map = _substitution_map(pairs)
    return Base.map(fi -> _subs(fi, map), f)
end

## ── Numerator and denominator ───────────────────────────────────────────────

# Numeric coefficient and the powers of a canonical denominator. Denominators
# built here are products of powers, so every factor splits into `base^k`.
function _den_powers(d::Expression)::Tuple{ComplexF64, Dict{Expression, Int}}
    coeff = one(ComplexF64)
    powers = Dict{Expression, Int}()
    factors = @match d begin
        SymExpr.EMul(args) => args::Vector{Expression}
        _ => Expression[d]
    end
    for f in factors
        v = expr_number(f)
        if v.found
            coeff *= v.val
            continue
        end
        (base, k) = _split_power(f)
        powers[base] = get(powers, base, 0) + k
    end
    return coeff, powers
end

# Put a sum over the common denominator built from each base's highest power.
function _num_den_add(args::Vector{Expression})::Tuple{Expression, Expression}
    nums = Vector{Expression}(undef, length(args))
    dens = Vector{Dict{Expression, Int}}(undef, length(args))
    max_powers = Dict{Expression, Int}()
    for (i, a) in enumerate(args)
        (p, q) = num_den(a)
        (c, powers) = _den_powers(q)
        nums[i] = isone(c) ? p : _emul(Expression[SymExpr.ENum(inv(c)), p])
        dens[i] = powers
        for (base, k) in powers
            max_powers[base] = max(get(max_powers, base, 0), k)
        end
    end
    isempty(max_powers) && return _eadd(nums), one(Expression)

    for i in eachindex(nums)
        factors = Expression[nums[i]]
        for (base, k) in max_powers
            missing_power = k - get(dens[i], base, 0)
            missing_power > 0 && push!(factors, _epow(base, missing_power))
        end
        nums[i] = _emul(factors)
    end
    return _eadd(nums), _emul(Expression[_epow(b, k) for (b, k) in max_powers])
end

"""
    num_den(f::Expression) -> (num, den)

Numerator and denominator of `f`, such that `f` equals `num / den`.

Denominator bases are compared structurally and never factored, so `(x - 1)^2`
and `x^2 - 2x + 1` count as different bases: `num_den((x - 1)^2 / (x^2 - 2x + 1))`
returns `x^2 - 2x + 1` as its denominator.

```julia
@var x y
num_den(x / (y - 1) + y)   # (x + y*(-1 + y), -1 + y)
```
"""
function num_den(f::Expression)::Tuple{Expression, Expression}
    return @match f begin
        SymExpr.EAdd(args) => _num_den_add(args::Vector{Expression})
        SymExpr.EMul(args) => begin
            num = one(Expression)
            den = one(Expression)
            for a in args::Vector{Expression}
                (p, q) = num_den(a)
                num = _emul(Expression[num, p])
                den = _emul(Expression[den, q])
            end
            (num, den)
        end
        SymExpr.EPow(base, exp) => begin
            (p, q) = num_den(base::Expression)
            exp > 0 ? (_epow(p, exp), _epow(q, exp)) : (_epow(q, -exp), _epow(p, -exp))
        end
        # Literals, variables, a non-integer power and the unary functions have no
        # rational normal form, so they are their own numerator.
        _ => (f, one(Expression))
    end
end

## ── Degrees and polynomiality ───────────────────────────────────────────────

# Structural degree bound under `weights`, which gives the degree of each
# variable and leaves every unlisted symbol at 0. `_NOT_POLYNOMIAL` marks an
# expression that is not polynomial in those variables, or that uses one of
# unknown (negative) weight; real bounds are non-negative.
const _NOT_POLYNOMIAL = (-1, -1)

function _degree_bounds(
        e::Expression, weights::Dict{Symbol, Int},
    )::Tuple{Int, Int}
    return @match e begin
        SymExpr.ENum(_) => (0, 0)
        SymExpr.EVar(name) => begin
            w = get(weights, name, 0)
            w < 0 ? _NOT_POLYNOMIAL : (w, w)
        end
        SymExpr.EAdd(args) => begin
            lo = typemax(Int)
            hi = 0
            for a in args::Vector{Expression}
                bounds = _degree_bounds(a, weights)
                bounds == _NOT_POLYNOMIAL && return _NOT_POLYNOMIAL
                lo = min(lo, bounds[1])
                hi = max(hi, bounds[2])
            end
            (lo, hi)
        end
        SymExpr.EMul(args) => begin
            lo = 0
            hi = 0
            for a in args::Vector{Expression}
                bounds = _degree_bounds(a, weights)
                bounds == _NOT_POLYNOMIAL && return _NOT_POLYNOMIAL
                lo += bounds[1]
                hi += bounds[2]
            end
            (lo, hi)
        end
        SymExpr.EPow(base, exp) => begin
            bounds = _degree_bounds(base::Expression, weights)
            bounds == _NOT_POLYNOMIAL && return _NOT_POLYNOMIAL
            bounds == (0, 0) && return (0, 0)
            exp < 0 && return _NOT_POLYNOMIAL
            (exp * bounds[1], exp * bounds[2])
        end
        SymExpr.ERPow(base, _) => begin
            bounds = _degree_bounds(base::Expression, weights)
            bounds == _NOT_POLYNOMIAL && return _NOT_POLYNOMIAL
            bounds == (0, 0) ? (0, 0) : _NOT_POLYNOMIAL
        end
        SymExpr.EFn(_, arg) => begin
            bounds = _degree_bounds(arg::Expression, weights)
            bounds == _NOT_POLYNOMIAL && return _NOT_POLYNOMIAL
            bounds == (0, 0) ? (0, 0) : _NOT_POLYNOMIAL
        end
    end
end

"""
    is_polynomial(f::Expression, vars) -> Bool

Whether `f` is a polynomial in `vars`.
"""
is_polynomial(f::Expression, vars::AbstractVector{Expression})::Bool =
    _degree_bounds(f, _unit_weights(vars)) != _NOT_POLYNOMIAL

_unit_weights(vars::AbstractVector{Expression})::Dict{Symbol, Int} =
    Dict{Symbol, Int}(Symbol(v) => 1 for v in vars)

"""
    degree(f::Expression, vars) -> Int

Upper bound on the total degree of `f` in `vars`. Returns `-1` when `f` is not
polynomial in `vars`.
"""
function degree(f::Expression, vars::AbstractVector{Expression})::Int
    bounds = _degree_bounds(f, _unit_weights(vars))
    bounds == _NOT_POLYNOMIAL && return -1
    return bounds[2]
end

_expression_degrees(
    exprs::AbstractVector{Expression}, vars::AbstractVector{Expression},
)::Tuple{Vector{Int}, Bool} = _weighted_degrees(exprs, _unit_weights(vars))

# Per-expression degree bound and homogeneity under `weights`, in one pass.
function _weighted_degrees(
        exprs::AbstractVector{Expression}, weights::Dict{Symbol, Int},
    )::Tuple{Vector{Int}, Bool}
    degs = Vector{Int}(undef, length(exprs))
    homogeneous = true
    for i in eachindex(exprs)
        bounds = _degree_bounds(exprs[i], weights)
        if bounds == _NOT_POLYNOMIAL
            degs[i] = -1
            homogeneous = false
        else
            degs[i] = bounds[2]
            homogeneous &= bounds[1] == bounds[2]
        end
    end
    return degs, homogeneous
end

"""
    has_real_coefficients(f::Expression) -> Bool

Whether every numeric literal in `f` is real.
"""
function has_real_coefficients(e::Expression)::Bool
    return @match e begin
        SymExpr.ENum(val) => iszero(imag(val))
        SymExpr.EVar(_) => true
        SymExpr.EAdd(args) => all(has_real_coefficients, args::Vector{Expression})
        SymExpr.EMul(args) => all(has_real_coefficients, args::Vector{Expression})
        SymExpr.EPow(base, _) => has_real_coefficients(base::Expression)
        SymExpr.ERPow(base, exp) =>
            isreal(exp) && has_real_coefficients(base::Expression)
        SymExpr.EFn(_, arg) => has_real_coefficients(arg::Expression)
    end
end

"""
    expression_scale(f::Expression) -> Float64

Magnitude of `f` with every variable set to `1` and every numeric literal replaced by
its absolute value. This bounds the ℓ1 norm of the coefficients of a polynomial from
above, exactly when nothing cancels: `(x - 1)^10` scales as `1024` against a largest
coefficient of `252`. For a rational function it is the ratio of the two bounds. `sin`
and `cos` contribute `1`, being bounded by it.
"""
function expression_scale(e::Expression)::Float64
    return @match e begin
        SymExpr.ENum(val) => abs(val)
        SymExpr.EVar(_) => 1.0
        SymExpr.EAdd(args) => begin
            s = 0.0
            for a in args::Vector{Expression}
                s += expression_scale(a)
            end
            s
        end
        SymExpr.EMul(args) => begin
            s = 1.0
            for a in args::Vector{Expression}
                s *= expression_scale(a)
            end
            s
        end
        SymExpr.EPow(base, exp) => expression_scale(base::Expression)^exp
        SymExpr.ERPow(base, exp) => expression_scale(base::Expression)^real(exp)
        SymExpr.EFn(kind, arg) =>
            kind == SUnaryKind.UNARY_SQRT ? sqrt(expression_scale(arg::Expression)) : 1.0
    end
end

## ── Display ─────────────────────────────────────────────────────────────────

function _show_number(io::IO, val::ComplexF64)::Nothing
    if iszero(imag(val))
        r = real(val)
        # Only integral values below `maxintfloat` are `Int`-representable.
        print(io, (isinteger(r) && abs(r) ≤ maxintfloat(Float64)) ? Int(r) : r)
    else
        print(io, "(", val, ")")
    end
    return nothing
end

function _show_expr(io::IO, e::Expression, prec::Int)::Nothing
    @match e begin
        SymExpr.ENum(val) => _show_number(io, val)
        SymExpr.EVar(name) => print(io, name)
        SymExpr.EAdd(args) => begin
            prec > 1 && print(io, "(")
            for (i, a) in enumerate(args::Vector{Expression})
                i > 1 && print(io, " + ")
                _show_expr(io, a, 1)
            end
            prec > 1 && print(io, ")")
        end
        SymExpr.EMul(args) => begin
            prec > 2 && print(io, "(")
            for (i, a) in enumerate(args::Vector{Expression})
                i > 1 && print(io, "*")
                _show_expr(io, a, 2)
            end
            prec > 2 && print(io, ")")
        end
        SymExpr.EPow(base, exp) => begin
            _show_expr(io, base::Expression, 3)
            print(io, "^", exp)
        end
        SymExpr.ERPow(base, exp) => begin
            _show_expr(io, base::Expression, 3)
            print(io, "^")
            _show_number(io, exp)
        end
        SymExpr.EFn(kind, arg) => begin
            print(io, _unary_name(kind), "(")
            _show_expr(io, arg::Expression, 0)
            print(io, ")")
        end
    end
    return nothing
end

Base.show(io::IO, e::Expression) = _show_expr(io, e, 0)
Base.show(io::IO, ::Type{Expression}) = print(io, "Expression")

## ── Lowering to SExpr ───────────────────────────────────────────────────────

"""
    expression_to_sexpr(e, var_to_idx, param_to_idx) -> SExprT

Lower an `Expression` to the tape compiler's `SExpr` IR. Every free symbol must
appear in `var_to_idx` or `param_to_idx`.
"""
function expression_to_sexpr(
        e::Expression,
        var_to_idx::Dict{Symbol, Int},
        param_to_idx::Dict{Symbol, Int},
    )::SExprT
    return @match e begin
        SymExpr.ENum(val) => SExpr.SConst(val)
        SymExpr.EVar(name) => begin
            idx = get(var_to_idx, name, 0)
            idx > 0 && return SExpr.SVar(idx)
            idx = get(param_to_idx, name, 0)
            idx > 0 && return SExpr.SParam(idx)
            throw(_unknown_symbol_error(name))
        end
        SymExpr.EAdd(args) => _canonical_add(
            SExprT[
                expression_to_sexpr(a, var_to_idx, param_to_idx)
                    for a in args::Vector{Expression}
            ],
        )
        SymExpr.EMul(args) => _canonical_mul(
            SExprT[
                expression_to_sexpr(a, var_to_idx, param_to_idx)
                    for a in args::Vector{Expression}
            ],
        )
        SymExpr.EPow(base, exp) => SExpr.SPow(
            expression_to_sexpr(base::Expression, var_to_idx, param_to_idx), exp,
        )
        SymExpr.ERPow(base, exp) => _canonical_rpow(
            expression_to_sexpr(base::Expression, var_to_idx, param_to_idx), exp,
        )
        SymExpr.EFn(kind, arg) => _canonical_unary(
            kind, expression_to_sexpr(arg::Expression, var_to_idx, param_to_idx),
        )
    end
end

## ── MultivariatePolynomials interoperability ────────────────────────────────

function Expression(poly::MP.AbstractPolynomialLike)::Expression
    Base.@nospecialize poly
    terms = Expression[]
    for term in MP.terms(poly)
        coeff = ComplexF64(MP.coefficient(term))
        iszero(coeff) && continue
        factors = Expression[SymExpr.ENum(coeff)]
        mono = MP.monomial(term)
        for (var, exp) in zip(MP.variables(mono), MP.exponents(mono))
            exp == 0 && continue
            push!(factors, _epow(variable(Symbol(var)), exp))
        end
        push!(terms, _emul(factors))
    end
    return _eadd(terms)
end

Expression(v::MP.AbstractVariable)::Expression = variable(Symbol(v))

function Expression(r::MP.RationalPoly)::Expression
    Base.@nospecialize r
    return Expression(numerator(r)) / Expression(denominator(r))
end

Base.convert(::Type{Expression}, p::MP.AbstractPolynomialLike)::Expression = Expression(p)
Base.convert(::Type{Expression}, r::MP.RationalPoly)::Expression = Expression(r)
