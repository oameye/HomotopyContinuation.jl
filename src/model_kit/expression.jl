## Expression: symbolic frontend for non-polynomial input.
#
# `Expression` is a `Number` subtype, so ordinary Julia arithmetic, `sum`,
# broadcasting and matrix products build expression trees. It covers everything
# the tape can execute that `MultivariatePolynomials` cannot express: division,
# negative integer powers, `sqrt`, `sin` and `cos`.
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

    """Unary function application: sqrt, sin or cos."""
    struct EFn
        kind::SUnaryKind.T
        arg::SymExpr
    end
end

@derive SymExpr[Eq]

const Expression = typeof(SymExpr.ENum(zero(ComplexF64)))
const ENumStorage = variant_storage_type(SymExpr.ENum)
const EVarStorage = variant_storage_type(SymExpr.EVar)
const EAddStorage = variant_storage_type(SymExpr.EAdd)
const EMulStorage = variant_storage_type(SymExpr.EMul)
const EPowStorage = variant_storage_type(SymExpr.EPow)
const EFnStorage = variant_storage_type(SymExpr.EFn)

@inline expr_storage(e::Expression) = variant_storage(e)

@inline _owned_expr_args(args::AbstractVector{<:Expression})::Vector{Expression} =
    collect(Expression, args)

# Compound nodes are hashed structurally and used as Dict keys, so they must own
# their child storage rather than alias a caller vector.
@inline SymExpr.EAdd(args::AbstractVector{<:Expression}) =
    invoke(SymExpr.EAdd, Tuple{Any}, _owned_expr_args(args))
@inline SymExpr.EMul(args::AbstractVector{<:Expression}) =
    invoke(SymExpr.EMul, Tuple{Any}, _owned_expr_args(args))

## ── Hashing and ordering ────────────────────────────────────────────────────

function _fold_expr_hash(seed, args)::UInt
    h = hash(seed, zero(UInt))
    for a in args
        h = hash(a, h)
    end
    return h
end

function Base.hash(e::Expression, h::UInt)::UInt
    storage = expr_storage(e)
    if storage isa ENumStorage
        return hash(storage.val, hash(:ENum, h))
    elseif storage isa EVarStorage
        return hash(storage.name, hash(:EVar, h))
    elseif storage isa EAddStorage
        return hash(_fold_expr_hash(:EAdd, storage.args), h)
    elseif storage isa EMulStorage
        return hash(_fold_expr_hash(:EMul, storage.args), h)
    elseif storage isa EPowStorage
        return hash(storage.exp, hash(storage.base, hash(:EPow, h)))
    else # EFnStorage
        return hash(storage.arg, hash(storage.kind, hash(:EFn, h)))
    end
end

_expr_lt(a::Expression, b::Expression)::Bool = hash(a) < hash(b)

## ── Predicates and accessors ────────────────────────────────────────────────

"""Numeric value of `e`, or `nothing` when `e` is not a literal."""
@inline function expr_number(e::Expression)::Union{Nothing, ComplexF64}
    storage = expr_storage(e)
    storage isa ENumStorage && return storage.val
    return nothing
end

is_number(e::Expression)::Bool = expr_storage(e) isa ENumStorage
is_variable(e::Expression)::Bool = expr_storage(e) isa EVarStorage

function Base.Symbol(e::Expression)::Symbol
    storage = expr_storage(e)
    storage isa EVarStorage ||
        throw(ArgumentError("expression $(e) is not a variable"))
    return storage.name
end

Base.iszero(e::Expression)::Bool =
    (s = expr_storage(e); s isa ENumStorage && iszero(s.val))
Base.isone(e::Expression)::Bool =
    (s = expr_storage(e); s isa ENumStorage && isone(s.val))

Base.zero(::Type{Expression})::Expression = SymExpr.ENum(zero(ComplexF64))
Base.one(::Type{Expression})::Expression = SymExpr.ENum(one(ComplexF64))
Base.zero(::Expression)::Expression = zero(Expression)
Base.one(::Expression)::Expression = one(Expression)

# Numeric literals are conjugated; variables stand for real symbols, since the
# tape has no conjugation instruction.
function Base.conj(e::Expression)::Expression
    storage = expr_storage(e)
    if storage isa ENumStorage
        return SymExpr.ENum(conj(storage.val))
    elseif storage isa EVarStorage
        return e
    elseif storage isa EAddStorage
        return _eadd(Expression[conj(a) for a in storage.args])
    elseif storage isa EMulStorage
        return _emul(Expression[conj(a) for a in storage.args])
    elseif storage isa EPowStorage
        return _epow(conj(storage.base), storage.exp)
    else # EFnStorage
        return _efn(storage.kind, conj(storage.arg))
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

function _flatten_eadd!(
        terms::Vector{Expression}, const_sum::Base.RefValue{ComplexF64}, e::Expression,
    )::Nothing
    storage = expr_storage(e)
    if storage isa EAddStorage
        for child in storage.args
            _flatten_eadd!(terms, const_sum, child)
        end
    elseif storage isa ENumStorage
        const_sum[] += storage.val
    else
        push!(terms, e)
    end
    return nothing
end

"""Split a product into its leading numeric coefficient and the rest."""
function _split_coefficient(e::Expression)::Tuple{ComplexF64, Expression}
    storage = expr_storage(e)
    if storage isa EMulStorage
        args = storage.args
        head = expr_storage(args[1])
        if head isa ENumStorage
            rest = args[2:end]
            return head.val, (length(rest) == 1 ? rest[1] : SymExpr.EMul(rest))
        end
    end
    return one(ComplexF64), e
end

function _eadd(args::Vector{Expression})::Expression
    terms = Expression[]
    const_sum = Ref(zero(ComplexF64))
    for a in args
        _flatten_eadd!(terms, const_sum, a)
    end

    # Collect like terms: coefficients of structurally identical bases add up.
    bases = Expression[]
    coeffs = ComplexF64[]
    index = Dict{Expression, Int}()
    for t in terms
        (c, base) = _split_coefficient(t)
        k = get(index, base, 0)
        if k == 0
            push!(bases, base)
            push!(coeffs, c)
            index[base] = length(bases)
        else
            coeffs[k] += c
        end
    end

    collected = Expression[]
    nested = false
    for (base, c) in zip(bases, coeffs)
        iszero(c) && continue
        if isone(c)
            # `base` can be a sum (`2 * (x + y)` splits to `x + y`), which has to
            # be re-flattened rather than nested.
            nested |= expr_storage(base) isa EAddStorage
            push!(collected, base)
        else
            push!(collected, _emul(Expression[SymExpr.ENum(c), base]))
        end
    end
    if nested
        !iszero(const_sum[]) && push!(collected, SymExpr.ENum(const_sum[]))
        return _eadd(collected)
    end

    sort!(collected; lt = _expr_lt)
    !iszero(const_sum[]) && pushfirst!(collected, SymExpr.ENum(const_sum[]))
    isempty(collected) && return zero(Expression)
    length(collected) == 1 && return collected[1]
    return SymExpr.EAdd(collected)
end

function _flatten_emul!(
        factors::Vector{Expression}, coeff::Base.RefValue{ComplexF64}, e::Expression,
    )::Nothing
    storage = expr_storage(e)
    if storage isa EMulStorage
        for child in storage.args
            _flatten_emul!(factors, coeff, child)
        end
    elseif storage isa ENumStorage
        coeff[] *= storage.val
    else
        push!(factors, e)
    end
    return nothing
end

"""Split a factor into its base and integer exponent."""
@inline function _split_power(e::Expression)::Tuple{Expression, Int}
    storage = expr_storage(e)
    storage isa EPowStorage && return (storage.base, storage.exp)
    return (e, 1)
end

function _emul(args::Vector{Expression})::Expression
    factors = Expression[]
    coeff = Ref(one(ComplexF64))
    for a in args
        _flatten_emul!(factors, coeff, a)
    end
    iszero(coeff[]) && return zero(Expression)

    # Collect powers: repeated bases have their exponents added.
    bases = Expression[]
    exps = Int[]
    index = Dict{Expression, Int}()
    for f in factors
        (base, k) = _split_power(f)
        j = get(index, base, 0)
        if j == 0
            push!(bases, base)
            push!(exps, k)
            index[base] = length(bases)
        else
            exps[j] += k
        end
    end

    collected = Expression[]
    for (base, k) in zip(bases, exps)
        k == 0 && continue
        push!(collected, _epow(base, k))
    end

    # `_epow` can fold a base to a literal (e.g. (1/2)^-1), so re-absorb them.
    kept = Expression[]
    for c in collected
        v = expr_number(c)
        if v === nothing
            push!(kept, c)
        else
            coeff[] *= v
        end
    end
    iszero(coeff[]) && return zero(Expression)

    sort!(kept; lt = _expr_lt)
    isone(coeff[]) || pushfirst!(kept, SymExpr.ENum(coeff[]))
    isempty(kept) && return one(Expression)
    length(kept) == 1 && return kept[1]
    return SymExpr.EMul(kept)
end

function _epow(base::Expression, k::Int)::Expression
    k == 0 && return one(Expression)
    k == 1 && return base
    storage = expr_storage(base)
    if storage isa ENumStorage
        return SymExpr.ENum(op_pow_int(storage.val, k))
    elseif storage isa EPowStorage
        return _epow(storage.base, storage.exp * k)
    elseif storage isa EMulStorage
        # (a*b)^k = a^k * b^k keeps powers next to their base for CSE.
        return _emul(Expression[_epow(a, k) for a in storage.args])
    end
    return SymExpr.EPow(base, k)
end

function _efn(kind::SUnaryKind.T, arg::Expression)::Expression
    v = expr_number(arg)
    v === nothing || return SymExpr.ENum(apply_unary(kind, v))
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

# The tape has no general power instruction.
function Base.:^(a::Expression, b::Expression)::Expression
    v = expr_number(b)
    if v !== nothing && iszero(imag(v)) && isinteger(real(v))
        return _epow(a, Int(real(v)))
    end
    throw(ArgumentError("only integer exponents are supported, got $(b)"))
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

function _build_var(var, unique::Bool)
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
    storage = expr_storage(e)
    if storage isa EVarStorage
        if !(storage.name in seen)
            push!(seen, storage.name)
            push!(acc, e)
        end
    elseif storage isa EAddStorage
        for a in storage.args
            _collect_expr_variables!(acc, seen, a)
        end
    elseif storage isa EMulStorage
        for a in storage.args
            _collect_expr_variables!(acc, seen, a)
        end
    elseif storage isa EPowStorage
        _collect_expr_variables!(acc, seen, storage.base)
    elseif storage isa EFnStorage
        _collect_expr_variables!(acc, seen, storage.arg)
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
    storage = expr_storage(e)
    if storage isa ENumStorage
        return zero(Expression)
    elseif storage isa EVarStorage
        return storage.name === v ? one(Expression) : zero(Expression)
    elseif storage isa EAddStorage
        return _eadd(Expression[_differentiate(a, v) for a in storage.args])
    elseif storage isa EMulStorage
        args = storage.args
        terms = Expression[]
        for i in eachindex(args)
            da = _differentiate(args[i], v)
            iszero(da) && continue
            factors = Expression[da]
            for j in eachindex(args)
                j == i && continue
                push!(factors, args[j])
            end
            push!(terms, _emul(factors))
        end
        return _eadd(terms)
    elseif storage isa EPowStorage
        db = _differentiate(storage.base, v)
        iszero(db) && return zero(Expression)
        k = storage.exp
        return _emul(
            Expression[SymExpr.ENum(ComplexF64(k)), _epow(storage.base, k - 1), db],
        )
    else # EFnStorage
        da = _differentiate(storage.arg, v)
        iszero(da) && return zero(Expression)
        kind = storage.kind
        if kind == SUnaryKind.UNARY_SQRT
            # d/dx sqrt(a) = a' / (2 sqrt(a))
            return _emul(
                Expression[
                    SymExpr.ENum(ComplexF64(0.5)),
                    _epow(_efn(SUnaryKind.UNARY_SQRT, storage.arg), -1),
                    da,
                ],
            )
        elseif kind == SUnaryKind.UNARY_SIN
            return _emul(Expression[_efn(SUnaryKind.UNARY_COS, storage.arg), da])
        else
            return _emul(
                Expression[
                    SymExpr.ENum(-one(ComplexF64)),
                    _efn(SUnaryKind.UNARY_SIN, storage.arg),
                    da,
                ],
            )
        end
    end
end

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
    storage = expr_storage(e)
    if storage isa ENumStorage
        return e
    elseif storage isa EVarStorage
        return get(map, storage.name, e)
    elseif storage isa EAddStorage
        return _eadd(Expression[_subs(a, map) for a in storage.args])
    elseif storage isa EMulStorage
        return _emul(Expression[_subs(a, map) for a in storage.args])
    elseif storage isa EPowStorage
        return _epow(_subs(storage.base, map), storage.exp)
    else # EFnStorage
        return _efn(storage.kind, _subs(storage.arg, map))
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

## ── Degrees and polynomiality ───────────────────────────────────────────────

# Structural degree bound in `vars`. `nothing` marks an expression that is not
# polynomial in `vars` (a negative power or a unary function of a variable).
function _degree_bounds(e::Expression, vars::Set{Symbol})::Union{Nothing, Tuple{Int, Int}}
    storage = expr_storage(e)
    if storage isa ENumStorage
        return (0, 0)
    elseif storage isa EVarStorage
        return storage.name in vars ? (1, 1) : (0, 0)
    elseif storage isa EAddStorage
        lo = typemax(Int)
        hi = 0
        for a in storage.args
            bounds = _degree_bounds(a, vars)
            bounds === nothing && return nothing
            lo = min(lo, bounds[1])
            hi = max(hi, bounds[2])
        end
        return (lo, hi)
    elseif storage isa EMulStorage
        lo = 0
        hi = 0
        for a in storage.args
            bounds = _degree_bounds(a, vars)
            bounds === nothing && return nothing
            lo += bounds[1]
            hi += bounds[2]
        end
        return (lo, hi)
    elseif storage isa EPowStorage
        bounds = _degree_bounds(storage.base, vars)
        bounds === nothing && return nothing
        bounds == (0, 0) && return (0, 0)
        storage.exp < 0 && return nothing
        return (storage.exp * bounds[1], storage.exp * bounds[2])
    else # EFnStorage
        bounds = _degree_bounds(storage.arg, vars)
        bounds === nothing && return nothing
        return bounds == (0, 0) ? (0, 0) : nothing
    end
end

"""
    is_polynomial(f::Expression, vars) -> Bool

Whether `f` is a polynomial in `vars`.
"""
is_polynomial(f::Expression, vars::AbstractVector{Expression})::Bool =
    _degree_bounds(f, Set{Symbol}(Symbol(v) for v in vars)) !== nothing

"""
    degree(f::Expression, vars) -> Int

Upper bound on the total degree of `f` in `vars`. Returns `-1` when `f` is not
polynomial in `vars`.
"""
function degree(f::Expression, vars::AbstractVector{Expression})::Int
    bounds = _degree_bounds(f, Set{Symbol}(Symbol(v) for v in vars))
    bounds === nothing && return -1
    return bounds[2]
end

function _expression_degrees(
        exprs::AbstractVector{Expression}, vars::AbstractVector{Expression},
    )::Tuple{Vector{Int}, Bool}
    var_set = Set{Symbol}(Symbol(v) for v in vars)
    degs = Vector{Int}(undef, length(exprs))
    homogeneous = true
    for i in eachindex(exprs)
        bounds = _degree_bounds(exprs[i], var_set)
        if bounds === nothing
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
    storage = expr_storage(e)
    if storage isa ENumStorage
        return iszero(imag(storage.val))
    elseif storage isa EVarStorage
        return true
    elseif storage isa EAddStorage
        for a in storage.args
            has_real_coefficients(a) || return false
        end
        return true
    elseif storage isa EMulStorage
        for a in storage.args
            has_real_coefficients(a) || return false
        end
        return true
    elseif storage isa EPowStorage
        return has_real_coefficients(storage.base)
    else # EFnStorage
        return has_real_coefficients(storage.arg)
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
    storage = expr_storage(e)
    if storage isa ENumStorage
        _show_number(io, storage.val)
    elseif storage isa EVarStorage
        print(io, storage.name)
    elseif storage isa EAddStorage
        prec > 1 && print(io, "(")
        for (i, a) in enumerate(storage.args)
            i > 1 && print(io, " + ")
            _show_expr(io, a, 1)
        end
        prec > 1 && print(io, ")")
    elseif storage isa EMulStorage
        prec > 2 && print(io, "(")
        for (i, a) in enumerate(storage.args)
            i > 1 && print(io, "*")
            _show_expr(io, a, 2)
        end
        prec > 2 && print(io, ")")
    elseif storage isa EPowStorage
        _show_expr(io, storage.base, 3)
        print(io, "^", storage.exp)
    else # EFnStorage
        kind = storage.kind
        name = kind == SUnaryKind.UNARY_SQRT ? "sqrt" :
            kind == SUnaryKind.UNARY_SIN ? "sin" : "cos"
        print(io, name, "(")
        _show_expr(io, storage.arg, 0)
        print(io, ")")
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
    storage = expr_storage(e)
    if storage isa ENumStorage
        return SExpr.SConst(storage.val)
    elseif storage isa EVarStorage
        idx = get(var_to_idx, storage.name, 0)
        idx > 0 && return SExpr.SVar(idx)
        idx = get(param_to_idx, storage.name, 0)
        idx > 0 && return SExpr.SParam(idx)
        throw(
            ArgumentError(
                "symbol $(storage.name) is neither a variable nor a parameter of the system",
            ),
        )
    elseif storage isa EAddStorage
        return _canonical_add(
            SExprT[expression_to_sexpr(a, var_to_idx, param_to_idx) for a in storage.args],
        )
    elseif storage isa EMulStorage
        return _canonical_mul(
            SExprT[expression_to_sexpr(a, var_to_idx, param_to_idx) for a in storage.args],
        )
    elseif storage isa EPowStorage
        return SExpr.SPow(
            expression_to_sexpr(storage.base, var_to_idx, param_to_idx), storage.exp,
        )
    else # EFnStorage
        return _canonical_unary(
            storage.kind, expression_to_sexpr(storage.arg, var_to_idx, param_to_idx),
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
