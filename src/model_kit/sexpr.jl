## SExpr types, hash/==, canonicalization, and poly_to_sexpr
#
# SExpr is an S-expression IR used as an intermediate representation between
# the MultivariatePolynomials input and the CSE algorithm. It mirrors the
# canonical forms produced by SymEngine's Add/Mul/Pow/FunctionSymbol types.

## ── SExpr types ─────────────────────────────────────────────────────────────

abstract type SExpr end

"""Constant value."""
struct SConst <: SExpr
    val::ComplexF64
end

"""Variable reference (1-based index)."""
struct SVar <: SExpr
    idx::Int
end

"""Parameter reference (1-based index)."""
struct SParam <: SExpr
    idx::Int
end

"""CSE temporary (assigned by tree_cse)."""
struct STmp <: SExpr
    id::Int
end

_collect_args(args::AbstractVector) = collect(SExpr, args)

function _fold_hash(seed, args)::UInt
    h = hash(seed, zero(UInt))
    for a in args
        h = hash(a, h)
    end
    return h
end

"""Addition: sum of args (n-ary, n >= 2). Hash is cached at construction."""
struct SAdd <: SExpr
    args::Vector{SExpr}
    _hash::UInt
end
function SAdd(args::AbstractVector)
    vargs = _collect_args(args)
    return SAdd(vargs, _fold_hash(:SAdd, vargs))
end

"""Multiplication: product of args (n-ary, n >= 2). Hash is cached at construction."""
struct SMul <: SExpr
    args::Vector{SExpr}
    _hash::UInt
end
function SMul(args::AbstractVector)
    vargs = _collect_args(args)
    return SMul(vargs, _fold_hash(:SMul, vargs))
end

"""Integer power: base^exp where exp is a positive integer. Hash is cached at construction."""
struct SPow <: SExpr
    base::SExpr
    exp::Int
    _hash::UInt
end
function SPow(base::SExpr, exp::Int)
    return SPow(base, exp, hash(exp, hash(base, hash(:SPow, zero(UInt)))))
end

"""Negation: -arg. Hash is cached at construction."""
struct SNeg <: SExpr
    arg::SExpr
    _hash::UInt
end
function SNeg(arg::SExpr)
    return SNeg(arg, hash(arg, hash(:SNeg, zero(UInt))))
end

"""
Kind of unevaluated function symbol used by opt_cse.
Corresponds to SymEngine's FunctionSymbol name ("add", "mul", "pow").
"""
@enumx SFuncKind::Int8 begin
    SFUNC_ADD
    SFUNC_MUL
    SFUNC_POW
end

"""
Unevaluated function symbol — placeholder created by opt_cse to represent
factored common arguments without triggering canonical-form collapse.
`kind` is SFuncKind.SFUNC_ADD, SFUNC_MUL, or SFUNC_POW.
Corresponds to SymEngine's FunctionSymbol.
"""
struct SFuncSym <: SExpr
    kind::SFuncKind.T
    args::Vector{SExpr}
    _hash::UInt
end
function SFuncSym(kind::SFuncKind.T, args::AbstractVector{<:SExpr})
    vargs = _collect_args(args)
    return SFuncSym(kind, vargs, _fold_hash((:SFuncSym, kind), vargs))
end

## ── Hashing and equality ────────────────────────────────────────────────────

function Base.hash(e::SConst, h::UInt)::UInt
    return hash(e.val, hash(:SConst, h))
end
function Base.hash(e::SVar, h::UInt)::UInt
    return hash(e.idx, hash(:SVar, h))
end
function Base.hash(e::SParam, h::UInt)::UInt
    return hash(e.idx, hash(:SParam, h))
end
function Base.hash(e::STmp, h::UInt)::UInt
    return hash(e.id, hash(:STmp, h))
end
function Base.hash(e::SAdd, h::UInt)::UInt
    return hash(e._hash, h)
end
function Base.hash(e::SMul, h::UInt)::UInt
    return hash(e._hash, h)
end
function Base.hash(e::SPow, h::UInt)::UInt
    return hash(e._hash, h)
end
function Base.hash(e::SNeg, h::UInt)::UInt
    return hash(e._hash, h)
end
function Base.hash(e::SFuncSym, h::UInt)::UInt
    return hash(e._hash, h)
end

Base.:(==)(a::SConst, b::SConst) = a.val == b.val
Base.:(==)(a::SVar, b::SVar) = a.idx == b.idx
Base.:(==)(a::SParam, b::SParam) = a.idx == b.idx
Base.:(==)(a::STmp, b::STmp) = a.id == b.id
Base.:(==)(a::SPow, b::SPow) = a._hash == b._hash && a.exp == b.exp && a.base == b.base
Base.:(==)(a::SNeg, b::SNeg) = a._hash == b._hash && a.arg == b.arg

function _same_args(args_a::Vector{SExpr}, args_b::Vector{SExpr})::Bool
    length(args_a) == length(args_b) || return false
    return all(i -> args_a[i] == args_b[i], eachindex(args_a))
end

Base.:(==)(a::SAdd, b::SAdd) = a._hash == b._hash && _same_args(a.args, b.args)
Base.:(==)(a::SMul, b::SMul) = a._hash == b._hash && _same_args(a.args, b.args)
Base.:(==)(a::SFuncSym, b::SFuncSym) =
    a._hash == b._hash && a.kind == b.kind && _same_args(a.args, b.args)
Base.:(==)(::SExpr, ::SExpr) = false

## ── SExpr helpers ───────────────────────────────────────────────────────────

_is_atom(e::SExpr)::Bool = e isa SConst || e isa SVar || e isa SParam || e isa STmp

"""Get the arguments (children) of a compound expression."""
_get_args(::SExpr)::Vector{SExpr} = SExpr[]
_get_args(e::SAdd)::Vector{SExpr} = e.args
_get_args(e::SMul)::Vector{SExpr} = e.args
_get_args(e::SPow)::Vector{SExpr} = SExpr[e.base]
_get_args(e::SNeg)::Vector{SExpr} = SExpr[e.arg]
_get_args(e::SFuncSym)::Vector{SExpr} = e.args

_rebuild_expr(expr::SExpr, ::Vector{SExpr})::SExpr = expr
_rebuild_expr(::SAdd, args::Vector{SExpr})::SExpr = _canonical_add(args)
_rebuild_expr(::SMul, args::Vector{SExpr})::SExpr = _canonical_mul(args)
_rebuild_expr(expr::SPow, args::Vector{SExpr})::SExpr = SPow(args[1], expr.exp)
_rebuild_expr(::SNeg, args::Vector{SExpr})::SExpr = SNeg(args[1])

function _rebuild_expr(expr::SFuncSym, args::Vector{SExpr})::SExpr
    if expr.kind == SFuncKind.SFUNC_ADD
        return _canonical_add(args)
    elseif expr.kind == SFuncKind.SFUNC_MUL
        return _canonical_mul(args)
    elseif expr.kind == SFuncKind.SFUNC_POW && length(args) == 2 && args[2] isa SConst
        return SPow(args[1], Int(real(args[2].val)))
    end
    return SFuncSym(expr.kind, args)
end

_sexpr_lt(a::SExpr, b::SExpr)::Bool = hash(a) < hash(b)

"""
Extract the "base expression" of an Add term, stripping the leading coefficient.
E.g., Mul(3+i, v1, v2) → the base is Mul(v1, v2) (the product without coef).
A bare variable SVar(1) stays as is. A Pow stays as is.
"""
function _add_term_base(e::SExpr)::SExpr
    if e isa SMul && !isempty(e.args) && e.args[1] isa SConst
        rest = e.args[2:end]
        return length(rest) == 1 ? rest[1] : SMul(rest)
    end
    return e
end

function _flatten_add_arg!(
        flat_args::Vector{SExpr},
        const_sum::Base.RefValue{ComplexF64},
        arg::SExpr,
    )::Nothing
    if arg isa SAdd
        for child in arg.args
            _flatten_add_arg!(flat_args, const_sum, child)
        end
    elseif arg isa SConst
        const_sum[] += arg.val
    else
        push!(flat_args, arg)
    end
    return nothing
end

"""
Canonicalize an Add expression.

1. Flatten nested Adds and collect constants
2. Sort by `_sexpr_lt`
3. Reconstruct: constant first (if non-zero), then sorted terms
"""
function _canonical_add(args::Vector{SExpr})::SExpr
    flat_args = SExpr[]
    const_sum = Ref(zero(ComplexF64))
    for arg in args
        _flatten_add_arg!(flat_args, const_sum, arg)
    end
    sort!(flat_args; lt = _sexpr_lt)
    if !iszero(const_sum[])
        pushfirst!(flat_args, SConst(const_sum[]))
    end
    if isempty(flat_args)
        return SConst(zero(ComplexF64))
    elseif length(flat_args) == 1
        return flat_args[1]
    else
        return SAdd(flat_args)
    end
end

function _flatten_mul_arg!(
        flat_args::Vector{SExpr},
        coeff::Base.RefValue{ComplexF64},
        arg::SExpr,
    )::Nothing
    if arg isa SMul
        for child in arg.args
            _flatten_mul_arg!(flat_args, coeff, child)
        end
    elseif arg isa SConst
        coeff[] *= arg.val
    elseif arg isa SNeg
        coeff[] = -coeff[]
        _flatten_mul_arg!(flat_args, coeff, arg.arg)
    else
        push!(flat_args, arg)
    end
    return nothing
end

function _canonical_mul(args::Vector{SExpr})::SExpr
    flat_args = SExpr[]
    coeff = Ref(one(ComplexF64))
    for arg in args
        _flatten_mul_arg!(flat_args, coeff, arg)
    end

    iszero(coeff[]) && return SConst(zero(ComplexF64))

    sort!(flat_args; lt = _sexpr_lt)
    if coeff[] != one(ComplexF64)
        pushfirst!(flat_args, SConst(coeff[]))
    end

    if isempty(flat_args)
        return SConst(one(ComplexF64))
    elseif length(flat_args) == 1
        return flat_args[1]
    else
        return SMul(flat_args)
    end
end

## ── Polynomial → SExpr conversion ──────────────────────────────────────────

"""
    poly_to_sexpr(poly, var_to_idx, param_to_idx) -> SExpr

Convert a MultivariatePolynomials polynomial to an SExpr tree.
Produces the same structure as SymEngine's canonical forms:
- Each term becomes SMul([coeff, factors...]) matching Mul.get_args()
- Coefficient -1 uses SNeg (matching SymEngine's neg() simplification)
- The sum becomes SAdd([terms...]) matching Add.get_args()
"""
function poly_to_sexpr(
        poly::MP.AbstractPolynomialLike,
        var_to_idx::Dict{Symbol, Int},
        param_to_idx::Dict{Symbol, Int},
    )::SExpr
    terms = SExpr[]
    for term in MP.terms(poly)
        raw_coeff = MP.coefficient(term)
        coeff = ComplexF64(raw_coeff)
        iszero(coeff) && continue
        mono = MP.monomial(term)

        factors = SExpr[]
        for (var, exp) in zip(MP.variables(mono), MP.exponents(mono))
            exp == 0 && continue
            sym = Symbol(var)
            idx_var = get(var_to_idx, sym, 0)
            idx_param = get(param_to_idx, sym, 0)
            base = idx_var > 0 ? SVar(idx_var) : SParam(idx_param)
            push!(factors, exp == 1 ? base : SPow(base, exp))
        end

        if isempty(factors)
            # Pure constant term
            push!(terms, SConst(coeff))
        elseif coeff == one(ComplexF64)
            # Monomial with coefficient 1: just the factors
            push!(terms, length(factors) == 1 ? factors[1] : SMul(factors))
        elseif coeff == -one(ComplexF64)
            # Represent -1 as a Mul coefficient so negative-Mul handling
            # matches SymEngine's bvisit(const Mul&).
            pushfirst!(factors, SConst(coeff))
            push!(terms, SMul(factors))
        else
            # General coefficient: SMul([coeff, factor1, factor2, ...])
            # Matches SymEngine's Mul(coef, {base: exp, ...}).get_args()
            pushfirst!(factors, SConst(coeff))
            push!(terms, SMul(factors))
        end
    end

    if isempty(terms)
        return SConst(zero(ComplexF64))
    elseif length(terms) == 1
        return terms[1]
    else
        sort!(terms; lt = (a, b) -> _sexpr_lt(_add_term_base(a), _add_term_base(b)))
        return SAdd(terms)
    end
end
