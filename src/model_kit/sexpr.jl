## SExpr — algebraic data type for the symbolic IR.
#
# SExpr is an S-expression IR used as an intermediate representation between
# the MultivariatePolynomials input and the CSE algorithm. Uses Moshi.jl for
# tagged-union storage — all variants are one concrete type, eliminating
# dynamic dispatch in the CSE pipeline.

## ── SFuncKind enum (used by SFuncSym variant) ─────────────────────────────

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
Unary function applied by an `SUnary` node. Each kind lowers to one arity-1
`OpType`.
"""
@enumx SUnaryKind::Int8 begin
    UNARY_SQRT
    UNARY_SIN
    UNARY_COS
end

@inline function unary_op_type(kind::SUnaryKind.T)::OpType.T
    kind == SUnaryKind.UNARY_SQRT && return OpType.OP_SQRT
    kind == SUnaryKind.UNARY_SIN && return OpType.OP_SIN
    return OpType.OP_COS
end

@inline function apply_unary(kind::SUnaryKind.T, val::ComplexF64)::ComplexF64
    kind == SUnaryKind.UNARY_SQRT && return sqrt(val)
    kind == SUnaryKind.UNARY_SIN && return sin(val)
    return cos(val)
end

## ── SExpr ADT ─────────────────────────────────────────────────────────────

@data SExpr begin
    """Constant value."""
    struct SConst
        val::ComplexF64
    end

    """Variable reference (1-based index)."""
    struct SVar
        idx::Int
    end

    """Parameter reference (1-based index)."""
    struct SParam
        idx::Int
    end

    """CSE temporary (assigned by tree_cse)."""
    struct STmp
        id::Int
    end

    """Addition: sum of args (n-ary, n >= 2)."""
    struct SAdd
        args::Vector{SExpr}
    end

    """Multiplication: product of args (n-ary, n >= 2)."""
    struct SMul
        args::Vector{SExpr}
    end

    """Integer power: base^exp where exp is a positive integer."""
    struct SPow
        base::SExpr
        exp::Int
    end

    """Negation: -arg."""
    struct SNeg
        arg::SExpr
    end

    """Unary function application: sqrt, sin or cos."""
    struct SUnary
        kind::SUnaryKind.T
        arg::SExpr
    end

    """Unevaluated function symbol — placeholder created by opt_cse."""
    struct SFuncSym
        kind::SFuncKind.T
        args::Vector{SExpr}
    end
end

@derive SExpr[Eq]

## ── Concrete type alias ───────────────────────────────────────────────────
# Moshi creates one concrete type for all variants. We alias it for use in
# Vector{SExprT}, Dict{SExprT,...}, Set{SExprT}, and function signatures.

const SExprT = typeof(SExpr.SConst(zero(ComplexF64)))
const SConstStorage = variant_storage_type(SExpr.SConst)
const SVarStorage = variant_storage_type(SExpr.SVar)
const SParamStorage = variant_storage_type(SExpr.SParam)
const STmpStorage = variant_storage_type(SExpr.STmp)
const SAddStorage = variant_storage_type(SExpr.SAdd)
const SMulStorage = variant_storage_type(SExpr.SMul)
const SPowStorage = variant_storage_type(SExpr.SPow)
const SNegStorage = variant_storage_type(SExpr.SNeg)
const SUnaryStorage = variant_storage_type(SExpr.SUnary)
const SFuncSymStorage = variant_storage_type(SExpr.SFuncSym)
const _EMPTY_SEXPR_VEC = SExprT[]

@inline _owned_args(args::AbstractVector{<:SExprT})::Vector{SExprT} = collect(SExprT, args)

# Compound expressions are hashed structurally and used as Dict/Set keys in CSE.
# They must therefore own stable child storage instead of aliasing caller vectors.
@inline SExpr.SAdd(args::AbstractVector{<:SExprT}) = invoke(SExpr.SAdd, Tuple{Any}, _owned_args(args))
@inline SExpr.SMul(args::AbstractVector{<:SExprT}) = invoke(SExpr.SMul, Tuple{Any}, _owned_args(args))
@inline SExpr.SFuncSym(kind::SFuncKind.T, args::AbstractVector{<:SExprT}) =
    invoke(SExpr.SFuncSym, Tuple{Any, Any}, kind, _owned_args(args))

@inline sexpr_storage(expr::SExprT) = variant_storage(expr)

## ── Hashing (matches old symbol-seeded hash for CSE ordering stability) ──

function _fold_hash(seed, args)::UInt
    h = hash(seed, zero(UInt))
    for a in args
        h = hash(a, h)
    end
    return h
end

function Base.hash(e::SExprT, h::UInt)::UInt
    storage = sexpr_storage(e)
    if storage isa SConstStorage
        return hash(storage.val, hash(:SConst, h))
    elseif storage isa SVarStorage
        return hash(storage.idx, hash(:SVar, h))
    elseif storage isa SParamStorage
        return hash(storage.idx, hash(:SParam, h))
    elseif storage isa STmpStorage
        return hash(storage.id, hash(:STmp, h))
    elseif storage isa SAddStorage
        return hash(_fold_hash(:SAdd, storage.args), h)
    elseif storage isa SMulStorage
        return hash(_fold_hash(:SMul, storage.args), h)
    elseif storage isa SPowStorage
        return hash(hash(storage.exp, hash(storage.base, hash(:SPow, zero(UInt)))), h)
    elseif storage isa SNegStorage
        return hash(storage.arg, hash(:SNeg, h))
    elseif storage isa SUnaryStorage
        return hash(storage.arg, hash(storage.kind, hash(:SUnary, h)))
    else # SFuncSymStorage
        return hash(_fold_hash((:SFuncSym, storage.kind), storage.args), h)
    end
end

## ── SExpr helpers ───────────────────────────────────────────────────────────

@inline _is_atom_storage(::Union{SConstStorage, SVarStorage, SParamStorage, STmpStorage}) = true
@inline _is_atom_storage(::Any) = false
@inline _is_atom(e::SExprT)::Bool = _is_atom_storage(sexpr_storage(e))

"""Get the arguments (children) of a compound expression."""
@inline _get_args_storage(storage::Union{SConstStorage, SVarStorage, SParamStorage, STmpStorage}) =
    _EMPTY_SEXPR_VEC
@inline _get_args_storage(storage::SAddStorage) = storage.args
@inline _get_args_storage(storage::SMulStorage) = storage.args
@inline _get_args_storage(storage::SPowStorage) = SExprT[storage.base]
@inline _get_args_storage(storage::SNegStorage) = SExprT[storage.arg]
@inline _get_args_storage(storage::SUnaryStorage) = SExprT[storage.arg]
@inline _get_args_storage(storage::SFuncSymStorage) = storage.args
@inline _get_args(e::SExprT)::Vector{SExprT} = _get_args_storage(sexpr_storage(e))

@inline _rebuild_expr_storage(storage::SAddStorage, args::Vector{SExprT})::SExprT =
    _canonical_add(args)
@inline _rebuild_expr_storage(storage::SMulStorage, args::Vector{SExprT})::SExprT =
    _canonical_mul(args)
@inline _rebuild_expr_storage(storage::SPowStorage, args::Vector{SExprT})::SExprT =
    SExpr.SPow(args[1], storage.exp)
@inline _rebuild_expr_storage(storage::SNegStorage, args::Vector{SExprT})::SExprT =
    SExpr.SNeg(args[1])
@inline _rebuild_expr_storage(storage::SUnaryStorage, args::Vector{SExprT})::SExprT =
    SExpr.SUnary(storage.kind, args[1])
@inline function _rebuild_expr_storage(
        storage::SFuncSymStorage,
        args::Vector{SExprT},
    )::SExprT
    if storage.kind == SFuncKind.SFUNC_ADD
        return _canonical_add(args)
    elseif storage.kind == SFuncKind.SFUNC_MUL
        return _canonical_mul(args)
    elseif storage.kind == SFuncKind.SFUNC_POW && length(args) == 2
        exponent_storage = sexpr_storage(args[2])
        if exponent_storage isa SConstStorage
            return SExpr.SPow(args[1], Int(real(exponent_storage.val)))
        end
    end
    return SExpr.SFuncSym(storage.kind, args)
end
@inline _rebuild_expr_storage(::Union{SConstStorage, SVarStorage, SParamStorage, STmpStorage}, args::Vector{SExprT})::SExprT =
    error("atom expressions should not be rebuilt")
@inline _rebuild_expr(expr::SExprT, args::Vector{SExprT})::SExprT =
    _is_atom(expr) ? expr : _rebuild_expr_storage(sexpr_storage(expr), args)

@inline _complex_lt(a::ComplexF64, b::ComplexF64)::Bool =
    real(a) < real(b) || (real(a) == real(b) && imag(a) < imag(b))
@inline _sexpr_kind_lt(a::SFuncKind.T, b::SFuncKind.T)::Bool = Int(a) < Int(b)
@inline _sexpr_kind_lt(a::SUnaryKind.T, b::SUnaryKind.T)::Bool = Int(a) < Int(b)

@inline _sexpr_tag_order(::SConstStorage)::UInt8 = 0x01
@inline _sexpr_tag_order(::SVarStorage)::UInt8 = 0x02
@inline _sexpr_tag_order(::SParamStorage)::UInt8 = 0x03
@inline _sexpr_tag_order(::STmpStorage)::UInt8 = 0x04
@inline _sexpr_tag_order(::SAddStorage)::UInt8 = 0x05
@inline _sexpr_tag_order(::SMulStorage)::UInt8 = 0x06
@inline _sexpr_tag_order(::SPowStorage)::UInt8 = 0x07
@inline _sexpr_tag_order(::SNegStorage)::UInt8 = 0x08
@inline _sexpr_tag_order(::SUnaryStorage)::UInt8 = 0x09
@inline _sexpr_tag_order(::SFuncSymStorage)::UInt8 = 0x0a

function _sexpr_args_lt(a_args::Vector{SExprT}, b_args::Vector{SExprT})::Bool
    n = min(length(a_args), length(b_args))
    for i in 1:n
        a = a_args[i]
        b = b_args[i]
        a == b && continue
        return _sexpr_struct_lt(a, b)
    end
    return length(a_args) < length(b_args)
end

function _sexpr_struct_lt(a::SExprT, b::SExprT)::Bool
    a_storage = sexpr_storage(a)
    b_storage = sexpr_storage(b)
    a_tag = _sexpr_tag_order(a_storage)
    b_tag = _sexpr_tag_order(b_storage)
    a_tag == b_tag || return a_tag < b_tag

    if a_storage isa SConstStorage
        b_storage_typed = b_storage::SConstStorage
        return _complex_lt(a_storage.val, b_storage_typed.val)
    elseif a_storage isa SVarStorage
        b_storage_typed = b_storage::SVarStorage
        return a_storage.idx < b_storage_typed.idx
    elseif a_storage isa SParamStorage
        b_storage_typed = b_storage::SParamStorage
        return a_storage.idx < b_storage_typed.idx
    elseif a_storage isa STmpStorage
        b_storage_typed = b_storage::STmpStorage
        return a_storage.id < b_storage_typed.id
    elseif a_storage isa SAddStorage
        b_storage_typed = b_storage::SAddStorage
        return _sexpr_args_lt(a_storage.args, b_storage_typed.args)
    elseif a_storage isa SMulStorage
        b_storage_typed = b_storage::SMulStorage
        return _sexpr_args_lt(a_storage.args, b_storage_typed.args)
    elseif a_storage isa SPowStorage
        b_storage_typed = b_storage::SPowStorage
        if a_storage.base == b_storage_typed.base
            return a_storage.exp < b_storage_typed.exp
        end
        return _sexpr_struct_lt(a_storage.base, b_storage_typed.base)
    elseif a_storage isa SNegStorage
        b_storage_typed = b_storage::SNegStorage
        return _sexpr_struct_lt(a_storage.arg, b_storage_typed.arg)
    elseif a_storage isa SUnaryStorage
        b_storage_typed = b_storage::SUnaryStorage
        if a_storage.kind != b_storage_typed.kind
            return _sexpr_kind_lt(a_storage.kind, b_storage_typed.kind)
        end
        return _sexpr_struct_lt(a_storage.arg, b_storage_typed.arg)
    elseif a_storage isa SFuncSymStorage
        b_storage_typed = b_storage::SFuncSymStorage
        if a_storage.kind != b_storage_typed.kind
            return _sexpr_kind_lt(a_storage.kind, b_storage_typed.kind)
        end
        return _sexpr_args_lt(a_storage.args, b_storage_typed.args)
    else
        error("Unhandled SExpr storage in _sexpr_struct_lt: $(typeof(a_storage))")
    end
end

_sexpr_lt(a::SExprT, b::SExprT)::Bool = hash(a) < hash(b)
_add_term_lt(a::SExprT, b::SExprT)::Bool = _sexpr_lt(_add_term_base(a), _add_term_base(b))

"""Return `empty_val` for 0 args, the single arg for 1, or `constructor(args)` for many."""
function _wrap_args(args::Vector{SExprT}, empty_val::SExprT, constructor::F)::SExprT where {F}
    isempty(args) && return empty_val
    length(args) == 1 && return args[1]
    return constructor(args)
end

"""
Extract the "base expression" of an Add term, stripping the leading coefficient.
"""
function _add_term_base(e::SExprT)::SExprT
    storage = sexpr_storage(e)
    if storage isa SMulStorage
        args = storage.args
        if !isempty(args)
            first_storage = sexpr_storage(args[1])
            if first_storage isa SConstStorage
                rest = args[2:end]
                return length(rest) == 1 ? rest[1] : SExpr.SMul(rest)
            end
        end
    end
    return e
end

function _flatten_add_arg!(
        flat_args::Vector{SExprT},
        const_sum::Base.RefValue{ComplexF64},
        arg::SExprT,
    )::Nothing
    storage = sexpr_storage(arg)
    if storage isa SAddStorage
        for child in storage.args
            _flatten_add_arg!(flat_args, const_sum, child)
        end
    elseif storage isa SConstStorage
        const_sum[] += storage.val
    else
        push!(flat_args, arg)
    end
    return nothing
end

function _canonical_add(args::Vector{SExprT})::SExprT
    flat_args = SExprT[]
    const_sum = Ref(zero(ComplexF64))
    for arg in args
        _flatten_add_arg!(flat_args, const_sum, arg)
    end
    sort!(flat_args; lt = _sexpr_lt)
    !iszero(const_sum[]) && pushfirst!(flat_args, SExpr.SConst(const_sum[]))
    return _wrap_args(flat_args, SExpr.SConst(zero(ComplexF64)), SExpr.SAdd)
end

function _flatten_mul_arg!(
        flat_args::Vector{SExprT},
        coeff::Base.RefValue{ComplexF64},
        arg::SExprT,
    )::Nothing
    storage = sexpr_storage(arg)
    if storage isa SMulStorage
        for child in storage.args
            _flatten_mul_arg!(flat_args, coeff, child)
        end
    elseif storage isa SConstStorage
        coeff[] *= storage.val
    elseif storage isa SNegStorage
        coeff[] = -coeff[]
        _flatten_mul_arg!(flat_args, coeff, storage.arg)
    else
        push!(flat_args, arg)
    end
    return nothing
end

function _canonical_mul(args::Vector{SExprT})::SExprT
    flat_args = SExprT[]
    coeff = Ref(one(ComplexF64))
    for arg in args
        _flatten_mul_arg!(flat_args, coeff, arg)
    end
    iszero(coeff[]) && return SExpr.SConst(zero(ComplexF64))
    sort!(flat_args; lt = _sexpr_lt)
    coeff[] != one(ComplexF64) && pushfirst!(flat_args, SExpr.SConst(coeff[]))
    return _wrap_args(flat_args, SExpr.SConst(one(ComplexF64)), SExpr.SMul)
end

function _canonical_unary(kind::SUnaryKind.T, arg::SExprT)::SExprT
    storage = sexpr_storage(arg)
    storage isa SConstStorage && return SExpr.SConst(apply_unary(kind, storage.val))
    return SExpr.SUnary(kind, arg)
end

## ── Polynomial → SExpr conversion ──────────────────────────────────────────

"""
    poly_to_sexpr(poly, var_to_idx, param_to_idx) -> SExprT

Convert a MultivariatePolynomials polynomial to an SExpr tree.
"""
function poly_to_sexpr(
        poly::MP.AbstractPolynomialLike,
        var_to_idx::Dict{Symbol, Int},
        param_to_idx::Dict{Symbol, Int},
    )::SExprT
    Base.@nospecialize poly
    terms = SExprT[]
    for term in MP.terms(poly)
        raw_coeff = MP.coefficient(term)
        coeff = ComplexF64(raw_coeff)
        iszero(coeff) && continue
        mono = MP.monomial(term)

        factors = SExprT[]
        for (var, exp) in zip(MP.variables(mono), MP.exponents(mono))
            exp == 0 && continue
            sym = Symbol(var)
            idx_var = get(var_to_idx, sym, 0)
            idx_param = get(param_to_idx, sym, 0)
            base = idx_var > 0 ? SExpr.SVar(idx_var) : SExpr.SParam(idx_param)
            push!(factors, exp == 1 ? base : SExpr.SPow(base, exp))
        end

        if isempty(factors)
            push!(terms, SExpr.SConst(coeff))
        elseif coeff == one(ComplexF64)
            push!(terms, length(factors) == 1 ? factors[1] : SExpr.SMul(factors))
        else
            pushfirst!(factors, SExpr.SConst(coeff))
            push!(terms, SExpr.SMul(factors))
        end
    end

    length(terms) > 1 && sort!(terms; lt = _add_term_lt)
    return _wrap_args(terms, SExpr.SConst(zero(ComplexF64)), SExpr.SAdd)
end
