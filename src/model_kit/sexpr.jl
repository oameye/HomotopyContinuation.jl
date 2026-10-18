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
    UNARY_EXP
    UNARY_LOG
    UNARY_TAN
    UNARY_ASIN
    UNARY_ACOS
    UNARY_SINH
    UNARY_COSH
    UNARY_TANH
end

@inline function unary_op_type(kind::SUnaryKind.T)::OpType.T
    kind == SUnaryKind.UNARY_SQRT && return OpType.OP_SQRT
    kind == SUnaryKind.UNARY_SIN && return OpType.OP_SIN
    kind == SUnaryKind.UNARY_COS && return OpType.OP_COS
    kind == SUnaryKind.UNARY_EXP && return OpType.OP_EXP
    kind == SUnaryKind.UNARY_LOG && return OpType.OP_LOG
    kind == SUnaryKind.UNARY_TAN && return OpType.OP_TAN
    kind == SUnaryKind.UNARY_ASIN && return OpType.OP_ASIN
    kind == SUnaryKind.UNARY_ACOS && return OpType.OP_ACOS
    kind == SUnaryKind.UNARY_SINH && return OpType.OP_SINH
    kind == SUnaryKind.UNARY_COSH && return OpType.OP_COSH
    return OpType.OP_TANH
end

@inline function apply_unary(kind::SUnaryKind.T, val::ComplexF64)::ComplexF64
    kind == SUnaryKind.UNARY_SQRT && return sqrt(val)
    kind == SUnaryKind.UNARY_SIN && return sin(val)
    kind == SUnaryKind.UNARY_COS && return cos(val)
    kind == SUnaryKind.UNARY_EXP && return exp(val)
    kind == SUnaryKind.UNARY_LOG && return log(val)
    kind == SUnaryKind.UNARY_TAN && return tan(val)
    kind == SUnaryKind.UNARY_ASIN && return asin(val)
    kind == SUnaryKind.UNARY_ACOS && return acos(val)
    kind == SUnaryKind.UNARY_SINH && return sinh(val)
    kind == SUnaryKind.UNARY_COSH && return cosh(val)
    return tanh(val)
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

    """Power with a numeric non-integer exponent."""
    struct SRPow
        base::SExpr
        exp::ComplexF64
    end

    """Negation: -arg."""
    struct SNeg
        arg::SExpr
    end

    """Unary function application."""
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
const _EMPTY_SEXPR_VEC = SExprT[]

@inline _owned_args(args::AbstractVector{<:SExprT})::Vector{SExprT} = collect(SExprT, args)

# Compound expressions are hashed structurally and used as Dict/Set keys in CSE.
# They must therefore own stable child storage instead of aliasing caller vectors.
@inline SExpr.SAdd(args::AbstractVector{<:SExprT}) = invoke(SExpr.SAdd, Tuple{Any}, _owned_args(args))
@inline SExpr.SMul(args::AbstractVector{<:SExprT}) = invoke(SExpr.SMul, Tuple{Any}, _owned_args(args))
@inline SExpr.SFuncSym(kind::SFuncKind.T, args::AbstractVector{<:SExprT}) =
    invoke(SExpr.SFuncSym, Tuple{Any, Any}, kind, _owned_args(args))

## ── Hashing (matches old symbol-seeded hash for CSE ordering stability) ──

function _fold_hash(seed, args)::UInt
    h = hash(seed, zero(UInt))
    for a in args
        h = hash(a, h)
    end
    return h
end

function Base.hash(e::SExprT, h::UInt)::UInt
    return @match e begin
        SExpr.SConst(val) => hash(val, hash(:SConst, h))
        SExpr.SVar(idx) => hash(idx, hash(:SVar, h))
        SExpr.SParam(idx) => hash(idx, hash(:SParam, h))
        SExpr.STmp(id) => hash(id, hash(:STmp, h))
        SExpr.SAdd(args) => hash(_fold_hash(:SAdd, args), h)
        SExpr.SMul(args) => hash(_fold_hash(:SMul, args), h)
        SExpr.SPow(base, exp) =>
            hash(hash(exp, hash(base, hash(:SPow, zero(UInt)))), h)
        SExpr.SRPow(base, exp) =>
            hash(hash(exp, hash(base, hash(:SRPow, zero(UInt)))), h)
        SExpr.SNeg(arg) => hash(arg, hash(:SNeg, h))
        SExpr.SUnary(kind, arg) => hash(arg, hash(kind, hash(:SUnary, h)))
        SExpr.SFuncSym(kind, args) =>
            hash(_fold_hash((:SFuncSym, kind), args), h)
    end
end

## ── SExpr helpers ───────────────────────────────────────────────────────────

@inline _is_atom(e::SExprT)::Bool = @match e begin
    SExpr.SConst(_) => true
    SExpr.SVar(_) => true
    SExpr.SParam(_) => true
    SExpr.STmp(_) => true
    _ => false
end

"""Get the arguments (children) of a compound expression."""
@inline _get_args(e::SExprT)::Vector{SExprT} = @match e begin
    SExpr.SAdd(args) => args::Vector{SExprT}
    SExpr.SMul(args) => args::Vector{SExprT}
    SExpr.SFuncSym(_, args) => args::Vector{SExprT}
    SExpr.SPow(base, _) => SExprT[base::SExprT]
    SExpr.SRPow(base, _) => SExprT[base::SExprT]
    SExpr.SNeg(arg) => SExprT[arg::SExprT]
    SExpr.SUnary(_, arg) => SExprT[arg::SExprT]
    _ => _EMPTY_SEXPR_VEC
end

@inline function _rebuild_funcsym(
        kind::SFuncKind.T, args::Vector{SExprT},
    )::SExprT
    kind == SFuncKind.SFUNC_ADD && return _canonical_add(args)
    kind == SFuncKind.SFUNC_MUL && return _canonical_mul(args)
    if kind == SFuncKind.SFUNC_POW && length(args) == 2
        @match args[2] begin
            SExpr.SConst(val) => return SExpr.SPow(args[1], Int(real(val)))
            _ => nothing
        end
    end
    return SExpr.SFuncSym(kind, args)
end

@inline _rebuild_expr(expr::SExprT, args::Vector{SExprT})::SExprT = @match expr begin
    SExpr.SAdd(_) => _canonical_add(args)
    SExpr.SMul(_) => _canonical_mul(args)
    SExpr.SPow(_, exp) => SExpr.SPow(args[1], exp)
    SExpr.SRPow(_, exp) => SExpr.SRPow(args[1], exp)
    SExpr.SNeg(_) => SExpr.SNeg(args[1])
    SExpr.SUnary(kind, _) => SExpr.SUnary(kind, args[1])
    SExpr.SFuncSym(kind, _) => _rebuild_funcsym(kind, args)
    # Atoms have no children, so `_rebuild_expr` returns them unchanged.
    _ => expr
end

@inline _complex_lt(a::ComplexF64, b::ComplexF64)::Bool =
    real(a) < real(b) || (real(a) == real(b) && imag(a) < imag(b))
@inline _sexpr_kind_lt(a::SFuncKind.T, b::SFuncKind.T)::Bool = Int(a) < Int(b)
@inline _sexpr_kind_lt(a::SUnaryKind.T, b::SUnaryKind.T)::Bool = Int(a) < Int(b)

@inline _sexpr_tag_order(e::SExprT)::UInt8 = @match e begin
    SExpr.SConst(_) => 0x01
    SExpr.SVar(_) => 0x02
    SExpr.SParam(_) => 0x03
    SExpr.STmp(_) => 0x04
    SExpr.SAdd(_) => 0x05
    SExpr.SMul(_) => 0x06
    SExpr.SPow(_, _) => 0x07
    SExpr.SNeg(_) => 0x08
    SExpr.SUnary(_, _) => 0x09
    SExpr.SFuncSym(_, _) => 0x0a
    SExpr.SRPow(_, _) => 0x0b
end

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
    a_tag = _sexpr_tag_order(a)
    b_tag = _sexpr_tag_order(b)
    a_tag == b_tag || return a_tag < b_tag

    # Same tag on both sides, so each arm binds one variant's fields from each.
    return @match (a, b) begin
        (SExpr.SConst(x), SExpr.SConst(y)) => _complex_lt(x, y)
        (SExpr.SVar(x), SExpr.SVar(y)) => x < y
        (SExpr.SParam(x), SExpr.SParam(y)) => x < y
        (SExpr.STmp(x), SExpr.STmp(y)) => x < y
        (SExpr.SAdd(xs), SExpr.SAdd(ys)) =>
            _sexpr_args_lt(xs, ys)
        (SExpr.SMul(xs), SExpr.SMul(ys)) =>
            _sexpr_args_lt(xs, ys)
        (SExpr.SPow(xb, xe), SExpr.SPow(yb, ye)) => begin
            p = xb
            q = yb
            p == q ? xe < ye : _sexpr_struct_lt(p, q)
        end
        (SExpr.SRPow(xb, xe), SExpr.SRPow(yb, ye)) => begin
            p = xb
            q = yb
            p == q ? _complex_lt(xe, ye) : _sexpr_struct_lt(p, q)
        end
        (SExpr.SNeg(x), SExpr.SNeg(y)) => _sexpr_struct_lt(x, y)
        (SExpr.SUnary(xk, x), SExpr.SUnary(yk, y)) =>
            xk != yk ? _sexpr_kind_lt(xk, yk) : _sexpr_struct_lt(x, y)
        (SExpr.SFuncSym(xk, xs), SExpr.SFuncSym(yk, ys)) =>
            xk != yk ? _sexpr_kind_lt(xk, yk) :
            _sexpr_args_lt(xs, ys)
        _ => error("Unhandled SExpr variant in _sexpr_struct_lt")
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
    @match e begin
        SExpr.SMul(args) => begin
            xs = args
            isempty(xs) || @match xs[1] begin
                SExpr.SConst(_) => begin
                    rest = xs[2:end]
                    return length(rest) == 1 ? rest[1] : SExpr.SMul(rest)
                end
                _ => nothing
            end
        end
        _ => nothing
    end
    return e
end

function _flatten_add_arg!(
        flat_args::Vector{SExprT},
        const_sum::Base.RefValue{ComplexF64},
        arg::SExprT,
    )::Nothing
    @match arg begin
        SExpr.SAdd(args) => for child in args
            _flatten_add_arg!(flat_args, const_sum, child)
        end
        SExpr.SConst(val) => (const_sum[] += val)
        _ => push!(flat_args, arg)
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
    @match arg begin
        SExpr.SMul(args) => for child in args
            _flatten_mul_arg!(flat_args, coeff, child)
        end
        SExpr.SConst(val) => (coeff[] *= val)
        SExpr.SNeg(a) => begin
            coeff[] = -coeff[]
            _flatten_mul_arg!(flat_args, coeff, a)
        end
        _ => push!(flat_args, arg)
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
    return @match arg begin
        SExpr.SConst(val) => SExpr.SConst(apply_unary(kind, val))
        _ => SExpr.SUnary(kind, arg)
    end
end

function _canonical_rpow(base::SExprT, exp::ComplexF64)::SExprT
    return @match base begin
        SExpr.SConst(val) => SExpr.SConst(val^exp)
        _ => SExpr.SRPow(base, exp)
    end
end

## ── Polynomial → SExpr conversion ──────────────────────────────────────────

@noinline _unknown_symbol_error(sym::Symbol)::ArgumentError = ArgumentError(
    "symbol $sym is neither a variable nor a parameter of the system",
)

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
            base = if idx_var > 0
                SExpr.SVar(idx_var)
            else
                idx_param = get(param_to_idx, sym, 0)
                idx_param > 0 || throw(_unknown_symbol_error(sym))
                SExpr.SParam(idx_param)
            end
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
