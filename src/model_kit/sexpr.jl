## SExpr — hash-consed expression graph for the symbolic IR.
#
# Every node lives in one `SGraph` arena and is named by an `Int32` id. Nodes are
# interned on construction, so structurally equal expressions share one id and
# the equality, hashing and set membership CSE relies on are integer operations
# on plain containers.

## ── SFuncKind enum (used by SFuncSym nodes) ───────────────────────────────

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

## ── Graph ─────────────────────────────────────────────────────────────────

"""Node kind of the symbolic IR."""
@enumx SKind::UInt8 begin
    S_CONST
    S_VAR
    S_PARAM
    S_TMP
    S_ADD
    S_MUL
    S_POW
    S_RPOW
    S_NEG
    S_UNARY
    S_FUNCSYM
end

# `int` holds the variable or parameter index, the temporary id or the integer
# exponent; `val` the constant or the numeric exponent; `sub` the unary or
# function-symbol kind. Children are `args[first:first+nargs-1]`, which holds the
# single operand of a power, negation or unary node. `hash` is the seed-free part
# of the structural hash of compound nodes.
struct SNode
    kind::SKind.T
    sub::Int8
    first::Int32
    nargs::Int32
    int::Int
    val::ComplexF64
    hash::UInt
end

struct SKey
    kind::SKind.T
    sub::Int8
    int::Int
    val::ComplexF64
    args::Vector{Int32}
end

Base.hash(k::SKey, h::UInt) =
    hash(k.args, hash(k.val, hash(k.int, hash(k.sub, hash(UInt8(k.kind), h)))))
Base.isequal(a::SKey, b::SKey) =
    a.kind == b.kind && a.sub == b.sub && a.int == b.int && isequal(a.val, b.val) &&
    a.args == b.args

"""
    SGraph()

Arena of interned `SExpr` nodes. A node id is only meaningful in its graph.
"""
struct SGraph
    nodes::Vector{SNode}
    args::Vector{Int32}
    intern::Dict{SKey, Int32}
end

SGraph() = SGraph(SNode[], Int32[], Dict{SKey, Int32}())

@inline node_kind(g::SGraph, e::Int32)::SKind.T = g.nodes[e].kind
@inline node_int(g::SGraph, e::Int32)::Int = g.nodes[e].int
@inline node_val(g::SGraph, e::Int32)::ComplexF64 = g.nodes[e].val
@inline node_unary_kind(g::SGraph, e::Int32)::SUnaryKind.T = SUnaryKind.T(g.nodes[e].sub)
@inline node_func_kind(g::SGraph, e::Int32)::SFuncKind.T = SFuncKind.T(g.nodes[e].sub)
@inline node_nargs(g::SGraph, e::Int32)::Int = Int(g.nodes[e].nargs)
@inline node_arg(g::SGraph, e::Int32, i::Int)::Int32 = g.args[g.nodes[e].first + i - 1]
@inline is_kind(g::SGraph, e::Int32, kind::SKind.T)::Bool = node_kind(g, e) == kind

"""Children of a node, in order; empty for atoms."""
function get_args(g::SGraph, e::Int32)::Vector{Int32}
    node = g.nodes[e]
    return g.args[node.first:(node.first + node.nargs - 1)]
end

@inline function is_atom(g::SGraph, e::Int32)::Bool
    k = node_kind(g, e)
    return k == SKind.S_CONST || k == SKind.S_VAR || k == SKind.S_PARAM ||
        k == SKind.S_TMP
end

## ── Structural hash ──────────────────────────────────────────────────────────
#
# `node_hash(g, e, h)` is the structural hash of the expression tree below `e`
# with seed `h`. It orders sums and products canonically.

function fold_hash(g::SGraph, seed, args::Vector{Int32})::UInt
    h = hash(seed, zero(UInt))
    for a in args
        h = node_hash(g, a, h)
    end
    return h
end

function node_hash(g::SGraph, e::Int32, h::UInt)::UInt
    node = g.nodes[e]
    k = node.kind
    k == SKind.S_CONST && return hash(node.val, hash(:SConst, h))
    k == SKind.S_VAR && return hash(node.int, hash(:SVar, h))
    k == SKind.S_PARAM && return hash(node.int, hash(:SParam, h))
    k == SKind.S_TMP && return hash(node.int, hash(:STmp, h))
    k == SKind.S_NEG && return node_hash(g, node_arg(g, e, 1), hash(:SNeg, h))
    k == SKind.S_UNARY && return node_hash(
        g, node_arg(g, e, 1), hash(node.sub, hash(:SUnary, h)),
    )
    return hash(node.hash, h)
end

struct NodeRef
    graph::SGraph
    id::Int32
end

Base.hash(n::NodeRef, h::UInt) = node_hash(n.graph, n.id, h)

node_hash(g::SGraph, e::Int32)::UInt = hash(NodeRef(g, e))

function intern!(
        g::SGraph, kind::SKind.T, sub::Int8, int::Int, val::ComplexF64,
        args::Vector{Int32},
    )::Int32
    key = SKey(kind, sub, int, val, args)
    existing = get(g.intern, key, Int32(0))
    existing != Int32(0) && return existing
    first = Int32(length(g.args) + 1)
    append!(g.args, args)
    h = if kind == SKind.S_ADD
        fold_hash(g, :SAdd, args)
    elseif kind == SKind.S_MUL
        fold_hash(g, :SMul, args)
    elseif kind == SKind.S_FUNCSYM
        fold_hash(g, (:SFuncSym, sub), args)
    elseif kind == SKind.S_POW
        hash(int, node_hash(g, args[1], hash(:SPow, zero(UInt))))
    elseif kind == SKind.S_RPOW
        hash(val, node_hash(g, args[1], hash(:SRPow, zero(UInt))))
    else
        zero(UInt)
    end
    push!(g.nodes, SNode(kind, sub, first, Int32(length(args)), int, val, h))
    id = Int32(length(g.nodes))
    g.intern[key] = id
    return id
end

const NO_ARGS = Int32[]

sconst!(g::SGraph, val::ComplexF64)::Int32 =
    intern!(g, SKind.S_CONST, Int8(0), 0, val, NO_ARGS)
svar!(g::SGraph, idx::Int)::Int32 =
    intern!(g, SKind.S_VAR, Int8(0), idx, zero(ComplexF64), NO_ARGS)
sparam!(g::SGraph, idx::Int)::Int32 =
    intern!(g, SKind.S_PARAM, Int8(0), idx, zero(ComplexF64), NO_ARGS)
stmp!(g::SGraph, id::Int)::Int32 =
    intern!(g, SKind.S_TMP, Int8(0), id, zero(ComplexF64), NO_ARGS)
sadd!(g::SGraph, args::Vector{Int32})::Int32 =
    intern!(g, SKind.S_ADD, Int8(0), 0, zero(ComplexF64), args)
smul!(g::SGraph, args::Vector{Int32})::Int32 =
    intern!(g, SKind.S_MUL, Int8(0), 0, zero(ComplexF64), args)
spow!(g::SGraph, base::Int32, exp::Int)::Int32 =
    intern!(g, SKind.S_POW, Int8(0), exp, zero(ComplexF64), Int32[base])
srpow!(g::SGraph, base::Int32, exp::ComplexF64)::Int32 =
    intern!(g, SKind.S_RPOW, Int8(0), 0, exp, Int32[base])
sneg!(g::SGraph, arg::Int32)::Int32 =
    intern!(g, SKind.S_NEG, Int8(0), 0, zero(ComplexF64), Int32[arg])
sunary!(g::SGraph, kind::SUnaryKind.T, arg::Int32)::Int32 =
    intern!(g, SKind.S_UNARY, Int8(kind), 0, zero(ComplexF64), Int32[arg])
sfuncsym!(g::SGraph, kind::SFuncKind.T, args::Vector{Int32})::Int32 =
    intern!(g, SKind.S_FUNCSYM, Int8(kind), 0, zero(ComplexF64), args)

## ── Canonical construction ──────────────────────────────────────────────────

"""Sort node ids by structural hash, stably."""
function sort_by_hash!(g::SGraph, ids::Vector{Int32})::Vector{Int32}
    keyed = Tuple{UInt, Int32}[(node_hash(g, e), e) for e in ids]
    stable_sort!(keyed, isless)
    for i in eachindex(ids)
        ids[i] = keyed[i][2]
    end
    return ids
end

function rebuild_funcsym(g::SGraph, kind::SFuncKind.T, args::Vector{Int32})::Int32
    kind == SFuncKind.SFUNC_ADD && return canonical_add(g, args)
    kind == SFuncKind.SFUNC_MUL && return canonical_mul(g, args)
    if kind == SFuncKind.SFUNC_POW && length(args) == 2 && is_kind(g, args[2], SKind.S_CONST)
        return spow!(g, args[1], Int(real(node_val(g, args[2]))))
    end
    return sfuncsym!(g, kind, args)
end

function rebuild_expr(g::SGraph, expr::Int32, args::Vector{Int32})::Int32
    k = node_kind(g, expr)
    k == SKind.S_ADD && return canonical_add(g, args)
    k == SKind.S_MUL && return canonical_mul(g, args)
    k == SKind.S_POW && return spow!(g, args[1], node_int(g, expr))
    k == SKind.S_RPOW && return srpow!(g, args[1], node_val(g, expr))
    k == SKind.S_NEG && return sneg!(g, args[1])
    k == SKind.S_UNARY && return sunary!(g, node_unary_kind(g, expr), args[1])
    k == SKind.S_FUNCSYM && return rebuild_funcsym(g, node_func_kind(g, expr), args)
    # Atoms have no children, so `rebuild_expr` returns them unchanged.
    return expr
end

"""Return `empty_val` for 0 args, the single arg for 1, or `constructor(g, args)` for many."""
function wrap_args(
        g::SGraph, args::Vector{Int32}, empty_val::Int32, constructor::F,
    )::Int32 where {F}
    isempty(args) && return empty_val
    length(args) == 1 && return args[1]
    return constructor(g, args)
end

function flatten_add_arg!(
        g::SGraph, flat_args::Vector{Int32}, const_sum::Base.RefValue{ComplexF64},
        arg::Int32,
    )::Nothing
    k = node_kind(g, arg)
    if k == SKind.S_ADD
        for child in get_args(g, arg)
            flatten_add_arg!(g, flat_args, const_sum, child)
        end
    elseif k == SKind.S_CONST
        const_sum[] += node_val(g, arg)
    else
        push!(flat_args, arg)
    end
    return nothing
end

function canonical_add(g::SGraph, args::Vector{Int32})::Int32
    flat_args = Int32[]
    const_sum = Ref(zero(ComplexF64))
    for arg in args
        flatten_add_arg!(g, flat_args, const_sum, arg)
    end
    sort_by_hash!(g, flat_args)
    !iszero(const_sum[]) && pushfirst!(flat_args, sconst!(g, const_sum[]))
    return wrap_args(g, flat_args, sconst!(g, zero(ComplexF64)), sadd!)
end

function flatten_mul_arg!(
        g::SGraph, flat_args::Vector{Int32}, coeff::Base.RefValue{ComplexF64},
        arg::Int32,
    )::Nothing
    k = node_kind(g, arg)
    if k == SKind.S_MUL
        for child in get_args(g, arg)
            flatten_mul_arg!(g, flat_args, coeff, child)
        end
    elseif k == SKind.S_CONST
        coeff[] *= node_val(g, arg)
    elseif k == SKind.S_NEG
        coeff[] = -coeff[]
        flatten_mul_arg!(g, flat_args, coeff, node_arg(g, arg, 1))
    else
        push!(flat_args, arg)
    end
    return nothing
end

function canonical_mul(g::SGraph, args::Vector{Int32})::Int32
    flat_args = Int32[]
    coeff = Ref(one(ComplexF64))
    for arg in args
        flatten_mul_arg!(g, flat_args, coeff, arg)
    end
    iszero(coeff[]) && return sconst!(g, zero(ComplexF64))
    sort_by_hash!(g, flat_args)
    coeff[] != one(ComplexF64) && pushfirst!(flat_args, sconst!(g, coeff[]))
    return wrap_args(g, flat_args, sconst!(g, one(ComplexF64)), smul!)
end

function canonical_unary(g::SGraph, kind::SUnaryKind.T, arg::Int32)::Int32
    is_kind(g, arg, SKind.S_CONST) && return sconst!(g, apply_unary(kind, node_val(g, arg)))
    return sunary!(g, kind, arg)
end

function canonical_rpow(g::SGraph, base::Int32, exp::ComplexF64)::Int32
    is_kind(g, base, SKind.S_CONST) && return sconst!(g, node_val(g, base)^exp)
    return srpow!(g, base, exp)
end

@noinline unknown_symbol_error(sym::Symbol)::ArgumentError = ArgumentError(
    "symbol $sym is neither a variable nor a parameter of the system",
)
