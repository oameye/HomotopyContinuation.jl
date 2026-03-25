## CSE (Common Subexpression Elimination) — SymEngine algorithm in Julia
#
# A 1:1 port of SymEngine's cse.cpp two-phase algorithm:
#   Phase 1 (opt_cse): factor common arguments in Add/Mul nodes
#   Phase 2 (tree_cse): eliminate repeated subexpressions
#
# Reference: https://github.com/symengine/symengine/blob/master/symengine/cse.cpp

## ── SExpr types ─────────────────────────────────────────────────────────────

abstract type SExpr end

"""Constant value. `is_real_int` tracks whether this came from an integer coefficient
(matching SymEngine's Integer type) vs a complex one (ComplexDouble)."""
struct SConst <: SExpr
    val::ComplexF64
    is_real_int::Bool
end
SConst(val::ComplexF64) = SConst(val, iszero(imag(val)) && isinteger(real(val)))

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

"""Addition: sum of args (n-ary, n >= 2)."""
struct SAdd <: SExpr
    args::Vector{SExpr}
end

"""Multiplication: product of args (n-ary, n >= 2)."""
struct SMul <: SExpr
    args::Vector{SExpr}
end

"""Integer power: base^exp where exp is a positive integer."""
struct SPow <: SExpr
    base::SExpr
    exp::Int
end

"""Negation: -arg."""
struct SNeg <: SExpr
    arg::SExpr
end

"""
Unevaluated function symbol — placeholder created by opt_cse to represent
factored common arguments without triggering canonical-form collapse.
`name` is "add", "mul", or "pow".
Corresponds to SymEngine's FunctionSymbol.
"""
struct SFuncSym <: SExpr
    name::String
    args::Vector{SExpr}
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
    h = hash(:SAdd, h)
    for a in e.args
        h = hash(a, h)
    end
    return h
end
function Base.hash(e::SMul, h::UInt)::UInt
    h = hash(:SMul, h)
    for a in e.args
        h = hash(a, h)
    end
    return h
end
function Base.hash(e::SPow, h::UInt)::UInt
    return hash(e.exp, hash(e.base, hash(:SPow, h)))
end
function Base.hash(e::SNeg, h::UInt)::UInt
    return hash(e.arg, hash(:SNeg, h))
end
function Base.hash(e::SFuncSym, h::UInt)::UInt
    h = hash(e.name, hash(:SFuncSym, h))
    for a in e.args
        h = hash(a, h)
    end
    return h
end

Base.:(==)(a::SConst, b::SConst) = a.val == b.val
Base.:(==)(a::SVar, b::SVar) = a.idx == b.idx
Base.:(==)(a::SParam, b::SParam) = a.idx == b.idx
Base.:(==)(a::STmp, b::STmp) = a.id == b.id
Base.:(==)(a::SPow, b::SPow) = a.exp == b.exp && a.base == b.base
Base.:(==)(a::SNeg, b::SNeg) = a.arg == b.arg
function Base.:(==)(a::SAdd, b::SAdd)
    length(a.args) == length(b.args) || return false
    return all(i -> a.args[i] == b.args[i], eachindex(a.args))
end
function Base.:(==)(a::SMul, b::SMul)
    length(a.args) == length(b.args) || return false
    return all(i -> a.args[i] == b.args[i], eachindex(a.args))
end
function Base.:(==)(a::SFuncSym, b::SFuncSym)
    a.name == b.name || return false
    length(a.args) == length(b.args) || return false
    return all(i -> a.args[i] == b.args[i], eachindex(a.args))
end
Base.:(==)(::SExpr, ::SExpr) = false

## ── SExpr helpers ───────────────────────────────────────────────────────────

_is_atom(e::SExpr)::Bool = e isa SConst || e isa SVar || e isa SParam || e isa STmp

"""
Get the arguments (children) of a compound expression.
Corresponds to SymEngine's get_args().
"""
function _get_args(e::SExpr)::Vector{SExpr}
    if e isa SAdd
        return e.args
    elseif e isa SMul
        return e.args
    elseif e isa SPow
        return SExpr[e.base]
    elseif e isa SNeg
        return SExpr[e.arg]
    elseif e isa SFuncSym
        return e.args
    else
        return SExpr[]
    end
end

"""
    _sexpr_type_code(e) -> Int

Return a type ordering code matching SymEngine's TypeID enum ordering:
  INTEGER(1) < RATIONAL(2) < COMPLEX(3) < ... < SYMBOL(14) < MUL(16) < ADD(17) < POW(18) < ...
We map our SExpr types to match.
"""
function _sexpr_type_code(e::SExpr)::Int
    e isa SConst && return 1     # maps to INTEGER/RATIONAL/COMPLEX etc.
    e isa SVar && return 14      # SYMENGINE_SYMBOL
    e isa SParam && return 14    # also a symbol
    e isa STmp && return 14      # CSE temps are Symbol("x0") in SymEngine — same type
    e isa SMul && return 16      # SYMENGINE_MUL
    e isa SAdd && return 17      # SYMENGINE_ADD
    e isa SPow && return 18      # SYMENGINE_POW
    e isa SNeg && return 16      # SNeg is Mul(-1, x) in SymEngine
    e isa SFuncSym && return 100 # FunctionSymbol comes much later
    return 999
end

"""
    _sexpr_cmp(a, b) -> Int  (-1, 0, 1)

Compare two SExpr values using SymEngine's `Basic::__cmp__` ordering:
1. Compare type codes first (SymEngine's TypeID enum order)
2. Within same type, use the type-specific `compare()` logic

This is critical for matching SymEngine's `set_basic` iteration order,
which determines the processing order in `match_common_args`.
"""
function _sexpr_cmp(a::SExpr, b::SExpr)::Int
    ta = _sexpr_type_code(a)
    tb = _sexpr_type_code(b)
    ta != tb && return ta < tb ? -1 : 1
    return _sexpr_compare(a, b)
end

# Type-specific compare (called when type codes match)
function _sexpr_compare(a::SConst, b::SConst)::Int
    # SymEngine compares numbers: Integer < Rational < Complex etc.
    # For our ComplexF64 constants, compare real part first, then imaginary
    ra, rb = real(a.val), real(b.val)
    ra != rb && return ra < rb ? -1 : 1
    ia, ib = imag(a.val), imag(b.val)
    ia != ib && return ia < ib ? -1 : 1
    return 0
end

function _sexpr_compare(a::SVar, b::SVar)::Int
    # Symbol::compare: lexicographic by name string
    # Our SVar stores an index; we need to compare the actual variable names
    # that would be used. Since we don't have names here, compare by index
    # which gives the same order as creation-order variable names (v1 < v2 < ...)
    a.idx != b.idx && return a.idx < b.idx ? -1 : 1
    return 0
end

function _sexpr_compare(a::SParam, b::SParam)::Int
    a.idx != b.idx && return a.idx < b.idx ? -1 : 1
    return 0
end

function _sexpr_compare(a::STmp, b::STmp)::Int
    a.id != b.id && return a.id < b.id ? -1 : 1
    return 0
end

function _sexpr_compare(a::SMul, b::SMul)::Int
    # Mul::compare: dict_.size() → coef → unified_compare(dict_)
    # In SymEngine, Mul stores coef + dict{base: exp}.
    # Our SMul stores flat args [coef, base1, base2, ...].
    # The "dict size" is number of non-coefficient args.
    # The "coef" is the leading SConst (if present).
    a_coef, a_bases = _split_mul_coef_bases(a)
    b_coef, b_bases = _split_mul_coef_bases(b)

    # Compare dict size (number of base factors)
    length(a_bases) != length(b_bases) && return length(a_bases) < length(b_bases) ? -1 : 1
    # Compare coef
    cc = _sexpr_cmp_number(a_coef, b_coef)
    cc != 0 && return cc
    # Compare dict entries (sorted by base __cmp__)
    # SymEngine's dict is map<Basic,Basic> ordered by __cmp__
    # We sort bases by __cmp__ and compare pairwise
    sa = sort(a_bases; lt = (x, y) -> _sexpr_cmp(x, y) < 0)
    sb = sort(b_bases; lt = (x, y) -> _sexpr_cmp(x, y) < 0)
    for i in eachindex(sa)
        c = _sexpr_cmp(sa[i], sb[i])
        c != 0 && return c
    end
    return 0
end

function _sexpr_compare(a::SAdd, b::SAdd)::Int
    # Add::compare: dict_.size() → coef → unified_compare(ordered_dict)
    a_coef, a_terms = _split_add_coef_terms(a)
    b_coef, b_terms = _split_add_coef_terms(b)
    length(a_terms) != length(b_terms) && return length(a_terms) < length(b_terms) ? -1 : 1
    cc = _sexpr_cmp_number(a_coef, b_coef)
    cc != 0 && return cc
    sa = sort(a_terms; lt = (x, y) -> _sexpr_cmp(x, y) < 0)
    sb = sort(b_terms; lt = (x, y) -> _sexpr_cmp(x, y) < 0)
    for i in eachindex(sa)
        c = _sexpr_cmp(sa[i], sb[i])
        c != 0 && return c
    end
    return 0
end

function _sexpr_compare(a::SPow, b::SPow)::Int
    # Pow::compare: base first, then exponent
    c = _sexpr_cmp(a.base, b.base)
    c != 0 && return c
    a.exp != b.exp && return a.exp < b.exp ? -1 : 1
    return 0
end

function _sexpr_compare(a::SNeg, b::SNeg)::Int
    return _sexpr_cmp(a.arg, b.arg)
end

function _sexpr_compare(a::SFuncSym, b::SFuncSym)::Int
    a.name != b.name && return a.name < b.name ? -1 : 1
    length(a.args) != length(b.args) && return length(a.args) < length(b.args) ? -1 : 1
    for i in eachindex(a.args)
        c = _sexpr_cmp(a.args[i], b.args[i])
        c != 0 && return c
    end
    return 0
end

# Cross-type comparison for same type code (all "symbol-like" types have code 14)
# In SymEngine, STmp is Symbol("x0"), SVar is Symbol("v1"), etc.
# Symbol comparison is lexicographic by name: "v1" < "v2" < "x0" < "x1"
# We approximate: SVar/SParam sort before STmp (since "v" < "x"),
# and within each, sort by index/id.
function _sexpr_compare(a::SExpr, b::SExpr)::Int
    # Same type handled above; this handles cross-type with same type_code
    ra = _symbol_sort_rank(a)
    rb = _symbol_sort_rank(b)
    ra != rb && return ra < rb ? -1 : 1
    return 0
end

# Rank for symbol-like types matching SymEngine's name-based ordering.
# In SymEngine: params are "p1","p2"..., vars are "v1","v2"..., temps are "x0","x1"...
# Lexicographic: "p" < "v" < "x" → SParam < SVar < STmp
_symbol_sort_rank(e::SParam) = (0, e.idx)
_symbol_sort_rank(e::SVar) = (1, e.idx)
_symbol_sort_rank(e::STmp) = (2, e.id)
_symbol_sort_rank(::SExpr) = (3, 0)

# Helpers for Mul/Add splitting (matching SymEngine internal representation)
function _split_mul_coef_bases(m::SMul)
    if !isempty(m.args) && m.args[1] isa SConst
        return m.args[1].val, m.args[2:end]
    end
    return one(ComplexF64), m.args
end

function _split_add_coef_terms(a::SAdd)
    coef = zero(ComplexF64)
    terms = SExpr[]
    for arg in a.args
        if arg isa SConst
            coef += arg.val
        else
            push!(terms, arg)
        end
    end
    return coef, terms
end

function _sexpr_cmp_number(a::ComplexF64, b::ComplexF64)::Int
    # SymEngine compares numbers by __cmp__ which orders Integer < Rational < Complex etc.
    # For same-type integers: compare by value
    # For ComplexF64: compare real first, then imag
    ra, rb = real(a), real(b)
    ra != rb && return ra < rb ? -1 : 1
    ia, ib = imag(a), imag(b)
    ia != ib && return ia < ib ? -1 : 1
    return 0
end

# Convenience: sort key that uses _sexpr_cmp for sort()
_sexpr_lt(a::SExpr, b::SExpr)::Bool = _sexpr_cmp(a, b) < 0

"""
Extract the "base expression" of an Add term, stripping the leading coefficient.
SymEngine's Add stores terms as dict{base: coef}, so ordering is by base only.
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

function _add_term_coeff(e::SExpr)::ComplexF64
    if e isa SMul && !isempty(e.args) && e.args[1] isa SConst
        return e.args[1].val
    end
    return one(ComplexF64)
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
Canonicalize an Add expression matching SymEngine's internal representation.

SymEngine's Add stores: `coef_ + dict_{base_expr → numeric_coeff}`.
The dict is a `umap_basic_num` (unordered), but `get_args()` iterates it
in the hash bucket order. To match, we:
1. Flatten nested Adds and collect constants into coef
2. Decompose each term into (base_expr, numeric_coeff) pairs
3. Group by base, summing coefficients (like SymEngine's canonicalization)
4. Sort by base using `_sexpr_cmp` (matching `map_basic_num` ordered iteration)
5. Reconstruct: constant first (if non-zero), then coeff*base for each pair
"""
function _canonical_add(args::Vector{SExpr})::SExpr
    flat_args = SExpr[]
    const_sum = Ref(zero(ComplexF64))
    for arg in args
        _flatten_add_arg!(flat_args, const_sum, arg)
    end

    # Build dict{base → coeff} matching SymEngine's Add internal representation
    # Each term is decomposed: SMul([c, base...]) → base=SMul(base...) or base...[1], coeff=c
    # Non-Mul terms: base=term, coeff=1
    dict = Dict{SExpr, ComplexF64}()
    for term in flat_args
        base = _add_term_base(term)
        coeff = _add_term_coeff(term)
        dict[base] = get(dict, base, zero(ComplexF64)) + coeff
    end

    # Sort bases by __cmp__ (matching SymEngine's map_basic_num key ordering)
    sorted_bases = sort!(collect(keys(dict)); lt = _sexpr_lt)

    # Reconstruct args: constant first, then coeff*base for each entry
    result = SExpr[]
    if !iszero(const_sum[])
        push!(result, SConst(const_sum[]))
    end
    for base in sorted_bases
        c = dict[base]
        iszero(c) && continue
        if isone(c)
            push!(result, base)
        else
            # Reconstruct as SMul([coeff, base_factors...])
            if base isa SMul
                push!(result, SMul(SExpr[SConst(c), base.args...]))
            else
                push!(result, SMul(SExpr[SConst(c), base]))
            end
        end
    end

    if isempty(result)
        return SConst(zero(ComplexF64))
    elseif length(result) == 1
        return result[1]
    else
        return SAdd(result)
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
        # Track whether this is a "real integer" coefficient (like SymEngine's Integer type)
        # vs a complex one (like SymEngine's ComplexDouble). This matters for is_minus_one().
        coeff_is_real_int = raw_coeff isa Real && isinteger(raw_coeff)
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
            push!(terms, SConst(coeff, coeff_is_real_int))
        elseif coeff == one(ComplexF64)
            # Monomial with coefficient 1: just the factors
            push!(terms, length(factors) == 1 ? factors[1] : SMul(factors))
        elseif coeff == -one(ComplexF64)
            # Represent -1 as a Mul coefficient so negative-Mul handling
            # matches SymEngine's bvisit(const Mul&).
            pushfirst!(factors, SConst(coeff, coeff_is_real_int))
            push!(terms, SMul(factors))
        else
            # General coefficient: SMul([coeff, factor1, factor2, ...])
            # Matches SymEngine's Mul(coef, {base: exp, ...}).get_args()
            pushfirst!(factors, SConst(coeff, coeff_is_real_int))
            push!(terms, SMul(factors))
        end
    end

    if isempty(terms)
        return SConst(zero(ComplexF64))
    elseif length(terms) == 1
        return terms[1]
    else
        # Sort terms by base expression (without coefficient) to match
        # SymEngine's Add dict{base→coeff} key ordering
        sort!(terms; lt = (a, b) -> _sexpr_cmp(_add_term_base(a), _add_term_base(b)) < 0)
        return SAdd(terms)
    end
end

## ── Phase 1: opt_cse ────────────────────────────────────────────────────────

# ── FuncArgTracker ──────────────────────────────────────────────────────────
# Direct translation of SymEngine's FuncArgTracker class.

mutable struct FuncArgTracker
    const value_numbers::Dict{SExpr, UInt32}
    const value_number_to_value::Vector{SExpr}
    const arg_to_funcset::Vector{Set{UInt32}}
    const func_to_argset::Vector{Set{UInt32}}
end

function FuncArgTracker(
        funcs::Vector{Pair{SExpr, Vector{SExpr}}},
    )::FuncArgTracker
    value_numbers = Dict{SExpr, UInt32}()
    value_number_to_value = SExpr[]
    arg_to_funcset = Set{UInt32}[]
    func_to_argset = Set{UInt32}[]

    # Pre-allocate arg_to_funcset entries
    resize!(arg_to_funcset, length(funcs))
    for i in eachindex(arg_to_funcset)
        arg_to_funcset[i] = Set{UInt32}()
    end

    for (func_i, (_, func_args)) in enumerate(funcs)
        func_argset = Set{UInt32}()
        for func_arg in func_args
            arg_number = _get_or_add_value_number!(
                value_numbers, value_number_to_value, arg_to_funcset, func_arg,
            )
            push!(func_argset, arg_number)
            push!(arg_to_funcset[arg_number], UInt32(func_i))
        end
        push!(func_to_argset, func_argset)
    end

    return FuncArgTracker(value_numbers, value_number_to_value, arg_to_funcset, func_to_argset)
end

function _get_or_add_value_number!(
        value_numbers::Dict{SExpr, UInt32},
        value_number_to_value::Vector{SExpr},
        arg_to_funcset::Vector{Set{UInt32}},
        value::SExpr,
    )::UInt32
    existing = get(value_numbers, value, UInt32(0))
    if existing != UInt32(0)
        return existing
    end
    push!(value_number_to_value, value)
    push!(arg_to_funcset, Set{UInt32}())
    vn = UInt32(length(value_number_to_value))
    value_numbers[value] = vn
    return vn
end

"""
Get values in sorted index order.
C++ std::set iterates in sorted order; Julia Set does not.
We sort explicitly to match SymEngine behavior.
"""
function _get_args_in_value_order(
        tracker::FuncArgTracker,
        argset::Union{Set{UInt32}, Vector{UInt32}},
    )::Vector{SExpr}
    sorted = sort!(collect(UInt32, argset))
    return SExpr[tracker.value_number_to_value[i] for i in sorted]
end

function _stop_arg_tracking!(tracker::FuncArgTracker, func_i::UInt32)::Nothing
    for arg in tracker.func_to_argset[func_i]
        delete!(tracker.arg_to_funcset[arg], func_i)
    end
    return nothing
end

"""
Find other functions sharing >= 2 arguments with the given argset.
Only considers functions with index > min_func_i.
Returns Dict(func_index => count_of_shared_args).
Corresponds to SymEngine's FuncArgTracker::get_common_arg_candidates.
"""
function _get_common_arg_candidates(
        tracker::FuncArgTracker,
        argset::Set{UInt32},
        min_func_i::UInt32,
    )::Dict{UInt32, UInt32}
    count_map = Dict{UInt32, UInt32}()

    # SymEngine sorts funcsets by size for performance.
    # We collect and sort similarly.
    funcsets = [tracker.arg_to_funcset[arg] for arg in argset]
    sort!(funcsets; by = length)

    for funcset in funcsets
        for func_i in funcset
            func_i > min_func_i || continue
            count_map[func_i] = get(count_map, func_i, UInt32(0)) + UInt32(1)
        end
    end

    # Keep only entries with count >= 2
    filter!(p -> p.second >= UInt32(2), count_map)
    return count_map
end

"""
Find functions in restrict_to_funcset that have ALL arguments in argset.
Corresponds to SymEngine's FuncArgTracker::get_subset_candidates.
"""
function _get_subset_candidates(
        tracker::FuncArgTracker,
        argset::Vector{UInt32},
        restrict_to_funcset,
    )::Vector{UInt32}
    indices = sort!(collect(UInt32, restrict_to_funcset))
    for arg in argset
        new_indices = UInt32[]
        for idx in indices
            if idx in tracker.arg_to_funcset[arg]
                push!(new_indices, idx)
            end
        end
        indices = new_indices
    end
    return indices
end

function _update_func_argset!(
        tracker::FuncArgTracker,
        func_i::UInt32,
        new_args::Vector{UInt32},
    )::Nothing
    old_args = tracker.func_to_argset[func_i]
    new_set = Set{UInt32}(new_args)

    for arg in old_args
        if !(arg in new_set)
            delete!(tracker.arg_to_funcset[arg], func_i)
        end
    end

    for arg in new_set
        if !(arg in old_args)
            push!(tracker.arg_to_funcset[arg], func_i)
        end
    end

    tracker.func_to_argset[func_i] = new_set
    return nothing
end

"""Insert `number` into a sorted vector if not already present."""
function _add_to_sorted_vec!(vec::Vector{UInt32}, number::UInt32)::Nothing
    idx = searchsortedfirst(vec, number)
    if idx > length(vec) || vec[idx] != number
        insert!(vec, idx, number)
    end
    return nothing
end

# ── match_common_args! ──────────────────────────────────────────────────────
# Direct translation of SymEngine's match_common_args function.

function match_common_args!(
        func_class::String,
        funcs_::Vector{SExpr},
        opt_subs::Dict{SExpr, SExpr},
    )::Nothing
    isempty(funcs_) && return nothing

    # Build (expression, args) pairs.
    # SymEngine: funcs comes from set_as_vec(muls/adds) which iterates set_basic
    # in __cmp__ order. Then std::sort by arg count (NOT stable in C++, but the
    # input is already __cmp__-sorted, so same-size groups preserve __cmp__ order
    # in practice on most implementations).
    # We sort by __cmp__ first, then stable-sort by arg count to match.
    funcs = Pair{SExpr, Vector{SExpr}}[e => _get_args(e) for e in funcs_]
    sort!(funcs; lt = (a, b) -> _sexpr_lt(a.first, b.first))
    sort!(funcs; by = p -> length(p.second), alg = Base.Sort.MergeSort)

    tracker = FuncArgTracker(funcs)

    changed = Set{UInt32}()

    for i_raw in 1:length(funcs)
        i = UInt32(i_raw)

        candidates_counts = _get_common_arg_candidates(
            tracker, tracker.func_to_argset[i], i,
        )

        # Sort candidates by match count (ascending), then by index
        # "This makes us try combining smaller matches first." — SymEngine
        candidates = sort!(
            collect(keys(candidates_counts));
            by = j -> (candidates_counts[j], j),
        )

        ci = 1
        while ci <= length(candidates)
            j = candidates[ci]
            ci += 1

            # Intersect arg sets
            com_args = sort!(collect(intersect(tracker.func_to_argset[i], tracker.func_to_argset[j])))

            length(com_args) >= 2 || continue

            diff_i = sort!(collect(setdiff(tracker.func_to_argset[i], com_args)))

            local com_func_number::UInt32

            if !isempty(diff_i)
                # "com_func needs to be unevaluated to allow for recursive matches."
                com_func = SFuncSym(func_class, _get_args_in_value_order(tracker, com_args))
                com_func_number = _get_or_add_value_number!(
                    tracker.value_numbers, tracker.value_number_to_value,
                    tracker.arg_to_funcset, com_func,
                )
                _add_to_sorted_vec!(diff_i, com_func_number)
                _update_func_argset!(tracker, i, diff_i)
                push!(changed, i)
            else
                # "Treat the whole expression as a CSE."
                # SymEngine comment: "The reason this needs to be done is somewhat
                # subtle. Within tree_cse(), to_eliminate only contains expressions
                # that are seen more than once. The problem is unevaluated expressions
                # do not compare equal to the evaluated equivalent. So tree_cse()
                # won't mark funcs[i] as a CSE if we use an unevaluated version."
                com_func_number = _get_or_add_value_number!(
                    tracker.value_numbers, tracker.value_number_to_value,
                    tracker.arg_to_funcset, funcs[i].first,
                )
            end

            diff_j = sort!(collect(setdiff(tracker.func_to_argset[j], com_args)))
            _add_to_sorted_vec!(diff_j, com_func_number)
            _update_func_argset!(tracker, j, diff_j)
            push!(changed, j)

            # Also update all subset candidates
            for k in _get_subset_candidates(tracker, com_args, candidates[ci:end])
                diff_k = sort!(collect(setdiff(tracker.func_to_argset[k], com_args)))
                _add_to_sorted_vec!(diff_k, com_func_number)
                _update_func_argset!(tracker, k, diff_k)
                push!(changed, k)
            end
        end

        if i in changed
            opt_subs[funcs[i].first] = SFuncSym(
                func_class,
                _get_args_in_value_order(tracker, tracker.func_to_argset[i]),
            )
        end
        _stop_arg_tracking!(tracker, i)
    end

    return nothing
end

# ── OptsCSEVisitor ──────────────────────────────────────────────────────────
# Direct translation of SymEngine's OptsCSEVisitor class.

"""
    _opts_cse_visit!(expr, adds, muls, opt_subs, seen)

Visitor that collects Add/Mul nodes and handles negative coefficients/exponents.
Direct translation of SymEngine's OptsCSEVisitor.
"""
function _opts_cse_visit!(
        expr::SExpr,
        adds::Set{SExpr},
        muls::Set{SExpr},
        opt_subs::Dict{SExpr, SExpr},
        seen::Set{SExpr},
    )::Nothing
    expr in seen && return nothing
    push!(seen, expr)

    if expr isa SAdd
        # bvisit(const Add &x)
        for a in expr.args
            _opts_cse_visit!(a, adds, muls, opt_subs, seen)
        end
        push!(adds, expr)

    elseif expr isa SMul
        # bvisit(const Mul &x)
        for a in expr.args
            _opts_cse_visit!(a, adds, muls, opt_subs, seen)
        end
        # Check for negative coefficient
        # SymEngine: if (x.get_coef()->is_negative())
        # IMPORTANT: SymEngine's is_negative() returns true ONLY for real negative
        # numbers (Integer, Rational, RealDouble), NEVER for Complex.
        # So we must check: imaginary part is zero AND real part is negative.
        if !isempty(expr.args) && expr.args[1] isa SConst &&
                imag(expr.args[1].val) == 0 && real(expr.args[1].val) < 0
            neg_coeff = expr.args[1].val
            pos_coeff = -neg_coeff
            # Compute neg(expr): flip the coefficient
            if length(expr.args) == 2 && isone(pos_coeff)
                # neg(Mul(-1, x)) = x — SymEngine simplifies this
                neg_expr = expr.args[2]
            else
                if isone(pos_coeff)
                    pos_args = expr.args[2:end]
                else
                    pos_args = copy(expr.args)
                    pos_args[1] = SConst(pos_coeff)
                end
                neg_expr = length(pos_args) == 1 ? pos_args[1] : SMul(pos_args)
            end
            # SymEngine: if (not is_a<Symbol>(*neg_expr))
            # Skip when negation simplifies to an atom (like a variable)
            if !_is_atom(neg_expr)
                opt_subs[expr] = SFuncSym("mul", SExpr[SConst(-one(ComplexF64)), neg_expr])
                push!(seen, neg_expr)
                # SymEngine: expr = neg_expr; if (is_a<Mul>(*expr)) muls.insert(expr)
                # Note: using Set ensures no duplicates (matching SymEngine's set_basic)
                if neg_expr isa SMul
                    push!(muls, neg_expr)
                end
            else
                # neg_expr is an atom, treat original as regular Mul
                push!(muls, expr)
            end
        else
            push!(muls, expr)
        end

    elseif expr isa SPow
        # bvisit(const Pow &x)
        _opts_cse_visit!(expr.base, adds, muls, opt_subs, seen)
        # SymEngine: check if exponent is negative
        if expr.exp < 0
            # pow(base, -n) → FuncSym("pow", [pow(base, n), -1])
            opt_subs[expr] = SFuncSym("pow", SExpr[SPow(expr.base, -expr.exp), SConst(ComplexF64(-1))])
        end

    elseif expr isa SNeg
        # SNeg is our representation for SymEngine's Mul(-1, x) where neg simplifies to atom
        _opts_cse_visit!(expr.arg, adds, muls, opt_subs, seen)

    elseif expr isa SFuncSym
        # bvisit(const Basic &x) — generic case for compound expressions
        for a in expr.args
            _opts_cse_visit!(a, adds, muls, opt_subs, seen)
        end
    end
    # Atoms (SConst, SVar, SParam, STmp) — nothing to do
    return nothing
end

"""
    opt_cse(exprs) -> Dict{SExpr, SExpr}

Phase 1: find optimization opportunities in Add/Mul/Pow nodes.
Direct translation of SymEngine's opt_cse function.
"""
function opt_cse(exprs::Vector{SExpr})::Dict{SExpr, SExpr}
    opt_subs = Dict{SExpr, SExpr}()
    # SymEngine uses set_basic (ordered set) for adds/muls — ensures uniqueness
    adds = Set{SExpr}()
    muls = Set{SExpr}()
    seen = Set{SExpr}()

    for e in exprs
        _opts_cse_visit!(e, adds, muls, opt_subs, seen)
    end

    # Convert to vectors for match_common_args (SymEngine: set_as_vec)
    adds_vec = collect(SExpr, adds)
    muls_vec = collect(SExpr, muls)
    sort!(adds_vec; lt = _sexpr_lt)
    sort!(muls_vec; lt = _sexpr_lt)
    match_common_args!("add", adds_vec, opt_subs)
    match_common_args!("mul", muls_vec, opt_subs)

    return opt_subs
end

## ── Phase 2: tree_cse ───────────────────────────────────────────────────────

"""
    _find_repeated!(expr, seen, to_eliminate, opt_subs, excluded_symbols)

Walk expression tree, applying opt_subs, and mark expressions seen 2+ times.
Direct translation of SymEngine's find_repeated lambda in tree_cse.

Key logic (matching SymEngine exactly):
1. Skip atoms (Numbers)
2. Track symbols (for name collision avoidance)
3. If already seen → mark for elimination, return
4. Add to seen
5. Replace expr with opt_subs[expr] if present
6. Get args of (possibly replaced) expr and recurse into each
"""
function _find_repeated!(
        expr::SExpr,
        seen::Set{SExpr},
        to_eliminate::Set{SExpr},
        opt_subs::Dict{SExpr, SExpr},
        excluded_symbols::Set{SExpr},
    )::Nothing
    # SymEngine: if (is_a_Number(*expr) ...) return;
    if expr isa SConst
        return nothing
    end

    # SymEngine: if (is_a<Symbol>(*expr)) { excluded_symbols.insert(expr); }
    # Note: SymEngine does NOT return here — symbols can be CSE'd.
    # But for our use case, variables are direct slot reads in the interpreter,
    # so CSE-ing them would just add useless IDENTITY instructions.
    # We return early as an optimization.
    if expr isa SVar || expr isa SParam
        push!(excluded_symbols, expr)
    end

    if expr isa STmp
        return nothing
    end

    # SymEngine: if (seen_subexp.find(expr) != seen_subexp.end()) { to_eliminate.insert(expr); return; }
    if expr in seen
        push!(to_eliminate, expr)
        return nothing
    end

    # SymEngine: seen_subexp.insert(expr);
    push!(seen, expr)

    # SymEngine: auto iter = opt_subs.find(expr);
    #            if (iter != opt_subs.end()) { expr = iter->second; }
    # Replace expr with opt_subs version, then get args of REPLACED version
    actual = get(opt_subs, expr, expr)

    # SymEngine: vec_basic args = expr->get_args();
    #            for (auto &arg : args) { find_repeated(arg); }
    for arg in _get_args(actual)
        _find_repeated!(arg, seen, to_eliminate, opt_subs, excluded_symbols)
    end

    return nothing
end

"""
    _rebuild(orig_expr, to_eliminate, opt_subs, subs, replacements, next_id) -> SExpr

Rebuild expression tree, replacing repeated subexpressions with temporaries.
Direct translation of SymEngine's RebuildVisitor::apply.

Key logic (matching SymEngine exactly):
1. Atoms pass through
2. Check subs cache → return if found
3. Apply opt_subs (keep orig_expr for to_eliminate check)
4. Rebuild children of (possibly replaced) expression
5. If orig_expr in to_eliminate → create temporary, cache in subs
6. Return rebuilt expression
"""
function _rebuild(
        orig_expr::SExpr,
        to_eliminate::Set{SExpr},
        opt_subs::Dict{SExpr, SExpr},
        subs::Dict{SExpr, SExpr},
        replacements::Vector{Pair{SExpr, SExpr}},
        next_id::Base.RefValue{Int},
    )::SExpr
    # SymEngine: if (is_a_Atom(*expr)) return expr;
    _is_atom(orig_expr) && return orig_expr

    # SymEngine: auto iter = subs.find(expr); if (iter != subs.end()) return iter->second;
    sub = get(subs, orig_expr, nothing)
    sub !== nothing && return sub

    # SymEngine: auto iter2 = opt_subs.find(expr);
    #            if (iter2 != opt_subs.end()) { expr = iter2->second; }
    expr = get(opt_subs, orig_expr, orig_expr)

    # SymEngine: expr->accept(*this); auto new_expr = result_;
    # This calls the visitor which rebuilds children
    new_expr = _rebuild_children(expr, to_eliminate, opt_subs, subs, replacements, next_id)

    # SymEngine: if (to_eliminate.find(orig_expr) != to_eliminate.end())
    if orig_expr in to_eliminate
        next_id[] += 1
        tmp = STmp(next_id[])
        subs[orig_expr] = tmp
        push!(replacements, tmp => new_expr)
        return tmp
    end

    return new_expr
end

"""
Rebuild the children of an expression.
Corresponds to SymEngine's TransformVisitor + RebuildVisitor::bvisit(FunctionSymbol).
"""
function _rebuild_children(
        expr::SExpr,
        to_eliminate::Set{SExpr},
        opt_subs::Dict{SExpr, SExpr},
        subs::Dict{SExpr, SExpr},
        replacements::Vector{Pair{SExpr, SExpr}},
        next_id::Base.RefValue{Int},
    )::SExpr
    if _is_atom(expr)
        return expr
    elseif expr isa SAdd
        new_args = SExpr[_rebuild(a, to_eliminate, opt_subs, subs, replacements, next_id) for a in expr.args]
        return _canonical_add(new_args)
    elseif expr isa SMul
        new_args = SExpr[_rebuild(a, to_eliminate, opt_subs, subs, replacements, next_id) for a in expr.args]
        return _canonical_mul(new_args)
    elseif expr isa SPow
        new_base = _rebuild(expr.base, to_eliminate, opt_subs, subs, replacements, next_id)
        return SPow(new_base, expr.exp)
    elseif expr isa SNeg
        new_arg = _rebuild(expr.arg, to_eliminate, opt_subs, subs, replacements, next_id)
        return SNeg(new_arg)
    elseif expr isa SFuncSym
        # SymEngine's RebuildVisitor::bvisit(const FunctionSymbol &x)
        # Rebuild args, then evaluate the function symbol back to a real expression
        new_args = SExpr[_rebuild(a, to_eliminate, opt_subs, subs, replacements, next_id) for a in expr.args]
        if expr.name == "add"
            return _canonical_add(new_args)
        elseif expr.name == "mul"
            return _canonical_mul(new_args)
        elseif expr.name == "pow" && length(new_args) == 2
            # SymEngine: result_ = pow(newargs[0], newargs[1]);
            if new_args[2] isa SConst
                return SPow(new_args[1], Int(real(new_args[2].val)))
            end
            return SFuncSym(expr.name, new_args)
        else
            return SFuncSym(expr.name, new_args)
        end
    else
        return expr
    end
end

"""
    tree_cse(exprs, opt_subs) -> (replacements, reduced_exprs)

Phase 2: find and eliminate repeated subexpressions.
Direct translation of SymEngine's tree_cse function.
"""
function tree_cse(
        exprs::Vector{SExpr},
        opt_subs::Dict{SExpr, SExpr},
    )::Tuple{Vector{Pair{SExpr, SExpr}}, Vector{SExpr}}
    to_eliminate = Set{SExpr}()
    seen = Set{SExpr}()
    excluded_symbols = Set{SExpr}()

    for e in exprs
        _find_repeated!(e, seen, to_eliminate, opt_subs, excluded_symbols)
    end

    subs = Dict{SExpr, SExpr}()
    replacements = Pair{SExpr, SExpr}[]

    next_id = Ref(0)
    reduced_exprs = SExpr[]
    for e in exprs
        push!(reduced_exprs, _rebuild(e, to_eliminate, opt_subs, subs, replacements, next_id))
    end

    return replacements, reduced_exprs
end

## ── Main CSE entry point ────────────────────────────────────────────────────

"""
    cse(exprs::Vector{SExpr}) -> (replacements, reduced_exprs)

Run Common Subexpression Elimination on a list of expressions.
Direct translation of SymEngine's cse function.
"""
function cse(exprs::Vector{SExpr})::Tuple{Vector{Pair{SExpr, SExpr}}, Vector{SExpr}}
    # Phase 1: find optimization opportunities (common argument matching)
    opt_subs = opt_cse(exprs)

    # Phase 2: eliminate repeated subexpressions
    return tree_cse(exprs, opt_subs)
end

## ── SExpr → IR compilation ──────────────────────────────────────────────────

## ── SExpr → IR compilation (1:1 port of HC v2 intermediate_representation.jl) ─

"""
State for compiling SExpr trees to IR instructions.
Mirrors HC v2's IntermediateRepresentation builder.
"""
mutable struct SExprCompiler
    const ir_stmts::Vector{IRStatement}
    ref_counter::Int
    const constants_list::Vector{ComplexF64}
    const constants_map::Dict{ComplexF64, ComplexF64}
    # CSE temp id → definition (compiled lazily on first use)
    const cse_defs::Dict{Int, SExpr}
    # CSE temp id → compiled ref (memoized)
    const cse_compiled::Dict{Int, IRStatementArg}
    const var_syms::Vector{Symbol}
    const param_syms::Vector{Symbol}
end

function SExprCompiler(var_syms::Vector{Symbol}, param_syms::Vector{Symbol})
    return SExprCompiler(
        IRStatement[],
        0,
        ComplexF64[],
        Dict{ComplexF64, ComplexF64}(),
        Dict{Int, SExpr}(),
        Dict{Int, IRStatementArg}(),
        var_syms,
        param_syms,
    )
end

function _get_constant!(c::SExprCompiler, val::ComplexF64)::ComplexF64
    if !haskey(c.constants_map, val)
        push!(c.constants_list, val)
        c.constants_map[val] = val
    end
    return val
end

function _add_op!(c::SExprCompiler, op::OpType.T, args::IRStatementArg...)::IRStatementRef
    c.ref_counter += 1
    ref = IRStatementRef(c.ref_counter)
    push!(c.ir_stmts, IRStatement(op, ref, args...))
    return ref
end

## ── Arithmetic helpers (exact port of HC v2 add!/neg!/sub!/mul!/etc.) ─────

function _ir_add!(c::SExprCompiler, @nospecialize(a), @nospecialize(b))
    if isnothing(a)
        isnothing(b) ? nothing : b
    else
        isnothing(b) ? a : _add_op!(c, OpType.OP_ADD, a, b)
    end
end

function _ir_neg!(c::SExprCompiler, @nospecialize(a))
    isnothing(a) ? nothing : _add_op!(c, OpType.OP_NEG, a)
end

function _ir_sub!(c::SExprCompiler, @nospecialize(a), @nospecialize(b))
    if isnothing(a)
        isnothing(b) ? nothing : _add_op!(c, OpType.OP_NEG, b)
    else
        isnothing(b) ? a : _add_op!(c, OpType.OP_SUB, a, b)
    end
end

_is_one(a::ComplexF64)::Bool = a == one(ComplexF64)
_is_one(a::IRStatementRef)::Bool = false
_is_one(::Nothing)::Bool = false
_is_one(::Symbol)::Bool = false

_is_minus_one(a::ComplexF64)::Bool = a == -one(ComplexF64)
_is_minus_one(a::IRStatementRef)::Bool = false
_is_minus_one(::Nothing)::Bool = false
_is_minus_one(::Symbol)::Bool = false

_is_zero(a::ComplexF64)::Bool = iszero(a)
_is_zero(::IRStatementRef)::Bool = false
_is_zero(::Nothing)::Bool = false
_is_zero(::Symbol)::Bool = false

function _ir_mul!(c::SExprCompiler, @nospecialize(a), @nospecialize(b))
    (isnothing(a) || isnothing(b)) && return nothing
    if _is_one(a)
        return b
    elseif _is_one(b)
        return a
    elseif _is_minus_one(a)
        return _add_op!(c, OpType.OP_NEG, b)
    elseif _is_minus_one(b)
        return _add_op!(c, OpType.OP_NEG, a)
    elseif a isa ComplexF64 && a == ComplexF64(2)
        return _add_op!(c, OpType.OP_ADD, b, b)
    end
    return _add_op!(c, OpType.OP_MUL, a, b)
end

function _ir_muladd!(c::SExprCompiler, @nospecialize(a), @nospecialize(b), @nospecialize(d))
    (isnothing(a) || isnothing(b)) && return d
    isnothing(d) && return _ir_mul!(c, a, b)
    _is_one(a) && return _ir_add!(c, b, d)
    _is_one(b) && return _ir_add!(c, a, d)
    return _add_op!(c, OpType.OP_MULADD, a, b, d)
end


function _ir_mulmuladd!(c::SExprCompiler, a, b, d, e)
    return _add_op!(c, OpType.OP_MULMULADD, a, b, d, e)
end

function _ir_div!(c::SExprCompiler, @nospecialize(a), @nospecialize(b))
    isnothing(a) && return nothing
    _is_one(b) ? a : _add_op!(c, OpType.OP_DIV, a, b)
end

function _ir_sqr!(c::SExprCompiler, @nospecialize(a))
    isnothing(a) && return nothing
    return _add_op!(c, OpType.OP_SQR, a)
end

function _ir_pow!(c::SExprCompiler, @nospecialize(a), k::Int)
    k == 0 && return (_get_constant!(c, one(ComplexF64)); one(ComplexF64))
    k == 1 && return a
    k == 2 && return _ir_sqr!(c, a)
    k == 3 && return _add_op!(c, OpType.OP_CB, a)
    k == -1 && return _add_op!(c, OpType.OP_INV, a)
    k == -2 && return _add_op!(c, OpType.OP_INVSQR, a)
    return _add_op!(c, OpType.OP_POW_INT, a, ComplexF64(k))
end

## ── Main dispatcher (mirrors HC v2's expr_to_ir_statements!) ──────────────

function _sexpr_to_ir!(c::SExprCompiler, expr::SExpr)::IRStatementArg
    if expr isa SConst
        _get_constant!(c, expr.val)
        return expr.val
    elseif expr isa SVar
        return c.var_syms[expr.idx]
    elseif expr isa SParam
        return c.param_syms[expr.idx]
    elseif expr isa STmp
        cached = get(c.cse_compiled, expr.id, nothing)
        cached !== nothing && return cached
        val = _sexpr_to_ir!(c, c.cse_defs[expr.id])
        if val isa IRStatementRef
            c.cse_compiled[expr.id] = val
        end
        return val
    elseif expr isa SPow
        base = _sexpr_to_ir!(c, expr.base)
        return _ir_pow!(c, base, expr.exp)
    elseif expr isa SMul
        return _process_mul!(c, expr)
    elseif expr isa SAdd
        return _process_sum!(c, expr)
    elseif expr isa SNeg
        return _ir_neg!(c, _sexpr_to_ir!(c, expr.arg))
    elseif expr isa SFuncSym
        if expr.name == "add"
            return _process_sum!(c, SAdd(expr.args))
        elseif expr.name == "mul"
            return _process_mul!(c, SMul(expr.args))
        else
            error("Unknown SFuncSym: $(expr.name)")
        end
    else
        error("Unknown SExpr type: $(typeof(expr))")
    end
end

## ── Mul processing (mirrors HC v2's process_mul! + split_into_num_denom! + prod_parts!) ─

function _split_off_minus_one(expr::SExpr)::Tuple{Int, SExpr}
    if expr isa SMul && !isempty(expr.args) && expr.args[1] isa SConst
        c = expr.args[1]
        # SymEngine's is_minus_one only returns true for Integer(-1), not ComplexDouble(-1+0i).
        # We match this by checking the is_real_int flag.
        if c.val == -one(ComplexF64) && c.is_real_int
            rest = expr.args[2:end]
            return -1, length(rest) == 1 ? rest[1] : SMul(rest)
        end
    end
    return 1, expr
end

function _split_into_num_denom!(c::SExprCompiler, expr::SMul)
    nums = Any[]
    denoms = Any[]
    for arg in expr.args
        if arg isa SPow && arg.exp < 0
            push!(denoms, _ir_pow!(c, _sexpr_to_ir!(c, arg.base), -arg.exp))
        elseif arg isa SPow
            push!(nums, _ir_pow!(c, _sexpr_to_ir!(c, arg.base), arg.exp))
        else
            push!(nums, _sexpr_to_ir!(c, arg))
        end
    end
    return nums, denoms
end

function _prod_parts!(c::SExprCompiler, exs::Vector)
    isempty(exs) && return nothing
    # Multiply from reverse to detect 2*x (constants are in front)
    parts = reverse(Any[e for e in exs])
    while length(parts) > 1
        if length(parts) >= 4
            a = pop!(parts); b = pop!(parts); d = pop!(parts); e = pop!(parts)
            push!(parts, _add_op!(c, OpType.OP_MUL4, a, b, d, e))
        elseif length(parts) == 3
            a = pop!(parts); b = pop!(parts); d = pop!(parts)
            push!(parts, _add_op!(c, OpType.OP_MUL3, a, b, d))
        elseif length(parts) == 2
            a = pop!(parts); b = pop!(parts)
            push!(parts, _ir_mul!(c, a, b))
        end
    end
    return parts[1]
end

function _process_mul!(c::SExprCompiler, expr::SMul)::IRStatementArg
    (m, expr2) = _split_off_minus_one(expr)
    if !(expr2 isa SMul)
        return _ir_mul!(c, ComplexF64(m), _sexpr_to_ir!(c, expr2))
    end
    nums, denoms = _split_into_num_denom!(c, expr2)
    num_prod = _prod_parts!(c, nums)
    denom_prod = _prod_parts!(c, denoms)
    if isnothing(num_prod)
        ref = _add_op!(c, OpType.OP_INV, denom_prod)
    elseif isnothing(denom_prod)
        ref = num_prod
    else
        ref = _ir_div!(c, num_prod, denom_prod)
    end
    return _ir_mul!(c, ComplexF64(m), ref)
end

## ── Add processing (mirrors HC v2's process_sum! + reduce_to_at_most_two_multiplicants! + sum_products!) ─

function _split_into_positives_negatives(expr::SAdd)
    positives = SExpr[]
    negatives = SExpr[]
    for arg in expr.args
        (sign, val) = _split_off_minus_one(arg)
        if sign == -1
            push!(negatives, val)
        else
            push!(positives, val)
        end
    end
    return positives, negatives
end

function _reduce_to_at_most_two_multiplicants!(c::SExprCompiler, expr::SExpr)
    if expr isa SMul
        args = expr.args
        if length(args) == 2
            return (
                _sexpr_to_ir!(c, args[1]),
                _sexpr_to_ir!(c, args[2]),
            )
        elseif length(args) == 1
            return (_sexpr_to_ir!(c, args[1]), nothing)
        elseif length(args) > 2
            prefix = SMul(args[1:(end - 1)])
            v2 = _sexpr_to_ir!(c, prefix)
            return (_sexpr_to_ir!(c, args[end]), v2)
        end
    end
    return (_sexpr_to_ir!(c, expr), nothing)
end

function _sum_products!(c::SExprCompiler, tuples::Vector)
    isempty(tuples) && return nothing

    singles = IRStatementArg[first(t) for t in tuples if isnothing(t[2])]
    pairs = [t for t in tuples if !isnothing(t[2])]
    n = length(pairs)

    for k in 1:2:(n - 1)
        (a, b) = pairs[k]
        (d, e) = pairs[k + 1]
        push!(singles, _ir_mulmuladd!(c, a, b, d, e))
    end

    if isodd(n)
        if isempty(singles)
            return _ir_mul!(c, pairs[n][1], pairs[n][2])
        end
        (a, b) = pairs[n]
        d = pop!(singles)
        push!(singles, _ir_muladd!(c, a, b, d))
    end

    while length(singles) > 1
        if length(singles) >= 4
            a = pop!(singles); b = pop!(singles); d = pop!(singles); e = pop!(singles)
            push!(singles, _add_op!(c, OpType.OP_ADD4, a, b, d, e))
        elseif length(singles) == 3
            a = pop!(singles); b = pop!(singles); d = pop!(singles)
            push!(singles, _add_op!(c, OpType.OP_ADD3, a, b, d))
        elseif length(singles) == 2
            a = pop!(singles); b = pop!(singles)
            push!(singles, _add_op!(c, OpType.OP_ADD, a, b))
        end
    end
    return singles[1]
end

function _process_sum!(c::SExprCompiler, expr::SAdd)::IRStatementArg
    pos, neg = _split_into_positives_negatives(expr)
    pos_reduced = [_reduce_to_at_most_two_multiplicants!(c, e) for e in pos]
    neg_reduced = [_reduce_to_at_most_two_multiplicants!(c, e) for e in neg]

    if length(pos_reduced) == 1 && length(neg_reduced) == 1
        (a, b) = pos_reduced[1]
        (d, e) = neg_reduced[1]
        if !isnothing(b) && !isnothing(e)
            return _add_op!(c, OpType.OP_MULMULSUB, a, b, d, e)
        elseif !isnothing(b)
            return _add_op!(c, OpType.OP_MULSUB, a, b, d)
        elseif !isnothing(e)
            return _add_op!(c, OpType.OP_SUBMUL, d, e, a)
        else
            return _add_op!(c, OpType.OP_SUB, a, d)
        end
    elseif length(neg_reduced) == 1
        a = _sum_products!(c, pos_reduced)
        (d, e) = neg_reduced[1]
        if !isnothing(e)
            return _add_op!(c, OpType.OP_SUBMUL, d, e, a)
        else
            return _add_op!(c, OpType.OP_SUB, a, d)
        end
    end

    pos_sum = _sum_products!(c, pos_reduced)
    neg_sum = _sum_products!(c, neg_reduced)
    return _ir_sub!(c, pos_sum, neg_sum)
end

## ── Full compilation pipeline ───────────────────────────────────────────────

"""
    compile_cse_to_ir(replacements, reduced_exprs, var_syms, param_syms)
        -> (ir_stmts, constants_list, result_refs)

Compile CSE output to IR instructions. 1:1 port of HC v2 logic.
"""
function compile_cse_to_ir(
        replacements::Vector{Pair{SExpr, SExpr}},
        reduced_exprs::Vector{SExpr},
        var_syms::Vector{Symbol},
        param_syms::Vector{Symbol},
    )::Tuple{Vector{IRStatement}, Vector{ComplexF64}, Vector{IRStatementArg}}
    compiler = SExprCompiler(var_syms, param_syms)

    # Register CSE definitions (compiled lazily on first use, like HC v2's pse)
    for (tmp, definition) in replacements
        @assert tmp isa STmp
        compiler.cse_defs[tmp.id] = definition
    end

    # Compile reduced expressions
    result_refs = IRStatementArg[]
    for expr in reduced_exprs
        push!(result_refs, _sexpr_to_ir!(compiler, expr))
    end

    return compiler.ir_stmts, compiler.constants_list, result_refs
end
