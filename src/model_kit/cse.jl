## CSE (Common Subexpression Elimination) — SymEngine algorithm in Julia
#
# A 1:1 port of SymEngine's cse.cpp two-phase algorithm:
#   Phase 1 (opt_cse): factor common arguments in Add/Mul nodes
#   Phase 2 (tree_cse): eliminate repeated subexpressions
#
# Reference: https://github.com/symengine/symengine/blob/master/symengine/cse.cpp

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

## ── Phase 1: opt_cse ────────────────────────────────────────────────────────

# ── FuncArgTracker ──────────────────────────────────────────────────────────
# Direct translation of SymEngine's FuncArgTracker class.

struct FuncArgTracker
    value_numbers::Dict{SExpr, UInt32}
    value_number_to_value::Vector{SExpr}
    arg_to_funcset::Vector{Set{UInt32}}
    func_to_argset::Vector{Set{UInt32}}
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

## ── SExpr → Instruction compilation (TapeCompiler) ──────────────────────────

"""
State for compiling SExpr trees directly to `Instruction` values with tape indices.
"""
mutable struct TapeCompiler
    const instructions::Vector{Instruction}
    const constants::Vector{ComplexF64}
    const constants_map::Dict{ComplexF64, Int32}  # value → tape slot
    const var_slots::Vector{Int32}                 # var index → tape slot
    const param_slots::Vector{Int32}               # param index → tape slot
    const cse_defs::Dict{Int, SExpr}               # STmp id → definition
    const cse_slots::Dict{Int, Int32}              # STmp id → tape slot (memoized)
    next_slot::Int32
    # Known constant slots for optimization
    one_slot::Int32
    minus_one_slot::Int32
    two_slot::Int32
end

const _SLOT_NONE = Int32(0)

function TapeCompiler(nvars::Int, nparams::Int)
    return TapeCompiler(
        Instruction[],
        ComplexF64[],
        Dict{ComplexF64, Int32}(),
        Vector{Int32}(undef, nvars),
        Vector{Int32}(undef, nparams),
        Dict{Int, SExpr}(),
        Dict{Int, Int32}(),
        Int32(0),
        _SLOT_NONE,
        _SLOT_NONE,
        _SLOT_NONE,
    )
end

## ── Core helpers ────────────────────────────────────────────────────────────

"""Register a constant and return its tape slot (1-based, relative to constants block)."""
function _get_constant_slot!(c::TapeCompiler, val::ComplexF64)::Int32
    slot = get(c.constants_map, val, _SLOT_NONE)
    slot != _SLOT_NONE && return slot
    push!(c.constants, val)
    slot = Int32(length(c.constants))
    c.constants_map[val] = slot
    if val == one(ComplexF64)
        c.one_slot = slot
    elseif val == -one(ComplexF64)
        c.minus_one_slot = slot
    elseif val == ComplexF64(2)
        c.two_slot = slot
    end
    return slot
end

"""Emit an instruction and return the output tape slot."""
function _emit!(c::TapeCompiler, op::OpType.T, a1::Int32)::Int32
    c.next_slot += Int32(1)
    slot = c.next_slot
    push!(c.instructions, Instruction((a1, a1, a1, a1), op, slot))
    return slot
end

function _emit!(c::TapeCompiler, op::OpType.T, a1::Int32, a2::Int32)::Int32
    c.next_slot += Int32(1)
    slot = c.next_slot
    push!(c.instructions, Instruction((a1, a2, a2, a2), op, slot))
    return slot
end

function _emit!(c::TapeCompiler, op::OpType.T, a1::Int32, a2::Int32, a3::Int32)::Int32
    c.next_slot += Int32(1)
    slot = c.next_slot
    push!(c.instructions, Instruction((a1, a2, a3, a3), op, slot))
    return slot
end

function _emit!(
        c::TapeCompiler, op::OpType.T, a1::Int32, a2::Int32, a3::Int32, a4::Int32,
    )::Int32
    c.next_slot += Int32(1)
    slot = c.next_slot
    push!(c.instructions, Instruction((a1, a2, a3, a4), op, slot))
    return slot
end

## ── Slot-based predicates ───────────────────────────────────────────────────

_is_one_slot(c::TapeCompiler, s::Int32)::Bool =
    c.one_slot != _SLOT_NONE && s == c.one_slot
_is_minus_one_slot(c::TapeCompiler, s::Int32)::Bool =
    c.minus_one_slot != _SLOT_NONE && s == c.minus_one_slot
_is_two_slot(c::TapeCompiler, s::Int32)::Bool =
    c.two_slot != _SLOT_NONE && s == c.two_slot

## ── Arithmetic helpers ──────────────────────────────────────────────────────

function _tape_add!(c::TapeCompiler, a::Int32, b::Int32)::Int32
    return _emit!(c, OpType.OP_ADD, a, b)
end

function _tape_neg!(c::TapeCompiler, a::Int32)::Int32
    return _emit!(c, OpType.OP_NEG, a)
end

function _tape_sub!(c::TapeCompiler, a::Int32, b::Int32)::Int32
    return _emit!(c, OpType.OP_SUB, a, b)
end

function _tape_mul!(c::TapeCompiler, a::Int32, b::Int32)::Int32
    _is_one_slot(c, a) && return b
    _is_one_slot(c, b) && return a
    _is_minus_one_slot(c, a) && return _emit!(c, OpType.OP_NEG, b)
    _is_minus_one_slot(c, b) && return _emit!(c, OpType.OP_NEG, a)
    _is_two_slot(c, a) && return _emit!(c, OpType.OP_ADD, b, b)
    return _emit!(c, OpType.OP_MUL, a, b)
end

function _tape_muladd!(c::TapeCompiler, a::Int32, b::Int32, d::Int32)::Int32
    _is_one_slot(c, a) && return _tape_add!(c, b, d)
    _is_one_slot(c, b) && return _tape_add!(c, a, d)
    return _emit!(c, OpType.OP_MULADD, a, b, d)
end

function _tape_mulmuladd!(
        c::TapeCompiler, a::Int32, b::Int32, d::Int32, e::Int32,
    )::Int32
    return _emit!(c, OpType.OP_MULMULADD, a, b, d, e)
end

function _tape_div!(c::TapeCompiler, a::Int32, b::Int32)::Int32
    return _is_one_slot(c, b) ? a : _emit!(c, OpType.OP_DIV, a, b)
end

function _tape_sqr!(c::TapeCompiler, a::Int32)::Int32
    return _emit!(c, OpType.OP_SQR, a)
end

function _tape_pow!(c::TapeCompiler, a::Int32, k::Int)::Int32
    if k == 0
        return _get_constant_slot!(c, one(ComplexF64))
    elseif k == 1
        return a
    elseif k == 2
        return _tape_sqr!(c, a)
    elseif k == 3
        return _emit!(c, OpType.OP_CB, a)
    elseif k == -1
        return _emit!(c, OpType.OP_INV, a)
    elseif k == -2
        return _emit!(c, OpType.OP_INVSQR, a)
    else
        # OP_POW_INT: second arg is the integer exponent stored directly
        c.next_slot += Int32(1)
        slot = c.next_slot
        push!(
            c.instructions,
            Instruction((a, Int32(k), Int32(k), Int32(k)), OpType.OP_POW_INT, slot),
        )
        return slot
    end
end

## ── Main dispatcher ─────────────────────────────────────────────────────────

"""Compile an SExpr to a tape slot, returning the Int32 slot index."""
function _compile!(c::TapeCompiler, expr::SExpr)::Int32
    if expr isa SConst
        return _get_constant_slot!(c, expr.val)
    elseif expr isa SVar
        return c.var_slots[expr.idx]
    elseif expr isa SParam
        return c.param_slots[expr.idx]
    elseif expr isa STmp
        cached = get(c.cse_slots, expr.id, _SLOT_NONE)
        cached != _SLOT_NONE && return cached
        slot = _compile!(c, c.cse_defs[expr.id])
        c.cse_slots[expr.id] = slot
        return slot
    elseif expr isa SPow
        base = _compile!(c, expr.base)
        return _tape_pow!(c, base, expr.exp)
    elseif expr isa SMul
        return _compile_mul!(c, expr)
    elseif expr isa SAdd
        return _compile_sum!(c, expr)
    elseif expr isa SNeg
        return _tape_neg!(c, _compile!(c, expr.arg))
    elseif expr isa SFuncSym
        if expr.name == "add"
            return _compile_sum!(c, SAdd(expr.args))
        elseif expr.name == "mul"
            return _compile_mul!(c, SMul(expr.args))
        else
            error("Unknown SFuncSym: $(expr.name)")
        end
    else
        error("Unknown SExpr type: $(typeof(expr))")
    end
end

## ── Mul processing ──────────────────────────────────────────────────────────

function _split_off_minus_one(expr::SExpr)::Tuple{Int, SExpr}
    if expr isa SMul && !isempty(expr.args) && expr.args[1] isa SConst
        cv = expr.args[1]
        if cv.val == -one(ComplexF64)
            rest = expr.args[2:end]
            return -1, length(rest) == 1 ? rest[1] : SMul(rest)
        end
    end
    return 1, expr
end

function _compile_split_into_num_denom!(c::TapeCompiler, expr::SMul)
    nums = Int32[]
    denoms = Int32[]
    for arg in expr.args
        if arg isa SPow && arg.exp < 0
            push!(denoms, _tape_pow!(c, _compile!(c, arg.base), -arg.exp))
        elseif arg isa SPow
            push!(nums, _tape_pow!(c, _compile!(c, arg.base), arg.exp))
        else
            push!(nums, _compile!(c, arg))
        end
    end
    return nums, denoms
end

function _compile_prod_parts!(c::TapeCompiler, exs::Vector{Int32})::Int32
    isempty(exs) && return _SLOT_NONE
    parts = reverse(copy(exs))
    while length(parts) > 1
        if length(parts) >= 4
            a = pop!(parts); b = pop!(parts); d = pop!(parts); e = pop!(parts)
            push!(parts, _emit!(c, OpType.OP_MUL4, a, b, d, e))
        elseif length(parts) == 3
            a = pop!(parts); b = pop!(parts); d = pop!(parts)
            push!(parts, _emit!(c, OpType.OP_MUL3, a, b, d))
        elseif length(parts) == 2
            a = pop!(parts); b = pop!(parts)
            push!(parts, _tape_mul!(c, a, b))
        end
    end
    return parts[1]
end

function _compile_mul!(c::TapeCompiler, expr::SMul)::Int32
    (m, expr2) = _split_off_minus_one(expr)
    m_slot = _get_constant_slot!(c, ComplexF64(m))
    if !(expr2 isa SMul)
        return _tape_mul!(c, m_slot, _compile!(c, expr2))
    end
    nums, denoms = _compile_split_into_num_denom!(c, expr2)
    num_prod = _compile_prod_parts!(c, nums)
    denom_prod = _compile_prod_parts!(c, denoms)
    local ref::Int32
    if num_prod == _SLOT_NONE
        ref = _emit!(c, OpType.OP_INV, denom_prod)
    elseif denom_prod == _SLOT_NONE
        ref = num_prod
    else
        ref = _tape_div!(c, num_prod, denom_prod)
    end
    return _tape_mul!(c, m_slot, ref)
end

## ── Add processing ──────────────────────────────────────────────────────────

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

function _compile_reduce_to_at_most_two!(
        c::TapeCompiler, expr::SExpr,
    )::Tuple{Int32, Int32}
    if expr isa SMul
        args = expr.args
        if length(args) == 2
            return (_compile!(c, args[1]), _compile!(c, args[2]))
        elseif length(args) == 1
            return (_compile!(c, args[1]), _SLOT_NONE)
        elseif length(args) > 2
            prefix = SMul(args[1:(end - 1)])
            v2 = _compile!(c, prefix)
            return (_compile!(c, args[end]), v2)
        end
    end
    return (_compile!(c, expr), _SLOT_NONE)
end

function _compile_sum_products!(
        c::TapeCompiler, tuples::Vector{Tuple{Int32, Int32}},
    )::Int32
    isempty(tuples) && return _SLOT_NONE

    singles = Int32[first(t) for t in tuples if t[2] == _SLOT_NONE]
    pairs = [t for t in tuples if t[2] != _SLOT_NONE]
    n = length(pairs)

    for k in 1:2:(n - 1)
        (a, b) = pairs[k]
        (d, e) = pairs[k + 1]
        push!(singles, _tape_mulmuladd!(c, a, b, d, e))
    end

    if isodd(n)
        if isempty(singles)
            return _tape_mul!(c, pairs[n][1], pairs[n][2])
        end
        (a, b) = pairs[n]
        d = pop!(singles)
        push!(singles, _tape_muladd!(c, a, b, d))
    end

    while length(singles) > 1
        if length(singles) >= 4
            a = pop!(singles); b = pop!(singles); d = pop!(singles); e = pop!(singles)
            push!(singles, _emit!(c, OpType.OP_ADD4, a, b, d, e))
        elseif length(singles) == 3
            a = pop!(singles); b = pop!(singles); d = pop!(singles)
            push!(singles, _emit!(c, OpType.OP_ADD3, a, b, d))
        elseif length(singles) == 2
            a = pop!(singles); b = pop!(singles)
            push!(singles, _emit!(c, OpType.OP_ADD, a, b))
        end
    end
    return singles[1]
end

function _compile_sum!(c::TapeCompiler, expr::SAdd)::Int32
    pos, neg = _split_into_positives_negatives(expr)
    pos_reduced =
        Tuple{Int32, Int32}[_compile_reduce_to_at_most_two!(c, e) for e in pos]
    neg_reduced =
        Tuple{Int32, Int32}[_compile_reduce_to_at_most_two!(c, e) for e in neg]

    # Case 1: single positive, single negative — try maximal fusion
    if length(pos_reduced) == 1 && length(neg_reduced) == 1
        (a, b) = pos_reduced[1]
        (d, e) = neg_reduced[1]
        if b != _SLOT_NONE && e != _SLOT_NONE
            return _emit!(c, OpType.OP_MULMULSUB, a, b, d, e)
        elseif b != _SLOT_NONE
            return _emit!(c, OpType.OP_MULSUB, a, b, d)
        elseif e != _SLOT_NONE
            return _emit!(c, OpType.OP_SUBMUL, d, e, a)
        else
            return _emit!(c, OpType.OP_SUB, a, d)
        end
    end

    # Case 2: multiple positives, single negative
    if length(neg_reduced) == 1
        a = _compile_sum_products!(c, pos_reduced)
        (d, e) = neg_reduced[1]
        if e != _SLOT_NONE
            return _emit!(c, OpType.OP_SUBMUL, d, e, a)
        else
            return _emit!(c, OpType.OP_SUB, a, d)
        end
    end

    # Case 3: single positive, multiple negatives
    if length(pos_reduced) == 1
        neg_sum = _compile_sum_products!(c, neg_reduced)
        (a, b) = pos_reduced[1]
        if b != _SLOT_NONE
            return _emit!(c, OpType.OP_MULSUB, a, b, neg_sum)
        else
            return _emit!(c, OpType.OP_SUB, a, neg_sum)
        end
    end

    # Case 4: multiple positives, multiple negatives — interleave for MULMULSUB
    # Pair positive products with negative products for MULMULSUB(a,b,c,d) = a*b - c*d
    pos_pairs = Tuple{Int32, Int32}[t for t in pos_reduced if t[2] != _SLOT_NONE]
    pos_singles = Int32[t[1] for t in pos_reduced if t[2] == _SLOT_NONE]
    neg_pairs = Tuple{Int32, Int32}[t for t in neg_reduced if t[2] != _SLOT_NONE]
    neg_singles = Int32[t[1] for t in neg_reduced if t[2] == _SLOT_NONE]

    # Pair positive products with negative products for MULMULSUB
    fused_results = Int32[]
    n_fused = min(length(pos_pairs), length(neg_pairs))
    for i in 1:n_fused
        (a, b) = pos_pairs[i]
        (d, e) = neg_pairs[i]
        push!(fused_results, _emit!(c, OpType.OP_MULMULSUB, a, b, d, e))
    end

    # Leftover positive products
    leftover_pos = Tuple{Int32, Int32}[]
    for i in (n_fused + 1):length(pos_pairs)
        push!(leftover_pos, pos_pairs[i])
    end
    for s in pos_singles
        push!(leftover_pos, (s, _SLOT_NONE))
    end

    # Leftover negative products
    leftover_neg = Tuple{Int32, Int32}[]
    for i in (n_fused + 1):length(neg_pairs)
        push!(leftover_neg, neg_pairs[i])
    end
    for s in neg_singles
        push!(leftover_neg, (s, _SLOT_NONE))
    end

    # Sum fused results + leftover positives
    all_pos_parts = Int32[]
    append!(all_pos_parts, fused_results)
    if !isempty(leftover_pos)
        pos_sum = _compile_sum_products!(c, leftover_pos)
        pos_sum != _SLOT_NONE && push!(all_pos_parts, pos_sum)
    end

    pos_total = _SLOT_NONE
    if length(all_pos_parts) == 1
        pos_total = all_pos_parts[1]
    elseif length(all_pos_parts) >= 2
        pos_total = _compile_sum_products!(
            c, Tuple{Int32, Int32}[(s, _SLOT_NONE) for s in all_pos_parts],
        )
    end

    # Sum leftover negatives
    neg_total = _SLOT_NONE
    if !isempty(leftover_neg)
        neg_total = _compile_sum_products!(c, leftover_neg)
    end

    # Final subtraction
    if pos_total == _SLOT_NONE
        return neg_total == _SLOT_NONE ? _SLOT_NONE : _tape_neg!(c, neg_total)
    elseif neg_total == _SLOT_NONE
        return pos_total
    else
        return _tape_sub!(c, pos_total, neg_total)
    end
end

## ── Entry point ─────────────────────────────────────────────────────────────

"""
    compile_to_instructions(replacements, reduced_exprs;
        nvars, nparams, output_dim, npolys,
        continuation_parameter_index=nothing) -> InstructionSequence

Compile CSE output directly to an optimized `InstructionSequence`.

Tape layout: constants | params | [cont_param] | variables | scratch | assignments
"""
function compile_to_instructions(
        replacements::Vector{Pair{SExpr, SExpr}},
        reduced_exprs::Vector{SExpr};
        nvars::Int,
        nparams::Int,
        output_dim::Int,
        npolys::Int,
        continuation_parameter_index::Union{Nothing, Int} = nothing,
    )::InstructionSequence
    compiler = TapeCompiler(nvars, nparams)

    # Register CSE definitions (compiled lazily on first use)
    for (tmp, definition) in replacements
        @assert tmp isa STmp
        compiler.cse_defs[tmp.id] = definition
    end

    # Use a two-pass approach with placeholder slots.
    # During compilation, constant slots use temporary 1-based indices.
    # Var/param slots use negative placeholders. Scratch uses high offsets.
    # After compilation, we remap everything to the final tape layout.

    scratch_offset = Int32(10000)

    # Assign placeholder slots for params and vars (negative indices)
    for i in 1:nparams
        compiler.param_slots[i] = Int32(-i)
    end
    for i in 1:nvars
        compiler.var_slots[i] = Int32(-(nparams + i))
    end
    compiler.next_slot = scratch_offset

    # Compile all reduced expressions
    result_slots = Int32[_compile!(compiler, expr) for expr in reduced_exprs]

    # Build final tape layout
    nconstants = length(compiler.constants)
    has_cont = !isnothing(continuation_parameter_index) ? 1 : 0
    input_block_size = nconstants + nparams + has_cont + nvars

    # Build remapping: old slot → new slot
    remap = Dict{Int32, Int32}()

    # Constants: temporary slot k → final slot k (identity for 1:nconstants)
    constants_range = 1:nconstants
    for k in 1:nconstants
        remap[Int32(k)] = Int32(k)
    end

    # Parameters: placeholder -i → nconstants + i
    parameters_range = (nconstants + 1):(nconstants + nparams)
    for i in 1:nparams
        remap[Int32(-i)] = Int32(nconstants + i)
    end

    # Continuation parameter
    local cont_param_tape_index::Union{Nothing, Int}
    if has_cont == 1
        cont_param_tape_index = nconstants + nparams + 1
    else
        cont_param_tape_index = nothing
    end

    # Variables: placeholder -(nparams+i) → nconstants + nparams + has_cont + i
    variables_start = nconstants + nparams + has_cont + 1
    variables_range = variables_start:(variables_start + nvars - 1)
    for i in 1:nvars
        remap[Int32(-(nparams + i))] = Int32(variables_start + i - 1)
    end

    # Scratch: (scratch_offset + k) → (input_block_size + k)
    nscratch = Int(compiler.next_slot - scratch_offset)
    for k in 1:nscratch
        remap[Int32(scratch_offset + k)] = Int32(input_block_size + k)
    end

    # Build assignment slots.
    # Non-scratch results (constants/vars/params) → use source slot directly (no IDENTITY).
    # First-use scratch outputs → remap to dedicated assignment slot.
    # Duplicate scratch outputs → IDENTITY to copy into a new assignment slot.
    nassignments = length(result_slots)
    scratch_output_set = Set{Int32}(instr.output for instr in compiler.instructions)
    claimed_slots = Dict{Int32, Int}()  # raw result_slot → use count
    identity_instructions = Instruction[]

    # Track which assignments need scratch-based dedicated slots vs direct references
    direct_assignments = Tuple{Int, Int32}[]       # (output_index, input_block_slot)
    scratch_assignment_count = 0
    scratch_assignment_indices = Int[]              # which output indices use scratch slots

    for (k, raw_slot) in enumerate(result_slots)
        seen = get(claimed_slots, raw_slot, 0)
        is_scratch_output = raw_slot ∈ scratch_output_set

        if is_scratch_output && seen == 0
            # First use of a scratch output: gets a dedicated assignment slot
            claimed_slots[raw_slot] = 1
            scratch_assignment_count += 1
            push!(scratch_assignment_indices, k)
        elseif is_scratch_output
            # Duplicate scratch output: needs IDENTITY to copy
            claimed_slots[raw_slot] = seen + 1
            scratch_assignment_count += 1
            push!(scratch_assignment_indices, k)
        else
            # Non-scratch output (constant/var/param): direct reference, no IDENTITY
            claimed_slots[raw_slot] = seen + 1
            remapped_source = Int32(get(remap, raw_slot, raw_slot))
            push!(direct_assignments, (k, remapped_source))
        end
    end

    # Assign contiguous scratch assignment slots
    assignments_start = input_block_size + nscratch + 1
    scratch_assignments_range = range(assignments_start; length = scratch_assignment_count)

    # Now emit remaps and IDENTITYs for scratch-based assignments
    claimed_slots_2 = Dict{Int32, Int}()
    scratch_assign_k = 0
    for output_k in scratch_assignment_indices
        raw_slot = result_slots[output_k]
        seen = get(claimed_slots_2, raw_slot, 0)
        scratch_assign_k += 1
        target_slot = Int32(assignments_start + scratch_assign_k - 1)

        if seen == 0
            # First use: remap scratch output directly to assignment slot
            claimed_slots_2[raw_slot] = 1
            remap[raw_slot] = target_slot
        else
            # Duplicate: emit IDENTITY
            claimed_slots_2[raw_slot] = seen + 1
            remapped_source = get(remap, raw_slot, raw_slot)
            push!(
                identity_instructions, Instruction(
                    (remapped_source, remapped_source, remapped_source, remapped_source),
                    OpType.OP_IDENTITY, target_slot,
                ),
            )
        end
    end

    # Now remap all core instructions using the final remap
    nstmts = length(compiler.instructions)
    core_instructions = Vector{Instruction}(undef, nstmts + length(identity_instructions))
    for (idx, instr) in enumerate(compiler.instructions)
        new_input = ntuple(Val(4)) do k
            if should_use_index_not_reference(instr.op, k)
                instr.input[k]
            else
                get(remap, instr.input[k], instr.input[k])
            end
        end
        new_output = get(remap, instr.output, instr.output)
        core_instructions[idx] = Instruction(new_input, instr.op, new_output)
    end

    # Append IDENTITY instructions (these already use remapped source slots)
    for (j, id_instr) in enumerate(identity_instructions)
        core_instructions[nstmts + j] = id_instr
    end

    # Run optimizer and register allocator
    instructions_opt = _optimize_instruction_order(core_instructions)
    instructions_final, space_needed, updated_scratch_range, updated_direct =
        _reduce_space(
        instructions_opt, input_block_size, scratch_assignments_range, direct_assignments,
    )

    # Build final assignments: combine scratch-based and direct assignments
    updated_assignments = Vector{Tuple{Int, Int}}(undef, nassignments)

    # Fill in scratch-based assignments from updated range
    for (j, tape_idx) in enumerate(updated_scratch_range)
        output_k = scratch_assignment_indices[j]
        updated_assignments[output_k] = (output_k, Int(tape_idx))
    end

    # Fill in direct assignments (these point to input-block slots, no remapping needed)
    for (output_k, tape_slot) in updated_direct
        updated_assignments[output_k] = (output_k, Int(tape_slot))
    end

    # Add STOP instruction
    n = space_needed
    push!(
        instructions_final, Instruction(
            (Int32(n), Int32(n), Int32(n), Int32(n)), OpType.OP_STOP, Int32(n),
        ),
    )

    # Split assignments into u (function values) and U (Jacobian entries)
    u_assignments = Tuple{Int, Int}[
        (i, k) for (i, k) in updated_assignments if i <= output_dim
    ]
    U_assignments = Tuple{Int, Int}[
        (i - output_dim, k) for (i, k) in updated_assignments if i > output_dim
    ]

    return InstructionSequence(
        instructions_final,
        copy(compiler.constants),
        constants_range,
        parameters_range,
        variables_range,
        cont_param_tape_index,
        updated_assignments,
        output_dim,
        space_needed,
        u_assignments,
        U_assignments,
        length(u_assignments) == output_dim,
        length(U_assignments) == output_dim * nvars,
    )
end
