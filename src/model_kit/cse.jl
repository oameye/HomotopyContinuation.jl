## CSE (Common Subexpression Elimination) — SymEngine algorithm in Julia
#
# A 1:1 port of SymEngine's cse.cpp two-phase algorithm:
#   Phase 1 (opt_cse): factor common arguments in Add/Mul nodes
#   Phase 2 (tree_cse): eliminate repeated subexpressions
#
# Reference: https://github.com/symengine/symengine/blob/master/symengine/cse.cpp

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
        func_class::SFuncKind.T,
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
        for a in _get_args(expr)
            _opts_cse_visit!(a, adds, muls, opt_subs, seen)
        end
        push!(adds, expr)

    elseif expr isa SMul
        # bvisit(const Mul &x)
        for a in _get_args(expr)
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
                opt_subs[expr] = SFuncSym(SFuncKind.SFUNC_MUL, SExpr[SConst(-one(ComplexF64)), neg_expr])
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
            opt_subs[expr] = SFuncSym(SFuncKind.SFUNC_POW, SExpr[SPow(expr.base, -expr.exp), SConst(ComplexF64(-1))])
        end

    elseif expr isa SNeg
        # SNeg is our representation for SymEngine's Mul(-1, x) where neg simplifies to atom
        _opts_cse_visit!(expr.arg, adds, muls, opt_subs, seen)

    elseif expr isa SFuncSym
        # bvisit(const Basic &x) — generic case for compound expressions
        for a in _get_args(expr)
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
    match_common_args!(SFuncKind.SFUNC_ADD, adds_vec, opt_subs)
    match_common_args!(SFuncKind.SFUNC_MUL, muls_vec, opt_subs)

    return opt_subs
end

## ── Phase 2: tree_cse ───────────────────────────────────────────────────────

"""
    _find_repeated!(expr, seen, to_eliminate, opt_subs)

Walk expression tree, applying opt_subs, and mark expressions seen 2+ times.
Direct translation of SymEngine's find_repeated lambda in tree_cse.

Key logic (matching SymEngine exactly):
1. Skip atoms (Numbers)
2. If already seen → mark for elimination, return
3. Add to seen
4. Replace expr with opt_subs[expr] if present
5. Get args of (possibly replaced) expr and recurse into each
"""
function _find_repeated!(
        expr::SExpr,
        seen::Set{SExpr},
        to_eliminate::Set{SExpr},
        opt_subs::Dict{SExpr, SExpr},
    )::Nothing
    # SymEngine: if (is_a_Number(*expr) ...) return;
    if expr isa SConst
        return nothing
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
        _find_repeated!(arg, seen, to_eliminate, opt_subs)
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
    end
    new_args = SExpr[
        _rebuild(a, to_eliminate, opt_subs, subs, replacements, next_id) for a in _get_args(expr)
    ]
    return _rebuild_expr(expr, new_args)
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

    for e in exprs
        _find_repeated!(e, seen, to_eliminate, opt_subs)
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
