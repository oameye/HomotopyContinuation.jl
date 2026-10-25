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
    value_numbers::Dict{Int32, Int32}
    value_number_to_value::Vector{Int32}
    arg_to_funcset::Vector{Set{Int32}}
    func_to_argset::Vector{Set{Int32}}
end

function FuncArgTracker(
        funcs::Vector{Pair{Int32, Vector{Int32}}},
    )::FuncArgTracker
    value_numbers = Dict{Int32, Int32}()
    value_number_to_value = Int32[]
    arg_to_funcset = Set{Int32}[]
    func_to_argset = Set{Int32}[]

    # Pre-allocate arg_to_funcset entries
    resize!(arg_to_funcset, length(funcs))
    for i in eachindex(arg_to_funcset)
        arg_to_funcset[i] = Set{Int32}()
    end

    for (func_i, (_, func_args)) in enumerate(funcs)
        func_argset = Set{Int32}()
        for func_arg in func_args
            arg_number = get_or_add_value_number!(
                value_numbers, value_number_to_value, arg_to_funcset, func_arg,
            )
            push!(func_argset, arg_number)
            push!(arg_to_funcset[arg_number], Int32(func_i))
        end
        push!(func_to_argset, func_argset)
    end

    return FuncArgTracker(value_numbers, value_number_to_value, arg_to_funcset, func_to_argset)
end

function get_or_add_value_number!(
        value_numbers::Dict{Int32, Int32},
        value_number_to_value::Vector{Int32},
        arg_to_funcset::Vector{Set{Int32}},
        value::Int32,
    )::Int32
    existing = get(value_numbers, value, Int32(0))
    if existing != Int32(0)
        return existing
    end
    push!(value_number_to_value, value)
    push!(arg_to_funcset, Set{Int32}())
    vn = Int32(length(value_number_to_value))
    value_numbers[value] = vn
    return vn
end

pair_arg_count(p::Pair{Int32, Vector{Int32}})::Int = length(p.second)

@inline function candidate_lt(a::Int32, b::Int32, candidate_counts::Dict{Int32, Int32})::Bool
    count_a = candidate_counts[a]
    count_b = candidate_counts[b]
    return count_a < count_b || (count_a == count_b && a < b)
end

"""
Get values in sorted index order.
C++ std::set iterates in sorted order; Julia Set does not.
We sort explicitly to match SymEngine behavior.
"""
function get_args_in_value_order(
        tracker::FuncArgTracker,
        argset::Union{Set{Int32}, Vector{Int32}},
    )::Vector{Int32}
    sorted = stable_sort!(collect(Int32, argset), isless)
    return Int32[tracker.value_number_to_value[i] for i in sorted]
end

function stop_arg_tracking!(tracker::FuncArgTracker, func_i::Int32)::Nothing
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
function get_common_arg_candidates(
        tracker::FuncArgTracker,
        argset::Set{Int32},
        min_func_i::Int32,
    )::Dict{Int32, Int32}
    count_map = Dict{Int32, Int32}()

    # SymEngine sorts funcsets by size for performance.
    # We collect and sort similarly.
    funcsets = [tracker.arg_to_funcset[arg] for arg in argset]
    stable_sort_by!(funcsets, length)

    for funcset in funcsets
        for func_i in funcset
            func_i > min_func_i || continue
            count_map[func_i] = get(count_map, func_i, Int32(0)) + Int32(1)
        end
    end

    # Keep only entries with count >= 2
    filter!(p -> p.second >= Int32(2), count_map)
    return count_map
end

"""
Find functions in restrict_to_funcset that have ALL arguments in argset.
Corresponds to SymEngine's FuncArgTracker::get_subset_candidates.
"""
function get_subset_candidates(
        tracker::FuncArgTracker,
        argset::Vector{Int32},
        restrict_to_funcset,
    )::Vector{Int32}
    indices = stable_sort!(collect(Int32, restrict_to_funcset), isless)
    for arg in argset
        new_indices = Int32[]
        for idx in indices
            if idx in tracker.arg_to_funcset[arg]
                push!(new_indices, idx)
            end
        end
        indices = new_indices
    end
    return indices
end

function update_func_argset!(
        tracker::FuncArgTracker,
        func_i::Int32,
        new_args::Vector{Int32},
    )::Nothing
    old_args = tracker.func_to_argset[func_i]
    new_set = Set{Int32}(new_args)

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
function add_to_sorted_vec!(vec::Vector{Int32}, number::Int32)::Nothing
    idx = searchsortedfirst(vec, number, Base.Order.Forward)
    if idx > length(vec) || vec[idx] != number
        insert!(vec, idx, number)
    end
    return nothing
end

# ── match_common_args! ──────────────────────────────────────────────────────
# Direct translation of SymEngine's match_common_args function.

function match_common_args!(
        g::SGraph,
        func_class::SFuncKind.T,
        funcs_::Vector{Int32},
        opt_subs::Dict{Int32, Int32},
    )::Nothing
    isempty(funcs_) && return nothing

    # Build (expression, args) pairs.
    # SymEngine: funcs comes from set_as_vec(muls/adds) which iterates set_basic
    # in __cmp__ order. Then std::sort by arg count (NOT stable in C++, but the
    # input is already __cmp__-sorted, so same-size groups preserve __cmp__ order
    # in practice on most implementations).
    # We sort by __cmp__ first, then stable-sort by arg count to match.
    sort_by_hash!(g, funcs_)
    funcs = Pair{Int32, Vector{Int32}}[e => get_args(g, e) for e in funcs_]
    stable_sort_by!(funcs, pair_arg_count)  # stable sort by arg count preserves __cmp__ order within groups

    tracker = FuncArgTracker(funcs)

    changed = Set{Int32}()

    for i_raw in 1:length(funcs)
        i = Int32(i_raw)

        candidates_counts = get_common_arg_candidates(
            tracker, tracker.func_to_argset[i], i,
        )

        # Sort candidates by match count (ascending), then by index
        # "This makes us try combining smaller matches first." — SymEngine
        candidates = stable_sort!(
            collect(keys(candidates_counts)),
            (a, b) -> candidate_lt(a, b, candidates_counts),
        )

        ci = 1
        while ci <= length(candidates)
            j = candidates[ci]
            ci += 1

            # Intersect arg sets
            com_args = stable_sort!(
                collect(intersect(tracker.func_to_argset[i], tracker.func_to_argset[j])),
                isless,
            )

            length(com_args) >= 2 || continue

            diff_i = stable_sort!(collect(setdiff(tracker.func_to_argset[i], com_args)), isless)

            local com_func_number::Int32

            if !isempty(diff_i)
                # "com_func needs to be unevaluated to allow for recursive matches."
                com_func = sfuncsym!(g, func_class, get_args_in_value_order(tracker, com_args))
                com_func_number = get_or_add_value_number!(
                    tracker.value_numbers, tracker.value_number_to_value,
                    tracker.arg_to_funcset, com_func,
                )
                add_to_sorted_vec!(diff_i, com_func_number)
                update_func_argset!(tracker, i, diff_i)
                push!(changed, i)
            else
                # "Treat the whole expression as a CSE."
                # SymEngine comment: "The reason this needs to be done is somewhat
                # subtle. Within tree_cse(), to_eliminate only contains expressions
                # that are seen more than once. The problem is unevaluated expressions
                # do not compare equal to the evaluated equivalent. So tree_cse()
                # won't mark funcs[i] as a CSE if we use an unevaluated version."
                com_func_number = get_or_add_value_number!(
                    tracker.value_numbers, tracker.value_number_to_value,
                    tracker.arg_to_funcset, funcs[i].first,
                )
            end

            diff_j = stable_sort!(collect(setdiff(tracker.func_to_argset[j], com_args)), isless)
            add_to_sorted_vec!(diff_j, com_func_number)
            update_func_argset!(tracker, j, diff_j)
            push!(changed, j)

            # Also update all subset candidates
            for k in get_subset_candidates(tracker, com_args, candidates[ci:end])
                diff_k = stable_sort!(collect(setdiff(tracker.func_to_argset[k], com_args)), isless)
                add_to_sorted_vec!(diff_k, com_func_number)
                update_func_argset!(tracker, k, diff_k)
                push!(changed, k)
            end
        end

        if i in changed
            opt_subs[funcs[i].first] = sfuncsym!(
                g, func_class,
                get_args_in_value_order(tracker, tracker.func_to_argset[i]),
            )
        end
        stop_arg_tracking!(tracker, i)
    end

    return nothing
end

# ── OptsCSEVisitor ──────────────────────────────────────────────────────────
# Direct translation of SymEngine's OptsCSEVisitor class.

"""
    opts_cse_visit!(g, expr, adds, muls, opt_subs, seen)

Visitor that collects Add/Mul nodes and handles negative coefficients/exponents.
Direct translation of SymEngine's OptsCSEVisitor.
"""
function opts_cse_visit!(
        g::SGraph,
        expr::Int32,
        adds::Set{Int32},
        muls::Set{Int32},
        opt_subs::Dict{Int32, Int32},
        seen::Set{Int32},
    )::Nothing
    expr in seen && return nothing
    push!(seen, expr)
    k = node_kind(g, expr)
    if k == SKind.S_ADD
        # bvisit(const Add &x)
        for a in get_args(g, expr)
            opts_cse_visit!(g, a, adds, muls, opt_subs, seen)
        end
        push!(adds, expr)
    elseif k == SKind.S_MUL
        # bvisit(const Mul &x)
        args = get_args(g, expr)
        for a in args
            opts_cse_visit!(g, a, adds, muls, opt_subs, seen)
        end
        if !isempty(args) && is_kind(g, args[1], SKind.S_CONST) &&
                imag(node_val(g, args[1])) == 0 && real(node_val(g, args[1])) < 0
            pos_coeff = -node_val(g, args[1])
            # Compute neg(expr): flip the coefficient
            if length(args) == 2 && isone(pos_coeff)
                # neg(Mul(-1, x)) = x — SymEngine simplifies this
                neg_expr = args[2]
            else
                if isone(pos_coeff)
                    pos_args = args[2:end]
                else
                    pos_args = copy(args)
                    pos_args[1] = sconst!(g, pos_coeff)
                end
                neg_expr = length(pos_args) == 1 ? pos_args[1] : smul!(g, pos_args)
            end
            # SymEngine: if (not is_a<Symbol>(*neg_expr))
            # Skip when negation simplifies to an atom (like a variable)
            if !is_atom(g, neg_expr)
                opt_subs[expr] = sfuncsym!(
                    g, SFuncKind.SFUNC_MUL,
                    Int32[sconst!(g, -one(ComplexF64)), neg_expr],
                )
                push!(seen, neg_expr)
                # SymEngine: expr = neg_expr; if (is_a<Mul>(*expr)) muls.insert(expr)
                is_kind(g, neg_expr, SKind.S_MUL) && push!(muls, neg_expr)
            else
                # neg_expr is an atom, treat original as regular Mul
                push!(muls, expr)
            end
        else
            push!(muls, expr)
        end
    elseif k == SKind.S_POW
        # bvisit(const Pow &x)
        b = node_arg(g, expr, 1)
        exp = node_int(g, expr)
        opts_cse_visit!(g, b, adds, muls, opt_subs, seen)
        # SymEngine: check if exponent is negative
        if exp < 0
            # pow(base, -n) → FuncSym("pow", [pow(base, n), -1])
            opt_subs[expr] = sfuncsym!(
                g, SFuncKind.SFUNC_POW,
                Int32[spow!(g, b, -exp), sconst!(g, ComplexF64(-1))],
            )
        end
    elseif k == SKind.S_RPOW || k == SKind.S_NEG || k == SKind.S_UNARY
        # SNeg is our representation for SymEngine's Mul(-1, x) where neg simplifies to atom
        opts_cse_visit!(g, node_arg(g, expr, 1), adds, muls, opt_subs, seen)
    elseif k == SKind.S_FUNCSYM
        # bvisit(const Basic &x) — generic case for compound expressions
        for a in get_args(g, expr)
            opts_cse_visit!(g, a, adds, muls, opt_subs, seen)
        end
    end
    # Atoms (SConst, SVar, SParam, STmp) — nothing to do
    return nothing
end

"""
    opt_cse(g, exprs) -> Dict{Int32, Int32}

Phase 1: find optimization opportunities in Add/Mul/Pow nodes.
Direct translation of SymEngine's opt_cse function.
"""
function opt_cse(g::SGraph, exprs::Vector{Int32})::Dict{Int32, Int32}
    opt_subs = Dict{Int32, Int32}()
    # SymEngine uses set_basic (ordered set) for adds/muls — ensures uniqueness
    adds = Set{Int32}()
    muls = Set{Int32}()
    seen = Set{Int32}()

    for e in exprs
        opts_cse_visit!(g, e, adds, muls, opt_subs, seen)
    end

    # Convert to vectors for match_common_args (SymEngine: set_as_vec)
    match_common_args!(g, SFuncKind.SFUNC_ADD, sort_by_hash!(g, collect(Int32, adds)), opt_subs)
    match_common_args!(g, SFuncKind.SFUNC_MUL, sort_by_hash!(g, collect(Int32, muls)), opt_subs)

    return opt_subs
end

## ── Phase 2: tree_cse ───────────────────────────────────────────────────────

"""
    find_repeated!(g, expr, seen, to_eliminate, opt_subs)

Walk expression tree, applying opt_subs, and mark expressions seen 2+ times.
Direct translation of SymEngine's find_repeated lambda in tree_cse.

Key logic (matching SymEngine exactly):
1. Skip atoms (Numbers)
2. If already seen → mark for elimination, return
3. Add to seen
4. Replace expr with opt_subs[expr] if present
5. Get args of (possibly replaced) expr and recurse into each
"""
function find_repeated!(
        g::SGraph,
        expr::Int32,
        seen::Set{Int32},
        to_eliminate::Set{Int32},
        opt_subs::Dict{Int32, Int32},
    )::Nothing
    # SymEngine: if (is_a_Number(*expr) ...) return;
    is_kind(g, expr, SKind.S_CONST) && return nothing
    is_kind(g, expr, SKind.S_TMP) && return nothing

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
    for arg in get_args(g, actual)
        find_repeated!(g, arg, seen, to_eliminate, opt_subs)
    end

    return nothing
end

"""
    rebuild_expression(g, orig_expr, to_eliminate, opt_subs, subs, replacements, next_id) -> Int32

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
function rebuild_expression(
        g::SGraph,
        orig_expr::Int32,
        to_eliminate::Set{Int32},
        opt_subs::Dict{Int32, Int32},
        subs::Dict{Int32, Int32},
        replacements::Vector{Pair{Int32, Int32}},
        next_id::Base.RefValue{Int},
    )::Int32
    # SymEngine: if (is_a_Atom(*expr)) return expr;
    is_atom(g, orig_expr) && return orig_expr

    # SymEngine: auto iter = subs.find(expr); if (iter != subs.end()) return iter->second;
    sub = get(subs, orig_expr, Int32(0))
    sub != Int32(0) && return sub

    # SymEngine: auto iter2 = opt_subs.find(expr);
    #            if (iter2 != opt_subs.end()) { expr = iter2->second; }
    expr = get(opt_subs, orig_expr, orig_expr)

    # SymEngine: expr->accept(*this); auto new_expr = result_;
    # This calls the visitor which rebuilds children
    new_expr = rebuild_children(g, expr, to_eliminate, opt_subs, subs, replacements, next_id)

    # SymEngine: if (to_eliminate.find(orig_expr) != to_eliminate.end())
    if orig_expr in to_eliminate
        next_id[] += 1
        tmp = stmp!(g, next_id[])
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
function rebuild_children(
        g::SGraph,
        expr::Int32,
        to_eliminate::Set{Int32},
        opt_subs::Dict{Int32, Int32},
        subs::Dict{Int32, Int32},
        replacements::Vector{Pair{Int32, Int32}},
        next_id::Base.RefValue{Int},
    )::Int32
    is_atom(g, expr) && return expr
    args = get_args(g, expr)
    new_args = Vector{Int32}(undef, length(args))
    for i in eachindex(args)
        new_args[i] = rebuild_expression(
            g, args[i], to_eliminate, opt_subs, subs, replacements, next_id,
        )
    end
    return rebuild_expr(g, expr, new_args)
end

"""
    tree_cse(g, exprs, opt_subs) -> (replacements, reduced_exprs)

Phase 2: find and eliminate repeated subexpressions.
Direct translation of SymEngine's tree_cse function.
"""
function tree_cse(
        g::SGraph,
        exprs::Vector{Int32},
        opt_subs::Dict{Int32, Int32},
    )::Tuple{Vector{Pair{Int32, Int32}}, Vector{Int32}}
    to_eliminate = Set{Int32}()
    seen = Set{Int32}()

    for e in exprs
        find_repeated!(g, e, seen, to_eliminate, opt_subs)
    end

    subs = Dict{Int32, Int32}()
    replacements = Pair{Int32, Int32}[]

    next_id = Ref(0)
    reduced_exprs = Int32[]
    for e in exprs
        push!(
            reduced_exprs,
            rebuild_expression(g, e, to_eliminate, opt_subs, subs, replacements, next_id),
        )
    end

    return replacements, reduced_exprs
end

## ── Main CSE entry point ────────────────────────────────────────────────────

"""
    cse(g, exprs) -> (replacements, reduced_exprs)

Run Common Subexpression Elimination on a list of expressions.
Direct translation of SymEngine's cse function.
"""
function cse(
        g::SGraph, exprs::Vector{Int32},
    )::Tuple{Vector{Pair{Int32, Int32}}, Vector{Int32}}
    # Phase 1: find optimization opportunities (common argument matching)
    opt_subs = opt_cse(g, exprs)

    # Phase 2: eliminate repeated subexpressions
    return tree_cse(g, exprs, opt_subs)
end
