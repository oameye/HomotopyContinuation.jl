## MP polynomial input helpers
#
# The public entry point for polynomial input is `System(polys; ...)`, which
# converts to `Expression`. This file finds the variables of that input.

## ── Variable discovery ──────────────────────────────────────────────────────

# NOTE: MP.variables is only defined on concrete polynomial types (DynamicPolynomials),
# not on the abstract MP.AbstractPolynomialLike. JET flags calls on abstract types.
# We call it on individual polynomials (which are always concrete at runtime).

# Canonical order is creation order, which DP exposes as the reverse of `isless`
# (first-created variable compares largest). Symbol-name fallback for types
# without a comparison.
@inline function lt_variable(a, b)
    if applicable(isless, a, b)
        return isless(b, a)
    end
    return string(Symbol(a)) < string(Symbol(b))
end

function append_unique!(
        acc::Vector{V}, seen::Set{V}, items::AbstractVector,
    )::Vector{V} where {V}
    for v in items
        v in seen && continue
        push!(seen, v)
        push!(acc, v)
    end
    return acc
end

@noinline function collect_variables(polys)
    Base.@nospecialize polys
    all_vars = empty(MP.variables(first(polys)))
    seen = Set(all_vars)
    for poly in polys
        append_unique!(all_vars, seen, MP.variables(poly))
    end
    stable_sort!(all_vars, lt_variable)
    return all_vars
end

# Return an empty vector with the correct variable element type
@noinline function empty_vars(polys)
    Base.@nospecialize polys
    return empty(MP.variables(first(polys)))
end

# `MP.variables` is undefined on `MP.RationalPoly`, so gather from numerator and
# denominator. `parameters` are matched by name: they may arrive as MP variables
# or as `Expression`s.
@noinline function rational_variables(polys, parameters)
    Base.@nospecialize polys parameters
    all_vars = empty(MP.variables(numerator(first(polys))))
    seen = Set(all_vars)
    for p in polys
        append_unique!(all_vars, seen, MP.variables(numerator(p)))
        append_unique!(all_vars, seen, MP.variables(denominator(p)))
    end
    stable_sort!(all_vars, lt_variable)
    isempty(parameters) && return all_vars
    param_names = Set{Symbol}(Symbol(p) for p in parameters)
    filter!(v -> !(Symbol(v) in param_names), all_vars)
    return all_vars
end

@noinline function effective_variables_excluding(polys, parameters)
    Base.@nospecialize polys parameters
    all_vars = collect_variables(polys)
    if isempty(parameters)
        return collect(all_vars)
    end
    param_set = Set(parameters)
    vars = empty(all_vars)
    for v in all_vars
        v in param_set && continue
        push!(vars, v)
    end
    return vars
end
