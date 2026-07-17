## MP polynomial input helpers
#
# The public entry point for polynomial input is `System(polys; ...)`. This file
# keeps only the pieces needed by `System` construction and by low-level tests of
# the polynomial lowering pipeline.

## ── Variable discovery ──────────────────────────────────────────────────────

# NOTE: MP.variables is only defined on concrete polynomial types (DynamicPolynomials),
# not on the abstract MP.AbstractPolynomialLike. JET flags calls on abstract types.
# We call it on individual polynomials (which are always concrete at runtime).

# Canonical order is creation order, which DP exposes as the reverse of `isless`
# (first-created variable compares largest). Symbol-name fallback for types
# without a comparison.
@inline function _lt_variable(a, b)
    if applicable(isless, a, b)
        return isless(b, a)
    end
    return string(Symbol(a)) < string(Symbol(b))
end

@noinline function _collect_variables(polys)
    Base.@nospecialize polys
    all_vars = empty(MP.variables(first(polys)))
    for poly in polys
        append!(all_vars, MP.variables(poly))
    end
    unique!(all_vars)
    _stable_sort!(all_vars, _lt_variable)
    return all_vars
end

# Return an empty vector with the correct variable element type
@noinline function _empty_vars(polys)
    Base.@nospecialize polys
    return empty(MP.variables(first(polys)))
end

@noinline function _effective_variables(polys, parameters)
    Base.@nospecialize polys parameters
    all_vars = _collect_variables(polys)
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

## ── Core pipeline ───────────────────────────────────────────────────────────

"""
    _build_instruction_sequence(polys, variables, parameters, include_jacobian)

Core pipeline: convert polynomials to SExpr trees → run CSE → compile to InstructionSequence.
"""
function _build_instruction_sequence(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
        parameters::AbstractVector,
        include_jacobian::Bool,
    )::InstructionSequence
    # TODO: switch to `_build_instruction_sequence_direct` once the direct
    # polynomial compiler has been validated more broadly for instruction
    # quality and hot-path runtime.
    return _build_instruction_sequence_via_sexpr(polys, variables, parameters, include_jacobian)
end
