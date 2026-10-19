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
@inline function lt_variable(a, b)
    if applicable(isless, a, b)
        return isless(b, a)
    end
    return string(Symbol(a)) < string(Symbol(b))
end

@noinline function collect_variables(polys)
    Base.@nospecialize polys
    all_vars = empty(MP.variables(first(polys)))
    for poly in polys
        append!(all_vars, MP.variables(poly))
    end
    unique!(all_vars)
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
    for p in polys
        append!(all_vars, MP.variables(numerator(p)))
        append!(all_vars, MP.variables(denominator(p)))
    end
    unique!(all_vars)
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

## ── Core pipeline ───────────────────────────────────────────────────────────

"""
    build_instruction_sequence(polys, variables, parameters, include_jacobian)

Core pipeline: convert polynomials to SExpr trees → run CSE → compile to InstructionSequence.
"""
function build_instruction_sequence(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
        parameters::AbstractVector,
        include_jacobian::Bool,
    )::InstructionSequence
    lowerer = if prefer_direct_polynomial_lowering(polys, variables, parameters)
        build_instruction_sequence_direct
    else
        build_instruction_sequence_via_sexpr
    end
    # Both lowerers are singleton function types, so Julia can otherwise
    # union-split through this construction-time policy even across the
    # unspecialized call boundary below.
    lowerer = Base.inferencebarrier(lowerer)
    return invoke_polynomial_lowerer(
        lowerer, polys, variables, parameters, include_jacobian,
    )
end

function prefer_direct_polynomial_lowering(polys, variables, parameters)::Bool
    length(variables) + length(parameters) <= 2 || return false
    nterms = 0
    for poly in polys
        nterms += length(MP.terms(poly))
        nterms <= 8 || return false
    end
    return true
end

# This is an intentional construction-time dispatch barrier. Keeping the
# selected lowering function unspecialized prevents inference from compiling
# both the direct and symbolic frontends for every System construction.
@noinline function invoke_polynomial_lowerer(
        lowerer::Function, polys, variables, parameters, include_jacobian::Bool,
    )::InstructionSequence
    Base.@nospecialize lowerer polys variables parameters
    return lowerer(polys, variables, parameters, include_jacobian)
end
