## MP polynomial → Interpreter pipeline
#
# Converts DynamicPolynomials input into an InstructionSequence and Interpreter
# via: polynomials → SExpr trees → CSE → IR → compile

## ── Internal generic builder ──────────────────────────────────────────────────

function _build_interpreter(
        ::Type{V},
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
        include_jacobian::Bool = false,
    ) where {V <: AbstractVector}
    seq = _build_instruction_sequence(polys, variables, parameters; include_jacobian = include_jacobian)
    var_syms = Symbol[Symbol(v) for v in variables]
    param_syms = Symbol[Symbol(p) for p in parameters]
    return Interpreter(V, seq; variables = var_syms, parameters = param_syms)
end

## ── User-facing API ─────────────────────────────────────────────────────────

"""
    build_interpreter(polys; parameters=[], variables=...)

Build an `Interpreter{Vector{ComplexF64}}` from a vector of MP polynomials.
"""
function build_interpreter(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    )
    return _build_interpreter(
        Vector{ComplexF64}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )
end

"""
    build_jacobian_interpreter(polys; parameters=[], variables=...)

Build an `Interpreter{Vector{ComplexF64}}` that evaluates both F and its Jacobian.
"""
function build_jacobian_interpreter(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    )
    return _build_interpreter(
        Vector{ComplexF64}, polys;
        parameters = parameters, variables = variables, include_jacobian = true,
    )
end

"""
    build_taylor_interpreter(polys, ::Val{K}; parameters=[], variables=...)

Build an `Interpreter{Vector{TruncatedTaylorSeries{K+1,ComplexF64}}}`.
"""
function build_taylor_interpreter(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        ::Val{K};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    ) where {K}
    return _build_interpreter(
        Vector{TruncatedTaylorSeries{K + 1, ComplexF64}}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )
end

"""
    build_df64_interpreter(polys; parameters=[], variables=...)

Build an `Interpreter{Vector{ComplexDF64}}`.
"""
function build_df64_interpreter(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    )
    return _build_interpreter(
        Vector{ComplexDF64}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )
end

## ── Variable discovery ──────────────────────────────────────────────────────

# NOTE: MP.variables is only defined on concrete polynomial types (DynamicPolynomials),
# not on the abstract MP.AbstractPolynomialLike. JET flags calls on abstract types.
# We call it on individual polynomials (which are always concrete at runtime).
@inline function _variable_creation_id(v)
    if hasfield(typeof(v), :variable_order)
        variable_order = getfield(v, :variable_order)
        if hasfield(typeof(variable_order), :order)
            order = getfield(variable_order, :order)
            if hasfield(typeof(order), :id)
                return getfield(order, :id)
            end
        end
    end
    return nothing
end

@inline function _lt_variable(a, b)
    id_a = _variable_creation_id(a)
    id_b = _variable_creation_id(b)
    if !(isnothing(id_a) || isnothing(id_b))
        return id_a < id_b
    end
    return string(Symbol(a)) < string(Symbol(b))
end

@noinline function _collect_variables(polys)
    all_vars = empty(MP.variables(first(polys)))
    for poly in polys
        append!(all_vars, MP.variables(poly))
    end
    unique!(all_vars)
    sort!(all_vars; lt = _lt_variable)
    return all_vars
end

# Return an empty vector with the correct variable element type
@noinline function _empty_vars(polys)
    return empty(MP.variables(first(polys)))
end

@noinline function _effective_variables(polys, parameters)
    all_vars = _collect_variables(polys)
    if isempty(parameters)
        return collect(all_vars)
    end
    param_set = Set(parameters)
    return [v for v in all_vars if v ∉ param_set]
end

## ── Core pipeline ───────────────────────────────────────────────────────────

"""
    _build_instruction_sequence(polys, variables, parameters; include_jacobian)

Core pipeline: convert polynomials to SExpr trees → run CSE → compile to InstructionSequence.
"""
function _build_instruction_sequence(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
        parameters::AbstractVector;
        include_jacobian::Bool,
    )::InstructionSequence
    nvars = length(variables)
    nparams = length(parameters)
    npolys = length(polys)

    # Build variable/parameter index maps for poly_to_sexpr
    var_to_idx = Dict{Symbol, Int}()
    for (i, v) in enumerate(variables)
        var_to_idx[Symbol(v)] = i
    end
    param_to_idx = Dict{Symbol, Int}()
    for (i, p) in enumerate(parameters)
        param_to_idx[Symbol(p)] = i
    end

    # Convert F polynomials to SExpr trees
    f_exprs = SExpr[poly_to_sexpr(p, var_to_idx, param_to_idx) for p in polys]

    # Convert Jacobian polynomials to SExpr trees (column-major order)
    jac_exprs = SExpr[]
    if include_jacobian
        for v in variables
            for p in polys
                dp = MP.differentiate(p, v)
                push!(jac_exprs, poly_to_sexpr(dp, var_to_idx, param_to_idx))
            end
        end
    end

    # Combine all expressions and run CSE
    all_exprs = vcat(f_exprs, jac_exprs)
    replacements, reduced_exprs = cse(all_exprs)

    # Compile directly to InstructionSequence
    return compile_to_instructions(
        replacements, reduced_exprs;
        nvars = nvars,
        nparams = nparams,
        output_dim = npolys,
        npolys = npolys,
    )
end
