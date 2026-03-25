## MP polynomial → Interpreter pipeline
#
# Converts DynamicPolynomials input into an InstructionSequence and Interpreter
# via: polynomials → SExpr trees → CSE → IR → compile

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
    seq = _build_instruction_sequence(polys, variables, parameters; include_jacobian = false)
    var_syms = Symbol[Symbol(v) for v in variables]
    param_syms = Symbol[Symbol(p) for p in parameters]
    return Interpreter(Vector{ComplexF64}, seq; variables = var_syms, parameters = param_syms)
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
    seq = _build_instruction_sequence(polys, variables, parameters; include_jacobian = true)
    var_syms = Symbol[Symbol(v) for v in variables]
    param_syms = Symbol[Symbol(p) for p in parameters]
    return Interpreter(Vector{ComplexF64}, seq; variables = var_syms, parameters = param_syms)
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
    seq = _build_instruction_sequence(polys, variables, parameters; include_jacobian = false)
    var_syms = Symbol[Symbol(v) for v in variables]
    param_syms = Symbol[Symbol(p) for p in parameters]
    return Interpreter(
        Vector{TruncatedTaylorSeries{K + 1, ComplexF64}}, seq;
        variables = var_syms, parameters = param_syms,
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
    seq = _build_instruction_sequence(polys, variables, parameters; include_jacobian = false)
    var_syms = Symbol[Symbol(v) for v in variables]
    param_syms = Symbol[Symbol(p) for p in parameters]
    return Interpreter(Vector{ComplexDF64}, seq; variables = var_syms, parameters = param_syms)
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
    return empty(_collect_variables(polys))
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

Core pipeline: convert polynomials to SExpr trees → run CSE → compile to IR → build InstructionSequence.
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

    var_syms = Symbol[Symbol(v) for v in variables]
    param_syms = Symbol[Symbol(p) for p in parameters]

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

    # Compile CSE output to IR
    ir_stmts, constants_list, result_refs = compile_cse_to_ir(
        replacements, reduced_exprs, var_syms, param_syms,
    )

    # Split result refs into F and Jacobian
    f_refs = result_refs[1:npolys]
    jac_refs = result_refs[(npolys + 1):end]

    # Ensure all result refs are IRStatementRefs (wrap constants/symbols in OP_IDENTITY)
    ref_counter = isempty(ir_stmts) ? 0 : maximum(s.target.i for s in ir_stmts)
    assigned_stmt_refs = Dict{IRStatementRef, Int}()

    function ensure_stmt_ref(ref::IRStatementArg)::IRStatementRef
        if ref isa IRStatementRef
            seen = get(assigned_stmt_refs, ref, 0)
            if seen == 0
                assigned_stmt_refs[ref] = 1
                return ref
            end
            assigned_stmt_refs[ref] = seen + 1
            ref_counter += 1
            r = IRStatementRef(ref_counter)
            push!(ir_stmts, IRStatement(OpType.OP_IDENTITY, r, ref))
            return r
        end
        # Wrap constant or symbol in an OP_IDENTITY instruction
        ref_counter += 1
        r = IRStatementRef(ref_counter)
        push!(ir_stmts, IRStatement(OpType.OP_IDENTITY, r, ref))
        return r
    end

    # Build assignments
    assignments = Tuple{Int, IRStatementArg}[]
    for (i, ref) in enumerate(f_refs)
        push!(assignments, (i, ensure_stmt_ref(ref)))
    end
    for (k, ref) in enumerate(jac_refs)
        push!(assignments, (npolys + k, ensure_stmt_ref(ref)))
    end

    ir = IntermediateRepresentation(ir_stmts, assignments, npolys)

    # Compile IR to InstructionSequence
    return build_instruction_sequence_from_ir(
        ir;
        nvars = nvars,
        nparams = nparams,
        nconstants = length(constants_list),
        constants = constants_list,
        variables = var_syms,
        parameters = param_syms,
    )
end
