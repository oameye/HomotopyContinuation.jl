## Expression lowering
#
# Front-end glue from `Expression` input to the tape pipeline:
#   Expression -> SExpr -> CSE -> tape compiler
#
# The Jacobian is produced by symbolic differentiation on `Expression`, which
# covers division, negative powers and the unary functions that
# `MP.differentiate` cannot express.

function _expression_index_map(values::AbstractVector{Expression})::Dict{Symbol, Int}
    index_map = Dict{Symbol, Int}()
    for (i, value) in enumerate(values)
        index_map[Symbol(value)] = i
    end
    return index_map
end

function _build_instruction_sequence_from_expressions(
        exprs::AbstractVector{Expression},
        variables::AbstractVector{Expression},
        parameters::AbstractVector{Expression},
        include_jacobian::Bool,
    )::InstructionSequence
    nvars = length(variables)
    nparams = length(parameters)
    output_dim = length(exprs)

    var_to_idx = _expression_index_map(variables)
    param_to_idx = _expression_index_map(parameters)

    all_exprs = Vector{SExprT}(undef, output_dim)
    for i in eachindex(exprs)
        all_exprs[i] = expression_to_sexpr(exprs[i], var_to_idx, param_to_idx)
    end

    if include_jacobian
        # Column-major: all equations for variable 1, then variable 2, ...
        for v in variables
            sym = Symbol(v)
            for f in exprs
                push!(
                    all_exprs,
                    expression_to_sexpr(_differentiate(f, sym), var_to_idx, param_to_idx),
                )
            end
        end
    end

    replacements, reduced_exprs = cse(all_exprs)
    return compile_to_instructions(replacements, reduced_exprs, nvars, nparams, output_dim)
end
