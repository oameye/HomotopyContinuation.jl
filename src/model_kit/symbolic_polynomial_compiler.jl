## Symbolic polynomial lowering
#
# This is the current default front-end for polynomial input:
#   polynomial -> SExpr -> CSE -> tape compiler
#
# The lower-level SExpr, CSE, and tape compiler implementations stay split into
# their own files. This file only owns the front-end glue from polynomial input
# to the symbolic pipeline.

function _symbol_index_map(values)::Dict{Symbol, Int}
    index_map = Dict{Symbol, Int}()
    for (i, value) in enumerate(values)
        index_map[Symbol(value)] = i
    end
    return index_map
end

function _sexprs_from_polys(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        var_to_idx::Dict{Symbol, Int},
        param_to_idx::Dict{Symbol, Int},
    )::Vector{SExprT}
    Base.@nospecialize polys
    exprs = Vector{SExprT}(undef, length(polys))
    for i in eachindex(polys)
        exprs[i] = poly_to_sexpr(polys[i], var_to_idx, param_to_idx)
    end
    return exprs
end

function _build_instruction_sequence_via_sexpr(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
        parameters::AbstractVector,
        include_jacobian::Bool,
    )::InstructionSequence
    Base.@nospecialize polys variables parameters
    nvars = length(variables)
    nparams = length(parameters)
    output_dim = length(polys)

    var_to_idx = _symbol_index_map(variables)
    param_to_idx = _symbol_index_map(parameters)

    f_exprs = _sexprs_from_polys(polys, var_to_idx, param_to_idx)

    jac_exprs = SExprT[]
    if include_jacobian
        for v in variables
            for p in polys
                dp = MP.differentiate(p, v)
                push!(jac_exprs, poly_to_sexpr(dp, var_to_idx, param_to_idx))
            end
        end
    end

    all_exprs = copy(f_exprs)
    append!(all_exprs, jac_exprs)
    replacements, reduced_exprs = cse(all_exprs)

    return compile_to_instructions(replacements, reduced_exprs, nvars, nparams, output_dim)
end
