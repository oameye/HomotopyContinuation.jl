## Direct polynomial lowering for small systems
#
# This backend lowers MultivariatePolynomials input directly to tape
# instructions without going through the SExpr + CSE pipeline.
#
# The public frontend selects this backend for small systems where avoiding the
# SExpr/CSE compiler materially reduces TTFX. Larger systems retain the symbolic
# compiler because its global CSE produces substantially smaller tapes.

struct MonomialKey
    data::Vector{Int32}
end

Base.isequal(a::MonomialKey, b::MonomialKey) = isequal(a.data, b.data)
Base.hash(key::MonomialKey, h::UInt) = hash(key.data, h)

struct PolynomialInstructionCompiler
    compiler::TapeCompiler
    slot_by_symbol::Dict{Symbol, Int32}
    power_slots::Dict{Tuple{Int32, Int}, Int32}
    monomial_slots::Dict{MonomialKey, Int32}
end

function _polynomial_instruction_compiler(
        variables::AbstractVector,
        parameters::AbstractVector,
    )::PolynomialInstructionCompiler
    compiler = TapeCompiler(length(variables), length(parameters))
    _initialize_placeholder_slots!(compiler, length(variables), length(parameters))
    slot_by_symbol = Dict{Symbol, Int32}()
    for i in eachindex(parameters)
        slot_by_symbol[Symbol(parameters[i])] = compiler.param_slots[i]
    end
    for i in eachindex(variables)
        slot_by_symbol[Symbol(variables[i])] = compiler.var_slots[i]
    end
    return PolynomialInstructionCompiler(
        compiler,
        slot_by_symbol,
        Dict{Tuple{Int32, Int}, Int32}(),
        Dict{MonomialKey, Int32}(),
    )
end

function _monomial_key(
        mono,
        slot_by_symbol::Dict{Symbol, Int32},
    )::MonomialKey
    data = Int32[]
    for (var, exp) in zip(MP.variables(mono), MP.exponents(mono))
        exp == 0 && continue
        push!(data, slot_by_symbol[Symbol(var)])
        push!(data, Int32(exp))
    end
    return MonomialKey(data)
end

function _power_slot!(
        state::PolynomialInstructionCompiler,
        base_slot::Int32,
        exp::Int,
    )::Int32
    exp == 1 && return base_slot
    key = (base_slot, exp)
    slot = get(state.power_slots, key, _SLOT_NONE)
    slot != _SLOT_NONE && return slot
    slot = _tape_pow!(state.compiler, base_slot, exp)
    state.power_slots[key] = slot
    return slot
end

function _monomial_slot!(
        state::PolynomialInstructionCompiler,
        mono,
    )::Int32
    key = _monomial_key(mono, state.slot_by_symbol)
    isempty(key.data) && return _get_constant_slot!(state.compiler, one(ComplexF64))

    slot = get(state.monomial_slots, key, _SLOT_NONE)
    slot != _SLOT_NONE && return slot

    factors = Int32[]
    for i in 1:2:length(key.data)
        push!(factors, _power_slot!(state, key.data[i], Int(key.data[i + 1])))
    end
    slot = _compile_prod_parts!(state.compiler, factors)
    state.monomial_slots[key] = slot
    return slot
end

function _compile_polynomial!(
        state::PolynomialInstructionCompiler,
        poly::MP.AbstractPolynomialLike,
    )::Int32
    Base.@nospecialize poly
    term_slots = Int32[]
    for term in MP.terms(poly)
        coeff = ComplexF64(MP.coefficient(term))
        iszero(coeff) && continue
        mono_slot = _monomial_slot!(state, MP.monomial(term))
        if coeff == one(ComplexF64)
            push!(term_slots, mono_slot)
        else
            coeff_slot = _get_constant_slot!(state.compiler, coeff)
            push!(term_slots, _tape_mul!(state.compiler, coeff_slot, mono_slot))
        end
    end
    isempty(term_slots) && return _get_constant_slot!(state.compiler, zero(ComplexF64))
    return _sum_slots!(state.compiler, term_slots)
end

function _build_instruction_sequence_direct(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
        parameters::AbstractVector,
        include_jacobian::Bool,
    )::InstructionSequence
    Base.@nospecialize polys variables parameters
    nvars = length(variables)
    nparams = length(parameters)
    output_dim = length(polys)

    state = _polynomial_instruction_compiler(variables, parameters)
    result_slots = Int32[]
    sizehint!(result_slots, output_dim * (include_jacobian ? (nvars + 1) : 1))

    for poly in polys
        push!(result_slots, _compile_polynomial!(state, poly))
    end

    if include_jacobian
        for v in variables
            for poly in polys
                push!(result_slots, _compile_polynomial!(state, MP.differentiate(poly, v)))
            end
        end
    end

    # The direct compiler emits every dependency before its consumer. Running
    # the generic DAG reorder here only rebuilds that ordering and adds cold
    # compiler work; the symbolic frontend retains the reorder.
    return _finalize_compiler(
        Val(false), state.compiler, result_slots, nvars, nparams, output_dim,
    )
end
