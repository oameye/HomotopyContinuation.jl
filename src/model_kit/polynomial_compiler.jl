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

# Monomial/power memoization shared by the polynomial and polyhedral support
# frontends, so identical powers and monomials compile to one tape slot.
# `MonomialKey.data` holds (variable_slot, exponent) pairs with nonzero
# exponents, ordered by descending slot; an empty key is the constant 1.
struct MonomialCache
    compiler::TapeCompiler
    power_slots::Dict{Tuple{Int32, Int}, Int32}
    monomial_slots::Dict{MonomialKey, Int32}
end

function MonomialCache(compiler::TapeCompiler)::MonomialCache
    return MonomialCache(
        compiler,
        Dict{Tuple{Int32, Int}, Int32}(),
        Dict{MonomialKey, Int32}(),
    )
end

function _power_slot!(cache::MonomialCache, base_slot::Int32, exp::Int)::Int32
    exp == 1 && return base_slot
    key = (base_slot, exp)
    slot = get(cache.power_slots, key, _SLOT_NONE)
    slot != _SLOT_NONE && return slot
    slot = _tape_pow!(cache.compiler, base_slot, exp)
    cache.power_slots[key] = slot
    return slot
end

function _monomial_slot!(cache::MonomialCache, key::MonomialKey)::Int32
    isempty(key.data) && return _get_constant_slot!(cache.compiler, one(ComplexF64))
    slot = get(cache.monomial_slots, key, _SLOT_NONE)
    slot != _SLOT_NONE && return slot

    factors = Int32[]
    for i in 1:2:length(key.data)
        push!(factors, _power_slot!(cache, key.data[i], Int(key.data[i + 1])))
    end
    slot = _compile_prod_parts!(cache.compiler, factors)
    cache.monomial_slots[key] = slot
    return slot
end

struct PolynomialInstructionCompiler
    cache::MonomialCache
    slot_by_symbol::Dict{Symbol, Int32}
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
    return PolynomialInstructionCompiler(MonomialCache(compiler), slot_by_symbol)
end

# Insertion-sorted in place: keys have a handful of pairs, and avoiding the
# generic sort keeps its specialization out of construction-time compilation.
function _monomial_key(
        mono,
        slot_by_symbol::Dict{Symbol, Int32},
    )::MonomialKey
    data = Int32[]
    for (var, exp) in zip(MP.variables(mono), MP.exponents(mono))
        exp == 0 && continue
        push!(data, slot_by_symbol[Symbol(var)])
        push!(data, Int32(exp))
        k = length(data) >> 1
        while k > 1 && data[2k - 3] < data[2k - 1]
            data[2k - 3], data[2k - 1] = data[2k - 1], data[2k - 3]
            data[2k - 2], data[2k] = data[2k], data[2k - 2]
            k -= 1
        end
    end
    return MonomialKey(data)
end

# Multiply `slot` by a literal coefficient. Coefficients that rewrite to a
# cheaper instruction never register a constant, leaving no dead tape slot.
function _coeff_mul_slot!(c::TapeCompiler, coeff::ComplexF64, slot::Int32)::Int32
    coeff == one(ComplexF64) && return slot
    _is_one_slot(c, slot) && return _get_constant_slot!(c, coeff)
    coeff == -one(ComplexF64) && return _tape_neg!(c, slot)
    coeff == ComplexF64(2) && return _tape_add!(c, slot, slot)
    return _tape_mul!(c, _get_constant_slot!(c, coeff), slot)
end

function _compile_polynomial!(
        state::PolynomialInstructionCompiler,
        poly::MP.AbstractPolynomialLike,
    )::Int32
    Base.@nospecialize poly
    compiler = state.cache.compiler
    term_slots = Int32[]
    for term in MP.terms(poly)
        coeff = ComplexF64(MP.coefficient(term))
        iszero(coeff) && continue
        key = _monomial_key(MP.monomial(term), state.slot_by_symbol)
        if isempty(key.data)
            push!(term_slots, _get_constant_slot!(compiler, coeff))
        else
            mono_slot = _monomial_slot!(state.cache, key)
            push!(term_slots, _coeff_mul_slot!(compiler, coeff, mono_slot))
        end
    end
    isempty(term_slots) && return _get_constant_slot!(compiler, zero(ComplexF64))
    return _sum_slots!(compiler, term_slots)
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
        Val(false), state.cache.compiler, result_slots, nvars, nparams, output_dim,
    )
end
