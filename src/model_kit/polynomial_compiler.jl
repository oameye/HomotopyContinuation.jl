## Monomial memoization for lowering supports straight to tape instructions

struct MonomialKey
    data::Vector{Int32}
end

Base.isequal(a::MonomialKey, b::MonomialKey) = isequal(a.data, b.data)
Base.hash(key::MonomialKey, h::UInt) = hash(key.data, h)

# Monomial/power memoization for the polyhedral support frontend, so identical
# powers and monomials compile to one tape slot.
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

function power_slot!(cache::MonomialCache, base_slot::Int32, exp::Int)::Int32
    exp == 1 && return base_slot
    key = (base_slot, exp)
    slot = get(cache.power_slots, key, SLOT_NONE)
    slot != SLOT_NONE && return slot
    slot = tape_pow!(cache.compiler, base_slot, exp)
    cache.power_slots[key] = slot
    return slot
end

function cached_monomial_slot!(cache::MonomialCache, key::MonomialKey)::Int32
    isempty(key.data) && return get_constant_slot!(cache.compiler, one(ComplexF64))
    slot = get(cache.monomial_slots, key, SLOT_NONE)
    slot != SLOT_NONE && return slot

    factors = Int32[]
    for i in 1:2:length(key.data)
        push!(factors, power_slot!(cache, key.data[i], Int(key.data[i + 1])))
    end
    slot = compile_prod_parts!(cache.compiler, factors)
    cache.monomial_slots[key] = slot
    return slot
end

# Multiply `slot` by a literal coefficient. Coefficients that rewrite to a
# cheaper instruction never register a constant, leaving no dead tape slot.
function coeff_mul_slot!(c::TapeCompiler, coeff::ComplexF64, slot::Int32)::Int32
    coeff == one(ComplexF64) && return slot
    is_one_slot(c, slot) && return get_constant_slot!(c, coeff)
    coeff == -one(ComplexF64) && return tape_neg!(c, slot)
    coeff == ComplexF64(2) && return tape_add!(c, slot, slot)
    return tape_mul!(c, get_constant_slot!(c, coeff), slot)
end
