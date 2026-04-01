## TapeCompiler — compiles SExpr trees to InstructionSequence
#
# Converts the output of cse() (replacements + reduced_exprs) into an
# optimized InstructionSequence with a concrete tape layout:
#   constants | params | [cont_param] | variables | scratch | assignments

## ── SExpr → Instruction compilation (TapeCompiler) ──────────────────────────

"""
State for compiling SExpr trees directly to `Instruction` values with tape indices.
"""
mutable struct TapeCompiler
    const instructions::Vector{Instruction}
    const constants::Vector{ComplexF64}
    const constants_map::Dict{ComplexF64, Int32}  # value → tape slot
    const var_slots::Vector{Int32}                 # var index → tape slot
    const param_slots::Vector{Int32}               # param index → tape slot
    const cse_defs::Dict{Int, SExprT}               # STmp id → definition
    const cse_slots::Dict{Int, Int32}              # STmp id → tape slot (memoized)
    next_slot::Int32
    # Known constant slots for optimization
    one_slot::Int32
    minus_one_slot::Int32
    two_slot::Int32
end

const _SLOT_NONE = Int32(0)

function TapeCompiler(nvars::Int, nparams::Int)
    return TapeCompiler(
        Instruction[],
        ComplexF64[],
        Dict{ComplexF64, Int32}(),
        Vector{Int32}(undef, nvars),
        Vector{Int32}(undef, nparams),
        Dict{Int, SExprT}(),
        Dict{Int, Int32}(),
        Int32(0),
        _SLOT_NONE,
        _SLOT_NONE,
        _SLOT_NONE,
    )
end

## ── Core helpers ────────────────────────────────────────────────────────────

"""Register a constant and return its tape slot (1-based, relative to constants block)."""
function _get_constant_slot!(c::TapeCompiler, val::ComplexF64)::Int32
    slot = get(c.constants_map, val, _SLOT_NONE)
    slot != _SLOT_NONE && return slot
    push!(c.constants, val)
    slot = Int32(length(c.constants))
    c.constants_map[val] = slot
    if val == one(ComplexF64)
        c.one_slot = slot
    elseif val == -one(ComplexF64)
        c.minus_one_slot = slot
    elseif val == ComplexF64(2)
        c.two_slot = slot
    end
    return slot
end

"""Emit an instruction and return the output tape slot. Unused input slots are filled with the last provided arg."""
function _emit!(c::TapeCompiler, op::OpType.T, a1::Int32, a2::Int32 = a1, a3::Int32 = a2, a4::Int32 = a3)::Int32
    c.next_slot += Int32(1)
    push!(c.instructions, _instruction((a1, a2, a3, a4), op, c.next_slot))
    return c.next_slot
end

## ── Slot-based predicates ───────────────────────────────────────────────────

_is_one_slot(c::TapeCompiler, s::Int32)::Bool =
    c.one_slot != _SLOT_NONE && s == c.one_slot
_is_minus_one_slot(c::TapeCompiler, s::Int32)::Bool =
    c.minus_one_slot != _SLOT_NONE && s == c.minus_one_slot
_is_two_slot(c::TapeCompiler, s::Int32)::Bool =
    c.two_slot != _SLOT_NONE && s == c.two_slot

## ── Arithmetic helpers ──────────────────────────────────────────────────────

_tape_add!(c::TapeCompiler, a::Int32, b::Int32)::Int32 = _emit!(c, OpType.OP_ADD, a, b)
_tape_neg!(c::TapeCompiler, a::Int32)::Int32 = _emit!(c, OpType.OP_NEG, a)
_tape_sub!(c::TapeCompiler, a::Int32, b::Int32)::Int32 = _emit!(c, OpType.OP_SUB, a, b)
_tape_div!(c::TapeCompiler, a::Int32, b::Int32)::Int32 =
    _is_one_slot(c, b) ? a : _emit!(c, OpType.OP_DIV, a, b)

function _tape_mul!(c::TapeCompiler, a::Int32, b::Int32)::Int32
    _is_one_slot(c, a) && return b
    _is_one_slot(c, b) && return a
    _is_minus_one_slot(c, a) && return _emit!(c, OpType.OP_NEG, b)
    _is_minus_one_slot(c, b) && return _emit!(c, OpType.OP_NEG, a)
    _is_two_slot(c, a) && return _emit!(c, OpType.OP_ADD, b, b)
    return _emit!(c, OpType.OP_MUL, a, b)
end

function _tape_muladd!(c::TapeCompiler, a::Int32, b::Int32, d::Int32)::Int32
    _is_one_slot(c, a) && return _tape_add!(c, b, d)
    _is_one_slot(c, b) && return _tape_add!(c, a, d)
    return _emit!(c, OpType.OP_MULADD, a, b, d)
end

function _tape_pow!(c::TapeCompiler, a::Int32, k::Int)::Int32
    if k == 0
        return _get_constant_slot!(c, one(ComplexF64))
    elseif k == 1
        return a
    elseif k == 2
        return _emit!(c, OpType.OP_SQR, a)
    elseif k == 3
        return _emit!(c, OpType.OP_CB, a)
    elseif k == -1
        return _emit!(c, OpType.OP_INV, a)
    elseif k == -2
        return _emit!(c, OpType.OP_INVSQR, a)
    else
        # OP_POW_INT: second arg is the integer exponent stored directly
        c.next_slot += Int32(1)
        slot = c.next_slot
        push!(
            c.instructions,
            _instruction((a, Int32(k), Int32(k), Int32(k)), OpType.OP_POW_INT, slot),
        )
        return slot
    end
end

## ── Main dispatcher ─────────────────────────────────────────────────────────

"""Compile an SExpr to a tape slot, returning the Int32 slot index."""
@inline _compile!(c::TapeCompiler, expr::SExprT)::Int32 =
    _compile_storage!(c, sexpr_storage(expr))

@inline _compile_storage!(c::TapeCompiler, storage::SConstStorage)::Int32 =
    _get_constant_slot!(c, storage.val)
@inline _compile_storage!(c::TapeCompiler, storage::SVarStorage)::Int32 =
    c.var_slots[storage.idx]
@inline _compile_storage!(c::TapeCompiler, storage::SParamStorage)::Int32 =
    c.param_slots[storage.idx]

function _compile_storage!(c::TapeCompiler, storage::STmpStorage)::Int32
    cached = get(c.cse_slots, storage.id, _SLOT_NONE)
    cached != _SLOT_NONE && return cached
    slot = _compile!(c, c.cse_defs[storage.id])
    c.cse_slots[storage.id] = slot
    return slot
end

function _compile_storage!(c::TapeCompiler, storage::SPowStorage)::Int32
    base_slot = _compile!(c, storage.base)
    return _tape_pow!(c, base_slot, storage.exp)
end

@inline _compile_storage!(c::TapeCompiler, storage::SMulStorage)::Int32 =
    _compile_mul!(c, storage.args)
@inline _compile_storage!(c::TapeCompiler, storage::SAddStorage)::Int32 =
    _compile_sum!(c, storage.args)
@inline _compile_storage!(c::TapeCompiler, storage::SNegStorage)::Int32 =
    _tape_neg!(c, _compile!(c, storage.arg))

function _compile_storage!(c::TapeCompiler, storage::SFuncSymStorage)::Int32
    if storage.kind == SFuncKind.SFUNC_ADD
        return _compile_sum!(c, storage.args)
    elseif storage.kind == SFuncKind.SFUNC_MUL
        return _compile_mul!(c, storage.args)
    end
    error("Unknown SFuncSym kind: $(storage.kind)")
end

## ── Mul processing ──────────────────────────────────────────────────────────

function _split_off_minus_one(args::Vector{SExprT})::Tuple{Int, Vector{SExprT}}
    if !isempty(args)
        coeff_storage = sexpr_storage(args[1])
        if coeff_storage isa SConstStorage && coeff_storage.val == -one(ComplexF64)
            return -1, args[2:end]
        end
    end
    return 1, args
end

function _compile_split_into_num_denom!(c::TapeCompiler, args::Vector{SExprT})
    nums = Int32[]
    denoms = Int32[]
    for arg in args
        storage = sexpr_storage(arg)
        if storage isa SPowStorage && storage.exp < 0
            push!(denoms, _tape_pow!(c, _compile!(c, storage.base), -storage.exp))
        elseif storage isa SPowStorage
            push!(nums, _tape_pow!(c, _compile!(c, storage.base), storage.exp))
        else
            push!(nums, _compile!(c, arg))
        end
    end
    return nums, denoms
end

"""Reduce a stack of slots by consuming 4/3/2 at a time with the given ops."""
function _tree_reduce!(c::TapeCompiler, parts::Vector{Int32}, op4::OpType.T, op3::OpType.T, op2::F)::Int32 where {F}
    while length(parts) > 1
        if length(parts) >= 4
            a = pop!(parts); b = pop!(parts); d = pop!(parts); e = pop!(parts)
            push!(parts, _emit!(c, op4, a, b, d, e))
        elseif length(parts) == 3
            a = pop!(parts); b = pop!(parts); d = pop!(parts)
            push!(parts, _emit!(c, op3, a, b, d))
        else
            a = pop!(parts); b = pop!(parts)
            push!(parts, op2(c, a, b))
        end
    end
    return parts[1]
end

function _compile_prod_parts!(c::TapeCompiler, exs::Vector{Int32})::Int32
    isempty(exs) && return _SLOT_NONE
    return _tree_reduce!(c, reverse(exs), OpType.OP_MUL4, OpType.OP_MUL3, _tape_mul!)
end

function _compile_mul!(c::TapeCompiler, args::Vector{SExprT})::Int32
    (m, args2) = _split_off_minus_one(args)
    m_slot = _get_constant_slot!(c, ComplexF64(m))
    if isempty(args2)
        return m_slot
    elseif length(args2) == 1
        return _tape_mul!(c, m_slot, _compile!(c, args2[1]))
    end
    nums, denoms = _compile_split_into_num_denom!(c, args2)
    num_prod = _compile_prod_parts!(c, nums)
    denom_prod = _compile_prod_parts!(c, denoms)
    local ref::Int32
    if num_prod == _SLOT_NONE
        ref = _emit!(c, OpType.OP_INV, denom_prod)
    elseif denom_prod == _SLOT_NONE
        ref = num_prod
    else
        ref = _tape_div!(c, num_prod, denom_prod)
    end
    return _tape_mul!(c, m_slot, ref)
end

## ── Add processing ──────────────────────────────────────────────────────────

function _split_into_positives_negatives(args::Vector{SExprT})
    positives = SExprT[]
    negatives = SExprT[]
    for arg in args
        storage = sexpr_storage(arg)
        if storage isa SMulStorage
            sign, values = _split_off_minus_one(storage.args)
            val = length(values) == 1 ? values[1] : SExpr.SMul(values)
        else
            sign = 1
            val = arg
        end
        if sign == -1
            push!(negatives, val)
        else
            push!(positives, val)
        end
    end
    return positives, negatives
end

function _compile_reduce_to_at_most_two!(
        c::TapeCompiler, expr::SExprT,
    )::Tuple{Int32, Int32}
    storage = sexpr_storage(expr)
    if storage isa SMulStorage
        args = storage.args
        if length(args) == 2
            return (_compile!(c, args[1]), _compile!(c, args[2]))
        elseif length(args) == 1
            return (_compile!(c, args[1]), _SLOT_NONE)
        elseif length(args) > 2
            prefix = SExpr.SMul(args[1:(end - 1)])
            v2 = _compile!(c, prefix)
            return (_compile!(c, args[end]), v2)
        end
    end
    return (_compile!(c, expr), _SLOT_NONE)
end

function _compile_sum_products!(
        c::TapeCompiler, tuples::Vector{Tuple{Int32, Int32}},
    )::Int32
    isempty(tuples) && return _SLOT_NONE

    singles = Int32[]
    pairs = Tuple{Int32, Int32}[]
    for tuple in tuples
        if tuple[2] == _SLOT_NONE
            push!(singles, first(tuple))
        else
            push!(pairs, tuple)
        end
    end
    n = length(pairs)

    for k in 1:2:(n - 1)
        (a, b) = pairs[k]
        (d, e) = pairs[k + 1]
        push!(singles, _emit!(c, OpType.OP_MULMULADD, a, b, d, e))
    end

    if isodd(n)
        if isempty(singles)
            return _tape_mul!(c, pairs[n][1], pairs[n][2])
        end
        (a, b) = pairs[n]
        d = pop!(singles)
        push!(singles, _tape_muladd!(c, a, b, d))
    end

    isempty(singles) && return _SLOT_NONE
    return _tree_reduce!(c, singles, OpType.OP_ADD4, OpType.OP_ADD3, _tape_add!)
end

"""Reduce a vector of slots by addition."""
function _sum_slots!(c::TapeCompiler, slots::Vector{Int32})::Int32
    isempty(slots) && return _SLOT_NONE
    return _tree_reduce!(c, slots, OpType.OP_ADD4, OpType.OP_ADD3, _tape_add!)
end

function _compile_sum!(c::TapeCompiler, args::Vector{SExprT})::Int32
    pos, neg = _split_into_positives_negatives(args)
    pos_reduced = Vector{Tuple{Int32, Int32}}(undef, length(pos))
    for i in eachindex(pos)
        pos_reduced[i] = _compile_reduce_to_at_most_two!(c, pos[i])
    end
    neg_reduced = Vector{Tuple{Int32, Int32}}(undef, length(neg))
    for i in eachindex(neg)
        neg_reduced[i] = _compile_reduce_to_at_most_two!(c, neg[i])
    end

    # Case 1: single positive, single negative — try maximal fusion
    if length(pos_reduced) == 1 && length(neg_reduced) == 1
        (a, b) = pos_reduced[1]
        (d, e) = neg_reduced[1]
        if b != _SLOT_NONE && e != _SLOT_NONE
            return _emit!(c, OpType.OP_MULMULSUB, a, b, d, e)
        elseif b != _SLOT_NONE
            return _emit!(c, OpType.OP_MULSUB, a, b, d)
        elseif e != _SLOT_NONE
            return _emit!(c, OpType.OP_SUBMUL, d, e, a)
        else
            return _emit!(c, OpType.OP_SUB, a, d)
        end
    end

    # Case 2: multiple positives, single negative
    if length(neg_reduced) == 1
        a = _compile_sum_products!(c, pos_reduced)
        (d, e) = neg_reduced[1]
        if e != _SLOT_NONE
            return _emit!(c, OpType.OP_SUBMUL, d, e, a)
        else
            return _emit!(c, OpType.OP_SUB, a, d)
        end
    end

    # Case 3: single positive, multiple negatives
    if length(pos_reduced) == 1
        neg_sum = _compile_sum_products!(c, neg_reduced)
        (a, b) = pos_reduced[1]
        if b != _SLOT_NONE
            return _emit!(c, OpType.OP_MULSUB, a, b, neg_sum)
        else
            return _emit!(c, OpType.OP_SUB, a, neg_sum)
        end
    end

    # Case 4: multiple positives, multiple negatives — pair for MULMULSUB
    pos_pairs = Tuple{Int32, Int32}[]
    neg_pairs = Tuple{Int32, Int32}[]
    pos_singles = Tuple{Int32, Int32}[]
    neg_singles = Tuple{Int32, Int32}[]
    for tuple in pos_reduced
        if tuple[2] == _SLOT_NONE
            push!(pos_singles, (tuple[1], _SLOT_NONE))
        else
            push!(pos_pairs, tuple)
        end
    end
    for tuple in neg_reduced
        if tuple[2] == _SLOT_NONE
            push!(neg_singles, (tuple[1], _SLOT_NONE))
        else
            push!(neg_pairs, tuple)
        end
    end
    n_fused = min(length(pos_pairs), length(neg_pairs))

    # Fuse matching pos/neg product pairs into MULMULSUB
    fused = Int32[]
    for i in 1:n_fused
        push!(fused, _emit!(c, OpType.OP_MULMULSUB, pos_pairs[i]..., neg_pairs[i]...))
    end

    # Collect leftover pairs + singles for each side
    leftover_pos = Tuple{Int32, Int32}[]
    leftover_neg = Tuple{Int32, Int32}[]
    for i in (n_fused + 1):length(pos_pairs)
        push!(leftover_pos, pos_pairs[i])
    end
    append!(leftover_pos, pos_singles)
    for i in (n_fused + 1):length(neg_pairs)
        push!(leftover_neg, neg_pairs[i])
    end
    append!(leftover_neg, neg_singles)

    # Sum: fused results + leftover positives
    if !isempty(leftover_pos)
        s = _compile_sum_products!(c, leftover_pos)
        s != _SLOT_NONE && push!(fused, s)
    end
    pos_total = _sum_slots!(c, fused)

    # Sum leftover negatives
    neg_total = isempty(leftover_neg) ? _SLOT_NONE : _compile_sum_products!(c, leftover_neg)

    # Final subtraction
    if pos_total == _SLOT_NONE
        return neg_total == _SLOT_NONE ? _SLOT_NONE : _tape_neg!(c, neg_total)
    elseif neg_total == _SLOT_NONE
        return pos_total
    else
        return _tape_sub!(c, pos_total, neg_total)
    end
end

## ── Entry point ─────────────────────────────────────────────────────────────

const _SCRATCH_OFFSET = Int32(10000)

function _initialize_placeholder_slots!(
        compiler::TapeCompiler,
        nvars::Int,
        nparams::Int,
    )::Nothing
    for i in 1:nparams
        compiler.param_slots[i] = Int32(-i)
    end
    for i in 1:nvars
        compiler.var_slots[i] = Int32(-(nparams + i))
    end
    compiler.next_slot = _SCRATCH_OFFSET
    return nothing
end

function _build_slot_remap(nconstants::Int, nparams::Int, nvars::Int, nscratch::Int)
    input_block_size = nconstants + nparams + nvars
    remap = Dict{Int32, Int32}()

    # Constants: identity mapping (k → k)
    for k in 1:nconstants
        remap[Int32(k)] = Int32(k)
    end
    # Params: negative placeholders → after constants
    for i in 1:nparams
        remap[Int32(-i)] = Int32(nconstants + i)
    end
    # Vars: negative placeholders → after params
    vars_start = nconstants + nparams + 1
    for i in 1:nvars
        remap[Int32(-(nparams + i))] = Int32(vars_start + i - 1)
    end
    # Scratch: high offsets → after input block
    for k in 1:nscratch
        remap[Int32(_SCRATCH_OFFSET + k)] = Int32(input_block_size + k)
    end

    return (
        remap = remap,
        constants_range = 1:nconstants,
        parameters_range = (nconstants + 1):(nconstants + nparams),
        variables_range = vars_start:(vars_start + nvars - 1),
        input_block_size = input_block_size,
    )
end

function _plan_assignment_slots!(
        remap::Dict{Int32, Int32},
        result_slots::Vector{Int32},
        instructions::Vector{Instruction},
        input_block_size::Int,
        nscratch::Int,
    )
    scratch_output_set = Set{Int32}()
    for instr in instructions
        push!(scratch_output_set, instruction_output(instr))
    end
    claimed_slots = Dict{Int32, Int}()
    direct_assignments = Tuple{Int, Int32}[]
    scratch_assignment_indices = Int[]

    for (k, raw_slot) in enumerate(result_slots)
        claimed_slots[raw_slot] = get(claimed_slots, raw_slot, 0) + 1
        if raw_slot ∈ scratch_output_set
            push!(scratch_assignment_indices, k)
        else
            push!(direct_assignments, (k, Int32(get(remap, raw_slot, raw_slot))))
        end
    end

    assignments_start = input_block_size + nscratch + 1
    scratch_assignments_range =
        assignments_start:(assignments_start + length(scratch_assignment_indices) - 1)
    identity_instructions = Instruction[]
    empty!(claimed_slots)

    for (scratch_assign_k, output_k) in enumerate(scratch_assignment_indices)
        raw_slot = result_slots[output_k]
        seen = get(claimed_slots, raw_slot, 0)
        claimed_slots[raw_slot] = seen + 1
        target_slot = Int32(assignments_start + scratch_assign_k - 1)

        if seen == 0
            remap[raw_slot] = target_slot
        else
            src = get(remap, raw_slot, raw_slot)
            push!(
                identity_instructions,
                _instruction((src, src, src, src), OpType.OP_IDENTITY, target_slot),
            )
        end
    end

    return (
        direct_assignments = direct_assignments,
        scratch_assignment_indices = scratch_assignment_indices,
        scratch_assignments_range = scratch_assignments_range,
        identity_instructions = identity_instructions,
        nassignments = length(result_slots),
    )
end

function _remap_instructions(
        instructions::Vector{Instruction},
        remap::Dict{Int32, Int32},
        identity_instructions::Vector{Instruction},
    )::Vector{Instruction}
    remapped = Vector{Instruction}(undef, length(instructions))
    for i in eachindex(instructions)
        remapped[i] = _remap_instruction(instructions[i], remap)
    end
    append!(remapped, identity_instructions)
    return remapped
end

function _build_assignments(
        nassignments::Int,
        updated_scratch_range,
        scratch_assignment_indices::Vector{Int},
        direct_assignments::Vector{Tuple{Int, Int32}},
    )::Vector{Tuple{Int, Int}}
    updated_assignments = Vector{Tuple{Int, Int}}(undef, nassignments)

    for (j, tape_idx) in enumerate(updated_scratch_range)
        output_k = scratch_assignment_indices[j]
        updated_assignments[output_k] = (output_k, Int(tape_idx))
    end

    for (output_k, tape_slot) in direct_assignments
        updated_assignments[output_k] = (output_k, Int(tape_slot))
    end

    return updated_assignments
end

function _finalize_compiler(
        compiler::TapeCompiler,
        result_slots::Vector{Int32},
        nvars::Int,
        nparams::Int,
        output_dim::Int,
    )::InstructionSequence
    nconstants = length(compiler.constants)
    nscratch = Int(compiler.next_slot - _SCRATCH_OFFSET)
    layout = _build_slot_remap(nconstants, nparams, nvars, nscratch)
    remap = layout.remap
    assignment_plan = _plan_assignment_slots!(
        remap, result_slots, compiler.instructions, layout.input_block_size, nscratch,
    )
    core_instructions = _remap_instructions(
        compiler.instructions,
        remap,
        assignment_plan.identity_instructions,
    )

    instructions_opt = _optimize_instruction_order(core_instructions)
    instructions_final, space_needed, updated_scratch_range, updated_direct =
        _reduce_space(
        instructions_opt,
        layout.input_block_size,
        assignment_plan.scratch_assignments_range,
        assignment_plan.direct_assignments,
    )

    updated_assignments = _build_assignments(
        assignment_plan.nassignments,
        updated_scratch_range,
        assignment_plan.scratch_assignment_indices,
        updated_direct,
    )

    n = Int32(space_needed)
    push!(instructions_final, _instruction((n, n, n, n), OpType.OP_STOP, n))

    u_assignments = Tuple{Int, Int}[]
    U_assignments = Tuple{Int, Int}[]
    for (i, k) in updated_assignments
        if i <= output_dim
            push!(u_assignments, (i, k))
        else
            push!(U_assignments, (i - output_dim, k))
        end
    end

    return InstructionSequence(
        instructions_final,
        copy(compiler.constants),
        layout.constants_range,
        layout.parameters_range,
        layout.variables_range,
        output_dim,
        space_needed,
        u_assignments,
        U_assignments,
        length(u_assignments) == output_dim,
        length(U_assignments) == output_dim * nvars,
    )
end

"""
    compile_to_instructions(replacements, reduced_exprs, nvars, nparams, output_dim)

Compile CSE output directly to an optimized `InstructionSequence`.

Tape layout: constants | params | variables | scratch | assignments
"""
function compile_to_instructions(
        replacements::Vector{Pair{SExprT, SExprT}},
        reduced_exprs::Vector{SExprT},
        nvars::Int,
        nparams::Int,
        output_dim::Int,
    )::InstructionSequence
    compiler = TapeCompiler(nvars, nparams)

    # Register CSE definitions (compiled lazily on first use)
    for (tmp, definition) in replacements
        tmp_storage = sexpr_storage(tmp)::STmpStorage
        compiler.cse_defs[tmp_storage.id] = definition
    end

    # Use a two-pass approach with placeholder slots.
    # During compilation, constant slots use temporary 1-based indices.
    # Var/param slots use negative placeholders. Scratch uses high offsets.
    # After compilation, we remap everything to the final tape layout.

    _initialize_placeholder_slots!(compiler, nvars, nparams)

    # Compile all reduced expressions
    result_slots = Vector{Int32}(undef, length(reduced_exprs))
    for i in eachindex(reduced_exprs)
        result_slots[i] = _compile!(compiler, reduced_exprs[i])
    end
    return _finalize_compiler(compiler, result_slots, nvars, nparams, output_dim)
end

