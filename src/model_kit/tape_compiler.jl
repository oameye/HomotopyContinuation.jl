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
    const cse_defs::Dict{Int, SExpr}               # STmp id → definition
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
        Dict{Int, SExpr}(),
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

"""Emit an instruction and return the output tape slot."""
function _emit!(c::TapeCompiler, op::OpType.T, a1::Int32)::Int32
    c.next_slot += Int32(1)
    slot = c.next_slot
    push!(c.instructions, Instruction((a1, a1, a1, a1), op, slot))
    return slot
end

function _emit!(c::TapeCompiler, op::OpType.T, a1::Int32, a2::Int32)::Int32
    c.next_slot += Int32(1)
    slot = c.next_slot
    push!(c.instructions, Instruction((a1, a2, a2, a2), op, slot))
    return slot
end

function _emit!(c::TapeCompiler, op::OpType.T, a1::Int32, a2::Int32, a3::Int32)::Int32
    c.next_slot += Int32(1)
    slot = c.next_slot
    push!(c.instructions, Instruction((a1, a2, a3, a3), op, slot))
    return slot
end

function _emit!(
        c::TapeCompiler, op::OpType.T, a1::Int32, a2::Int32, a3::Int32, a4::Int32,
    )::Int32
    c.next_slot += Int32(1)
    slot = c.next_slot
    push!(c.instructions, Instruction((a1, a2, a3, a4), op, slot))
    return slot
end

## ── Slot-based predicates ───────────────────────────────────────────────────

_is_one_slot(c::TapeCompiler, s::Int32)::Bool =
    c.one_slot != _SLOT_NONE && s == c.one_slot
_is_minus_one_slot(c::TapeCompiler, s::Int32)::Bool =
    c.minus_one_slot != _SLOT_NONE && s == c.minus_one_slot
_is_two_slot(c::TapeCompiler, s::Int32)::Bool =
    c.two_slot != _SLOT_NONE && s == c.two_slot

## ── Arithmetic helpers ──────────────────────────────────────────────────────

function _tape_add!(c::TapeCompiler, a::Int32, b::Int32)::Int32
    return _emit!(c, OpType.OP_ADD, a, b)
end

function _tape_neg!(c::TapeCompiler, a::Int32)::Int32
    return _emit!(c, OpType.OP_NEG, a)
end

function _tape_sub!(c::TapeCompiler, a::Int32, b::Int32)::Int32
    return _emit!(c, OpType.OP_SUB, a, b)
end

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

function _tape_mulmuladd!(
        c::TapeCompiler, a::Int32, b::Int32, d::Int32, e::Int32,
    )::Int32
    return _emit!(c, OpType.OP_MULMULADD, a, b, d, e)
end

function _tape_div!(c::TapeCompiler, a::Int32, b::Int32)::Int32
    return _is_one_slot(c, b) ? a : _emit!(c, OpType.OP_DIV, a, b)
end

function _tape_sqr!(c::TapeCompiler, a::Int32)::Int32
    return _emit!(c, OpType.OP_SQR, a)
end

function _tape_pow!(c::TapeCompiler, a::Int32, k::Int)::Int32
    if k == 0
        return _get_constant_slot!(c, one(ComplexF64))
    elseif k == 1
        return a
    elseif k == 2
        return _tape_sqr!(c, a)
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
            Instruction((a, Int32(k), Int32(k), Int32(k)), OpType.OP_POW_INT, slot),
        )
        return slot
    end
end

## ── Main dispatcher ─────────────────────────────────────────────────────────

"""Compile an SExpr to a tape slot, returning the Int32 slot index."""
function _compile!(c::TapeCompiler, expr::SExpr)::Int32
    if expr isa SConst
        return _get_constant_slot!(c, expr.val)
    elseif expr isa SVar
        return c.var_slots[expr.idx]
    elseif expr isa SParam
        return c.param_slots[expr.idx]
    elseif expr isa STmp
        cached = get(c.cse_slots, expr.id, _SLOT_NONE)
        cached != _SLOT_NONE && return cached
        slot = _compile!(c, c.cse_defs[expr.id])
        c.cse_slots[expr.id] = slot
        return slot
    elseif expr isa SPow
        base = _compile!(c, expr.base)
        return _tape_pow!(c, base, expr.exp)
    elseif expr isa SMul
        return _compile_mul!(c, expr)
    elseif expr isa SAdd
        return _compile_sum!(c, expr)
    elseif expr isa SNeg
        return _tape_neg!(c, _compile!(c, expr.arg))
    elseif expr isa SFuncSym
        if expr.kind == SFuncKind.SFUNC_ADD
            return _compile_sum!(c, SAdd(expr.args))
        elseif expr.kind == SFuncKind.SFUNC_MUL
            return _compile_mul!(c, SMul(expr.args))
        else
            error("Unknown SFuncSym kind: $(expr.kind)")
        end
    else
        error("Unknown SExpr type: $(typeof(expr))")
    end
end

## ── Mul processing ──────────────────────────────────────────────────────────

function _split_off_minus_one(expr::SExpr)::Tuple{Int, SExpr}
    if expr isa SMul && !isempty(expr.args) && expr.args[1] isa SConst
        cv = expr.args[1]
        if cv.val == -one(ComplexF64)
            rest = expr.args[2:end]
            return -1, length(rest) == 1 ? rest[1] : SMul(rest)
        end
    end
    return 1, expr
end

function _compile_split_into_num_denom!(c::TapeCompiler, expr::SMul)
    nums = Int32[]
    denoms = Int32[]
    for arg in expr.args
        if arg isa SPow && arg.exp < 0
            push!(denoms, _tape_pow!(c, _compile!(c, arg.base), -arg.exp))
        elseif arg isa SPow
            push!(nums, _tape_pow!(c, _compile!(c, arg.base), arg.exp))
        else
            push!(nums, _compile!(c, arg))
        end
    end
    return nums, denoms
end

function _compile_prod_parts!(c::TapeCompiler, exs::Vector{Int32})::Int32
    isempty(exs) && return _SLOT_NONE
    parts = reverse(copy(exs))
    while length(parts) > 1
        if length(parts) >= 4
            a = pop!(parts); b = pop!(parts); d = pop!(parts); e = pop!(parts)
            push!(parts, _emit!(c, OpType.OP_MUL4, a, b, d, e))
        elseif length(parts) == 3
            a = pop!(parts); b = pop!(parts); d = pop!(parts)
            push!(parts, _emit!(c, OpType.OP_MUL3, a, b, d))
        elseif length(parts) == 2
            a = pop!(parts); b = pop!(parts)
            push!(parts, _tape_mul!(c, a, b))
        end
    end
    return parts[1]
end

function _compile_mul!(c::TapeCompiler, expr::SMul)::Int32
    (m, expr2) = _split_off_minus_one(expr)
    m_slot = _get_constant_slot!(c, ComplexF64(m))
    if !(expr2 isa SMul)
        return _tape_mul!(c, m_slot, _compile!(c, expr2))
    end
    nums, denoms = _compile_split_into_num_denom!(c, expr2)
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

function _split_into_positives_negatives(expr::SAdd)
    positives = SExpr[]
    negatives = SExpr[]
    for arg in expr.args
        (sign, val) = _split_off_minus_one(arg)
        if sign == -1
            push!(negatives, val)
        else
            push!(positives, val)
        end
    end
    return positives, negatives
end

function _compile_reduce_to_at_most_two!(
        c::TapeCompiler, expr::SExpr,
    )::Tuple{Int32, Int32}
    if expr isa SMul
        args = expr.args
        if length(args) == 2
            return (_compile!(c, args[1]), _compile!(c, args[2]))
        elseif length(args) == 1
            return (_compile!(c, args[1]), _SLOT_NONE)
        elseif length(args) > 2
            prefix = SMul(args[1:(end - 1)])
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

    singles = Int32[first(t) for t in tuples if t[2] == _SLOT_NONE]
    pairs = [t for t in tuples if t[2] != _SLOT_NONE]
    n = length(pairs)

    for k in 1:2:(n - 1)
        (a, b) = pairs[k]
        (d, e) = pairs[k + 1]
        push!(singles, _tape_mulmuladd!(c, a, b, d, e))
    end

    if isodd(n)
        if isempty(singles)
            return _tape_mul!(c, pairs[n][1], pairs[n][2])
        end
        (a, b) = pairs[n]
        d = pop!(singles)
        push!(singles, _tape_muladd!(c, a, b, d))
    end

    while length(singles) > 1
        if length(singles) >= 4
            a = pop!(singles); b = pop!(singles); d = pop!(singles); e = pop!(singles)
            push!(singles, _emit!(c, OpType.OP_ADD4, a, b, d, e))
        elseif length(singles) == 3
            a = pop!(singles); b = pop!(singles); d = pop!(singles)
            push!(singles, _emit!(c, OpType.OP_ADD3, a, b, d))
        elseif length(singles) == 2
            a = pop!(singles); b = pop!(singles)
            push!(singles, _emit!(c, OpType.OP_ADD, a, b))
        end
    end
    return singles[1]
end

function _compile_sum!(c::TapeCompiler, expr::SAdd)::Int32
    pos, neg = _split_into_positives_negatives(expr)
    pos_reduced =
        Tuple{Int32, Int32}[_compile_reduce_to_at_most_two!(c, e) for e in pos]
    neg_reduced =
        Tuple{Int32, Int32}[_compile_reduce_to_at_most_two!(c, e) for e in neg]

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

    # Case 4: multiple positives, multiple negatives — interleave for MULMULSUB
    # Pair positive products with negative products for MULMULSUB(a,b,c,d) = a*b - c*d
    pos_pairs = Tuple{Int32, Int32}[t for t in pos_reduced if t[2] != _SLOT_NONE]
    pos_singles = Int32[t[1] for t in pos_reduced if t[2] == _SLOT_NONE]
    neg_pairs = Tuple{Int32, Int32}[t for t in neg_reduced if t[2] != _SLOT_NONE]
    neg_singles = Int32[t[1] for t in neg_reduced if t[2] == _SLOT_NONE]

    # Pair positive products with negative products for MULMULSUB
    fused_results = Int32[]
    n_fused = min(length(pos_pairs), length(neg_pairs))
    for i in 1:n_fused
        (a, b) = pos_pairs[i]
        (d, e) = neg_pairs[i]
        push!(fused_results, _emit!(c, OpType.OP_MULMULSUB, a, b, d, e))
    end

    # Leftover positive products
    leftover_pos = Tuple{Int32, Int32}[]
    for i in (n_fused + 1):length(pos_pairs)
        push!(leftover_pos, pos_pairs[i])
    end
    for s in pos_singles
        push!(leftover_pos, (s, _SLOT_NONE))
    end

    # Leftover negative products
    leftover_neg = Tuple{Int32, Int32}[]
    for i in (n_fused + 1):length(neg_pairs)
        push!(leftover_neg, neg_pairs[i])
    end
    for s in neg_singles
        push!(leftover_neg, (s, _SLOT_NONE))
    end

    # Sum fused results + leftover positives
    all_pos_parts = Int32[]
    append!(all_pos_parts, fused_results)
    if !isempty(leftover_pos)
        pos_sum = _compile_sum_products!(c, leftover_pos)
        pos_sum != _SLOT_NONE && push!(all_pos_parts, pos_sum)
    end

    pos_total = _SLOT_NONE
    if length(all_pos_parts) == 1
        pos_total = all_pos_parts[1]
    elseif length(all_pos_parts) >= 2
        pos_total = _compile_sum_products!(
            c, Tuple{Int32, Int32}[(s, _SLOT_NONE) for s in all_pos_parts],
        )
    end

    # Sum leftover negatives
    neg_total = _SLOT_NONE
    if !isempty(leftover_neg)
        neg_total = _compile_sum_products!(c, leftover_neg)
    end

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

"""
    compile_to_instructions(replacements, reduced_exprs;
        nvars, nparams, output_dim, npolys,
        continuation_parameter_index=nothing) -> InstructionSequence

Compile CSE output directly to an optimized `InstructionSequence`.

Tape layout: constants | params | [cont_param] | variables | scratch | assignments
"""
function compile_to_instructions(
        replacements::Vector{Pair{SExpr, SExpr}},
        reduced_exprs::Vector{SExpr};
        nvars::Int,
        nparams::Int,
        output_dim::Int,
        npolys::Int,
        continuation_parameter_index::Union{Nothing, Int} = nothing,
    )::InstructionSequence
    compiler = TapeCompiler(nvars, nparams)

    # Register CSE definitions (compiled lazily on first use)
    for (tmp, definition) in replacements
        @assert tmp isa STmp
        compiler.cse_defs[tmp.id] = definition
    end

    # Use a two-pass approach with placeholder slots.
    # During compilation, constant slots use temporary 1-based indices.
    # Var/param slots use negative placeholders. Scratch uses high offsets.
    # After compilation, we remap everything to the final tape layout.

    scratch_offset = Int32(10000)

    # Assign placeholder slots for params and vars (negative indices)
    for i in 1:nparams
        compiler.param_slots[i] = Int32(-i)
    end
    for i in 1:nvars
        compiler.var_slots[i] = Int32(-(nparams + i))
    end
    compiler.next_slot = scratch_offset

    # Compile all reduced expressions
    result_slots = Int32[_compile!(compiler, expr) for expr in reduced_exprs]

    # Build final tape layout
    nconstants = length(compiler.constants)
    has_cont = !isnothing(continuation_parameter_index) ? 1 : 0
    input_block_size = nconstants + nparams + has_cont + nvars

    # Build remapping: old slot → new slot
    remap = Dict{Int32, Int32}()

    # Constants: temporary slot k → final slot k (identity for 1:nconstants)
    constants_range = 1:nconstants
    for k in 1:nconstants
        remap[Int32(k)] = Int32(k)
    end

    # Parameters: placeholder -i → nconstants + i
    parameters_range = (nconstants + 1):(nconstants + nparams)
    for i in 1:nparams
        remap[Int32(-i)] = Int32(nconstants + i)
    end

    # Continuation parameter
    local cont_param_tape_index::Union{Nothing, Int}
    if has_cont == 1
        cont_param_tape_index = nconstants + nparams + 1
    else
        cont_param_tape_index = nothing
    end

    # Variables: placeholder -(nparams+i) → nconstants + nparams + has_cont + i
    variables_start = nconstants + nparams + has_cont + 1
    variables_range = variables_start:(variables_start + nvars - 1)
    for i in 1:nvars
        remap[Int32(-(nparams + i))] = Int32(variables_start + i - 1)
    end

    # Scratch: (scratch_offset + k) → (input_block_size + k)
    nscratch = Int(compiler.next_slot - scratch_offset)
    for k in 1:nscratch
        remap[Int32(scratch_offset + k)] = Int32(input_block_size + k)
    end

    # Build assignment slots.
    # Non-scratch results (constants/vars/params) → use source slot directly (no IDENTITY).
    # First-use scratch outputs → remap to dedicated assignment slot.
    # Duplicate scratch outputs → IDENTITY to copy into a new assignment slot.
    nassignments = length(result_slots)
    scratch_output_set = Set{Int32}(instr.output for instr in compiler.instructions)
    claimed_slots = Dict{Int32, Int}()  # raw result_slot → use count
    identity_instructions = Instruction[]

    # Track which assignments need scratch-based dedicated slots vs direct references
    direct_assignments = Tuple{Int, Int32}[]       # (output_index, input_block_slot)
    scratch_assignment_count = 0
    scratch_assignment_indices = Int[]              # which output indices use scratch slots

    for (k, raw_slot) in enumerate(result_slots)
        seen = get(claimed_slots, raw_slot, 0)
        is_scratch_output = raw_slot ∈ scratch_output_set

        if is_scratch_output && seen == 0
            # First use of a scratch output: gets a dedicated assignment slot
            claimed_slots[raw_slot] = 1
            scratch_assignment_count += 1
            push!(scratch_assignment_indices, k)
        elseif is_scratch_output
            # Duplicate scratch output: needs IDENTITY to copy
            claimed_slots[raw_slot] = seen + 1
            scratch_assignment_count += 1
            push!(scratch_assignment_indices, k)
        else
            # Non-scratch output (constant/var/param): direct reference, no IDENTITY
            claimed_slots[raw_slot] = seen + 1
            remapped_source = Int32(get(remap, raw_slot, raw_slot))
            push!(direct_assignments, (k, remapped_source))
        end
    end

    # Assign contiguous scratch assignment slots
    assignments_start = input_block_size + nscratch + 1
    scratch_assignments_range = range(assignments_start; length = scratch_assignment_count)

    # Now emit remaps and IDENTITYs for scratch-based assignments
    claimed_slots_2 = Dict{Int32, Int}()
    scratch_assign_k = 0
    for output_k in scratch_assignment_indices
        raw_slot = result_slots[output_k]
        seen = get(claimed_slots_2, raw_slot, 0)
        scratch_assign_k += 1
        target_slot = Int32(assignments_start + scratch_assign_k - 1)

        if seen == 0
            # First use: remap scratch output directly to assignment slot
            claimed_slots_2[raw_slot] = 1
            remap[raw_slot] = target_slot
        else
            # Duplicate: emit IDENTITY
            claimed_slots_2[raw_slot] = seen + 1
            remapped_source = get(remap, raw_slot, raw_slot)
            push!(
                identity_instructions, Instruction(
                    (remapped_source, remapped_source, remapped_source, remapped_source),
                    OpType.OP_IDENTITY, target_slot,
                ),
            )
        end
    end

    # Now remap all core instructions using the final remap
    nstmts = length(compiler.instructions)
    core_instructions = Vector{Instruction}(undef, nstmts + length(identity_instructions))
    for (idx, instr) in enumerate(compiler.instructions)
        new_input = ntuple(Val(4)) do k
            if should_use_index_not_reference(instr.op, k)
                instr.input[k]
            else
                get(remap, instr.input[k], instr.input[k])
            end
        end
        new_output = get(remap, instr.output, instr.output)
        core_instructions[idx] = Instruction(new_input, instr.op, new_output)
    end

    # Append IDENTITY instructions (these already use remapped source slots)
    for (j, id_instr) in enumerate(identity_instructions)
        core_instructions[nstmts + j] = id_instr
    end

    # Run optimizer and register allocator
    instructions_opt = _optimize_instruction_order(core_instructions)
    instructions_final, space_needed, updated_scratch_range, updated_direct =
        _reduce_space(
        instructions_opt, input_block_size, scratch_assignments_range, direct_assignments,
    )

    # Build final assignments: combine scratch-based and direct assignments
    updated_assignments = Vector{Tuple{Int, Int}}(undef, nassignments)

    # Fill in scratch-based assignments from updated range
    for (j, tape_idx) in enumerate(updated_scratch_range)
        output_k = scratch_assignment_indices[j]
        updated_assignments[output_k] = (output_k, Int(tape_idx))
    end

    # Fill in direct assignments (these point to input-block slots, no remapping needed)
    for (output_k, tape_slot) in updated_direct
        updated_assignments[output_k] = (output_k, Int(tape_slot))
    end

    # Add STOP instruction
    n = space_needed
    push!(
        instructions_final, Instruction(
            (Int32(n), Int32(n), Int32(n), Int32(n)), OpType.OP_STOP, Int32(n),
        ),
    )

    # Split assignments into u (function values) and U (Jacobian entries)
    u_assignments = Tuple{Int, Int}[
        (i, k) for (i, k) in updated_assignments if i <= output_dim
    ]
    U_assignments = Tuple{Int, Int}[
        (i - output_dim, k) for (i, k) in updated_assignments if i > output_dim
    ]

    return InstructionSequence(
        instructions_final,
        copy(compiler.constants),
        constants_range,
        parameters_range,
        variables_range,
        cont_param_tape_index,
        updated_assignments,
        output_dim,
        space_needed,
        u_assignments,
        U_assignments,
        length(u_assignments) == output_dim,
        length(U_assignments) == output_dim * nvars,
    )
end
