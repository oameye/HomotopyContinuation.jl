## IR (Intermediate Representation) data structures

struct IRStatementRef
    i::Int
end

const IRStatementArg = Union{Nothing, ComplexF64, Symbol, IRStatementRef}

struct IRStatement
    op::OpType.T
    target::IRStatementRef
    args::NTuple{4, IRStatementArg}
end

function IRStatement(
        op::OpType.T,
        target::IRStatementRef,
        a1::IRStatementArg,
    )
    arity(op) == 1 ||
        error("IRStatement has arity $(arity(op)) but called with 1 arg")
    return IRStatement(op, target, (a1, nothing, nothing, nothing))
end
function IRStatement(
        op::OpType.T,
        target::IRStatementRef,
        a1::IRStatementArg,
        a2::IRStatementArg,
    )
    arity(op) == 2 ||
        error("IRStatement has arity $(arity(op)) but called with 2 args")
    return IRStatement(op, target, (a1, a2, nothing, nothing))
end
function IRStatement(
        op::OpType.T,
        target::IRStatementRef,
        a1::IRStatementArg,
        a2::IRStatementArg,
        a3::IRStatementArg,
    )
    arity(op) == 3 ||
        error("IRStatement has arity $(arity(op)) but called with 3 args")
    return IRStatement(op, target, (a1, a2, a3, nothing))
end
function IRStatement(
        op::OpType.T,
        target::IRStatementRef,
        a1::IRStatementArg,
        a2::IRStatementArg,
        a3::IRStatementArg,
        a4::IRStatementArg,
    )
    arity(op) == 4 ||
        error("IRStatement has arity $(arity(op)) but called with 4 args")
    return IRStatement(op, target, (a1, a2, a3, a4))
end

struct IntermediateRepresentation
    statements::Vector{IRStatement}
    assignments::Vector{Tuple{Int, IRStatementArg}}
    output_dim::Int
end

## Instruction and InstructionSequence

struct Instruction
    input::NTuple{4, Int32}
    op::OpType.T
    output::Int32
end

struct InstructionSequence
    instructions::Vector{Instruction}
    constants::Vector{ComplexF64}
    constants_range::UnitRange{Int}
    parameters_range::UnitRange{Int}
    variables_range::UnitRange{Int}
    continuation_parameter_index::Union{Nothing, Int}
    assignments::Vector{Tuple{Int, Int}}
    output_dim::Int
    tape_space_needed::Int
    u_assignments::Vector{Tuple{Int, Int}}
    U_assignments::Vector{Tuple{Int, Int}}
    all_u_assigned::Bool
    all_U_assigned::Bool
end

Base.length(seq::InstructionSequence) = length(seq.instructions)
Base.iterate(seq::InstructionSequence) = iterate(seq.instructions)
Base.iterate(seq::InstructionSequence, state) = iterate(seq.instructions, state)

## IR → InstructionSequence compilation

"""
    build_instruction_sequence_from_ir(ir; nvars, nparams, nconstants, constants,
        continuation_parameter_index=nothing)

Compile an `IntermediateRepresentation` into an `InstructionSequence`.

Tape layout (1-indexed):
- `1:nconstants` — constants (pre-loaded from `constants` vector)
- `nconstants+1:nconstants+nparams` — parameters
- if `continuation_parameter_index` is not nothing, one extra slot after params
- next `nvars` slots — variables
- remaining slots — scratch registers for intermediate results
"""
function build_instruction_sequence_from_ir(
        ir::IntermediateRepresentation;
        nvars::Int,
        nparams::Int,
        nconstants::Int,
        constants::Vector{ComplexF64},
        continuation_parameter_index::Union{Nothing, Int} = nothing,
        variables::Vector{Symbol} = Symbol[],
        parameters::Vector{Symbol} = Symbol[],
    )
    # --- Build arg_index_map: map each IRStatementArg to a tape index ---
    arg_index_map = Dict{IRStatementArg, Int32}()
    tape_index = Int32(0)

    # Constants occupy tape positions 1:nconstants
    constants_range = 1:nconstants
    for k in 1:nconstants
        # Constants in the IR are ComplexF64 values; we map them by value
        # They were collected externally and passed as the `constants` vector.
        # We register each constant value by position.
        arg_index_map[constants[k]] = (tape_index += Int32(1))
    end

    # Parameters occupy the next nparams slots
    parameters_range = range(Int(tape_index) + 1; length = nparams)
    for v in parameters
        arg_index_map[v] = (tape_index += Int32(1))
    end
    if isempty(parameters)
        tape_index += Int32(nparams)
    end

    # Continuation parameter (if any)
    local cont_param_tape_index::Union{Nothing, Int}
    if !isnothing(continuation_parameter_index)
        tape_index += Int32(1)
        cont_param_tape_index = Int(tape_index)
    else
        cont_param_tape_index = nothing
    end

    # Variables occupy the next nvars slots
    variables_start = Int(tape_index) + 1
    for v in variables
        arg_index_map[v] = (tape_index += Int32(1))
    end
    if isempty(variables)
        tape_index += Int32(nvars)
    end
    variables_range = range(variables_start; length = nvars)

    input_block_size = Int(tape_index)

    # --- Map IR statement refs to tape indices ---
    # Each IR statement ref i refers to statement i in ir.statements.
    # Before optimization, statement i gets tape slot input_block_size + i.
    nstmts = length(ir.statements)
    for i in 1:nstmts
        ref = IRStatementRef(i)
        arg_index_map[ref] = Int32(input_block_size + i)
    end

    # Assignment targets also get pre-assigned slots (after scratch space)
    # These are updated during optimize step.
    nassignments = length(ir.assignments)
    assignments_start = input_block_size + nstmts + 1
    assignments_original_range = range(assignments_start; length = nassignments)
    for (k, (_, ref)) in enumerate(ir.assignments)
        if ref isa IRStatementRef
            # Will be updated later; for now, re-map to a slot after nstmts scratch
            arg_index_map[ref] = Int32(assignments_start + k - 1)
        end
    end

    # --- Build raw instructions ---
    instructions = Vector{Instruction}(undef, nstmts)
    for (stmt_idx, stmt) in enumerate(ir.statements)
        instr_op = stmt.op
        a1 = _resolve_ir_arg(stmt.args[1], instr_op, 1, arg_index_map, input_block_size)
        a2 = _resolve_ir_arg(stmt.args[2], instr_op, 2, arg_index_map, input_block_size)
        a3 = _resolve_ir_arg(stmt.args[3], instr_op, 3, arg_index_map, input_block_size)
        a4 = _resolve_ir_arg(stmt.args[4], instr_op, 4, arg_index_map, input_block_size)
        target_idx = get(arg_index_map, stmt.target, Int32(stmt.target.i + input_block_size))
        instructions[stmt_idx] = Instruction((a1, a2, a3, a4), instr_op, target_idx)
    end

    # --- Build initial assignment index range ---
    # assignments_original_range tracks where assignment targets currently live
    # (these slots are after the nstmts scratch area)
    # But we need to look at where the IR assignment refs actually point.
    # The v2 approach: assignment targets get their own reserved slots, separate from scratch.
    # After optimization, they get compacted into [input_block_size + max_scratch + 1 ..]

    # The assignment slots were mapped above as assignments_start+k-1,
    # but the actual target for each assignment is from ir.assignments[k][2] (an IRStatementRef).
    # We need to collect which tape indices are the "assignment outputs".
    assignment_tape_indices = Int32[
        get(arg_index_map, ref, Int32(0)) for (_, ref) in ir.assignments
            if ref isa IRStatementRef
    ]

    # For assignments that are direct constants/nothing, handle separately.
    # Build the full assignments range (as a UnitRange for the compactification step).
    # We use the pre-assigned slots from assignments_original_range.
    assignments_range = assignments_original_range

    # --- Optimize: reorder + compact ---
    instructions_opt = _optimize_instruction_order(instructions)
    instructions_final, space_needed, updated_assignments_range =
        _reduce_space(instructions_opt, input_block_size, assignments_range)

    # --- Build final assignments vector: (output_index, tape_index) ---
    updated_assignments = Vector{Tuple{Int, Int}}(undef, nassignments)
    for (k, ((out_idx, _), tape_idx)) in
        enumerate(zip(ir.assignments, updated_assignments_range))
        updated_assignments[k] = (out_idx, Int(tape_idx))
    end

    # --- Add STOP instruction ---
    n = space_needed
    push!(instructions_final, Instruction((Int32(n), Int32(n), Int32(n), Int32(n)), OpType.OP_STOP, Int32(n)))

    # --- Split assignments into u (function) and U (Jacobian) ---
    odim = ir.output_dim
    u_assignments = Tuple{Int, Int}[
        (i, k) for (i, k) in updated_assignments if i <= odim
    ]
    U_assignments = Tuple{Int, Int}[
        (i - odim, k) for (i, k) in updated_assignments if i > odim
    ]

    return InstructionSequence(
        instructions_final,
        copy(constants),
        constants_range,
        parameters_range,
        variables_range,
        cont_param_tape_index,
        updated_assignments,
        odim,
        space_needed,
        u_assignments,
        U_assignments,
        length(u_assignments) == odim,
        length(U_assignments) == odim * nvars,
    )
end

## Internal helpers

function _resolve_ir_arg(
        arg::IRStatementArg,
        instr_op::OpType.T,
        arg_pos::Int,
        arg_index_map::Dict{IRStatementArg, Int32},
        input_block_size::Int,
    )::Int32
    return if isnothing(arg)
        # Padding — use the previous arg's value (convention from v2)
        # Caller pads with the last valid value; here we just return 1 as safe default
        Int32(1)
    elseif should_use_index_not_reference(instr_op, arg_pos)
        # OP_POW_INT: second arg is an integer exponent stored directly
        Int32(real(arg::ComplexF64))
    elseif arg isa IRStatementRef
        get(arg_index_map, arg, Int32(arg.i + input_block_size))
    elseif arg isa ComplexF64
        get(arg_index_map, arg, Int32(1))
    else
        # Symbol — should be in arg_index_map
        get(arg_index_map, arg, Int32(1))
    end
end

"""
    _optimize_instruction_order(instructions)

Reorder instructions using a DAG-based topological sort for better data locality.
Builds a dependency graph (output → inputs direction), then does a DFS-based
reverse postorder starting from "root" instructions (those whose output is not
consumed by other instructions, i.e., assignment targets).
"""
function _optimize_instruction_order(
        instructions::Vector{Instruction},
    )::Vector{Instruction}
    isempty(instructions) && return instructions

    index_to_instr = Dict{Int32, Instruction}()
    vertices = Set{Int32}()
    vertex_list = Int32[]
    function register_vertex!(v::Int32)
        if v ∉ vertices
            push!(vertices, v)
            push!(vertex_list, v)
        end
        return v
    end
    in_degree = Dict{Int32, Int}()
    out_degree = Dict{Int32, Int}()
    children = Dict{Int32, Vector{Int32}}()

    for instr in instructions
        index_to_instr[instr.output] = instr
        register_vertex!(instr.output)
        get!(children, instr.output, Int32[])
        get!(in_degree, instr.output, 0)
        get!(out_degree, instr.output, 0)
        for k in 1:arity(instr.op)
            should_use_index_not_reference(instr.op, k) && continue
            input_idx = instr.input[k]
            register_vertex!(input_idx)
            push!(get!(children, instr.output, Int32[]), input_idx)
            in_degree[input_idx] = get(in_degree, input_idx, 0) + 1
            out_degree[instr.output] = get(out_degree, instr.output, 0) + 1
            get!(out_degree, input_idx, 0)
            get!(in_degree, instr.output, 0)
        end
    end

    listing = Int32[]
    sort!(vertex_list)
    roots = Int32[v for v in vertex_list if get(in_degree, v, 0) == 0]

    while !isempty(roots)
        root = pop!(roots)
        unlisted_nodes_without_parent = Int32[root]
        while !isempty(unlisted_nodes_without_parent)
            u = pop!(unlisted_nodes_without_parent)
            push!(listing, u)
            u_children = get(children, u, Int32[])
            if u in vertices
                delete!(vertices, u)
            end
            out_degree[u] = 0
            for v in u_children
                in_degree[v] = get(in_degree, v, 0) - 1
                if get(in_degree, v, 0) == 0 && get(out_degree, v, 0) > 0
                    push!(listing, v)
                    if v in vertices
                        delete!(vertices, v)
                    end
                    v_children = get(children, v, Int32[])
                    out_degree[v] = 0
                    for w in v_children
                        in_degree[w] = get(in_degree, w, 0) - 1
                    end
                end
            end

            if isempty(unlisted_nodes_without_parent)
                is_unlisted(v) =
                    get(in_degree, v, 0) == 0 &&
                    get(out_degree, v, 0) > 0 &&
                    v ∉ roots
                unlisted_nodes_without_parent =
                    Int32[v for v in vertex_list if v in vertices && is_unlisted(v)]
            end
        end
    end

    reverse!(listing)
    return [index_to_instr[v] for v in listing if haskey(index_to_instr, v)]
end

"""
    _reduce_space(instructions, input_block_size, assignments)

Apply register allocation to compact the tape indices used by instructions,
then remap all instruction inputs/outputs to the compacted indices.
"""
function _reduce_space(
        instructions::Vector{Instruction},
        input_block_size::Int,
        assignments::UnitRange{Int},
    )
    index_map, space_needed, updated_assignments =
        _index_compactification_mapping(instructions, input_block_size, assignments)

    remapped = map(instructions) do instr
        new_input = ntuple(Val(4)) do k
            if should_use_index_not_reference(instr.op, k)
                instr.input[k]
            else
                Int32(get(index_map, instr.input[k], instr.input[k]))
            end
        end
        new_output = Int32(index_map[instr.output])
        Instruction(new_input, instr.op, new_output)
    end

    return remapped, space_needed, updated_assignments
end

"""
    _index_compactification_mapping(instructions, input_block_size, assignments)

Linear-scan register allocation. Tracks the lifetime (last use) of each
intermediate register and reuses registers once their lifetime ends.
Assignment-target registers get dedicated slots at the end and are never reused.

Returns `(index_map, tape_space_needed, updated_assignments_range)`.
"""
function _index_compactification_mapping(
        instructions::Vector{Instruction},
        input_block_size::Int,
        assignments::UnitRange{Int},
    )
    used_indices = Set{Int32}()
    unused_indices = Vector{Int32}()

    index_map = Dict{Int32, Int32}()
    for idx in Int32(1):Int32(input_block_size)
        index_map[idx] = idx
    end

    get_register!() = begin
        if isempty(unused_indices)
            r = Int32(input_block_size + length(used_indices) + 1)
            push!(used_indices, r)
            return r
        end
        r = pop!(unused_indices)
        push!(used_indices, r)
        r
    end

    # Compute the last instruction index that reads each output register.
    output_lifetime_end = Dict{Int32, Int32}()
    for instr_idx in length(instructions):-1:1
        instr = instructions[instr_idx]
        for i in 1:arity(instr.op)
            should_use_index_not_reference(instr.op, i) && continue
            idx = instr.input[i]
            if !haskey(output_lifetime_end, idx)
                output_lifetime_end[idx] = Int32(instr_idx)
            end
        end
    end

    for (instr_idx, instr) in enumerate(instructions)
        # Allocate output register (skip if this is an assignment-target slot)
        if instr.output ∉ assignments
            index_map[instr.output] = get_register!()
        end

        # Free registers whose lifetime has ended
        for i in 1:arity(instr.op)
            should_use_index_not_reference(instr.op, i) && continue
            input_idx = instr.input[i]
            if input_idx > input_block_size &&
                    instr_idx == Int(get(output_lifetime_end, input_idx, Int32(0)))
                mapped_idx = get(index_map, input_idx, input_idx)
                if mapped_idx ∈ used_indices
                    pop!(used_indices, mapped_idx)
                    push!(unused_indices, mapped_idx)
                end
            end
        end
    end

    max_scratch = length(unused_indices)
    # Assignment slots come after scratch space
    updated_assignments =
        range(input_block_size + max_scratch + 1; length = length(assignments))
    for (k, orig_idx) in enumerate(assignments)
        index_map[Int32(orig_idx)] = Int32(input_block_size + max_scratch + k)
    end

    tape_space_needed = last(updated_assignments)
    return index_map, tape_space_needed, updated_assignments
end
