## Instruction and InstructionSequence

struct Instruction
    input::NTuple{4, Int32}
    op::OpType.T
    output::Int32
end

@inline instruction_op(instr::Instruction) = instr.op
@inline instruction_input(instr::Instruction) = instr.input
@inline instruction_output(instr::Instruction) = instr.output

@inline function _instruction(
        input::NTuple{4, <:Integer}, op::OpType.T, output::Integer,
    )::Instruction
    return Instruction(
        ntuple(Val(4)) do k
            Int32(input[k])
        end,
        op,
        Int32(output),
    )
end

struct InstructionSequence
    instructions::Vector{Instruction}
    constants::Vector{ComplexF64}
    constants_range::UnitRange{Int}
    parameters_range::UnitRange{Int}
    variables_range::UnitRange{Int}
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

## Internal helpers

@inline function _register_vertex!(
        vertices::Set{Int32},
        vertex_list::Vector{Int32},
        v::Int32,
    )::Int32
    if v ∉ vertices
        push!(vertices, v)
        push!(vertex_list, v)
    end
    return v
end

@inline function _is_unlisted_vertex(
        v::Int32,
        in_degree::Dict{Int32, Int},
        out_degree::Dict{Int32, Int},
        roots::Vector{Int32},
    )::Bool
    return get(in_degree, v, 0) == 0 &&
        get(out_degree, v, 0) > 0 &&
        v ∉ roots
end

function _take_register!(
        used_indices::Set{Int32},
        unused_indices::Vector{Int32},
        next_reg::Base.RefValue{Int32},
        max_reg::Base.RefValue{Int32},
    )::Int32
    if isempty(unused_indices)
        next_reg[] += Int32(1)
        r = next_reg[]
        push!(used_indices, r)
        max_reg[] = max(max_reg[], r)
        return r
    end
    r = pop!(unused_indices)
    push!(used_indices, r)
    return r
end

"""Remap a single instruction's inputs/outputs through a Dict, preserving immediate-value inputs."""
function _remap_instruction(
        instr::Instruction, remap::Dict{Int32, Int32}; output_fallback::Bool = true,
    )::Instruction
    op = instruction_op(instr)
    input = instruction_input(instr)
    new_input = ntuple(Val(4)) do k
        should_use_index_not_reference(op, k) && return input[k]
        Int32(get(remap, input[k], input[k]))
    end
    output = instruction_output(instr)
    new_output = output_fallback ? get(remap, output, output) : Int32(remap[output])
    return _instruction(new_input, op, new_output)
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
    in_degree = Dict{Int32, Int}()
    out_degree = Dict{Int32, Int}()
    children = Dict{Int32, Vector{Int32}}()

    for instr in instructions
        output = instruction_output(instr)
        op = instruction_op(instr)
        input = instruction_input(instr)
        index_to_instr[output] = instr
        _register_vertex!(vertices, vertex_list, output)
        get!(children, output, Int32[])
        get!(in_degree, output, 0)
        get!(out_degree, output, 0)
        for k in 1:arity(op)
            should_use_index_not_reference(op, k) && continue
            input_idx = input[k]
            _register_vertex!(vertices, vertex_list, input_idx)
            push!(get!(children, output, Int32[]), input_idx)
            in_degree[input_idx] = get(in_degree, input_idx, 0) + 1
            out_degree[output] = get(out_degree, output, 0) + 1
            get!(out_degree, input_idx, 0)
            get!(in_degree, output, 0)
        end
    end

    listing = Int32[]
    _stable_sort!(vertex_list, isless)
    roots = Int32[]
    for v in vertex_list
        get(in_degree, v, 0) == 0 && push!(roots, v)
    end

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
                for v in vertex_list
                    v in vertices || continue
                    _is_unlisted_vertex(v, in_degree, out_degree, roots) || continue
                    push!(unlisted_nodes_without_parent, v)
                end
            end
        end
    end

    reverse!(listing)
    ordered = Instruction[]
    for v in listing
        haskey(index_to_instr, v) || continue
        push!(ordered, index_to_instr[v])
    end
    return ordered
end

"""
    _reduce_space(instructions, input_block_size, scratch_assignments, direct_assignments)

Apply register allocation to compact the tape indices used by instructions,
then remap all instruction inputs/outputs to the compacted indices.

- `scratch_assignments`: `UnitRange{Int}` of tape slots that are scratch-based
  assignment targets (need dedicated post-scratch slots).
- `direct_assignments`: `Vector{Tuple{Int, Int32}}` of `(output_index, tape_slot)`
  for assignments pointing directly to input-block slots (no register needed).
"""
function _reduce_space(
        instructions::Vector{Instruction},
        input_block_size::Int,
        scratch_assignments::UnitRange{Int},
        direct_assignments::Vector{Tuple{Int, Int32}},
    )
    index_map, space_needed, updated_scratch_assignments =
        _index_compactification_mapping(instructions, input_block_size, scratch_assignments)

    remapped = Vector{Instruction}(undef, length(instructions))
    for i in eachindex(instructions)
        remapped[i] =
            _remap_instruction(instructions[i], index_map; output_fallback = false)
    end

    return remapped, space_needed, updated_scratch_assignments, direct_assignments
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
    next_reg = Ref(Int32(input_block_size))  # next fresh register to allocate
    max_reg = Ref(Int32(input_block_size))   # high-water mark for scratch registers

    index_map = Dict{Int32, Int32}()
    for idx in Int32(1):Int32(input_block_size)
        index_map[idx] = idx
    end

    # Compute the last instruction index that reads each output register.
    output_lifetime_end = Dict{Int32, Int32}()
    for instr_idx in length(instructions):-1:1
        instr = instructions[instr_idx]
        op = instruction_op(instr)
        input = instruction_input(instr)
        for i in 1:arity(op)
            should_use_index_not_reference(op, i) && continue
            idx = input[i]
            if !haskey(output_lifetime_end, idx)
                output_lifetime_end[idx] = Int32(instr_idx)
            end
        end
    end

    for (instr_idx, instr) in enumerate(instructions)
        op = instruction_op(instr)
        output = instruction_output(instr)
        input = instruction_input(instr)
        # Allocate output register (skip if this is an assignment-target slot)
        if output ∉ assignments
            index_map[output] = _take_register!(
                used_indices, unused_indices, next_reg, max_reg,
            )
        end

        # Free registers whose lifetime has ended
        for i in 1:arity(op)
            should_use_index_not_reference(op, i) && continue
            input_idx = input[i]
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

    # Assignment slots come after all scratch registers (using high-water mark)
    num_scratch = Int(max_reg[]) - input_block_size
    assignment_start = input_block_size + num_scratch + 1
    updated_assignments =
        assignment_start:(assignment_start + length(assignments) - 1)
    for (k, orig_idx) in enumerate(assignments)
        index_map[Int32(orig_idx)] = Int32(input_block_size + num_scratch + k)
    end

    tape_space_needed = isempty(updated_assignments) ? Int(max_reg[]) : last(updated_assignments)
    return index_map, tape_space_needed, updated_assignments
end
