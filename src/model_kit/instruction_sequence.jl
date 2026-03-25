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

## Internal helpers

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
    next_reg = Int32(input_block_size)  # next fresh register to allocate
    max_reg = Int32(input_block_size)   # high-water mark for scratch registers

    index_map = Dict{Int32, Int32}()
    for idx in Int32(1):Int32(input_block_size)
        index_map[idx] = idx
    end

    get_register!() = begin
        if isempty(unused_indices)
            next_reg += Int32(1)
            r = next_reg
            push!(used_indices, r)
            max_reg = max(max_reg, r)
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

    # Assignment slots come after all scratch registers (using high-water mark)
    num_scratch = Int(max_reg) - input_block_size
    updated_assignments =
        range(input_block_size + num_scratch + 1; length = length(assignments))
    for (k, orig_idx) in enumerate(assignments)
        index_map[Int32(orig_idx)] = Int32(input_block_size + num_scratch + k)
    end

    tape_space_needed = last(updated_assignments)
    return index_map, tape_space_needed, updated_assignments
end
