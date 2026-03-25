## Interpreter: tape-based execution engine for InstructionSequence

struct Interpreter{V <: AbstractVector}
    sequence::InstructionSequence
    tape::V
    variables::Vector{Symbol}
    parameters::Vector{Symbol}
end

function Base.show(io::IO, I::Interpreter{V}) where {V}
    return print(io, "Interpreter{", V, "} for ", length(I.sequence), " instructions")
end

## Construction

"""
    create_tape(::Type{V}, seq::InstructionSequence) where {V<:AbstractVector}

Allocate a zero-initialized tape of the correct size and pre-load constants.
"""
function create_tape(::Type{V}, seq::InstructionSequence) where {V <: AbstractVector}
    T = eltype(V)
    tape = V(undef, seq.tape_space_needed)
    @inbounds for j in eachindex(tape)
        tape[j] = zero(T)
    end
    @inbounds for (i, k) in enumerate(seq.constants_range)
        tape[k] = convert(T, seq.constants[i])
    end
    return tape
end

function Interpreter(
        ::Type{V},
        seq::InstructionSequence;
        variables::Vector{Symbol} = Symbol[],
        parameters::Vector{Symbol} = Symbol[],
    ) where {V <: AbstractVector}
    tape = create_tape(V, seq)
    return Interpreter{V}(seq, tape, variables, parameters)
end

## Inner execution loop — code generation
#
# We use @eval to build the generated functions at module load time,
# avoiding Julia 1.12 world age issues with @generated accessing
# helpers (nested_ifs, op_call, etc.) defined in the same compilation unit.

function _build_execute_instructions_inner(level::Int = 0)
    op_types = collect(instances(OpType.T))
    arity1 = filter(op -> arity(op) == 1, op_types)
    arity2 = filter(op -> arity(op) == 2 && op !== OpType.OP_POW_INT, op_types)
    arity3 = filter(op -> arity(op) == 3, op_types)
    arity4 = filter(op -> arity(op) == 4, op_types)

    # Order branches by frequency in polynomial evaluation:
    # MUL, MULADD, MULMULADD, ADD dominate typical tapes.
    # Put high-frequency ops first to minimize branch comparisons.
    # Priority order: arity2 (MUL, ADD, SUB) → arity3 (MULADD, MULSUB) →
    #                 arity4 (MULMULADD) → arity1 (SQR, CB, NEG) → POW_INT
    branches = [
        map(arity2) do op
            (
                :(op == $(op)), quote
                    t_2 = tape[arg_2]
                    tape[i] = $(op_call(op))(t_1, t_2)
                end,
            )
        end
        map(arity3) do op
            (
                :(op == $(op)), quote
                    t_2 = tape[arg_2]
                    t_3 = tape[arg_3]
                    tape[i] = $(op_call(op))(t_1, t_2, t_3)
                end,
            )
        end
        map(arity4) do op
            (
                :(op == $(op)), quote
                    t_2 = tape[arg_2]
                    t_3 = tape[arg_3]
                    t_4 = tape[arg_4]
                    tape[i] = $(op_call(op))(t_1, t_2, t_3, t_4)
                end,
            )
        end
        map(arity1) do op
            (:(op == $(op)), :(tape[i] = $(op_call(op))(t_1)))
        end
        [(:(op == $(OpType.OP_POW_INT)), :(tape[i] = $(op_call(OpType.OP_POW_INT))(t_1, arg_2)))]
    ]

    # Add one level of instruction recursion to reduce loop overhead
    if level < 1
        branches = map(branches) do (cond, code)
            (cond, :($code; $(_build_execute_instructions_inner(level + 1))))
        end
    end

    return quote
        Base.@_propagate_inbounds_meta
        instr = instructions[k += 1]
        op = instr.op
        arg_1, arg_2, arg_3, arg_4 = instr.input
        i = instr.output
        t_1 = tape[arg_1]
        $(
            nested_ifs(
                [
                    branches
                    [(:(op == $(OpType.OP_STOP)), :(break))]
                ]
            )
        )
    end
end

@generated function execute_instructions!(tape::AbstractVector, instructions::Vector{Instruction})
    return quote
        Base.@_propagate_inbounds_meta
        k = 0
        while true
            $(_build_execute_instructions_inner())
        end
    end
end

## execute! — evaluate system

Base.@propagate_inbounds function execute!(
        u::AbstractVector,
        I::Interpreter,
        x::AbstractVector,
    )
    isempty(I.sequence.parameters_range) ||
        error("Interpreter expects parameters; call execute!(u, I, x, p)")

    vars_range = I.sequence.variables_range
    @inbounds for (i, k) in enumerate(vars_range)
        I.tape[k] = x[i]
    end
    @inbounds execute_instructions!(I.tape, I.sequence.instructions)

    I.sequence.all_u_assigned || fill!(u, zero(eltype(u)))
    @inbounds for (i, k) in I.sequence.u_assignments
        u[i] = I.tape[k]
    end

    return u
end

Base.@propagate_inbounds function execute!(
        u::AbstractVector,
        I::Interpreter,
        x::AbstractVector,
        p::AbstractVector,
    )
    vars_range = I.sequence.variables_range
    params_range = I.sequence.parameters_range

    @inbounds for (i, k) in enumerate(params_range)
        I.tape[k] = p[i]
    end
    @inbounds for (i, k) in enumerate(vars_range)
        I.tape[k] = x[i]
    end
    @inbounds execute_instructions!(I.tape, I.sequence.instructions)

    I.sequence.all_u_assigned || fill!(u, zero(eltype(u)))
    @inbounds for (i, k) in I.sequence.u_assignments
        u[i] = I.tape[k]
    end

    return u
end

Base.@propagate_inbounds function execute!(
        u::AbstractVector,
        U::AbstractMatrix,
        I::Interpreter,
        x::AbstractVector,
    )
    isempty(I.sequence.parameters_range) ||
        error("Interpreter expects parameters; call execute!(u, U, I, x, p)")

    vars_range = I.sequence.variables_range
    @inbounds for (i, k) in enumerate(vars_range)
        I.tape[k] = x[i]
    end
    @inbounds execute_instructions!(I.tape, I.sequence.instructions)

    I.sequence.all_U_assigned || fill!(U, zero(eltype(U)))
    idx = CartesianIndices((I.sequence.output_dim, size(U, 2)))
    @inbounds for (j, k) in I.sequence.U_assignments
        U[idx[j]] = I.tape[k]
    end

    I.sequence.all_u_assigned || fill!(u, zero(eltype(u)))
    @inbounds for (i, k) in I.sequence.u_assignments
        u[i] = I.tape[k]
    end

    return u
end

Base.@propagate_inbounds function execute!(
        u::AbstractVector,
        U::AbstractMatrix,
        I::Interpreter,
        x::AbstractVector,
        p::AbstractVector,
    )
    vars_range = I.sequence.variables_range
    params_range = I.sequence.parameters_range

    @inbounds for (i, k) in enumerate(params_range)
        I.tape[k] = p[i]
    end
    @inbounds for (i, k) in enumerate(vars_range)
        I.tape[k] = x[i]
    end
    @inbounds execute_instructions!(I.tape, I.sequence.instructions)

    I.sequence.all_U_assigned || fill!(U, zero(eltype(U)))
    idx = CartesianIndices((I.sequence.output_dim, size(U, 2)))
    @inbounds for (j, k) in I.sequence.U_assignments
        U[idx[j]] = I.tape[k]
    end

    I.sequence.all_u_assigned || fill!(u, zero(eltype(u)))
    @inbounds for (i, k) in I.sequence.u_assignments
        u[i] = I.tape[k]
    end

    return u
end

## Taylor execution — code generation

"""
    taylor_op_call(op::OpType.T) -> Symbol

Map an `OpType` to the corresponding `taylor_op_*` function name.
Used at code-generation time by the Taylor instruction dispatch builder.
"""
function taylor_op_call(op::OpType.T)::Symbol
    return Symbol(:taylor_, op_call(op))
end

function _build_execute_taylor_instructions_inner()
    op_types = collect(instances(OpType.T))
    arity1 = filter(op -> arity(op) == 1, op_types)
    arity2 = filter(op -> arity(op) == 2 && op !== OpType.OP_POW_INT, op_types)
    arity3 = filter(op -> arity(op) == 3, op_types)
    arity4 = filter(op -> arity(op) == 4, op_types)

    # Same frequency-based ordering as execute_instructions!
    branches = [
        map(arity2) do op
            (
                :(op == $(op)), quote
                    t_1 = tape[arg_1]
                    t_2 = tape[arg_2]
                    tape[i] = $(taylor_op_call(op))(t_1, t_2)
                end,
            )
        end
        map(arity3) do op
            (
                :(op == $(op)), quote
                    t_1 = tape[arg_1]
                    t_2 = tape[arg_2]
                    t_3 = tape[arg_3]
                    tape[i] = $(taylor_op_call(op))(t_1, t_2, t_3)
                end,
            )
        end
        map(arity4) do op
            (
                :(op == $(op)), quote
                    t_1 = tape[arg_1]
                    t_2 = tape[arg_2]
                    t_3 = tape[arg_3]
                    t_4 = tape[arg_4]
                    tape[i] = $(taylor_op_call(op))(t_1, t_2, t_3, t_4)
                end,
            )
        end
        map(arity1) do op
            (
                :(op == $(op)), quote
                    t_1 = tape[arg_1]
                    tape[i] = $(taylor_op_call(op))(t_1)
                end,
            )
        end
        [
            (
                :(op == $(OpType.OP_POW_INT)), quote
                    t_1 = tape[arg_1]
                    tape[i] = $(taylor_op_call(OpType.OP_POW_INT))(t_1, arg_2)
                end,
            ),
        ]
    ]

    return quote
        Base.@_propagate_inbounds_meta
        instr = instructions[k += 1]
        op = instr.op
        arg_1, arg_2, arg_3, arg_4 = instr.input
        i = instr.output
        $(
            nested_ifs(
                [
                    branches
                    [(:(op == $(OpType.OP_STOP)), :(break))]
                ]
            )
        )
    end
end

@eval @inline function execute_taylor_instructions!(
        tape::AbstractVector,
        instructions::Vector{Instruction},
    )
    @inbounds begin
        k = 0
        while true
            $(_build_execute_taylor_instructions_inner())
        end
    end
    return nothing
end

## execute_taylor! — evaluate Taylor coefficients

"""
    execute_taylor!(u, ::Val{K}, I, tx, p) where K

Execute the interpreter on Taylor series inputs to compute Taylor coefficients.

- `u::AbstractVector` — output vector (length = output_dim), receives order-K coefficients
- `Val{K}` — the Taylor order to extract
- `I::Interpreter` — the interpreter (tape element type must be `TruncatedTaylorSeries{N,T}`)
- `tx::TaylorVector` — Taylor coefficients of the variables
- `p::AbstractVector` — parameter values (scalars, not Taylor series)

The tape stores `TruncatedTaylorSeries` values throughout execution.
Constants and parameters are promoted to order-0 Taylor series automatically
via the tape's element type conversion.
"""
Base.@propagate_inbounds function execute_taylor!(
        u::AbstractVector,
        ::Val{K},
        I::Interpreter,
        tx::TaylorVector,
        p::AbstractVector,
    ) where {K}
    vars_range = I.sequence.variables_range
    params_range = I.sequence.parameters_range
    TTS = eltype(I.tape)

    # Load parameters as order-0 Taylor series
    @inbounds for (i, k) in enumerate(params_range)
        I.tape[k] = convert(TTS, p[i])
    end
    # Load variables as full Taylor series
    @inbounds for (i, k) in enumerate(vars_range)
        I.tape[k] = tx[i]
    end
    @inbounds execute_taylor_instructions!(I.tape, I.sequence.instructions)

    # Extract order-K coefficients into u
    fill!(u, zero(eltype(u)))
    @inbounds for (i, k) in I.sequence.u_assignments
        u[i] = I.tape[k][K]
    end

    return u
end
