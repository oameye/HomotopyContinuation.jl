## Interpreter: tape-based execution engine for InstructionSequence

@data ExecInstruction begin
    struct Stop
        output::Int32
    end

    struct Cb
        arg_1::Int32
        output::Int32
    end

    struct Cos
        arg_1::Int32
        output::Int32
    end

    struct Inv
        arg_1::Int32
        output::Int32
    end

    struct InvNotZero
        arg_1::Int32
        output::Int32
    end

    struct InvSqr
        arg_1::Int32
        output::Int32
    end

    struct Neg
        arg_1::Int32
        output::Int32
    end

    struct Sin
        arg_1::Int32
        output::Int32
    end

    struct Sqr
        arg_1::Int32
        output::Int32
    end

    struct Sqrt
        arg_1::Int32
        output::Int32
    end

    struct Identity
        arg_1::Int32
        output::Int32
    end

    struct Add
        arg_1::Int32
        arg_2::Int32
        output::Int32
    end

    struct Div
        arg_1::Int32
        arg_2::Int32
        output::Int32
    end

    struct Mul
        arg_1::Int32
        arg_2::Int32
        output::Int32
    end

    struct Sub
        arg_1::Int32
        arg_2::Int32
        output::Int32
    end

    struct PowInt
        arg_1::Int32
        arg_2::Int32
        output::Int32
    end

    struct Add3
        arg_1::Int32
        arg_2::Int32
        arg_3::Int32
        output::Int32
    end

    struct Mul3
        arg_1::Int32
        arg_2::Int32
        arg_3::Int32
        output::Int32
    end

    struct MulAdd
        arg_1::Int32
        arg_2::Int32
        arg_3::Int32
        output::Int32
    end

    struct MulSub
        arg_1::Int32
        arg_2::Int32
        arg_3::Int32
        output::Int32
    end

    struct SubMul
        arg_1::Int32
        arg_2::Int32
        arg_3::Int32
        output::Int32
    end

    struct Add4
        arg_1::Int32
        arg_2::Int32
        arg_3::Int32
        arg_4::Int32
        output::Int32
    end

    struct Mul4
        arg_1::Int32
        arg_2::Int32
        arg_3::Int32
        arg_4::Int32
        output::Int32
    end

    struct MulMulAdd
        arg_1::Int32
        arg_2::Int32
        arg_3::Int32
        arg_4::Int32
        output::Int32
    end

    struct MulMulSub
        arg_1::Int32
        arg_2::Int32
        arg_3::Int32
        arg_4::Int32
        output::Int32
    end
end

const ExecInstructionT = typeof(ExecInstruction.Stop(Int32(0)))

const _EXEC_INSTRUCTION_SPECS = (
    (:Stop, :OP_STOP),
    (:Cb, :OP_CB),
    (:Cos, :OP_COS),
    (:Inv, :OP_INV),
    (:InvNotZero, :OP_INV_NOT_ZERO),
    (:InvSqr, :OP_INVSQR),
    (:Neg, :OP_NEG),
    (:Sin, :OP_SIN),
    (:Sqr, :OP_SQR),
    (:Sqrt, :OP_SQRT),
    (:Identity, :OP_IDENTITY),
    (:Add, :OP_ADD),
    (:Div, :OP_DIV),
    (:Mul, :OP_MUL),
    (:Sub, :OP_SUB),
    (:PowInt, :OP_POW_INT),
    (:Add3, :OP_ADD3),
    (:Mul3, :OP_MUL3),
    (:MulAdd, :OP_MULADD),
    (:MulSub, :OP_MULSUB),
    (:SubMul, :OP_SUBMUL),
    (:Add4, :OP_ADD4),
    (:Mul4, :OP_MUL4),
    (:MulMulAdd, :OP_MULMULADD),
    (:MulMulSub, :OP_MULMULSUB),
)

struct Interpreter{V <: AbstractVector}
    sequence::InstructionSequence
    instructions::Vector{ExecInstructionT}
    tape::V
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
        seq::InstructionSequence,
    ) where {V <: AbstractVector}
    instructions = _compile_exec_instructions(seq.instructions)
    tape = create_tape(V, seq)
    return Interpreter{V}(seq, instructions, tape)
end

## Inner execution loop — variant-based dispatch

@inline exec_instruction_storage(instr::ExecInstructionT) = variant_storage(instr)

function _compile_exec_instruction_call(variant::Symbol, op::OpType.T)::Expr
    ctor = Expr(:., :ExecInstruction, QuoteNode(variant))
    args = op == OpType.OP_STOP ? Any[:out] : Any[Symbol(:arg_, k) for k in 1:arity(op)]
    op == OpType.OP_STOP || push!(args, :out)
    return Expr(:call, ctor, args...)
end

function _compile_exec_instruction_branches()
    return [
        (
                :(op == $op),
                :(return $(_compile_exec_instruction_call(variant, op))),
            ) for (variant, op_name) in _EXEC_INSTRUCTION_SPECS
            for op in (getfield(OpType, op_name),)
    ]
end

@eval @inline function _compile_exec_instruction(instr::Instruction)::ExecInstructionT
    arg_1, arg_2, arg_3, arg_4 = instr.input
    out = instr.output
    op = instr.op
    $(
        nested_ifs(
            _compile_exec_instruction_branches(),
            :(error("Unknown instruction op: ", op)),
        )
    )
end

function _compile_exec_instructions(
        instructions::Vector{Instruction},
    )::Vector{ExecInstructionT}
    compiled = Vector{ExecInstructionT}(undef, length(instructions))
    @inbounds for i in eachindex(instructions)
        compiled[i] = _compile_exec_instruction(instructions[i])
    end
    return compiled
end

## Generated single-function execute loops
#
# Instead of dispatching to 25 separate methods via variant_storage (which returns
# a 25-way Union and triggers dynamic dispatch), we generate a single function with
# an if-elseif chain on `isa` checks. This compiles to tag comparisons — equivalent
# to the old enum-based dispatch, avoiding any vtable lookup.

function _build_execute_call(op::OpType.T, fn_name::Symbol)
    args = Expr[]
    for k in 1:arity(op)
        field = Symbol(:arg_, k)
        if should_use_index_not_reference(op, k)
            push!(args, :(s.$field))
        else
            push!(args, :(tape[s.$field]))
        end
    end
    return Expr(:call, fn_name, args...)
end

function _generate_execute_body(fn_mapper::Function)
    cond_body = Tuple{Expr, Expr}[]
    for (variant, op_name) in _EXEC_INSTRUCTION_SPECS
        storage_type = variant_storage_type(getfield(ExecInstruction, variant))
        op = getfield(OpType, op_name)
        cond = :(s isa $storage_type)
        if op == OpType.OP_STOP
            push!(cond_body, (cond, :(return nothing)))
        else
            call = _build_execute_call(op, fn_mapper(op))
            push!(cond_body, (cond, :(@inbounds tape[s.output] = $call)))
        end
    end
    return nested_ifs(cond_body)
end

let body = _generate_execute_body(op -> op_call(op))
    @eval Base.@propagate_inbounds function execute_instructions!(
            tape::AbstractVector,
            instructions::Vector{ExecInstructionT},
        )::Nothing
        @inbounds for instr in instructions
            s = exec_instruction_storage(instr)
            $body
        end
        return nothing
    end
end

let body = _generate_execute_body(op -> Symbol(:taylor_, op_call(op)))
    @eval Base.@propagate_inbounds function execute_taylor_instructions!(
            tape::AbstractVector,
            instructions::Vector{ExecInstructionT},
        )::Nothing
        @inbounds for instr in instructions
            s = exec_instruction_storage(instr)
            $body
        end
        return nothing
    end
end

## execute! helpers

Base.@propagate_inbounds function _load_inputs!(
        I::Interpreter, x::AbstractVector,
    )::Nothing
    @inbounds for (i, k) in enumerate(I.sequence.variables_range)
        I.tape[k] = x[i]
    end
    return nothing
end

Base.@propagate_inbounds function _load_inputs!(
        I::Interpreter, x::AbstractVector, p::AbstractVector,
    )::Nothing
    @inbounds for (i, k) in enumerate(I.sequence.parameters_range)
        I.tape[k] = p[i]
    end
    @inbounds for (i, k) in enumerate(I.sequence.variables_range)
        I.tape[k] = x[i]
    end
    return nothing
end

Base.@propagate_inbounds function _extract_u!(u::AbstractVector, I::Interpreter)::Nothing
    I.sequence.all_u_assigned || fill!(u, zero(eltype(u)))
    @inbounds for (i, k) in I.sequence.u_assignments
        u[i] = I.tape[k]
    end
    return nothing
end

Base.@propagate_inbounds function _extract_U!(U::AbstractMatrix, I::Interpreter)::Nothing
    I.sequence.all_U_assigned || fill!(U, zero(eltype(U)))
    idx = CartesianIndices((I.sequence.output_dim, size(U, 2)))
    @inbounds for (j, k) in I.sequence.U_assignments
        U[idx[j]] = I.tape[k]
    end
    return nothing
end

## execute! — evaluate system

Base.@propagate_inbounds function _execute_eval!(u::AbstractVector, I::Interpreter)
    @inbounds execute_instructions!(I.tape, I.instructions)
    _extract_u!(u, I)
    return u
end

Base.@propagate_inbounds function _execute_jac!(
        u::AbstractVector, U::AbstractMatrix, I::Interpreter,
    )
    @inbounds execute_instructions!(I.tape, I.instructions)
    _extract_U!(U, I)
    _extract_u!(u, I)
    return u
end

Base.@propagate_inbounds function execute!(
        u::AbstractVector, I::Interpreter, x::AbstractVector,
    )
    isempty(I.sequence.parameters_range) ||
        error("Interpreter expects parameters; call execute!(u, I, x, p)")
    _load_inputs!(I, x)
    return _execute_eval!(u, I)
end

Base.@propagate_inbounds function execute!(
        u::AbstractVector, I::Interpreter, x::AbstractVector, p::AbstractVector,
    )
    _load_inputs!(I, x, p)
    return _execute_eval!(u, I)
end

Base.@propagate_inbounds function execute!(
        u::AbstractVector, U::AbstractMatrix, I::Interpreter, x::AbstractVector,
    )
    isempty(I.sequence.parameters_range) ||
        error("Interpreter expects parameters; call execute!(u, U, I, x, p)")
    _load_inputs!(I, x)
    return _execute_jac!(u, U, I)
end

Base.@propagate_inbounds function execute!(
        u::AbstractVector, U::AbstractMatrix, I::Interpreter,
        x::AbstractVector, p::AbstractVector,
    )
    _load_inputs!(I, x, p)
    return _execute_jac!(u, U, I)
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
    TapeEltype = eltype(I.tape)

    # Load parameters as order-0 Taylor series
    @inbounds for (i, k) in enumerate(params_range)
        I.tape[k] = convert(TapeEltype, p[i])
    end
    # Load variables as full Taylor series
    @inbounds for (i, k) in enumerate(vars_range)
        I.tape[k] = tx[i]
    end
    @inbounds execute_taylor_instructions!(I.tape, I.instructions)

    # Extract order-K coefficients into u
    fill!(u, zero(eltype(u)))
    @inbounds for (i, k) in I.sequence.u_assignments
        u[i] = I.tape[k][K]
    end

    return u
end
