## Arbitrary-precision (Arb) tape interpreter.
##
## The generic `Interpreter{V}` uses value semantics (`tape[i] = op(...)`), which
## is incompatible with Arb's precision-carrying in-place mutation: every `Acb`
## carries its own working precision and every operation writes into a
## pre-allocated ball. This file provides a dedicated interpreter over an
## `AcbRefVector` tape that dispatches each `ExecInstruction` to an in-place
## `Arblib.*!` operation, mirroring the generic loop in `interpreter.jl`.
##
## Used only by the extended-precision Krawczyk fallback in `certify`; not a hot
## path, so the working precision is settable at run time via `setprecision!`.

# ─────────────────────────────────────────────────────────────────────────────
# In-place Acb operations (output ref `t`, input refs, scratch tuple `m`)
# ─────────────────────────────────────────────────────────────────────────────

# arity 1
Base.@propagate_inbounds function acb_op_cb!(t, x, m)
    Arblib.sqr!(m[1], x)
    return Arblib.mul!(t, m[1], x)
end
Base.@propagate_inbounds acb_op_cos!(t, x, m) = Arblib.cos!(t, x)
Base.@propagate_inbounds acb_op_inv!(t, x, m) = Arblib.inv!(t, x)
Base.@propagate_inbounds acb_op_inv_not_zero!(t, x, m) =
    iszero(x) ? Arblib.set!(t, x) : Arblib.inv!(t, x)
Base.@propagate_inbounds function acb_op_invsqr!(t, x, m)
    Arblib.sqr!(m[1], x)
    return Arblib.inv!(t, m[1])
end
Base.@propagate_inbounds acb_op_neg!(t, x, m) = Arblib.neg!(t, x)
Base.@propagate_inbounds acb_op_sin!(t, x, m) = Arblib.sin!(t, x)
Base.@propagate_inbounds acb_op_sqr!(t, x, m) = Arblib.sqr!(t, x)

# A ball meeting a branch cut breaks the Krawczyk hypotheses, so reject it as the
# `IComplex{Float64}` operations do. Arb answers on a narrow straddling ball.
_acb_on_negative_axis(x::Arblib.AcbOrRef)::Bool =
    Arblib.contains_negative(Arblib.realref(x)) &&
    Arblib.contains_zero(Arblib.imagref(x))

function _acb_on_inverse_trig_cut(x::Arblib.AcbOrRef)::Bool
    Arblib.contains_zero(Arblib.imagref(x)) || return false
    re = Arblib.realref(x)
    return Arblib.ubound(re) > 1 || Arblib.lbound(re) < -1
end

Base.@propagate_inbounds acb_op_sqrt!(t, x::Arblib.AcbOrRef, m) =
    _acb_on_negative_axis(x) ? Arblib.indeterminate!(t) : Arblib.sqrt!(t, x)
Base.@propagate_inbounds acb_op_identity!(t, x, m) = Arblib.set!(t, x)
Base.@propagate_inbounds acb_op_exp!(t, x, m) = Arblib.exp!(t, x)
Base.@propagate_inbounds acb_op_log!(t, x::Arblib.AcbOrRef, m) =
    _acb_on_negative_axis(x) ? Arblib.indeterminate!(t) : Arblib.log!(t, x)
Base.@propagate_inbounds acb_op_sinh!(t, x, m) = Arblib.sinh!(t, x)
Base.@propagate_inbounds acb_op_cosh!(t, x, m) = Arblib.cosh!(t, x)
Base.@propagate_inbounds acb_op_tan!(t, x, m) = Arblib.tan!(t, x)
Base.@propagate_inbounds acb_op_tanh!(t, x, m) = Arblib.tanh!(t, x)
Base.@propagate_inbounds acb_op_asin!(t, x::Arblib.AcbOrRef, m) =
    _acb_on_inverse_trig_cut(x) ? Arblib.indeterminate!(t) : Arblib.asin!(t, x)
Base.@propagate_inbounds acb_op_acos!(t, x::Arblib.AcbOrRef, m) =
    _acb_on_inverse_trig_cut(x) ? Arblib.indeterminate!(t) : Arblib.acos!(t, x)

# arity 2
Base.@propagate_inbounds acb_op_add!(t, x, y, m) = Arblib.add!(t, x, y)
Base.@propagate_inbounds acb_op_div!(t, x, y, m) = Arblib.div!(t, x, y)
Base.@propagate_inbounds acb_op_mul!(t, x, y, m) = Arblib.mul!(t, x, y)
Base.@propagate_inbounds acb_op_sub!(t, x, y, m) = Arblib.sub!(t, x, y)
Base.@propagate_inbounds acb_op_pow_int!(t, x, k, m) = Arblib.pow!(t, x, convert(Int, k))
Base.@propagate_inbounds acb_op_pow!(t, x::Arblib.AcbOrRef, y, m) =
    _acb_on_negative_axis(x) ? Arblib.indeterminate!(t) : Arblib.pow!(t, x, y)

# arity 3
Base.@propagate_inbounds function acb_op_add3!(t, x, y, z, m)
    Arblib.add!(m[1], x, y)
    return Arblib.add!(t, m[1], z)
end
Base.@propagate_inbounds function acb_op_mul3!(t, x, y, z, m)
    Arblib.mul!(m[1], x, y)
    return Arblib.mul!(t, m[1], z)
end
Base.@propagate_inbounds function acb_op_muladd!(t, x, y, z, m)
    Arblib.mul!(m[1], x, y)
    return Arblib.add!(t, m[1], z)
end
Base.@propagate_inbounds function acb_op_mulsub!(t, x, y, z, m)
    Arblib.mul!(m[1], x, y)
    return Arblib.sub!(t, m[1], z)
end
Base.@propagate_inbounds function acb_op_submul!(t, x, y, z, m)
    Arblib.mul!(m[1], x, y)
    return Arblib.sub!(t, z, m[1])
end

# arity 4
Base.@propagate_inbounds function acb_op_add4!(t, x, y, z, w, m)
    Arblib.add!(m[1], x, y)
    Arblib.add!(m[2], z, w)
    return Arblib.add!(t, m[1], m[2])
end
Base.@propagate_inbounds function acb_op_mul4!(t, x, y, z, w, m)
    Arblib.mul!(m[1], x, y)
    Arblib.mul!(m[2], z, w)
    return Arblib.mul!(t, m[1], m[2])
end
Base.@propagate_inbounds function acb_op_mulmuladd!(t, x, y, z, w, m)
    Arblib.mul!(m[1], x, y)
    Arblib.mul!(m[2], z, w)
    return Arblib.add!(t, m[1], m[2])
end
Base.@propagate_inbounds function acb_op_mulmulsub!(t, x, y, z, w, m)
    Arblib.mul!(m[1], x, y)
    Arblib.mul!(m[2], z, w)
    return Arblib.sub!(t, m[1], m[2])
end

# The in-place op function name for an `OpType` value.
acb_op_call(op::OpType.T)::Symbol = Symbol(:acb_, op_call(op), :!)

# ─────────────────────────────────────────────────────────────────────────────
# Generated in-place execution loop
# ─────────────────────────────────────────────────────────────────────────────

function _build_acb_execute_call(op::OpType.T, fn_name::Symbol)
    args = Any[:(tape[s.output])]
    for k in 1:arity(op)
        field = Symbol(:arg_, k)
        if should_use_index_not_reference(op, k)
            push!(args, :(s.$field))
        else
            push!(args, :(tape[s.$field]))
        end
    end
    push!(args, :m)
    return Expr(:call, fn_name, args...)
end

function _generate_acb_execute_body()
    cond_body = Tuple{Expr, Expr}[]
    for (variant, op_name) in _EXEC_INSTRUCTION_SPECS
        storage_type = variant_storage_type(getfield(ExecInstruction, variant))
        op = getfield(OpType, op_name)
        cond = :(s isa $storage_type)
        if op == OpType.OP_STOP
            push!(cond_body, (cond, :(return nothing)))
        else
            call = _build_acb_execute_call(op, acb_op_call(op))
            push!(cond_body, (cond, :(@inbounds $call)))
        end
    end
    return nested_ifs(cond_body)
end

let body = _generate_acb_execute_body()
    @eval Base.@propagate_inbounds function execute_acb_instructions!(
            tape::AcbRefVector,
            instructions::Vector{ExecInstructionT},
        )::Nothing
        # Two scratch balls at the tape's working precision.
        m = (copy(tape[1]), copy(tape[1]))
        @inbounds for instr in instructions
            s = exec_instruction_storage(instr)
            $body
        end
        return nothing
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# AcbInterpreter
# ─────────────────────────────────────────────────────────────────────────────

"""
    AcbInterpreter

Tape interpreter over an `AcbRefVector` for arbitrary-precision evaluation. The
working precision is mutable (`setprecision!` re-views the tape at a new
precision), so the `tape` field cannot be `const`.
"""
mutable struct AcbInterpreter
    const sequence::InstructionSequence
    const instructions::Vector{ExecInstructionT}
    tape::AcbRefVector
end

function Base.show(io::IO, I::AcbInterpreter)
    return print(io, "AcbInterpreter for ", length(I.sequence), " instructions")
end

function AcbInterpreter(seq::InstructionSequence; prec::Int = 128)
    instructions = _compile_exec_instructions(seq.instructions)
    tape = AcbRefVector(seq.tape_space_needed; prec = prec)
    @inbounds for (i, k) in enumerate(seq.constants_range)
        Arblib.set!(tape[k], seq.constants[i])
    end
    return AcbInterpreter(seq, instructions, tape)
end

"""
    setprecision!(I::AcbInterpreter, prec::Int)

Set the working precision of the interpreter's tape. Constants keep their exact
loaded midpoints (the tape data is shared; only the precision view changes).
"""
function setprecision!(I::AcbInterpreter, prec::Int)
    I.tape = Base.setprecision(I.tape, prec)
    return I
end

## input loading / output extraction

Base.@propagate_inbounds function _acb_load!(I::AcbInterpreter, x, ::Nothing)::Nothing
    @inbounds for (i, k) in enumerate(I.sequence.variables_range)
        Arblib.set!(I.tape[k], x[i])
    end
    return nothing
end
Base.@propagate_inbounds function _acb_load!(I::AcbInterpreter, x, p)::Nothing
    @inbounds for (i, k) in enumerate(I.sequence.parameters_range)
        Arblib.set!(I.tape[k], p[i])
    end
    @inbounds for (i, k) in enumerate(I.sequence.variables_range)
        Arblib.set!(I.tape[k], x[i])
    end
    return nothing
end

Base.@propagate_inbounds function _acb_extract_u!(u, I::AcbInterpreter)::Nothing
    I.sequence.all_u_assigned || Arblib.zero!(u)
    @inbounds for (i, k) in I.sequence.u_assignments
        Arblib.set!(u[i], I.tape[k])
    end
    return nothing
end
Base.@propagate_inbounds function _acb_extract_U!(U, I::AcbInterpreter)::Nothing
    I.sequence.all_U_assigned || Arblib.zero!(U)
    idx = CartesianIndices((I.sequence.output_dim, size(U, 2)))
    @inbounds for (j, k) in I.sequence.U_assignments
        Arblib.set!(U[idx[j]], I.tape[k])
    end
    return nothing
end

## execute! — evaluate system (`u`) and Jacobian (`U`)

Base.@propagate_inbounds function acb_execute!(
        u::AcbRefMatrix, I::AcbInterpreter, x, p,
    )
    _acb_load!(I, x, p)
    execute_acb_instructions!(I.tape, I.instructions)
    _acb_extract_u!(u, I)
    return u
end

Base.@propagate_inbounds function acb_execute!(
        u::Union{Nothing, AcbRefMatrix}, U::AcbRefMatrix, I::AcbInterpreter, x, p,
    )
    _acb_load!(I, x, p)
    execute_acb_instructions!(I.tape, I.instructions)
    _acb_extract_U!(U, I)
    isnothing(u) || _acb_extract_u!(u, I)
    return u
end
