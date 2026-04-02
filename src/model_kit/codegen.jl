## Code generation: InstructionSequence → Julia Expr for RuntimeGeneratedFunctions
#
# Converts tape-based instructions into straight-line Julia code.
# Constants become literals, variables become x[i], intermediates become τₖ.
# No tape vector needed — LLVM optimizes across the entire function.

function _op_globalref(op::OpType.T)::GlobalRef
    return GlobalRef(parentmodule(OpType), op_call(op))
end

function _build_tape_symbol_map(seq::InstructionSequence)::Dict{Int32, Any}
    tape_sym = Dict{Int32, Any}()
    for (i, k) in enumerate(seq.constants_range)
        tape_sym[Int32(k)] = seq.constants[i]
    end
    for (i, k) in enumerate(seq.variables_range)
        tape_sym[Int32(k)] = :(x[$i])
    end
    for (i, k) in enumerate(seq.parameters_range)
        tape_sym[Int32(k)] = :(p[$i])
    end
    return tape_sym
end

function _emit_instruction_body!(
        body::Vector{Any},
        tape_sym::Dict{Int32, Any},
        instructions::Vector{Instruction},
    )::Nothing
    for instr in instructions
        op = instruction_op(instr)
        inp = instruction_input(instr)
        out = instruction_output(instr)
        op == OpType.OP_STOP && break

        sym = Symbol(:τ, out)
        tape_sym[out] = sym
        fn = _op_globalref(op)

        ref(k::Int32) = tape_sym[k]
        rhs = if op == OpType.OP_POW_INT
            Expr(:call, fn, ref(inp[1]), Int(inp[2]))
        elseif arity(op) == 1
            Expr(:call, fn, ref(inp[1]))
        elseif arity(op) == 2
            Expr(:call, fn, ref(inp[1]), ref(inp[2]))
        elseif arity(op) == 3
            Expr(:call, fn, ref(inp[1]), ref(inp[2]), ref(inp[3]))
        else
            Expr(:call, fn, ref(inp[1]), ref(inp[2]), ref(inp[3]), ref(inp[4]))
        end
        push!(body, :($sym = $rhs))
    end
    return nothing
end

"""
    _instruction_sequence_to_eval_expr(seq::InstructionSequence) -> Expr

Generate `(u, x, p) -> nothing` as straight-line Julia code.
"""
function _instruction_sequence_to_eval_expr(seq::InstructionSequence)::Expr
    tape_sym = _build_tape_symbol_map(seq)
    body = Any[]
    _emit_instruction_body!(body, tape_sym, seq.instructions)

    ref(k::Int32) = tape_sym[k]
    if !seq.all_u_assigned
        push!(body, :(fill!(u, zero(eltype(u)))))
    end
    for (i, k) in seq.u_assignments
        push!(body, :(u[$i] = $(ref(Int32(k)))))
    end
    push!(body, :(return nothing))

    return :(
        (u, x, p) -> @inbounds begin
            $(body...)
        end
    )
end

"""
    _instruction_sequence_to_jac_expr(seq::InstructionSequence) -> Expr

Generate `(u, U, x, p) -> nothing` for simultaneous eval + Jacobian.
"""
function _instruction_sequence_to_jac_expr(seq::InstructionSequence)::Expr
    tape_sym = _build_tape_symbol_map(seq)
    body = Any[]
    _emit_instruction_body!(body, tape_sym, seq.instructions)

    ref(k::Int32) = tape_sym[k]

    # Extract Jacobian
    if !seq.all_U_assigned
        push!(body, :(fill!(U, zero(eltype(U)))))
    end
    for (j, k) in seq.U_assignments
        row = ((j - 1) % seq.output_dim) + 1
        col = ((j - 1) ÷ seq.output_dim) + 1
        push!(body, :(U[$row, $col] = $(ref(Int32(k)))))
    end

    # Extract u
    if !seq.all_u_assigned
        push!(body, :(fill!(u, zero(eltype(u)))))
    end
    for (i, k) in seq.u_assignments
        push!(body, :(u[$i] = $(ref(Int32(k)))))
    end
    push!(body, :(return nothing))

    return :(
        (u, U, x, p) -> @inbounds begin
            $(body...)
        end
    )
end

## ── Compiled evaluator builder ──────────────────────────────────────────────

function _build_compiled_evaluator(
        seq_eval::InstructionSequence,
        seq_jac::InstructionSequence,
        interp_df64::Interpreter{Vector{ComplexDF64}},
        interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}},
        interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}},
        interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}},
        neqs::Int,
        nvars::Int,
        nparams::Int,
    )::SystemEvaluator
    eval_fn = @RuntimeGeneratedFunction(_instruction_sequence_to_eval_expr(seq_eval))
    jac_fn = @RuntimeGeneratedFunction(_instruction_sequence_to_jac_expr(seq_jac))

    return SystemEvaluator(
        SysEvalFW((u, x, p) -> (eval_fn(u, x, p); nothing)),
        SysEvalDF64FW((u, x, p) -> (_execute_eval_fw!(u, interp_df64, x, p); nothing)),
        SysEvalJacFW((u, U, x, p) -> (jac_fn(u, U, x, p); nothing)),
        SysTaylor1FW((u, tx, p) -> (execute_taylor!(u, Val(1), interp_t1, tx, p); nothing)),
        SysTaylor2FW((u, tx, p) -> (execute_taylor!(u, Val(2), interp_t2, tx, p); nothing)),
        SysTaylor3FW((u, tx, p) -> (execute_taylor!(u, Val(3), interp_t3, tx, p); nothing)),
        (neqs, nvars),
        nparams,
    )
end
