## Code generation: InstructionSequence → Julia Expr for RuntimeGeneratedFunctions
#
# Converts tape-based instructions into straight-line Julia code.
# Constants become literals, variables become x[i], intermediates become τₖ.
# No tape vector needed — LLVM optimizes across the entire function.

function _op_globalref(op::OpType.T)::GlobalRef
    return GlobalRef(parentmodule(OpType), op_call(op))
end

function _taylor_op_globalref(op::OpType.T)::GlobalRef
    return GlobalRef(parentmodule(OpType), Symbol(:taylor_, op_call(op)))
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

# Shared preamble for Taylor tape symbol maps: constants → TTS locals, variables → TTS locals
function _taylor_preamble!(
        tape_sym::Dict{Int32, Any},
        preamble::Vector{Any},
        seq::InstructionSequence,
        N::Int,
    )::Nothing
    for (i, k) in enumerate(seq.constants_range)
        c = seq.constants[i]
        sym = Symbol(:_c, i)
        tape_sym[Int32(k)] = sym
        push!(preamble, :($sym = TruncatedTaylorSeries{$N, ComplexF64}($c)))
    end
    for (i, k) in enumerate(seq.variables_range)
        sym = Symbol(:_x, i)
        tape_sym[Int32(k)] = sym
        push!(preamble, :($sym = tx[$i]))
    end
    return nothing
end

# Scalar parameters: promote to order-0 TTS
function _build_taylor_tape_symbol_map(
        seq::InstructionSequence, ::Val{K},
    )::Tuple{Dict{Int32, Any}, Vector{Any}} where {K}
    N = K + 1
    tape_sym = Dict{Int32, Any}()
    preamble = Any[]
    _taylor_preamble!(tape_sym, preamble, seq, N)
    for (i, k) in enumerate(seq.parameters_range)
        sym = Symbol(:_p, i)
        tape_sym[Int32(k)] = sym
        push!(preamble, :($sym = TruncatedTaylorSeries{$N, ComplexF64}(p[$i])))
    end
    return tape_sym, preamble
end

# TaylorVector parameters: load full TTS from TaylorVector
function _build_taylor_param_tape_symbol_map(
        seq::InstructionSequence, ::Val{K},
    )::Tuple{Dict{Int32, Any}, Vector{Any}} where {K}
    N = K + 1
    tape_sym = Dict{Int32, Any}()
    preamble = Any[]
    _taylor_preamble!(tape_sym, preamble, seq, N)
    for (i, k) in enumerate(seq.parameters_range)
        sym = Symbol(:_p, i)
        tape_sym[Int32(k)] = sym
        push!(preamble, :($sym = tp[$i]))
    end
    return tape_sym, preamble
end

function _emit_instruction_body!(
        body::Vector{Any},
        tape_sym::Dict{Int32, Any},
        instructions::Vector{Instruction},
        globalref_fn::F = _op_globalref,
    )::Nothing where {F}
    for instr in instructions
        op = instruction_op(instr)
        inp = instruction_input(instr)
        out = instruction_output(instr)
        op == OpType.OP_STOP && break

        sym = Symbol(:τ, out)
        tape_sym[out] = sym
        fn = globalref_fn(op)

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

# Shared tail for Taylor expr generators: extract order-K coefficients into u
function _taylor_extract_output!(
        body::Vector{Any},
        tape_sym::Dict{Int32, Any},
        seq::InstructionSequence,
        K::Int,
    )::Nothing
    ref(k::Int32) = tape_sym[k]
    if !seq.all_u_assigned
        push!(body, :(fill!(u, zero(eltype(u)))))
    end
    for (i, k) in seq.u_assignments
        push!(body, :(u[$i] = $(ref(Int32(k)))[$K]))
    end
    push!(body, :(return nothing))
    return nothing
end

"""
    _instruction_sequence_to_taylor_expr(seq, Val(K)) -> Expr

Generate `(u, tx, p) -> nothing` as straight-line Taylor code for order K.
`p` is a scalar parameter vector; parameters are promoted to order-0 TTS.

All inputs are pre-bound to local variables in the preamble to avoid
repeated TaylorVector indexing.
"""
function _instruction_sequence_to_taylor_expr(
        seq::InstructionSequence, ::Val{K},
    )::Expr where {K}
    tape_sym, preamble = _build_taylor_tape_symbol_map(seq, Val(K))
    body = copy(preamble)
    _emit_instruction_body!(body, tape_sym, seq.instructions, _taylor_op_globalref)
    _taylor_extract_output!(body, tape_sym, seq, K)

    return :(
        (u, tx, p) -> @inbounds begin
            $(body...)
        end
    )
end

"""
    _instruction_sequence_to_taylor_param_expr(seq, Val(K)) -> Expr

Generate `(u, tx, tp) -> nothing` as straight-line Taylor code for order K.
`tp` is a `TaylorVector{K+1}` carrying full Taylor series for parameters.
This is the production path used by CoefficientHomotopy and ToricHomotopy.
"""
function _instruction_sequence_to_taylor_param_expr(
        seq::InstructionSequence, ::Val{K},
    )::Expr where {K}
    tape_sym, preamble = _build_taylor_param_tape_symbol_map(seq, Val(K))
    body = copy(preamble)
    _emit_instruction_body!(body, tape_sym, seq.instructions, _taylor_op_globalref)
    _taylor_extract_output!(body, tape_sym, seq, K)

    return :(
        (u, tx, tp) -> @inbounds begin
            $(body...)
        end
    )
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

# Interpreter-based Taylor FunctionWrappers
function _build_taylor_fws(
        interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}},
        interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}},
        interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}},
    )
    return (
        SysTaylor1FW((u, tx, p) -> (execute_taylor!(u, Val(1), interp_t1, tx, p); nothing)),
        SysTaylor2FW((u, tx, p) -> (execute_taylor!(u, Val(2), interp_t2, tx, p); nothing)),
        SysTaylor3FW((u, tx, p) -> (execute_taylor!(u, Val(3), interp_t3, tx, p); nothing)),
        SysTaylor1ParamFW((u, tx, tp) -> (execute_taylor!(u, Val(1), interp_t1, tx, tp); nothing)),
        SysTaylor2ParamFW((u, tx, tp) -> (execute_taylor!(u, Val(2), interp_t2, tx, tp); nothing)),
        SysTaylor3ParamFW((u, tx, tp) -> (execute_taylor!(u, Val(3), interp_t3, tx, tp); nothing)),
    )
end

# RGF-compiled Taylor FunctionWrappers
function _build_taylor_fws(seq_eval::InstructionSequence)
    _f1 = @RuntimeGeneratedFunction(_instruction_sequence_to_taylor_expr(seq_eval, Val(1)))
    _f2 = @RuntimeGeneratedFunction(_instruction_sequence_to_taylor_expr(seq_eval, Val(2)))
    _f3 = @RuntimeGeneratedFunction(_instruction_sequence_to_taylor_expr(seq_eval, Val(3)))
    _f1p = @RuntimeGeneratedFunction(_instruction_sequence_to_taylor_param_expr(seq_eval, Val(1)))
    _f2p = @RuntimeGeneratedFunction(_instruction_sequence_to_taylor_param_expr(seq_eval, Val(2)))
    _f3p = @RuntimeGeneratedFunction(_instruction_sequence_to_taylor_param_expr(seq_eval, Val(3)))
    return (
        SysTaylor1FW((u, tx, p) -> (_f1(u, tx, p); nothing)),
        SysTaylor2FW((u, tx, p) -> (_f2(u, tx, p); nothing)),
        SysTaylor3FW((u, tx, p) -> (_f3(u, tx, p); nothing)),
        SysTaylor1ParamFW((u, tx, tp) -> (_f1p(u, tx, tp); nothing)),
        SysTaylor2ParamFW((u, tx, tp) -> (_f2p(u, tx, tp); nothing)),
        SysTaylor3ParamFW((u, tx, tp) -> (_f3p(u, tx, tp); nothing)),
    )
end

function _build_compiled_evaluator(
        seq_eval::InstructionSequence,
        seq_jac::InstructionSequence,
        interp_df64::Interpreter{Vector{ComplexDF64}},
        taylor_fws::Tuple,
        neqs::Int,
        nvars::Int,
        nparams::Int,
    )::SystemEvaluator
    eval_fn = @RuntimeGeneratedFunction(_instruction_sequence_to_eval_expr(seq_eval))
    jac_fn = @RuntimeGeneratedFunction(_instruction_sequence_to_jac_expr(seq_jac))

    return SystemEvaluator(
        SysEvalFW((u, x, p) -> (eval_fn(u, x, p); nothing)),
        _lazy_df64_interpreted(interp_df64)...,
        SysEvalJacFW((u, U, x, p) -> (jac_fn(u, U, x, p); nothing)),
        taylor_fws...,
        (neqs, nvars),
        nparams,
    )
end
