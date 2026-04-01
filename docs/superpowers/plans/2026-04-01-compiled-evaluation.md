# Compiled Evaluation Backend — Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Add an opt-in RGF-compiled evaluation backend for eval and Jacobian. Measure whether it closes the eval kernel gap and what it costs in build time / TTFX. This is an experiment, not a guaranteed fix for the end-to-end solve gap.

**Architecture:** Two modes — `CompileMode.INTERPRETED` (current default, unchanged) and `CompileMode.COMPILED` (opt-in, RGF eval+jac, interpreter Taylor+DF64). Both produce the same concrete `SystemEvaluator`. Default stays `INTERPRETED` until TTFX is measured and accepted.

**Tech Stack:** RuntimeGeneratedFunctions.jl

---

## Design

### Two Modes

| Mode | Eval | Jacobian | Taylor | DF64 | Default? |
|------|------|----------|--------|------|----------|
| `INTERPRETED` | Interpreter | Interpreter | Interpreter | Interpreter | Yes |
| `COMPILED` | RGF | RGF | Interpreter | Interpreter | No (opt-in) |

Taylor stays interpreted in both modes — v2 proved interpreted Taylor is efficient (TruncatedTaylorSeries arithmetic handles all orders generically). DF64 stays interpreted because it's a rare refinement path. Compiled Taylor/DF64 can be added later if profiling justifies it.

Both modes produce `SystemEvaluator` — same concrete type, same FunctionWrapper slots.

### Success Criteria (must measure before changing defaults)

1. **Eval kernel speedup:** RGF eval should be >2x faster than interpreter on katsura-3/cyclic-7
2. **Jacobian kernel speedup:** RGF jac should be >1.5x faster than interpreter
3. **Build time cost:** `System(F; compile=CompileMode.COMPILED)` should be <2x slower than `INTERPRETED`
4. **TTFX cost:** fresh-session first `solve()` with `COMPILED` should be <20s (currently ~13s with `INTERPRETED`)
5. **End-to-end solve:** re-run `benchmark/compare/tracking.jl` with `COMPILED` and measure impact
6. **Zero allocations:** compiled eval/jac through FunctionWrapper must be 0 allocs

Only if criteria 1-4 are met does `COMPILED` become a candidate for default.

### GC Lifetime Model

RGF-compiled callables are captured by FunctionWrapper closures in `SystemEvaluator`. The `System` struct keeps `_interp_*` fields as GC roots (existing pattern). For `COMPILED` mode, the RGF functions are captured by the closures passed to FunctionWrapper — they stay alive as long as the `SystemEvaluator` does. No additional GC root fields needed; the closure captures are sufficient.

### Code Generation Strategy

`_instruction_sequence_to_eval_expr(seq)` walks the instruction list and emits:
- Constants → literal `ComplexF64` values in generated code
- Variables → `x[i]`
- Parameters → `p[i]`
- Intermediates → local variables `τₖ`
- Operations → `GlobalRef(HomotopyContinuationNext, :op_add)` etc.

The generated function signature: `(u, x, p) -> @inbounds begin ... end`
For Jacobian: `(u, U, x, p) -> @inbounds begin ... end`

No tape vector, no dispatch loop — straight-line Julia code that LLVM can optimize.

---

## File Changes

| File | Change |
|------|--------|
| `Project.toml` | Add RuntimeGeneratedFunctions |
| `src/HomotopyContinuationNext.jl` | Add RGF using/init, CompileMode enum, export |
| `src/model_kit/codegen.jl` | NEW — expr generation + RGF compilation |
| `src/core/system.jl` | Add `compile` kwarg, branch on mode |
| `test/codegen_test.jl` | NEW — expr correctness, mode parity, allocs |
| `benchmark/compile_modes.jl` | NEW — v3 compiled vs interpreted (raw + FW) |

---

## Tasks

### Task 1: Add RuntimeGeneratedFunctions + CompileMode enum

**Files:**
- Modify: `Project.toml`
- Modify: `src/HomotopyContinuationNext.jl`

- [ ] **Step 1: Add dep**

```sh
julia --project -e 'using Pkg; Pkg.add("RuntimeGeneratedFunctions")'
```

- [ ] **Step 2: Add to main module**

In `src/HomotopyContinuationNext.jl`, add with the other `using` statements:

```julia
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions, @RuntimeGeneratedFunction
RuntimeGeneratedFunctions.init(@__MODULE__)
```

Add `CompileMode` enum near the other `@enumx` definitions:

```julia
@enumx CompileMode::Int8 begin
    INTERPRETED
    COMPILED
end
```

Add to exports:

```julia
export CompileMode
```

- [ ] **Step 3: Verify**

```sh
julia --project -e 'using HomotopyContinuationNext; println(CompileMode.COMPILED)'
```

---

### Task 2: Implement `_instruction_sequence_to_eval_expr`

**Files:**
- Create: `src/model_kit/codegen.jl`
- Create: `test/codegen_test.jl`

- [ ] **Step 1: Write failing test**

Create `test/codegen_test.jl`:

```julia
using Test
using RuntimeGeneratedFunctions: @RuntimeGeneratedFunction
import HomotopyContinuationNext as HC
using HomotopyContinuationNext:
    System, Interpreter, InstructionSequence, CompileMode,
    _instruction_sequence_to_eval_expr,
    execute!
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}

@testset "Code Generation" begin
    @testset "eval expr matches interpreter: katsura-3" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        sys = System(F)
        seq = sys._interp_f64.sequence

        expr = _instruction_sequence_to_eval_expr(seq)
        @test expr isa Expr

        fn = @RuntimeGeneratedFunction(HC, expr)

        x = ComplexF64[0.3, 0.5, -0.2, 0.1]
        u_compiled = zeros(ComplexF64, 4)
        fn(u_compiled, x, ComplexF64[])

        u_interp = zeros(ComplexF64, 4)
        execute!(u_interp, sys._interp_f64, x)

        @test u_compiled ≈ u_interp atol = 1e-14
    end

    @testset "eval expr matches interpreter: parametric" begin
        @polyvar x a b
        sys = System([a * x^2 + b * x - 1]; parameters = [a, b])
        seq = sys._interp_f64.sequence

        expr = _instruction_sequence_to_eval_expr(seq)
        fn = @RuntimeGeneratedFunction(HC, expr)

        u_compiled = zeros(ComplexF64, 1)
        fn(u_compiled, ComplexF64[0.5], ComplexF64[2.0, 3.0])

        u_interp = zeros(ComplexF64, 1)
        execute!(u_interp, sys._interp_f64, ComplexF64[0.5], ComplexF64[2.0, 3.0])

        @test u_compiled ≈ u_interp atol = 1e-14
    end
end
```

- [ ] **Step 2: Run test, verify fails**

```sh
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/codegen_test.jl")'
```

- [ ] **Step 3: Implement**

Create `src/model_kit/codegen.jl`:

```julia
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
        body::Vector{Expr},
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

        ref = k -> tape_sym[k]
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
    _instruction_sequence_to_eval_expr(seq) -> Expr

Generate `(u, x, p) -> nothing` as straight-line Julia code.
"""
function _instruction_sequence_to_eval_expr(seq::InstructionSequence)::Expr
    tape_sym = _build_tape_symbol_map(seq)
    body = Expr[]
    _emit_instruction_body!(body, tape_sym, seq.instructions)

    ref = k -> tape_sym[k]
    if !seq.all_u_assigned
        push!(body, :(fill!(u, zero(eltype(u)))))
    end
    for (i, k) in seq.u_assignments
        push!(body, :(u[$i] = $(ref(Int32(k)))))
    end
    push!(body, :(return nothing))

    return :((u, x, p) -> @inbounds begin $(body...) end)
end
```

- [ ] **Step 4: Include in main module**

In `src/HomotopyContinuationNext.jl`, add after the interpreter include:

```julia
include("model_kit/codegen.jl")
```

- [ ] **Step 5: Run test, verify passes**

---

### Task 3: Implement `_instruction_sequence_to_jac_expr`

**Files:**
- Modify: `src/model_kit/codegen.jl`
- Modify: `test/codegen_test.jl`

- [ ] **Step 1: Add Jacobian test**

Add to `test/codegen_test.jl`:

```julia
    @testset "jac expr matches interpreter: katsura-3" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        sys = System(F)
        seq_jac = sys._interp_jac.sequence

        expr = HC._instruction_sequence_to_jac_expr(seq_jac)
        fn = @RuntimeGeneratedFunction(HC, expr)

        x = ComplexF64[0.3, 0.5, -0.2, 0.1]
        u_c = zeros(ComplexF64, 4)
        U_c = zeros(ComplexF64, 4, 4)
        fn(u_c, U_c, x, ComplexF64[])

        u_i = zeros(ComplexF64, 4)
        U_i = zeros(ComplexF64, 4, 4)
        execute!(u_i, U_i, sys._interp_jac, x)

        @test u_c ≈ u_i atol = 1e-14
        @test U_c ≈ U_i atol = 1e-14
    end
```

- [ ] **Step 2: Implement**

Add to `src/model_kit/codegen.jl`:

```julia
"""
    _instruction_sequence_to_jac_expr(seq) -> Expr

Generate `(u, U, x, p) -> nothing` for simultaneous eval + Jacobian.
"""
function _instruction_sequence_to_jac_expr(seq::InstructionSequence)::Expr
    tape_sym = _build_tape_symbol_map(seq)
    body = Expr[]
    _emit_instruction_body!(body, tape_sym, seq.instructions)

    ref = k -> tape_sym[k]

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

    return :((u, U, x, p) -> @inbounds begin $(body...) end)
end
```

- [ ] **Step 3: Run tests**

---

### Task 4: Wire into System() with `compile` kwarg

**Files:**
- Modify: `src/core/system.jl`
- Modify: `test/codegen_test.jl`

- [ ] **Step 1: Add mode parity test**

Add to `test/codegen_test.jl`:

```julia
    @testset "System(compile=COMPILED) matches INTERPRETED" begin
        @polyvar x y
        F_polys = [x^2 + y - 1, x * y - 2]

        sys_i = System(F_polys; compile = CompileMode.INTERPRETED)
        sys_c = System(F_polys; compile = CompileMode.COMPILED)

        xv = FSVec{ComplexF64}(ComplexF64[0.3, 0.7])
        pv = FSVec{ComplexF64}(ComplexF64[])
        u_i = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        u_c = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        U_i = HC.FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        U_c = HC.FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))

        HC.evaluate!(u_i, sys_i.evaluator, xv, pv)
        HC.evaluate!(u_c, sys_c.evaluator, xv, pv)
        @test u_i ≈ u_c atol = 1e-14

        HC.evaluate_and_jacobian!(u_i, U_i, sys_i.evaluator, xv, pv)
        HC.evaluate_and_jacobian!(u_c, U_c, sys_c.evaluator, xv, pv)
        @test u_i ≈ u_c atol = 1e-14
        @test U_i ≈ U_c atol = 1e-14
    end

    @testset "System(compile=COMPILED) zero allocations" begin
        @polyvar x y
        sys = System([x^2 + y - 1, x * y - 2]; compile = CompileMode.COMPILED)

        xv = FSVec{ComplexF64}(ComplexF64[0.3, 0.7])
        pv = FSVec{ComplexF64}(ComplexF64[])
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))

        HC.evaluate!(u, sys.evaluator, xv, pv)  # warmup
        @test (@allocated HC.evaluate!(u, sys.evaluator, xv, pv)) == 0
    end

    @testset "System default is INTERPRETED" begin
        @polyvar x y
        # Default compile mode should be INTERPRETED (no TTFX regression)
        sys = System([x^2 + y - 1, x * y - 2])
        # Just verify it works — default is INTERPRETED until TTFX benchmarked
        xv = FSVec{ComplexF64}(ComplexF64[0.3, 0.7])
        pv = FSVec{ComplexF64}(ComplexF64[])
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        HC.evaluate!(u, sys.evaluator, xv, pv)
        @test all(isfinite, Vector(u))
    end
```

- [ ] **Step 2: Modify System constructors**

In `src/core/system.jl`, add `compile` kwarg (default `CompileMode.INTERPRETED`):

```julia
function System(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters = nothing,
        variables = nothing,
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::System
    parameters === nothing && (parameters = _empty_vars(polys))
    variables === nothing && (variables = _effective_variables(polys, parameters))
    return System(polys, parameters, variables, compile)
end

function System(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        parameters::AbstractVector,
        variables::AbstractVector,
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::System
    neqs = length(polys)
    nvars = length(variables)
    nparams = length(parameters)
    return _build_compiled_system(polys, variables, parameters, neqs, nvars, nparams, compile)
end
```

- [ ] **Step 3: Modify `_build_compiled_system` to branch on mode**

Add `compile::CompileMode.T` parameter. Always build all interpreters (needed for Taylor, DF64, GC roots). Branch evaluator construction:

```julia
evaluator = if compile == CompileMode.INTERPRETED
    _build_system_evaluator(
        interp_f64, interp_df64, interp_jac,
        interp_t1, interp_t2, interp_t3,
        neqs, nvars, nparams,
    )
else
    _build_compiled_evaluator(
        seq_eval, seq_jac,
        interp_df64, interp_t1, interp_t2, interp_t3,
        neqs, nvars, nparams,
    )
end
```

- [ ] **Step 4: Implement `_build_compiled_evaluator`**

Add to `src/model_kit/codegen.jl`:

```julia
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
```

- [ ] **Step 5: Run all tests**

```sh
make test
```

---

### Task 5: Benchmark compiled vs interpreted

**Files:**
- Create: `benchmark/compile_modes.jl`

Separate benchmark file — does not overload `v2_modes.jl`.

- [ ] **Step 1: Create benchmark**

```julia
# Compare v3 compilation modes: INTERPRETED vs COMPILED
# Standalone: julia --project=benchmark benchmark/compile_modes.jl

using BenchmarkTools
using DynamicPolynomials: @polyvar
using HomotopyContinuationNext
using HomotopyContinuationNext: Interpreter, execute!, CompileMode
using FixedSizeArrays: FixedSizeArray

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

println("="^72)
println("  v3 Compile Modes: INTERPRETED vs COMPILED")
println("="^72)

# ── Helper ──
function _katsura(vars, n)
    lin = vars[1] + sum(2vars[i] for i in 2:(n + 1)) - 1
    eqs = [lin]
    for l in 0:(n - 1)
        eq = -vars[l + 1]
        for i in (-n):n
            j = l - i
            abs(i) <= n && abs(j) <= n && (eq += vars[abs(i) + 1] * vars[abs(j) + 1])
        end
        push!(eqs, eq)
    end
    return eqs
end

println("\n── Raw eval (no FunctionWrapper) ──")
for n in [3, 5, 7]
    @polyvar kv[1:(n + 1)]
    F = _katsura(kv, n)
    sys_i = System(F; compile = CompileMode.INTERPRETED)
    sys_c = System(F; compile = CompileMode.COMPILED)

    # Raw interpreter path
    I = sys_i._interp_f64
    x = ComplexF64.(randn(n + 1))
    u = zeros(ComplexF64, n + 1)
    execute!(u, I, x)  # warmup
    t_interp = @belapsed execute!($u, $I, $x)

    # Raw compiled path (through SystemEvaluator)
    xv = FSVec{ComplexF64}(x)
    pv = FSVec{ComplexF64}(ComplexF64[])
    uv = FSVec{ComplexF64}(zeros(ComplexF64, n + 1))
    evaluate!(uv, sys_c.evaluator, xv, pv)  # warmup
    t_compiled = @belapsed evaluate!($uv, $(sys_c.evaluator), $xv, $pv)

    # FW-wrapped interpreter for apples-to-apples
    uv2 = FSVec{ComplexF64}(zeros(ComplexF64, n + 1))
    evaluate!(uv2, sys_i.evaluator, xv, pv)  # warmup
    t_interp_fw = @belapsed evaluate!($uv2, $(sys_i.evaluator), $xv, $pv)

    println("  katsura-$n:")
    println("    interp raw:    $(round(t_interp * 1e9; digits=1)) ns")
    println("    interp FW:     $(round(t_interp_fw * 1e9; digits=1)) ns")
    println("    compiled FW:   $(round(t_compiled * 1e9; digits=1)) ns")
    println("    speedup (FW):  $(round(t_interp_fw / t_compiled; digits=2))x")
end

println("\n── Build time ──")
for n in [3, 5, 7]
    @polyvar kv[1:(n + 1)]
    F = _katsura(kv, n)
    t_interp = @belapsed System($F; compile = CompileMode.INTERPRETED)
    t_compiled = @belapsed System($F; compile = CompileMode.COMPILED)
    println("  katsura-$n: interp=$(round(t_interp/1e-6; digits=1))us  compiled=$(round(t_compiled/1e-6; digits=1))us  ratio=$(round(t_compiled/t_interp; digits=2))x")
end

println("\n── End-to-end solve (if compile mode propagates) ──")
for n in [3, 4, 5]
    @polyvar kv[1:(n + 1)]
    F = _katsura(kv, n)
    sys_i = System(F; compile = CompileMode.INTERPRETED)
    sys_c = System(F; compile = CompileMode.COMPILED)

    solve(sys_i)  # warmup
    solve(sys_c)
    t_i = @belapsed solve($sys_i)
    t_c = @belapsed solve($sys_c)

    println("  katsura-$n: interp=$(round(t_i*1e3; digits=2))ms  compiled=$(round(t_c*1e3; digits=2))ms  speedup=$(round(t_i/t_c; digits=2))x")
end
```

- [ ] **Step 2: Run and evaluate against success criteria**

```sh
julia --project=benchmark benchmark/compile_modes.jl
```

Check results against criteria from Design section. Record numbers.

- [ ] **Step 3: Run TTFX with compiled mode**

Edit `benchmark/compare/ttfx.jl` to also test `System(F; compile=CompileMode.COMPILED)` and measure fresh-session impact.

---

### Task 6: Decision point — change default or keep opt-in

Based on Task 5 results:

- If success criteria 1-4 met → consider changing default to `CompileMode.COMPILED`
- If TTFX regresses unacceptably → keep `INTERPRETED` as default
- If eval speedup is <2x → the RGF overhead may not justify the complexity
- Update docs with actual measured numbers regardless of decision

---
