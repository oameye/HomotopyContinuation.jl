# Compiled Taylor Backend (`CompileMode.COMPILED_ALL`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a third compilation mode `CompileMode.COMPILED_ALL` that compiles Taylor coefficient evaluation (orders 1-3) to native code via RuntimeGeneratedFunctions, completing the compilation pipeline alongside eval+jac.

**Architecture:** Extend `codegen.jl` with a Taylor expression generator that emits straight-line Taylor operation code. The generated code pre-binds all inputs (constants as TTS literals, parameters as order-0 TTS, variables from TaylorVector columns) to local variables before the instruction body, avoiding repeated reconstruction of TruncatedTaylorSeries from matrix loads. Operations call `taylor_op_*` functions on TTS values instead of scalar `op_*` functions. The key call path in production is `CoefficientHomotopy`/`ToricHomotopy` → `taylor!(u, Val(K), H.system, tx, H.coeffs)` — a parametric system where `p` is the coefficient vector, exercised on every tracker step.

**Tech Stack:** RuntimeGeneratedFunctions.jl, existing `taylor_op_*` functions from `taylor.jl`, `InstructionSequence` tape infrastructure.

**Profiling context:** Taylor evaluation is ~25-28% of the tracker hot loop in `CompileMode.COMPILED` mode (katsura-5/7). Compiled eval+jac already gets 3-6x kernel speedup. A similar speedup on Taylor would yield ~1.2-1.3x end-to-end improvement.

---

### Task 1: Add `COMPILED_ALL` enum variant

**Files:**
- Modify: `src/HomotopyContinuationNext.jl:19-22`

- [ ] **Step 1: Add the enum variant**

In `src/HomotopyContinuationNext.jl`, change the `CompileMode` enum:

```julia
@enumx CompileMode::Int8 begin
    INTERPRETED
    COMPILED
    COMPILED_ALL
end
```

- [ ] **Step 2: Verify the package still loads**

Run: `julia --project -e 'using HomotopyContinuationNext; println(HomotopyContinuationNext.CompileMode.COMPILED_ALL)'`
Expected: `COMPILED_ALL`

---

### Task 2: Write failing tests for compiled Taylor correctness

**Files:**
- Modify: `test/codegen_test.jl`

- [ ] **Step 1: Add Taylor expr generation tests**

Append to the `"Code Generation"` testset in `test/codegen_test.jl`. Add these imports at the top of the file alongside the existing ones:

```julia
using HomotopyContinuationNext:
    _instruction_sequence_to_taylor_expr,
    execute_taylor!, TaylorVector, TruncatedTaylorSeries, taylor!
```

Then append these testsets inside the `"Code Generation"` testset:

```julia
    # ── Taylor expr ──────────────────────────────────────────────────────

    @testset "taylor expr matches interpreter: katsura-3, order $K" for K in 1:3
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        sys = System(F)
        seq = sys._interp_f64.sequence
        N = K + 1  # TTS order: taylor_1 uses TTS{2}, etc.

        expr = _instruction_sequence_to_taylor_expr(seq, Val(K))
        @test expr isa Expr

        fn = @RuntimeGeneratedFunction(HC, expr)

        nvars = 4
        data = FSMat{ComplexF64}(randn(ComplexF64, N, nvars))
        tx = TaylorVector{N, ComplexF64}(data)
        p = ComplexF64[]

        u_compiled = zeros(ComplexF64, 4)
        fn(u_compiled, tx, p)

        interp = K == 1 ? sys._interp_t1 : K == 2 ? sys._interp_t2 : sys._interp_t3
        u_interp = zeros(ComplexF64, 4)
        execute_taylor!(u_interp, Val(K), interp, tx, p)

        @test u_compiled ≈ u_interp atol = 1.0e-12
    end

    @testset "taylor expr matches interpreter: parametric, order $K" for K in 1:3
        @polyvar x a b
        sys = System([a * x^2 + b * x - 1]; parameters = [a, b])
        seq = sys._interp_f64.sequence
        N = K + 1

        expr = _instruction_sequence_to_taylor_expr(seq, Val(K))
        fn = @RuntimeGeneratedFunction(HC, expr)

        data = FSMat{ComplexF64}(randn(ComplexF64, N, 1))
        tx = TaylorVector{N, ComplexF64}(data)
        p = ComplexF64[2.0, 3.0]

        u_compiled = zeros(ComplexF64, 1)
        fn(u_compiled, tx, p)

        interp = K == 1 ? sys._interp_t1 : K == 2 ? sys._interp_t2 : sys._interp_t3
        u_interp = zeros(ComplexF64, 1)
        execute_taylor!(u_interp, Val(K), interp, tx, p)

        @test u_compiled ≈ u_interp atol = 1.0e-12
    end
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `julia --project -e 'using TestEnv; TestEnv.activate(); include("test/codegen_test.jl")'`
Expected: FAIL — `_instruction_sequence_to_taylor_expr` is not defined yet.

---

### Task 3: Implement Taylor expression generator

**Files:**
- Modify: `src/model_kit/codegen.jl`

The key insight: the existing `_emit_instruction_body!` emits `τₖ = op_mul(τⱼ, τₗ)` using `_op_globalref(op)` which resolves to `op_mul`. For Taylor, we need `τₖ = taylor_op_mul(τⱼ, τₗ)` instead. We also need different input loading: constants and parameters must be promoted to order-0 TTS, and variables must be loaded from TaylorVector columns. Critically, all inputs must be **pre-bound to local variables** before the instruction body to avoid repeated TaylorVector indexing (each `tx[i]` reconstructs a fresh TTS from N matrix loads — see `taylor.jl:128`).

- [ ] **Step 1: Add `_taylor_op_globalref` helper**

Add to `src/model_kit/codegen.jl` after the existing `_op_globalref` function (line 9):

```julia
function _taylor_op_globalref(op::OpType.T)::GlobalRef
    return GlobalRef(parentmodule(OpType), Symbol(:taylor_, op_call(op)))
end
```

- [ ] **Step 2: Add `_build_taylor_tape_symbol_map`**

Add after `_build_tape_symbol_map` (after line 23). This builds the symbol map for Taylor codegen. Unlike `_build_tape_symbol_map` which inlines `x[i]` expressions, this binds all inputs to local variables — constants become named TTS locals, parameters become named TTS locals, and variables become named TTS locals loaded once from the TaylorVector. The function returns both the symbol map and the preamble statements:

```julia
function _build_taylor_tape_symbol_map(
        seq::InstructionSequence, ::Val{K},
    )::Tuple{Dict{Int32, Any}, Vector{Any}} where {K}
    N = K + 1  # TTS{N}: taylor order K needs N coefficients
    tape_sym = Dict{Int32, Any}()
    preamble = Any[]

    # Constants: bind to local TTS variables
    for (i, k) in enumerate(seq.constants_range)
        c = seq.constants[i]
        sym = Symbol(:_c, i)
        tape_sym[Int32(k)] = sym
        push!(preamble, :($sym = TruncatedTaylorSeries{$N, ComplexF64}($c)))
    end
    # Variables: load from TaylorVector once into local TTS variables
    for (i, k) in enumerate(seq.variables_range)
        sym = Symbol(:_x, i)
        tape_sym[Int32(k)] = sym
        push!(preamble, :($sym = tx[$i]))
    end
    # Parameters: bind to local TTS variables
    for (i, k) in enumerate(seq.parameters_range)
        sym = Symbol(:_p, i)
        tape_sym[Int32(k)] = sym
        push!(preamble, :($sym = TruncatedTaylorSeries{$N, ComplexF64}(p[$i])))
    end
    return tape_sym, preamble
end
```

- [ ] **Step 3: Add `_emit_taylor_instruction_body!`**

Add after `_emit_instruction_body!` (after line 55). This is identical to `_emit_instruction_body!` except it uses `_taylor_op_globalref` instead of `_op_globalref`:

```julia
function _emit_taylor_instruction_body!(
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
        fn = _taylor_op_globalref(op)

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
```

- [ ] **Step 4: Add `_instruction_sequence_to_taylor_expr`**

Add after the new `_emit_taylor_instruction_body!`:

```julia
"""
    _instruction_sequence_to_taylor_expr(seq::InstructionSequence, ::Val{K}) -> Expr

Generate `(u, tx, p) -> nothing` as straight-line Taylor code for order K.
`tx` is a `TaylorVector{K+1}`, `p` is a parameter vector (scalars).
The function computes order-K Taylor coefficients and writes them to `u`.

All inputs (constants, variables, parameters) are pre-bound to local variables
in the preamble to avoid repeated TaylorVector indexing (each `tx[i]` reconstructs
a TTS from N matrix loads).
"""
function _instruction_sequence_to_taylor_expr(
        seq::InstructionSequence, ::Val{K},
    )::Expr where {K}
    tape_sym, preamble = _build_taylor_tape_symbol_map(seq, Val(K))
    body = copy(preamble)
    _emit_taylor_instruction_body!(body, tape_sym, seq.instructions)

    # Extract order-K coefficients into output vector
    ref(k::Int32) = tape_sym[k]
    push!(body, :(fill!(u, zero(eltype(u)))))
    for (i, k) in seq.u_assignments
        push!(body, :(u[$i] = $(ref(Int32(k)))[$K]))
    end
    push!(body, :(return nothing))

    return :(
        (u, tx, p) -> @inbounds begin
            $(body...)
        end
    )
end
```

- [ ] **Step 5: Run the Taylor expr tests**

Run: `julia --project -e 'using TestEnv; TestEnv.activate(); include("test/codegen_test.jl")'`
Expected: All Taylor expr tests PASS (katsura-3 and parametric, orders 1-3).

---

### Task 4: Wire compiled Taylor into `_build_compiled_evaluator`

**Files:**
- Modify: `src/model_kit/codegen.jl:123-147`
- Modify: `src/core/system.jl:159-171`

- [ ] **Step 1: Add `_build_fully_compiled_evaluator` function**

Add to `src/model_kit/codegen.jl` after `_build_compiled_evaluator` (after line 147):

```julia
function _build_fully_compiled_evaluator(
        seq_eval::InstructionSequence,
        seq_jac::InstructionSequence,
        interp_df64::Interpreter{Vector{ComplexDF64}},
        neqs::Int,
        nvars::Int,
        nparams::Int,
    )::SystemEvaluator
    eval_fn = @RuntimeGeneratedFunction(_instruction_sequence_to_eval_expr(seq_eval))
    jac_fn = @RuntimeGeneratedFunction(_instruction_sequence_to_jac_expr(seq_jac))
    taylor1_fn = @RuntimeGeneratedFunction(_instruction_sequence_to_taylor_expr(seq_eval, Val(1)))
    taylor2_fn = @RuntimeGeneratedFunction(_instruction_sequence_to_taylor_expr(seq_eval, Val(2)))
    taylor3_fn = @RuntimeGeneratedFunction(_instruction_sequence_to_taylor_expr(seq_eval, Val(3)))

    return SystemEvaluator(
        SysEvalFW((u, x, p) -> (eval_fn(u, x, p); nothing)),
        SysEvalDF64FW((u, x, p) -> (_execute_eval_fw!(u, interp_df64, x, p); nothing)),
        SysEvalJacFW((u, U, x, p) -> (jac_fn(u, U, x, p); nothing)),
        SysTaylor1FW((u, tx, p) -> (taylor1_fn(u, tx, p); nothing)),
        SysTaylor2FW((u, tx, p) -> (taylor2_fn(u, tx, p); nothing)),
        SysTaylor3FW((u, tx, p) -> (taylor3_fn(u, tx, p); nothing)),
        (neqs, nvars),
        nparams,
    )
end
```

Note: DF64 evaluation still uses the interpreter — there's no DF64 codegen and it's not on the hot path.

- [ ] **Step 2: Add `COMPILED_ALL` branch in `_build_compiled_system`**

In `src/core/system.jl`, change the evaluator construction block (lines 159-171) from:

```julia
    evaluator = if compile == CompileMode.INTERPRETED
        _build_system_evaluator(
            interp_f64, interp_df64, interp_jac,
            interp_t1, interp_t2, interp_t3,
            neqs, nvars, nparams,
        )
    else  # CompileMode.COMPILED
        _build_compiled_evaluator(
            seq_eval, seq_jac,
            interp_df64, interp_t1, interp_t2, interp_t3,
            neqs, nvars, nparams,
        )
    end
```

to:

```julia
    evaluator = if compile == CompileMode.INTERPRETED
        _build_system_evaluator(
            interp_f64, interp_df64, interp_jac,
            interp_t1, interp_t2, interp_t3,
            neqs, nvars, nparams,
        )
    elseif compile == CompileMode.COMPILED
        _build_compiled_evaluator(
            seq_eval, seq_jac,
            interp_df64, interp_t1, interp_t2, interp_t3,
            neqs, nvars, nparams,
        )
    else  # CompileMode.COMPILED_ALL
        _build_fully_compiled_evaluator(
            seq_eval, seq_jac,
            interp_df64,
            neqs, nvars, nparams,
        )
    end
```

- [ ] **Step 3: Run the existing test suite**

Run: `make test`
Expected: All existing tests still pass. The new enum variant doesn't break anything.

---

### Task 5: Add end-to-end tests for `COMPILED_ALL`

**Files:**
- Modify: `test/codegen_test.jl`

- [ ] **Step 1: Add `COMPILED_ALL` system-level tests (parameter-free)**

Append to the `"Code Generation"` testset in `test/codegen_test.jl`:

```julia
    # ── COMPILED_ALL system-level tests ──────────────────────────────────

    @testset "System(compile=COMPILED_ALL) matches INTERPRETED" begin
        @polyvar x y
        F_polys = [x^2 + y - 1, x * y - 2]

        sys_i = System(F_polys; compile = CompileMode.INTERPRETED)
        sys_c = System(F_polys; compile = CompileMode.COMPILED_ALL)

        xv = FSVec{ComplexF64}(ComplexF64[0.3, 0.7])
        pv = FSVec{ComplexF64}(ComplexF64[])

        u_i = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        u_c = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u_i, sys_i.evaluator, xv, pv)
        evaluate!(u_c, sys_c.evaluator, xv, pv)
        @test u_i ≈ u_c atol = 1.0e-14

        U_i = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        U_c = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        evaluate_and_jacobian!(u_i, U_i, sys_i.evaluator, xv, pv)
        evaluate_and_jacobian!(u_c, U_c, sys_c.evaluator, xv, pv)
        @test u_i ≈ u_c atol = 1.0e-14
        @test U_i ≈ U_c atol = 1.0e-14

        # Taylor correctness through SystemEvaluator
        for K in 1:3
            N = K + 1
            data = FSMat{ComplexF64}(randn(ComplexF64, N, 2))
            tx = TaylorVector{N, ComplexF64}(data)
            u_ti = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            u_tc = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            taylor!(u_ti, Val(K), sys_i.evaluator, tx, pv)
            taylor!(u_tc, Val(K), sys_c.evaluator, tx, pv)
            @test u_ti ≈ u_tc atol = 1.0e-12
        end
    end
```

- [ ] **Step 2: Add parametric `COMPILED_ALL` Taylor tests (production call pattern)**

This is the critical test — it exercises the exact call path used by `CoefficientHomotopy` and `ToricHomotopy`, which call `taylor!(u, Val(K), H.system, tx, H.coeffs)` where `H.coeffs` is a non-empty parameter vector:

```julia
    @testset "System(compile=COMPILED_ALL) parametric Taylor matches INTERPRETED" begin
        @polyvar x y a b c d
        # Parametric system similar to what CoefficientHomotopy builds:
        # coefficients multiply monomials, parameters change along the homotopy path
        F_polys = [a * x^2 + b * x * y + c * y - d, b * x + a * y^2 - c]

        sys_i = System(F_polys; parameters = [a, b, c, d], compile = CompileMode.INTERPRETED)
        sys_c = System(F_polys; parameters = [a, b, c, d], compile = CompileMode.COMPILED_ALL)

        pv = FSVec{ComplexF64}(ComplexF64[1.5, -0.3, 2.1, 0.7])

        # Eval + jac correctness with parameters
        xv = FSVec{ComplexF64}(ComplexF64[0.3, 0.7])
        u_i = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        u_c = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u_i, sys_i.evaluator, xv, pv)
        evaluate!(u_c, sys_c.evaluator, xv, pv)
        @test u_i ≈ u_c atol = 1.0e-14

        # Taylor correctness with parameters — the production CoefficientHomotopy path
        for K in 1:3
            N = K + 1
            data = FSMat{ComplexF64}(randn(ComplexF64, N, 2))
            tx = TaylorVector{N, ComplexF64}(data)
            u_ti = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            u_tc = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            taylor!(u_ti, Val(K), sys_i.evaluator, tx, pv)
            taylor!(u_tc, Val(K), sys_c.evaluator, tx, pv)
            @test u_ti ≈ u_tc atol = 1.0e-12
        end
    end
```

- [ ] **Step 3: Add zero-allocation test for compiled Taylor**

Following the pattern from `test/codegen_test.jl:118` and `test/core_test.jl:660`:

```julia
    @testset "System(compile=COMPILED_ALL) zero allocations: eval + jac + taylor" begin
        @polyvar x y
        sys = System([x^2 + y - 1, x * y - 2]; compile = CompileMode.COMPILED_ALL)

        xv = FSVec{ComplexF64}(ComplexF64[0.3, 0.7])
        pv = FSVec{ComplexF64}(ComplexF64[])
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))

        function _test_compiled_all_allocs(sys, u, xv, pv)
            evaluate!(u, sys.evaluator, xv, pv)  # warmup
            return @allocated evaluate!(u, sys.evaluator, xv, pv)
        end
        @test _test_compiled_all_allocs(sys, u, xv, pv) == 0

        # Taylor zero-allocation checks
        for K in 1:3
            N = K + 1
            data = FSMat{ComplexF64}(randn(ComplexF64, N, 2))
            tx = TaylorVector{N, ComplexF64}(data)
            ut = FSVec{ComplexF64}(zeros(ComplexF64, 2))

            # Use function barrier to avoid top-level scope artifacts
            function _test_taylor_allocs(sys, ut, tx, pv, ::Val{K}) where {K}
                taylor!(ut, Val(K), sys.evaluator, tx, pv)  # warmup
                return @allocated taylor!(ut, Val(K), sys.evaluator, tx, pv)
            end
            @test _test_taylor_allocs(sys, ut, tx, pv, Val(K)) == 0
        end
    end
```

- [ ] **Step 4: Add solve-level correctness tests with residual checks**

```julia
    @testset "System(compile=COMPILED_ALL) solves correctly: residual check" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2]; compile = CompileMode.COMPILED_ALL)
        result = HC.solve(F)
        @test HC.nsolutions(result) >= 2
        for sol in HC.solutions(result)
            res = abs(sol[1]^2 + sol[2] - 1) + abs(sol[1] * sol[2] - 2)
            @test res < 1.0e-10
        end
    end

    @testset "System(compile=COMPILED_ALL) katsura-3: solution agreement" begin
        @polyvar x0 x1 x2 x3
        F_polys = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        r_i = HC.solve(System(F_polys; compile = CompileMode.INTERPRETED))
        r_c = HC.solve(System(F_polys; compile = CompileMode.COMPILED_ALL))
        @test HC.nsolutions(r_i) == HC.nsolutions(r_c)

        # Compare actual solution sets, not just counts
        sols_i = sort(HC.solutions(r_i); by = s -> real(s[1]))
        sols_c = sort(HC.solutions(r_c); by = s -> real(s[1]))
        for (si, sc) in zip(sols_i, sols_c)
            @test si ≈ sc atol = 1.0e-6
        end
    end
```

- [ ] **Step 5: Add parametric solve test (parameter homotopy through CoefficientHomotopy)**

This exercises the full production path: `solve(F, starts; start_parameters, target_parameters)` → `CoefficientHomotopy` → `taylor!(u, Val(K), H.system, tx, H.coeffs)`:

```julia
    @testset "System(compile=COMPILED_ALL) parameter homotopy: solution agreement" begin
        @polyvar x y a
        F_i = System([x^2 - a, y^2 - a]; parameters = [a], compile = CompileMode.INTERPRETED)
        F_c = System([x^2 - a, y^2 - a]; parameters = [a], compile = CompileMode.COMPILED_ALL)

        F_start = System([x^2 - 1, y^2 - 1]; compile = CompileMode.INTERPRETED)
        starts = HC.solutions(HC.solve(F_start))

        r_i = HC.solve(F_i, starts; start_parameters = [1.0], target_parameters = [4.0])
        r_c = HC.solve(F_c, starts; start_parameters = [1.0], target_parameters = [4.0])
        @test HC.nresults(r_i) == HC.nresults(r_c)

        sols_i = sort(HC.real_solutions(r_i); by = s -> (s[1], s[2]))
        sols_c = sort(HC.real_solutions(r_c); by = s -> (s[1], s[2]))
        @test length(sols_i) == length(sols_c)
        for (si, sc) in zip(sols_i, sols_c)
            @test si ≈ sc atol = 1.0e-6
        end
    end
```

- [ ] **Step 6: Run all codegen tests**

Run: `julia --project -e 'using TestEnv; TestEnv.activate(); include("test/codegen_test.jl")'`
Expected: All tests PASS.

- [ ] **Step 7: Run the full test suite**

Run: `make test`
Expected: All tests PASS.

---

### Task 6: Extend benchmark harness and update status doc

**Files:**
- Modify: `benchmark/compile_modes.jl`
- Modify: `implementation_docs/02_status.md`

- [ ] **Step 1: Add `COMPILED_ALL` to the compile modes benchmark**

In `benchmark/compile_modes.jl`, extend the standalone mode section. After the existing eval/jac/build/solve benchmark loops, add Taylor kernel benchmarks and `COMPILED_ALL` rows. Add the following after the `"End-to-end solve"` block (before the final `end`):

```julia
    println("\n── Taylor through SystemEvaluator ──")
    using HomotopyContinuationNext: taylor!, TaylorVector
    for n in [3, 5, 7]
        @polyvar kv[1:(n + 1)]
        F = _katsura(kv, n)
        sys_i = System(F; compile = CompileMode.INTERPRETED)
        sys_c = System(F; compile = CompileMode.COMPILED)
        sys_a = System(F; compile = CompileMode.COMPILED_ALL)

        m = n + 1
        pv = FSVec{ComplexF64}(ComplexF64[])
        for K in 1:3
            N = K + 1
            data = FSMat{ComplexF64}(randn(ComplexF64, N, m))
            tx = TaylorVector{N, ComplexF64}(data)
            u = FSVec{ComplexF64}(zeros(ComplexF64, m))

            taylor!(u, Val(K), sys_i.evaluator, tx, pv)
            taylor!(u, Val(K), sys_c.evaluator, tx, pv)
            taylor!(u, Val(K), sys_a.evaluator, tx, pv)

            t_i = @belapsed taylor!($u, Val($K), $(sys_i.evaluator), $tx, $pv)
            t_c = @belapsed taylor!($u, Val($K), $(sys_c.evaluator), $tx, $pv)
            t_a = @belapsed taylor!($u, Val($K), $(sys_a.evaluator), $tx, $pv)
            println("  katsura-$n taylor_$K: interp=$(round(t_i * 1.0e9; digits = 1))ns  compiled=$(round(t_c * 1.0e9; digits = 1))ns  all=$(round(t_a * 1.0e9; digits = 1))ns  speedup=$(round(t_i / t_a; digits = 2))x")
        end
    end

    println("\n── End-to-end solve (COMPILED_ALL) ──")
    for n in [3, 4, 5]
        @polyvar kv[1:(n + 1)]
        F = _katsura(kv, n)
        sys_c = System(F; compile = CompileMode.COMPILED)
        sys_a = System(F; compile = CompileMode.COMPILED_ALL)
        solve(sys_c); solve(sys_a)
        t_c = @belapsed solve($sys_c)
        t_a = @belapsed solve($sys_a)
        println("  katsura-$n: compiled=$(round(t_c * 1.0e3; digits = 2))ms  all=$(round(t_a * 1.0e3; digits = 2))ms  speedup=$(round(t_c / t_a; digits = 2))x")
    end

    println("\n── Build time (COMPILED_ALL) ──")
    for n in [3, 5, 7]
        @polyvar kv[1:(n + 1)]
        F = _katsura(kv, n)
        t_c = @belapsed System($F; compile = CompileMode.COMPILED)
        t_a = @belapsed System($F; compile = CompileMode.COMPILED_ALL)
        println("  katsura-$n: compiled=$(round(t_c / 1.0e-6; digits = 1))us  all=$(round(t_a / 1.0e-6; digits = 1))us  overhead=$(round(t_a / t_c; digits = 2))x")
    end
```

- [ ] **Step 2: Run the benchmark**

Run: `julia --project=benchmark benchmark/compile_modes.jl`
Record results.

- [ ] **Step 3: Update status doc**

Add `COMPILED_ALL` rows to the `v3 compiled vs v3 interpreted` table in `implementation_docs/02_status.md` with the actual benchmark results: Taylor kernel speedups, end-to-end solve speedups, and build overhead.

- [ ] **Step 4: Update the "Not Done" checklist**

In `implementation_docs/02_status.md`, move "Compiled Taylor backend" from "Not Done — Later" to "Done":

```markdown
- [x] Compiled Taylor backend — RGF codegen for Taylor (`CompileMode.COMPILED_ALL`)
```

---

### Task 7: Format and final verification

- [ ] **Step 1: Format all changed files**

Run: `make format`

- [ ] **Step 2: Run the full test suite one final time**

Run: `make test`
Expected: All tests PASS.

- [ ] **Step 3: Verify JET reports no new issues**

The JET test is part of `make test`. Confirm zero new issues in the output.
