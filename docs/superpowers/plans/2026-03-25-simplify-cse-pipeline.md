# Simplify CSE Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce the model_kit pipeline from ~2229 lines to ~1400 lines by deleting SymEngine-parity code and eliminating the IR intermediate layer, while keeping all 641 tests passing.

**Architecture:** The current pipeline is SExpr → CSE → IRStatement → Instruction. We eliminate the IR layer so it becomes SExpr → CSE → Instruction directly. The SExprCompiler emits Instruction values with tape indices instead of IRStatementRef values with symbolic references. We also delete ~185 lines of comparison infrastructure and ~40 lines of dict-based canonicalization that existed only for SymEngine ordering parity.

**Tech Stack:** Julia 1.11+, DynamicPolynomials, MultivariatePolynomials, EnumX, FixedSizeArrays

---

### Task 1: Delete `_sexpr_cmp`/`_sexpr_compare` infrastructure, replace with hash-based ordering

All `sort!` calls on `SExpr` values currently use `_sexpr_lt` which delegates to a 185-line `_sexpr_cmp`/`_sexpr_compare` system that replicates SymEngine's `Basic::__cmp__` ordering. The CSE algorithm only needs a *consistent* total order — any deterministic ordering works. Replace with hash-based comparison.

**Files:**
- Modify: `src/model_kit/cse.jl:152-340` (delete), `src/model_kit/cse.jl:340` (rewrite `_sexpr_lt`)
- Test: `make test` (641 tests must still pass, instruction counts may shift slightly)

- [ ] **Step 1: Replace `_sexpr_lt` with hash-based comparison**

In `src/model_kit/cse.jl`, delete everything from line 152 (`_sexpr_type_code`) through line 340 (`_sexpr_lt`), including all of: `_sexpr_type_code`, `_sexpr_cmp`, all 10 `_sexpr_compare` methods, `_symbol_sort_rank` (4 methods), `_split_mul_coef_bases`, `_split_add_coef_terms`, `_sexpr_cmp_number`, `_add_term_base`, `_add_term_coeff`, and `_sexpr_lt`.

Replace with these two functions:

```julia
"""Deterministic total order for SExpr values. Uses hash for consistency."""
_sexpr_lt(a::SExpr, b::SExpr)::Bool = hash(a) < hash(b)

"""Extract the base expression of an Add term (stripping leading coefficient)."""
function _add_term_base(e::SExpr)::SExpr
    if e isa SMul && !isempty(e.args) && e.args[1] isa SConst
        rest = e.args[2:end]
        return length(rest) == 1 ? rest[1] : SMul(rest)
    end
    return e
end
```

`_add_term_base` is still needed by `_canonical_add`.

- [ ] **Step 2: Run tests**

Run: `make test`
Expected: All tests pass. Some instruction counts in `_MAX_EVAL_INSTRS` may need updating if the new ordering produces slightly different (but equally valid) CSE groupings.

- [ ] **Step 3: Update instruction count bounds if needed**

If any `instruction_count_test` fails, check if the new count is close to the old one (within ~5 instructions). If so, update `_MAX_EVAL_INSTRS` in `test/instruction_count_test.jl` to the new values. The counts should not increase by more than ~5 for any system.

---

### Task 2: Remove `SConst.is_real_int` field

The `is_real_int` field on `SConst` exists only to distinguish `Integer(-1)` from `ComplexDouble(-1+0i)` for SymEngine's `is_minus_one()` behavior. No longer needed.

**Files:**
- Modify: `src/model_kit/cse.jl:15-20` (SConst struct), `src/model_kit/cse.jl` (poly_to_sexpr), `src/model_kit/cse.jl` (_split_off_minus_one)

- [ ] **Step 1: Simplify SConst struct**

In `src/model_kit/cse.jl`, replace:

```julia
struct SConst <: SExpr
    val::ComplexF64
    is_real_int::Bool
end
SConst(val::ComplexF64) = SConst(val, iszero(imag(val)) && isinteger(real(val)))
```

with:

```julia
struct SConst <: SExpr
    val::ComplexF64
end
```

- [ ] **Step 2: Remove `coeff_is_real_int` from `poly_to_sexpr`**

In `poly_to_sexpr` (around line 495), delete the line:
```julia
coeff_is_real_int = raw_coeff isa Real && isinteger(raw_coeff)
```

And change all `SConst(coeff, coeff_is_real_int)` back to `SConst(coeff)`.

- [ ] **Step 3: Simplify `_split_off_minus_one`**

In `_split_off_minus_one` (around line 1321), remove the `c.is_real_int` check. Change:

```julia
if c.val == -one(ComplexF64) && c.is_real_int
```

to:

```julia
if c.val == -one(ComplexF64)
```

This means `_split_off_minus_one` now always splits `-1*expr` into negative form, regardless of whether the -1 came from an integer or complex coefficient. This is simpler and equally correct.

- [ ] **Step 4: Run tests**

Run: `make test`
Expected: All tests pass. Instruction counts may shift slightly for dense_quad systems (which have complex coefficients).

- [ ] **Step 5: Update instruction count bounds if needed**

Same as Task 1 Step 3.

---

### Task 3: Simplify `_canonical_add`

The current `_canonical_add` does dict-based term deduplication (grouping by base expression, summing coefficients) to match SymEngine's internal `Add` representation. This is overkill — just flatten nested Adds, collect constants, and sort.

**Files:**
- Modify: `src/model_kit/cse.jl` (_canonical_add function, ~lines 392-439)

- [ ] **Step 1: Rewrite `_canonical_add`**

Replace the current `_canonical_add` (which uses `Dict{SExpr, ComplexF64}` for grouping) with:

```julia
function _canonical_add(args::Vector{SExpr})::SExpr
    flat_args = SExpr[]
    const_sum = Ref(zero(ComplexF64))
    for arg in args
        _flatten_add_arg!(flat_args, const_sum, arg)
    end
    sort!(flat_args; lt = _sexpr_lt)
    if !iszero(const_sum[])
        pushfirst!(flat_args, SConst(const_sum[]))
    end
    if isempty(flat_args)
        return SConst(zero(ComplexF64))
    elseif length(flat_args) == 1
        return flat_args[1]
    else
        return SAdd(flat_args)
    end
end
```

Also delete `_add_term_coeff` (no longer needed after removing the dict-based grouping). Keep `_add_term_base` (still used by `poly_to_sexpr`'s sort).

- [ ] **Step 2: Run tests**

Run: `make test`
Expected: All tests pass.

---

### Task 4: Eliminate the IR layer — compile SExpr directly to Instruction

This is the big refactor. Currently: `compile_cse_to_ir` (cse.jl) emits `IRStatement` values → `_build_instruction_sequence` (polynomial_input.jl) wraps them in `IntermediateRepresentation` → `build_instruction_sequence_from_ir` (instruction_sequence.jl) resolves symbolic refs to tape indices and emits `Instruction` values.

The new design: `compile_to_instructions` (cse.jl) directly emits `Instruction` values with tape indices. No IR types needed.

**Files:**
- Modify: `src/model_kit/cse.jl` (replace SExprCompiler + compile_cse_to_ir)
- Modify: `src/model_kit/instruction_sequence.jl` (delete IR types + build_instruction_sequence_from_ir, keep optimization passes)
- Modify: `src/model_kit/polynomial_input.jl` (simplify _build_instruction_sequence)
- Test: `make test`

#### Step-by-step:

- [ ] **Step 1: Define `TapeCompiler` struct in `cse.jl`**

Replace `SExprCompiler` with a new `TapeCompiler` that knows about tape layout from the start. Add this after the CSE algorithm code (replacing the old SExprCompiler section):

```julia
mutable struct TapeCompiler
    const instructions::Vector{Instruction}
    const constants::Vector{ComplexF64}
    const constants_map::Dict{ComplexF64, Int32}  # val → tape index
    const var_indices::Vector{Int32}               # var i → tape index
    const param_indices::Vector{Int32}             # param i → tape index
    const cse_defs::Dict{Int, SExpr}               # CSE temp id → definition
    const cse_compiled::Dict{Int, Int32}            # CSE temp id → tape index
    next_slot::Int32                                # next available tape slot
end
```

The key change: instead of emitting `IRStatementRef` and resolving later, we emit tape slot indices directly.

Constructor:
```julia
function TapeCompiler(;
        nvars::Int, nparams::Int,
        constants::Vector{ComplexF64},
        var_syms::Vector{Symbol}, param_syms::Vector{Symbol},
        continuation_parameter_index::Union{Nothing, Int} = nothing,
    )
    constants_map = Dict{ComplexF64, Int32}()
    slot = Int32(0)

    # Tape layout: constants | parameters | [continuation_param] | variables | scratch
    for (i, c) in enumerate(constants)
        slot += Int32(1)
        constants_map[c] = slot
    end
    constants_range = Int32(1):slot

    param_indices = Int32[]
    for _ in 1:nparams
        slot += Int32(1)
        push!(param_indices, slot)
    end
    parameters_range = (length(constants_map) > 0 ? last(constants_range) : Int32(0)) .+ (Int32(1):Int32(nparams))

    cont_idx = if !isnothing(continuation_parameter_index)
        slot += Int32(1)
        Int(slot)
    else
        nothing
    end

    var_indices = Int32[]
    variables_start = slot + Int32(1)
    for _ in 1:nvars
        slot += Int32(1)
        push!(var_indices, slot)
    end
    variables_range = variables_start:slot

    return TapeCompiler(
        Instruction[], constants, constants_map,
        var_indices, param_indices,
        Dict{Int, SExpr}(), Dict{Int, Int32}(),
        slot,
    ), constants_range, parameters_range, variables_range, cont_idx
end
```

- [ ] **Step 2: Port arithmetic helpers to emit `Instruction` directly**

Replace all `_ir_*!` functions and `_add_op!` with tape-index-based versions. The key helper:

```julia
function _emit!(c::TapeCompiler, op::OpType.T, args::Vararg{Int32})::Int32
    c.next_slot += Int32(1)
    out = c.next_slot
    input = ntuple(i -> i <= length(args) ? args[i] : Int32(1), Val(4))
    push!(c.instructions, Instruction(input, op, out))
    return out
end
```

Each `_ir_mul!`, `_ir_add!`, etc. becomes a function returning `Int32` (tape index) instead of `IRStatementArg`. The `_is_one`/`_is_minus_one` checks now compare tape indices against known constant positions.

Port `_sexpr_to_ir!` → `_compile!` which returns `Int32` (tape index). `SConst` → lookup in `constants_map`. `SVar(i)` → `var_indices[i]`. `STmp(id)` → lazy compile via `cse_compiled`.

- [ ] **Step 3: Port `_process_mul!` and `_process_sum!`**

Same logic as before but all values are `Int32` tape indices. `_reduce_to_at_most_two_multiplicants!` returns `Tuple{Int32, Int32}` where the second is `Int32(0)` for "no second factor" (instead of `nothing`).

- [ ] **Step 4: Write `compile_to_instructions` entry point**

```julia
function compile_to_instructions(
        replacements::Vector{Pair{SExpr, SExpr}},
        reduced_exprs::Vector{SExpr};
        nvars::Int, nparams::Int,
        constants::Vector{ComplexF64},
        var_syms::Vector{Symbol},
        param_syms::Vector{Symbol},
        output_dim::Int,
        include_jacobian::Bool,
        npolys::Int,
    )::InstructionSequence
    compiler, constants_range, parameters_range, variables_range, cont_idx =
        TapeCompiler(; nvars, nparams, constants, var_syms, param_syms)

    # Register CSE defs
    for (tmp, def) in replacements
        compiler.cse_defs[tmp.id] = def
    end

    # Compile all reduced expressions
    result_slots = Int32[]
    for expr in reduced_exprs
        push!(result_slots, _compile!(compiler, expr))
    end

    # Build assignments (same ensure_stmt_ref logic as before but with tape slots)
    # ... split into f_refs and jac_refs, create IDENTITY for duplicates ...

    # Run optimization passes (reuse existing _optimize_instruction_order + _reduce_space)
    instructions_opt = _optimize_instruction_order(compiler.instructions)
    input_block_size = Int(compiler.var_indices[end])  # or last variable slot
    # ... _reduce_space, build final InstructionSequence ...
end
```

- [ ] **Step 5: Simplify `_build_instruction_sequence` in `polynomial_input.jl`**

Replace the current function that goes SExpr → `compile_cse_to_ir` → wrap in IR → `build_instruction_sequence_from_ir` with a single call to `compile_to_instructions`.

- [ ] **Step 6: Delete IR types from `instruction_sequence.jl`**

Delete: `IRStatementRef`, `IRStatementArg`, `IRStatement` (4 constructors), `IntermediateRepresentation`, `build_instruction_sequence_from_ir`, `_resolve_ir_arg`.

Keep: `Instruction`, `InstructionSequence`, `_optimize_instruction_order`, `_reduce_space`, `_index_compactification_mapping`.

- [ ] **Step 7: Delete old `SExprCompiler` and `compile_cse_to_ir` from `cse.jl`**

Delete the entire section from `struct SExprCompiler` through `function compile_cse_to_ir` (approximately lines 1140-1519 of the current file).

- [ ] **Step 8: Run tests**

Run: `make test`
Expected: All 641 tests pass. Instruction counts should be identical (same optimization passes, same CSE, just no intermediate IR step).

- [ ] **Step 9: Update instruction count bounds if any shifted**

Check and update `_MAX_EVAL_INSTRS` in `test/instruction_count_test.jl` if needed.

- [ ] **Step 10: Verify benchmark performance**

Run: `julia --project=benchmark benchmark/interpreter_sweep.jl`
Expected: Same or better performance (fewer allocations during build, same runtime).

---

## Expected outcome

| Metric | Before | After |
|--------|--------|-------|
| `cse.jl` | 1519 lines | ~900 lines |
| `instruction_sequence.jl` | 481 lines | ~300 lines |
| `polynomial_input.jl` | 229 lines | ~180 lines |
| **Total pipeline** | **2229 lines** | **~1380 lines** |
| IR types | 6 (IRStatementRef, IRStatementArg, IRStatement, IntermediateRepresentation, SExprCompiler, + compile_cse_to_ir) | 1 (TapeCompiler + compile_to_instructions) |
| Test count | 641 pass | 641 pass |
| Runtime performance | Same | Same or better |
