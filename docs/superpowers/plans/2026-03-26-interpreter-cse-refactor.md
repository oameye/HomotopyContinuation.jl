# Interpreter CSE Refactor (Proposal A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify CSE, improve instruction fusion, and eliminate unnecessary IDENTITY instructions to match HC v2 performance on all benchmark cases while reducing code.

**Architecture:** Three targeted changes to the compilation pipeline in `src/model_kit/cse.jl` and `src/model_kit/polynomial_input.jl`. No changes to the interpreter execution loop, operations, taylor, or instruction_sequence files. The changes are: (1) bypass expensive `opt_cse` Phase 1 for small systems, (2) rewrite `_compile_sum!` to match HC v2's superior positive/negative fusion strategy, (3) eliminate unnecessary IDENTITY instructions by allowing assignments to reference non-scratch tape slots directly.

**Tech Stack:** Julia, MultivariatePolynomials, DynamicPolynomials, BenchmarkTools

---

## Current Performance Baseline

Before starting, record these numbers from `benchmark/interpreter_sweep.jl`:

| Case | Eval ratio | Jac ratio | Notes |
|------|-----------|-----------|-------|
| cyclic6 | 1.47 | 1.49 | Next faster |
| cyclic7 | 1.41 | 1.32 | Next faster |
| chain6 | 1.07 | **0.82** | HC faster on jac (Next has 71 vs HC's 61 instructions) |
| densequad6 | 1.23 | 0.99 | Tie on jac |
| sparse6_2 | **0.74** | 1.82 | HC faster on eval (Next has 94 vs HC's 90 instructions) |

Root causes identified:
- `opt_cse` costs 400us (66% of CSE time) for only 15 instruction savings on cyclic-6 jac
- `_compile_sum!` doesn't fuse as aggressively as HC v2 (fewer MULMULADD, more SUBMUL/MULADD)
- `compile_to_instructions` emits unnecessary IDENTITY ops (18 vs HC's 9 on chain6 jac)

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `src/model_kit/cse.jl` | Modify lines 1250-1288, 1380-1410 | Rewrite `_compile_sum!` fusion; eliminate IDENTITY emission for non-scratch slots |
| `src/model_kit/cse.jl` | Modify lines 900-906 | Add `opt_cse` bypass in `cse()` |
| `src/model_kit/instruction_sequence.jl` | Modify lines 127-231 | Change `_reduce_space` to accept explicit assignment slots instead of contiguous range |
| `test/instruction_count_test.jl` | Modify lines 141-173 | Update instruction count ceilings (should decrease) |
| `test/cse_refactor_test.jl` | Create | New test file for the three refactoring changes |

---

### Task 1: Add a correctness baseline test for the refactor

We need a test that verifies numerical correctness for all benchmark systems BEFORE we change anything. This test will catch any regressions during the refactor.

**Files:**
- Create: `test/cse_refactor_test.jl`

- [ ] **Step 1: Write the baseline correctness test file**

```julia
## CSE refactor correctness tests
#
# These tests verify numerical correctness of eval + jacobian across
# all benchmark systems. They must pass before AND after each refactor step.

using Test
using Random: MersenneTwister
import HomotopyContinuationNext as Next
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: differentiate as mp_diff

@polyvar _rx[1:8]

function _term(vars, coeff, exps)
    t = coeff
    for i in eachindex(exps)
        e = exps[i]
        e == 1 && (t *= vars[i])
        e > 1 && (t *= vars[i]^e)
    end
    return t
end
_poly(vars, spec) = reduce(+, (_term(vars, c, exps) for (c, exps) in spec))

function _cyclic_specs(n)
    specs = Vector{Vector{Tuple{ComplexF64, Vector{Int}}}}()
    for k in 1:(n - 1)
        poly = Tuple{ComplexF64, Vector{Int}}[]
        for start in 1:n
            exps = zeros(Int, n)
            for off in 0:(k - 1)
                exps[mod1(start + off, n)] += 1
            end
            push!(poly, (1.0 + 0im, exps))
        end
        push!(specs, poly)
    end
    push!(specs, [(1.0 + 0im, ones(Int, n)), (-1.0 + 0im, zeros(Int, n))])
    return specs
end

function _chain_specs(n)
    specs = Vector{Vector{Tuple{ComplexF64, Vector{Int}}}}(undef, n)
    for j in 1:n
        jp = mod1(j + 1, n)
        jm = mod1(j - 1, n)
        poly = Tuple{ComplexF64, Vector{Int}}[]
        e = zeros(Int, n); e[j] = 2; push!(poly, (1.0 + 0im, copy(e)))
        fill!(e, 0); e[jp] = 2; push!(poly, (1.0 + 0im, copy(e)))
        fill!(e, 0); e[j] = 1; e[jp] = 1; push!(poly, (2.0 + 0im, copy(e)))
        fill!(e, 0); e[j] = 1; e[jm] = 1; push!(poly, (-1.0 + 0im, copy(e)))
        fill!(e, 0); e[j] = 1; push!(poly, (1.0 + 0im, copy(e)))
        push!(poly, (-1.0 + 0im, zeros(Int, n)))
        specs[j] = poly
    end
    return specs
end

function _random_sparse_specs(rng, n, m; terms_per_poly = 10)
    specs = Vector{Vector{Tuple{ComplexF64, Vector{Int}}}}(undef, m)
    for j in 1:m
        poly = Tuple{ComplexF64, Vector{Int}}[]
        for _ in 1:terms_per_poly
            deg = rand(rng, 2:4)
            exps = zeros(Int, n)
            for _ in 1:deg
                exps[rand(rng, 1:n)] += 1
            end
            push!(poly, (ComplexF64(rand(rng, [-2, -1, 1, 2]), rand(rng, [-1, 0, 1])), exps))
        end
        push!(poly, (ComplexF64(rand(rng, -2:2), rand(rng, [-1, 0, 1])), zeros(Int, n)))
        specs[j] = poly
    end
    return specs
end

function _verify_system(name, specs)
    n = length(first(first(specs))[2])
    nv = collect(_rx[1:n])
    polys = [_poly(nv, s) for s in specs]
    m = length(specs)
    x = ComplexF64.(randn(n))

    I_eval = Next.build_interpreter(polys)
    I_jac = Next.build_jacobian_interpreter(polys)

    u = zeros(ComplexF64, m)
    U = zeros(ComplexF64, m, n)

    Next.execute!(u, I_eval, x)
    u_mp = ComplexF64[p(nv => x) for p in polys]
    @test maximum(abs.(u .- u_mp)) < 1.0e-10

    Next.execute!(u, U, I_jac, x)
    J_mp = zeros(ComplexF64, m, n)
    for j in 1:n, i in 1:m
        dp = mp_diff(polys[i], nv[j])
        J_mp[i, j] = dp(nv => x)
    end
    @test maximum(abs.(u .- u_mp)) < 1.0e-10
    @test maximum(abs.(U .- J_mp)) < 1.0e-10
end

@testset "CSE refactor correctness" begin
    @testset "cyclic-$n" for n in 3:7
        _verify_system("cyclic_$n", _cyclic_specs(n))
    end
    @testset "chain-$n" for n in 3:7
        _verify_system("chain_$n", _chain_specs(n))
    end
    @testset "sparse6_$seed" for seed in 1:8
        _verify_system("sparse6_$seed", _random_sparse_specs(MersenneTwister(seed), 6, 6; terms_per_poly = 10))
    end
end
```

- [ ] **Step 2: Run the test to verify it passes on current code**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/cse_refactor_test.jl")'`

Expected: All tests PASS (this is a baseline — current code must be correct).

- [ ] **Step 3: Commit**

```
git add test/cse_refactor_test.jl
git commit -m "test: add CSE refactor correctness baseline tests"
```

---

### Task 2: Bypass `opt_cse` for small expression counts

`opt_cse` (Phase 1 of CSE) does pairwise argument matching across all Add/Mul nodes. It's O(n^2) and costs 400us for cyclic-6 jacobian (42 expressions), saving only 15 instructions. For small-to-medium systems, the cost isn't worth the savings.

**Files:**
- Modify: `src/model_kit/cse.jl:900-906` (the `cse()` function)

- [ ] **Step 1: Write a test that verifies opt_cse bypass produces correct results**

Add to `test/cse_refactor_test.jl`:

```julia
@testset "opt_cse bypass correctness" begin
    # Verify that cse() with and without opt_cse produces numerically identical results
    # for a system where opt_cse would normally fire
    @testset "cyclic-6 without opt_cse" begin
        _verify_system("cyclic_6_bypass", _cyclic_specs(6))
    end
end
```

This test passes already (it's the same as the baseline). The point is it must STILL pass after we add the bypass.

- [ ] **Step 2: Run test to confirm it passes before modification**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/cse_refactor_test.jl")'`

Expected: PASS

- [ ] **Step 3: Modify `cse()` to bypass `opt_cse` for small expression counts**

In `src/model_kit/cse.jl`, replace lines 900-906:

```julia
function cse(exprs::Vector{SExpr})::Tuple{Vector{Pair{SExpr, SExpr}}, Vector{SExpr}}
    # Phase 1: find optimization opportunities (common argument matching)
    opt_subs = opt_cse(exprs)

    # Phase 2: eliminate repeated subexpressions
    return tree_cse(exprs, opt_subs)
end
```

With:

```julia
"""
    cse(exprs::Vector{SExpr}) -> (replacements, reduced_exprs)

Run Common Subexpression Elimination on a list of expressions.
Direct translation of SymEngine's cse function.

For small expression counts (< 50), Phase 1 (opt_cse) is skipped because the
O(n^2) FuncArgTracker argument matching cost outweighs the modest instruction
savings. Phase 2 (tree_cse) alone handles repeated subexpression elimination.
"""
const _OPT_CSE_THRESHOLD = 50

function cse(exprs::Vector{SExpr})::Tuple{Vector{Pair{SExpr, SExpr}}, Vector{SExpr}}
    # Phase 1: find optimization opportunities (common argument matching)
    # Skip for small systems where the O(n^2) cost exceeds the benefit
    if length(exprs) >= _OPT_CSE_THRESHOLD
        opt_subs = opt_cse(exprs)
    else
        opt_subs = Dict{SExpr, SExpr}()
    end

    # Phase 2: eliminate repeated subexpressions
    return tree_cse(exprs, opt_subs)
end
```

- [ ] **Step 4: Run all tests to verify correctness is maintained**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/cse_refactor_test.jl")'`

Expected: All tests PASS. The instruction counts in `test/instruction_count_test.jl` may change slightly (up to +15 on cyclic-6, the measured savings of opt_cse). If any count exceeds the ceiling, we'll adjust ceilings in Task 5 after measuring the net impact.

- [ ] **Step 5: Also run the full instruction count test to check for regressions**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/instruction_count_test.jl")'`

Record which tests fail (if any) and by how many instructions. We'll address these in Task 5.

- [ ] **Step 6: Commit**

```
git add src/model_kit/cse.jl
git commit -m "perf: bypass opt_cse for small expression counts (<50)"
```

---

### Task 3: Eliminate unnecessary IDENTITY instructions

Currently, `compile_to_instructions` forces ALL output assignments into a contiguous range at the end of the tape. When an output is a constant, variable, parameter, or a CSE temp already used by another output, it emits an IDENTITY instruction to copy the value into the assignment slot. HC v2 avoids this by letting assignments point to any tape slot.

The fix: instead of emitting IDENTITY for non-scratch results, record the source tape slot directly in the assignment list. This requires changing `_reduce_space` and `_index_compactification_mapping` to accept a vector of explicit assignment slots instead of a contiguous range.

**Files:**
- Modify: `src/model_kit/cse.jl:1380-1474` (assignment handling in `compile_to_instructions`)
- Modify: `src/model_kit/instruction_sequence.jl:127-231` (`_reduce_space` and `_index_compactification_mapping`)

- [ ] **Step 1: Write a test that counts IDENTITY instructions**

Add to `test/cse_refactor_test.jl`:

```julia
@testset "IDENTITY reduction" begin
    @testset "chain-6 jac IDENTITY count" begin
        nv = collect(_rx[1:6])
        polys = [_poly(nv, s) for s in _chain_specs(6)]
        I_jac = Next.build_jacobian_interpreter(polys)
        identity_count = count(
            instr -> instr.op == Next.OpType.OP_IDENTITY,
            I_jac.sequence.instructions,
        )
        # Before fix: 18 IDENTITYs. After fix: should be <= 10.
        # HC v2 achieves 9.
        @test identity_count <= 10
    end

    @testset "constant/variable outputs skip IDENTITY" begin
        # f1 = x1, f2 = x2 — pure variable outputs should not need IDENTITY
        exprs = Next.SExpr[Next.SVar(1), Next.SVar(2)]
        replacements, reduced = Next.cse(exprs)
        seq = Next.compile_to_instructions(
            replacements, reduced;
            nvars = 2, nparams = 0, output_dim = 2, npolys = 2,
        )
        # The only instruction should be STOP. No IDENTITYs needed.
        non_stop = filter(i -> i.op != Next.OpType.OP_STOP, seq.instructions)
        identity_count = count(i -> i.op == Next.OpType.OP_IDENTITY, non_stop)
        @test identity_count == 0
        # But assignments must still work correctly
        @test seq.all_u_assigned
        @test length(seq.u_assignments) == 2
    end
end
```

- [ ] **Step 2: Run the test — it should FAIL on current code**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/cse_refactor_test.jl")'`

Expected: The "chain-6 jac IDENTITY count" test FAILS (currently 18 > 10). The "constant/variable outputs skip IDENTITY" test FAILS (currently emits 2 IDENTITYs).

- [ ] **Step 3: Modify `_index_compactification_mapping` to accept explicit assignment slots**

In `src/model_kit/instruction_sequence.jl`, replace the function signature and assignment handling (lines 159-231):

```julia
"""
    _index_compactification_mapping(instructions, input_block_size, assignment_slots)

Linear-scan register allocation. Tracks the lifetime (last use) of each
intermediate register and reuses registers once their lifetime ends.
Assignment-target registers get dedicated slots at the end and are never reused.

`assignment_slots` is a `Vector{Tuple{Int32, Int32}}` of `(output_index, tape_slot)` pairs.
Slots in the input block (constants/params/vars) are left as-is. Scratch slots
get remapped to the assignment region after all scratch registers.

Returns `(index_map, tape_space_needed, updated_assignment_slots)`.
"""
function _index_compactification_mapping(
        instructions::Vector{Instruction},
        input_block_size::Int,
        assignment_slots::Vector{Tuple{Int32, Int32}},
    )
    used_indices = Set{Int32}()
    unused_indices = Vector{Int32}()
    next_reg = Int32(input_block_size)
    max_reg = Int32(input_block_size)

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

    # Collect scratch-based assignment slots that need dedicated registers
    scratch_assignment_outputs = Set{Int32}()
    for (_, slot) in assignment_slots
        if slot > Int32(input_block_size)
            push!(scratch_assignment_outputs, slot)
        end
    end

    # Compute the last instruction index that reads each output register
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
        # Allocate output register (skip if this is a scratch assignment target)
        if instr.output ∉ scratch_assignment_outputs
            index_map[instr.output] = get_register!()
        end

        # Free registers whose lifetime has ended
        for i in 1:arity(instr.op)
            should_use_index_not_reference(instr.op, i) && continue
            input_idx = instr.input[i]
            if input_idx > input_block_size &&
                    instr_idx == Int(get(output_lifetime_end, input_idx, Int32(0)))
                mapped_idx = get(index_map, input_idx, input_idx)
                if mapped_idx in used_indices
                    pop!(used_indices, mapped_idx)
                    push!(unused_indices, mapped_idx)
                end
            end
        end
    end

    # Scratch assignment slots get dedicated positions after all scratch registers
    num_scratch = Int(max_reg) - input_block_size
    assign_counter = 0
    for (_, slot) in assignment_slots
        if slot > Int32(input_block_size)
            assign_counter += 1
            index_map[slot] = Int32(input_block_size + num_scratch + assign_counter)
        end
    end

    # Build updated assignment slots using the index map
    updated_assignment_slots = Tuple{Int32, Int32}[
        (out_idx, Int32(get(index_map, slot, slot)))
        for (out_idx, slot) in assignment_slots
    ]

    tape_space_needed = input_block_size + num_scratch + assign_counter
    return index_map, tape_space_needed, updated_assignment_slots
end
```

- [ ] **Step 4: Update `_reduce_space` to match the new signature**

In `src/model_kit/instruction_sequence.jl`, replace `_reduce_space` (lines 127-148):

```julia
"""
    _reduce_space(instructions, input_block_size, assignment_slots)

Apply register allocation to compact the tape indices used by instructions,
then remap all instruction inputs/outputs to the compacted indices.

`assignment_slots` is a `Vector{Tuple{Int32, Int32}}` of `(output_index, tape_slot)`.
"""
function _reduce_space(
        instructions::Vector{Instruction},
        input_block_size::Int,
        assignment_slots::Vector{Tuple{Int32, Int32}},
    )
    index_map, space_needed, updated_assignment_slots =
        _index_compactification_mapping(instructions, input_block_size, assignment_slots)

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

    return remapped, space_needed, updated_assignment_slots
end
```

- [ ] **Step 5: Update `compile_to_instructions` to skip IDENTITY for non-scratch outputs**

In `src/model_kit/cse.jl`, replace the assignment handling block (lines 1378-1474). The key changes:
1. Non-scratch results (constants, vars, params) don't get IDENTITY — their source slot goes directly into the assignment list
2. Duplicate scratch results still get IDENTITY (unavoidable — two outputs need the same computed value)
3. The assignment list passed to `_reduce_space` is a `Vector{Tuple{Int32, Int32}}` instead of `UnitRange`

Replace from `# Create assignment slots as a contiguous range after scratch.` (line 1379) through the end of the function:

```julia
    # Build assignment slots — avoid IDENTITY for non-scratch outputs.
    # Scratch outputs that are first-use get remapped directly to an assignment slot.
    # Non-scratch outputs (constants/vars/params) use their original remapped slot.
    # Duplicate scratch outputs get an IDENTITY instruction.
    scratch_output_set = Set{Int32}(instr.output for instr in compiler.instructions)
    claimed_slots = Dict{Int32, Int}()  # raw result_slot => use count
    identity_instructions = Instruction[]
    nassignments = length(result_slots)

    # Temporary assignment target counter for scratch-based assignments
    assignment_target_counter = Int32(0)
    scratch_offset_base = Int32(10000)  # will be remapped later

    # Build raw assignment list: (output_index, tape_slot)
    raw_assignment_slots = Vector{Tuple{Int32, Int32}}(undef, nassignments)

    for (k, raw_slot) in enumerate(result_slots)
        seen = get(claimed_slots, raw_slot, 0)
        is_scratch_output = raw_slot in scratch_output_set

        if is_scratch_output && seen == 0
            # First use of a scratch output: remap directly to assignment target
            claimed_slots[raw_slot] = 1
            assignment_target_counter += Int32(1)
            target_slot = scratch_offset_base + assignment_target_counter
            remap[raw_slot] = target_slot
            raw_assignment_slots[k] = (Int32(k), target_slot)
        elseif is_scratch_output
            # Duplicate scratch output: need IDENTITY to copy
            claimed_slots[raw_slot] = seen + 1
            assignment_target_counter += Int32(1)
            target_slot = scratch_offset_base + assignment_target_counter
            remapped_source = get(remap, raw_slot, raw_slot)
            push!(
                identity_instructions, Instruction(
                    (remapped_source, remapped_source, remapped_source, remapped_source),
                    OpType.OP_IDENTITY, target_slot,
                ),
            )
            raw_assignment_slots[k] = (Int32(k), target_slot)
        else
            # Non-scratch output (constant/var/param): use original slot directly
            claimed_slots[raw_slot] = seen + 1
            remapped_source = get(remap, raw_slot, raw_slot)
            raw_assignment_slots[k] = (Int32(k), remapped_source)
        end
    end

    # Now remap all core instructions using the final remap
    nstmts = length(compiler.instructions)
    core_instructions = Vector{Instruction}(undef, nstmts + length(identity_instructions))
    for (idx, instr) in enumerate(compiler.instructions)
        new_input = ntuple(Val(4)) do k
            if should_use_index_not_reference(instr.op, k)
                instr.input[k]
            else
                get(remap, instr.input[k], instr.input[k])
            end
        end
        new_output = get(remap, instr.output, instr.output)
        core_instructions[idx] = Instruction(new_input, instr.op, new_output)
    end

    # Append IDENTITY instructions (these already use remapped source slots)
    for (j, id_instr) in enumerate(identity_instructions)
        core_instructions[nstmts + j] = id_instr
    end

    # Run optimizer and register allocator
    instructions_opt = _optimize_instruction_order(core_instructions)

    # Remap raw_assignment_slots through the same remap that core_instructions used
    remapped_assignment_slots = Tuple{Int32, Int32}[
        (out_idx, Int32(get(remap, slot, slot)))
        for (out_idx, slot) in raw_assignment_slots
    ]

    instructions_final, space_needed, updated_assignment_slots =
        _reduce_space(instructions_opt, input_block_size, remapped_assignment_slots)

    # Build final assignments from updated slots
    updated_assignments = Vector{Tuple{Int, Int}}(undef, nassignments)
    for (out_idx, tape_slot) in updated_assignment_slots
        updated_assignments[Int(out_idx)] = (Int(out_idx), Int(tape_slot))
    end

    # Add STOP instruction
    n = space_needed
    push!(
        instructions_final, Instruction(
            (Int32(n), Int32(n), Int32(n), Int32(n)), OpType.OP_STOP, Int32(n),
        ),
    )

    # Split assignments into u (function values) and U (Jacobian entries)
    u_assignments = Tuple{Int, Int}[
        (i, k) for (i, k) in updated_assignments if i <= output_dim
    ]
    U_assignments = Tuple{Int, Int}[
        (i - output_dim, k) for (i, k) in updated_assignments if i > output_dim
    ]

    return InstructionSequence(
        instructions_final,
        copy(compiler.constants),
        constants_range,
        parameters_range,
        variables_range,
        cont_param_tape_index,
        updated_assignments,
        output_dim,
        space_needed,
        u_assignments,
        U_assignments,
        length(u_assignments) == output_dim,
        length(U_assignments) == output_dim * nvars,
    )
end
```

**IMPORTANT:** This replaces everything from the `# Create assignment slots...` comment through the closing `end` of `compile_to_instructions`. The lines above this (constants_range, parameters_range, variables_range, remap construction for constants/params/vars, scratch remapping, and the `result_slots` computation) remain unchanged.

- [ ] **Step 6: Run the IDENTITY reduction test**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/cse_refactor_test.jl")'`

Expected: All tests PASS, including the new IDENTITY count tests.

- [ ] **Step 7: Run ALL existing tests to verify nothing is broken**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/interpreter_test.jl")' && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/instruction_sequence_test.jl")' && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/polynomial_input_test.jl")'`

Expected: All PASS. If any fail, debug the assignment slot mapping — the most likely issue is an off-by-one in the `remap` or `raw_assignment_slots` construction.

- [ ] **Step 8: Commit**

```
git add src/model_kit/cse.jl src/model_kit/instruction_sequence.jl test/cse_refactor_test.jl
git commit -m "perf: eliminate unnecessary IDENTITY instructions in assignment handling"
```

---

### Task 4: Improve `_compile_sum!` fusion

The current `_compile_sum!` separates terms into positive and negative groups, then compiles each group independently with `_compile_sum_products!`, and finally SUBs the results. This misses fusion opportunities:

1. When there are multiple positives AND multiple negatives, products from opposite groups could be paired into MULMULSUB (a*b - c*d) instructions.
2. Single non-product negative terms could be absorbed into SUBMUL fusions with positive products.

HC v2's `process_sum!` handles these cases by trying MULMULSUB before falling through to separate-then-subtract.

**Files:**
- Modify: `src/model_kit/cse.jl:1250-1288` (the `_compile_sum!` function)

- [ ] **Step 1: Write a test that checks for improved fusion**

Add to `test/cse_refactor_test.jl`:

```julia
@testset "Improved sum fusion" begin
    @testset "a*b - c*d uses MULMULSUB" begin
        # Expression: x1*x2 - x3*x4
        # Should produce MULMULSUB, not MUL + MUL + SUB
        exprs = Next.SExpr[
            Next.SAdd([
                Next.SMul([Next.SVar(1), Next.SVar(2)]),
                Next.SMul([Next.SConst(-1.0 + 0im), Next.SVar(3), Next.SVar(4)]),
            ]),
        ]
        replacements, reduced = Next.cse(exprs)
        seq = Next.compile_to_instructions(
            replacements, reduced;
            nvars = 4, nparams = 0, output_dim = 1, npolys = 1,
        )
        ops = [instr.op for instr in seq.instructions]
        has_mulmulsub = Next.OpType.OP_MULMULSUB in ops
        has_submul = Next.OpType.OP_SUBMUL in ops
        # Either MULMULSUB or SUBMUL is acceptable fusion
        @test has_mulmulsub || has_submul
    end

    @testset "a*b + c*d - e*f uses MULMULADD + SUBMUL" begin
        # Expression: x1*x2 + x3*x4 - x5*x6
        # Best: MULMULADD(x1,x2,x3,x4) then SUBMUL(x5,x6,result)
        exprs = Next.SExpr[
            Next.SAdd([
                Next.SMul([Next.SVar(1), Next.SVar(2)]),
                Next.SMul([Next.SVar(3), Next.SVar(4)]),
                Next.SMul([Next.SConst(-1.0 + 0im), Next.SVar(5), Next.SVar(6)]),
            ]),
        ]
        replacements, reduced = Next.cse(exprs)
        seq = Next.compile_to_instructions(
            replacements, reduced;
            nvars = 6, nparams = 0, output_dim = 1, npolys = 1,
        )
        # Should use at most 3 non-STOP instructions (MULMULADD + SUBMUL + STOP or similar)
        non_stop = filter(i -> i.op != Next.OpType.OP_STOP, seq.instructions)
        @test length(non_stop) <= 3
    end
end
```

- [ ] **Step 2: Run the test — the second test may FAIL on current code**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/cse_refactor_test.jl")'`

Expected: The `a*b - c*d` test likely passes (the 1-pos + 1-neg case is already handled). The `a*b + c*d - e*f` test may fail if current code produces 4+ instructions.

- [ ] **Step 3: Rewrite `_compile_sum!` with improved fusion**

In `src/model_kit/cse.jl`, replace the `_compile_sum!` function (lines 1250-1288):

```julia
function _compile_sum!(c::TapeCompiler, expr::SAdd)::Int32
    pos, neg = _split_into_positives_negatives(expr)
    pos_reduced =
        Tuple{Int32, Int32}[_compile_reduce_to_at_most_two!(c, e) for e in pos]
    neg_reduced =
        Tuple{Int32, Int32}[_compile_reduce_to_at_most_two!(c, e) for e in neg]

    # Case 1: single positive, single negative — try maximal fusion
    if length(pos_reduced) == 1 && length(neg_reduced) == 1
        (a, b) = pos_reduced[1]
        (d, e) = neg_reduced[1]
        if b != _SLOT_NONE && e != _SLOT_NONE
            return _emit!(c, OpType.OP_MULMULSUB, a, b, d, e)
        elseif b != _SLOT_NONE
            return _emit!(c, OpType.OP_MULSUB, a, b, d)
        elseif e != _SLOT_NONE
            return _emit!(c, OpType.OP_SUBMUL, d, e, a)
        else
            return _emit!(c, OpType.OP_SUB, a, d)
        end
    end

    # Case 2: multiple positives, single negative — sum positives, then fuse subtraction
    if length(neg_reduced) == 1
        a = _compile_sum_products!(c, pos_reduced)
        (d, e) = neg_reduced[1]
        if e != _SLOT_NONE
            return _emit!(c, OpType.OP_SUBMUL, d, e, a)
        else
            return _emit!(c, OpType.OP_SUB, a, d)
        end
    end

    # Case 3: single positive, multiple negatives — sum negatives, then fuse subtraction
    if length(pos_reduced) == 1
        b_sum = _compile_sum_products!(c, neg_reduced)
        (a, b) = pos_reduced[1]
        if b != _SLOT_NONE
            return _emit!(c, OpType.OP_MULSUB, a, b, b_sum)
        else
            return _emit!(c, OpType.OP_SUB, a, b_sum)
        end
    end

    # Case 4: multiple positives, multiple negatives — interleave for MULMULSUB pairing
    # Pair up positive products with negative products for MULMULSUB(a,b,c,d) = a*b - c*d
    # Remaining terms go through sum_products + SUB
    pos_pairs = Tuple{Int32, Int32}[t for t in pos_reduced if t[2] != _SLOT_NONE]
    pos_singles = Int32[t[1] for t in pos_reduced if t[2] == _SLOT_NONE]
    neg_pairs = Tuple{Int32, Int32}[t for t in neg_reduced if t[2] != _SLOT_NONE]
    neg_singles = Int32[t[1] for t in neg_reduced if t[2] == _SLOT_NONE]

    # Pair positive products with negative products for MULMULSUB
    fused_results = Int32[]
    n_fused = min(length(pos_pairs), length(neg_pairs))
    for i in 1:n_fused
        (a, b) = pos_pairs[i]
        (d, e) = neg_pairs[i]
        push!(fused_results, _emit!(c, OpType.OP_MULMULSUB, a, b, d, e))
    end

    # Leftover positive products go through sum_products
    leftover_pos = Tuple{Int32, Int32}[]
    for i in (n_fused + 1):length(pos_pairs)
        push!(leftover_pos, pos_pairs[i])
    end
    for s in pos_singles
        push!(leftover_pos, (s, _SLOT_NONE))
    end

    # Leftover negative products go through sum_products
    leftover_neg = Tuple{Int32, Int32}[]
    for i in (n_fused + 1):length(neg_pairs)
        push!(leftover_neg, neg_pairs[i])
    end
    for s in neg_singles
        push!(leftover_neg, (s, _SLOT_NONE))
    end

    # Sum up fused results + leftover positives
    all_pos_parts = Int32[]
    append!(all_pos_parts, fused_results)
    if !isempty(leftover_pos)
        pos_sum = _compile_sum_products!(c, leftover_pos)
        pos_sum != _SLOT_NONE && push!(all_pos_parts, pos_sum)
    end

    # Sum all positive contributions
    pos_total = _SLOT_NONE
    if length(all_pos_parts) == 1
        pos_total = all_pos_parts[1]
    elseif length(all_pos_parts) >= 2
        # Wrap as singles for sum_products
        pos_total = _compile_sum_products!(
            c, Tuple{Int32, Int32}[(s, _SLOT_NONE) for s in all_pos_parts],
        )
    end

    # Sum leftover negatives
    neg_total = _SLOT_NONE
    if !isempty(leftover_neg)
        neg_total = _compile_sum_products!(c, leftover_neg)
    end

    # Final subtraction
    if pos_total == _SLOT_NONE
        return neg_total == _SLOT_NONE ? _SLOT_NONE : _tape_neg!(c, neg_total)
    elseif neg_total == _SLOT_NONE
        return pos_total
    else
        return _tape_sub!(c, pos_total, neg_total)
    end
end
```

- [ ] **Step 4: Run ALL tests**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/cse_refactor_test.jl")' && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/interpreter_test.jl")' && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/polynomial_input_test.jl")'`

Expected: All PASS.

- [ ] **Step 5: Commit**

```
git add src/model_kit/cse.jl test/cse_refactor_test.jl
git commit -m "perf: improve _compile_sum! fusion with MULMULSUB interleaving"
```

---

### Task 5: Update instruction count ceilings and run full benchmark

After all three optimizations, instruction counts will have changed. Some will decrease (fewer IDENTITY, better fusion), some may increase slightly (no opt_cse). Update the recorded ceilings in `test/instruction_count_test.jl`.

**Files:**
- Modify: `test/instruction_count_test.jl:141-173`

- [ ] **Step 1: Run the instruction count test to see which ceilings need updating**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/instruction_count_test.jl")'`

Record the actual instruction counts for every system. For each test that fails:
- If the new count is LOWER: lower the ceiling (this is an improvement!)
- If the new count is HIGHER: raise the ceiling, but only if the increase is <= 5 instructions AND the runtime benchmark shows no regression

- [ ] **Step 2: Update the `_MAX_EVAL_INSTRS` dictionary**

In `test/instruction_count_test.jl`, update lines 141-173 with the new instruction counts. The exact values depend on what the tests report — this cannot be pre-computed because the three changes interact.

**Guidelines for new ceilings:**
- Set each ceiling to the actual count + 1 (small headroom)
- If a count went UP by more than 5, investigate — the `opt_cse` bypass might be hurting that case, and the threshold might need tuning

- [ ] **Step 3: Run all tests to verify the updated ceilings pass**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'using TestEnv; TestEnv.activate(); include("test/instruction_count_test.jl")'`

Expected: All PASS.

- [ ] **Step 4: Run the benchmark sweep to measure performance impact**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=benchmark benchmark/interpreter_sweep.jl`

Record the new ratios and compare against the baseline from the top of this plan. The key metrics:
- chain6 jac ratio: should improve from 0.82 toward >= 1.0
- sparse6_2 eval ratio: should improve from 0.74 toward >= 0.90
- No case should regress by more than 5%

- [ ] **Step 5: Format code**

Run: `make format`

- [ ] **Step 6: Commit**

```
git add test/instruction_count_test.jl
git commit -m "test: update instruction count ceilings after CSE refactor"
```

---

### Task 6: Run full test suite and final validation

**Files:** (no changes — validation only)

- [ ] **Step 1: Run make test**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && make test`

Expected: All tests pass (Aqua, JET, ExplicitImports, all unit tests).

- [ ] **Step 2: Run the benchmark sweep one final time and record results**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=benchmark benchmark/interpreter_sweep.jl`

Print the results table in the same format as the baseline. Compare:

| Case | Before | After | Change |
|------|--------|-------|--------|
| chain6 jac | 0.82 | ??? | |
| sparse6_2 eval | 0.74 | ??? | |
| (all others) | ... | ... | |

- [ ] **Step 3: If any benchmark regressed significantly (> 10%), diagnose**

Check if the `_OPT_CSE_THRESHOLD` of 50 needs adjustment. For systems that regressed:
1. Compare instruction counts before/after
2. If opt_cse was helping that specific case, consider lowering the threshold or making it adaptive

- [ ] **Step 4: Verify no regressions in build time**

Run in Julia REPL:
```julia
using BenchmarkTools, DynamicPolynomials: @polyvar
using HomotopyContinuationNext: build_jacobian_interpreter
@polyvar x1 x2 x3 x4 x5 x6
F = [x1+x2+x3+x4+x5+x6, x1*x2+x2*x3+x3*x4+x4*x5+x5*x6+x6*x1,
     x1*x2*x3+x2*x3*x4+x3*x4*x5+x4*x5*x6+x5*x6*x1+x6*x1*x2,
     x1*x2*x3*x4+x2*x3*x4*x5+x3*x4*x5*x6+x4*x5*x6*x1+x5*x6*x1*x2+x6*x1*x2*x3,
     x1*x2*x3*x4*x5+x2*x3*x4*x5*x6+x3*x4*x5*x6*x1+x4*x5*x6*x1*x2+x5*x6*x1*x2*x3+x6*x1*x2*x3*x4,
     x1*x2*x3*x4*x5*x6-1]
@btime build_jacobian_interpreter($F)
```

Expected: Build time should decrease (from ~1005us to ~600us due to opt_cse bypass).
