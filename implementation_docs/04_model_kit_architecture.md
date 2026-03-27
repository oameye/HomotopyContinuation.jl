# Model Kit Architecture

Interpreter pipeline in `src/model_kit/` — converts DynamicPolynomials input into a tape-based evaluator for polynomial systems and their Jacobians.

## Pipeline

```
DynamicPolynomials (@polyvar)
    | active path: poly_to_sexpr() + MP.differentiate() for Jacobian
    v
SExpr trees (SAdd/SMul/SPow/SConst/SVar/SParam)
    | cse() = opt_cse() + tree_cse()
    v
CSE output: (replacements, reduced_exprs)
    | compile_to_instructions()  [TapeCompiler + fusion + register allocation]
    v
InstructionSequence
    | compile to ExecInstruction variants + Interpreter(V, seq)
    v
execute!(u, I, x) / execute!(u, U, I, x) / execute_taylor!(...)
```

The public polynomial entry point is `System(polys; ...)`.
`polynomial_input.jl` now only provides variable discovery and low-level
instruction-sequence construction helpers. `_build_instruction_sequence_via_sexpr`
is the active lowering path. `polynomial_compiler.jl` contains an experimental
direct compiler that is intentionally not the default yet.

## File map

| File | Lines | Role |
|------|------:|------|
| `operations.jl` | 157 | `OpType` enum (25 ops, arity 0-4), `@inline` scalar `op_*` implementations |
| `taylor.jl` | 513 | `TruncatedTaylorSeries{N,T}` (aliased as `TTS`), `TaylorVector{N,T}`, `@generated` `taylor_op_*` recurrences, `_cauchy_product_exprs` shared helper |
| `sexpr.jl` | 290 | SExpr type hierarchy, hash/==, `_canonical_add`/`_canonical_mul`, `_wrap_args` helper, `poly_to_sexpr` |
| `cse.jl` | 561 | SymEngine CSE port: `FuncArgTracker`, `opt_cse`, `tree_cse`, `cse` |
| `tape_compiler.jl` | 598 | `TapeCompiler`, `_compile!`, instruction fusion, `_tree_reduce!` shared helper, `compile_to_instructions` |
| `instruction_sequence.jl` | 245 | `Instruction`, `InstructionSequence`, `_remap_instruction`, DAG reorder, linear-scan register allocator |
| `interpreter.jl` | 407 | `Interpreter{V}`, `ExecInstruction` variants, `execute!` (4 overloads), `execute_taylor!` |
| `symbolic_polynomial_compiler.jl` | - | Active symbolic MP polynomial lowering path |
| `polynomial_compiler.jl` | - | Experimental direct MP polynomial lowering path |
| `polynomial_input.jl` | 186 | User API, MP variable discovery, pipeline orchestrator |
| **Total** | **~2960** | |

## SExpr types (`sexpr.jl`)

```
SExpr (abstract)
  SConst   — ComplexF64 constant
  SVar     — variable reference (1-based index)
  SParam   — parameter reference (1-based index)
  STmp     — CSE temporary (assigned by tree_cse)
  SAdd     — n-ary sum (args::Vector{SExpr}), cached hash, constructor copies args
  SMul     — n-ary product (args::Vector{SExpr}), cached hash, constructor copies args
  SPow     — integer power (base::SExpr, exp::Int), cached hash
  SNeg     — negation (arg::SExpr), cached hash
  SFuncSym — unevaluated function placeholder (only used inside opt_cse), SFuncKind enum
```

Constructors accept `Vector{<:SExpr}` (not `AbstractVector`). All compound types cache `_hash::UInt` at construction via `_fold_hash`. Equality uses hash-first short-circuit. `_wrap_args` helper handles the 0/1/many pattern shared by `_canonical_add`, `_canonical_mul`, and `poly_to_sexpr`.

## CSE algorithm (`cse.jl`)

Direct port of SymEngine's `cse.cpp`, two phases:

**Phase 1 — `opt_cse`:** `FuncArgTracker` value-numbers all Add/Mul arguments, `match_common_args!` finds pairs sharing >= 2 args and factors them into `SFuncSym` placeholders. O(n^2) in Add/Mul node count. Takes ~400us for cyclic-6 jac (42 exprs). Skipping it was tested and caused 30-65% instruction count regressions — it is essential.

**Phase 2 — `tree_cse`:** `_find_repeated!` marks expressions seen 2+ times, `_rebuild` replaces them with `STmp` temporaries.

Output: `(replacements::Vector{Pair{SExpr,SExpr}}, reduced_exprs::Vector{SExpr})`.

## Compilation (`tape_compiler.jl`)

`compile_to_instructions` converts CSE output to `InstructionSequence`.

**Tape layout:** `[ constants | params | variables | scratch | assignments ]`

**Two-pass slot scheme:** During compilation, constants get temporary 1-based slots, params get negative indices `-i`, variables get `-(nparams+i)`, scratch starts at offset `10001`. After compilation, `_build_slot_remap` remaps everything to the final contiguous layout.

**`_emit!`** — single function with default args (`a2=a1, a3=a2, a4=a3`) replaces 4 overloads.

**`_tree_reduce!`** — shared helper for the pop-4/3/2 tree reduction pattern, used by both `_compile_prod_parts!` (MUL4/MUL3/MUL) and `_compile_sum_products!` (ADD4/ADD3/ADD).

**Instruction fusion** (`_compile_sum!`): Splits terms into positive/negative groups:
- Case 1 (1+, 1-): MULMULSUB / MULSUB / SUBMUL / SUB
- Case 2 (N+, 1-): sum_products + SUBMUL / SUB
- Case 3 (1+, N-): sum_products(neg) + MULSUB / SUB
- Case 4 (N+, N-): pairs pos/neg products for MULMULSUB, leftovers via sum_products + SUB

`_compile_mul!`: Splits num/denom, reduces via `_tree_reduce!`, elides multiply-by-1/-1.

**`_remap_instruction`** — shared helper (in `instruction_sequence.jl`) used by both `_remap_instructions` in tape_compiler and `_reduce_space` in instruction_sequence.

**Post-compilation:** `_optimize_instruction_order` (DAG topological sort), then `_reduce_space` (linear-scan register allocation).

## Interpreter (`interpreter.jl`)

`Interpreter{V}` is parameterized by tape type: `Vector{ComplexF64}`, `Vector{ComplexDF64}`, or `Vector{TruncatedTaylorSeries{N,ComplexF64}}`.

Internal helpers (`_load_inputs!`, `_extract_u!`, `_extract_U!`) take `I::Interpreter` directly — the tape type is constrained by the struct parameter, not `AbstractVector`. User-facing `execute!` methods keep `AbstractVector` for `u`, `x`, `p` since callers may pass `Vector` or `FSVec`.

The runtime loop does not execute raw `Instruction` values directly. At construction time, each `Instruction` is compiled into an `ExecInstruction` variant (`Add`, `Mul`, `MulAdd`, etc.). The hot loop then dispatches on `variant_storage(instr)` and writes results into the pre-allocated tape. This keeps the loop allocation-free while avoiding the large generated opcode tree used in earlier revisions.

`execute_taylor_instructions!` uses `@eval` instead of `@generated` due to Julia 1.12 world-age constraints.

Outputs extracted via `u_assignments`/`U_assignments` — `Vector{Tuple{Int,Int}}` mapping `(output_index, tape_slot)`.

## OpType reference

| Arity | Operations | Semantics |
|------:|------------|-----------|
| 0 | `STOP` | Halt |
| 1 | `CB`, `COS`, `IDENTITY`, `INV`, `INV_NOT_ZERO`, `INVSQR`, `NEG`, `SIN`, `SQR`, `SQRT` | Unary |
| 2 | `ADD`, `DIV`, `MUL`, `SUB`, `POW_INT` | Binary (POW_INT: 2nd arg is literal int) |
| 3 | `ADD3`, `MUL3`, `MULADD`(a*b+c), `MULSUB`(a*b-c), `SUBMUL`(c-a*b) | Ternary fused |
| 4 | `ADD4`, `MUL4`, `MULMULADD`(a*b+c*d), `MULMULSUB`(a*b-c*d) | Quaternary fused |

## Taylor arithmetic (`taylor.jl`)

`TruncatedTaylorSeries{N,T}` (aliased as `TTS{N,T}`) wraps `NTuple{N,T}`. All `taylor_op_*` are `@generated`, unrolling completely for compile-time `N`.

**`_cauchy_product_exprs(N)`** — shared helper generates the Cauchy product expressions for orders 1..N, used by `taylor_op_mul`, `taylor_op_muladd`, `taylor_op_mulsub`, `taylor_op_submul`.

Key recurrences: Cauchy product (mul), quotient rule (div), logarithmic differentiation (pow), coupled sin/cos.

Composite ops (`add3`, `add4`, `mul3`, `mul4`, `mulmuladd`, `mulmulsub`) are one-line `@inline` delegations to primitive ops.

`TaylorVector{N,T}` stores `n` series as `FSMat{T}` of shape `N x n`. `vectors()` is `@generated` to return a tuple of views per order.

## Performance (2026-03-26)

### Steady-state execution

| Benchmark | Time |
|---|---|
| eval_katsura3 (4×4) | 57 ns |
| eval_cyclic7 (7×7) | 132 ns |
| jac_katsura3 (4×4) | 104 ns |
| jac_cyclic7 (7×7) | 419 ns |
| taylor_katsura3 (order 3) | 107 ns |

### Build time

| Benchmark | Time |
|---|---|
| build_jac_katsura3 | 155 μs |

### Runtime vs HC v2 (ratio > 1.0 = Next faster)

| Metric | Min | Median | Max |
|--------|----:|-------:|----:|
| Eval | 0.88x | 1.04x | 1.94x |
| Jac | 0.96x | 1.24x | 1.79x |
| Build | 7.2x | 10.0x | 15.3x |

Execution is zero-allocation, dominated by arithmetic + `getindex`/`setindex!`.

## System boundary

The model kit now feeds a `System` that stores both:

- the compiled evaluator state (`InstructionSequence`, interpreters, wrappers)
- the original MP source (`polys`, `variables`, `parameters`)

This is intentional. MP polynomials are the symbolic source of truth for this branch. Metadata such as `variables(F)`, `parameters(F)`, and `is_homogeneous(F)` should be derived from that stored source rather than from a revived v2-style `Expression` layer.

## Design decisions and what was tried

**1. `opt_cse` bypass for small systems — REVERTED.** Threshold of 50 expressions tested. Instruction counts regressed 30-65%. `opt_cse` is essential regardless of system size.

**2. Eliminating IDENTITY instructions — SHIPPED.** Non-scratch results point directly to their input-block slot instead of emitting IDENTITY copies.

**3. MULMULSUB interleaving — SHIPPED.** Case 3 (1 pos, N neg) and Case 4 (interleave pos/neg for MULMULSUB pairing) in `_compile_sum!`.

**4. Cauchy product deduplication — SHIPPED.** `_cauchy_product_exprs(N)` helper shared by mul/muladd/mulsub/submul. Saved ~60 lines in taylor.jl.

**5. Tree-reduce helper — SHIPPED.** `_tree_reduce!` shared by prod and sum reduction. Eliminates duplicated pop-4/3/2 pattern.

**6. `_remap_instruction` helper — SHIPPED.** Shared between `_remap_instructions` (tape_compiler) and `_reduce_space` (instruction_sequence).

## Known limitations

1. **Magic constant 10000**: Scratch slot placeholder base. Systems with >10000 constants+params+vars would collide (unlikely in practice).
2. **DynamicPolynomials introspection**: `_variable_creation_id` uses `hasfield`/`getfield` reflection into `variable_order.order.id` for deterministic ordering. Fragile across DP versions.
3. **No compiled mode**: Next is interpreter-only. HC v2 has both interpreted and compiled. The interpreter is within ~5% of compiled for most systems.
4. **No exports**: All symbols accessed via `HomotopyContinuationNext.foo` or explicit `using HomotopyContinuationNext: foo`. Exports will be chosen later.
