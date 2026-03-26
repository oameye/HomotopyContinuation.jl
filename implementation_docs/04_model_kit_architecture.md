# Model Kit Architecture

Interpreter pipeline in `src/model_kit/` — converts DynamicPolynomials input into a tape-based evaluator for polynomial systems and their Jacobians.

## Pipeline

```
DynamicPolynomials (@polyvar)
    | poly_to_sexpr() + MP.differentiate() for Jacobian
    v
SExpr trees (SAdd/SMul/SPow/SConst/SVar/SParam)
    | cse() = opt_cse() + tree_cse()
    v
CSE output: (replacements, reduced_exprs)
    | compile_to_instructions()  [TapeCompiler + fusion + register allocation]
    v
InstructionSequence
    | Interpreter(V, seq)
    v
execute!(u, I, x) / execute!(u, U, I, x) / execute_taylor!(...)
```

Entry points: `build_interpreter`, `build_jacobian_interpreter`, `build_taylor_interpreter`, `build_df64_interpreter` in `polynomial_input.jl`.

## File map

| File | Lines | Role |
|------|------:|------|
| `operations.jl` | 230 | `OpType` enum (27 ops, arity 0-4) and `@inline` scalar `op_*` implementations |
| `taylor.jl` | 622 | `TruncatedTaylorSeries{N,T}`, `TaylorVector{N,T}`, `@generated` `taylor_op_*` recurrences |
| `instruction_sequence.jl` | 238 | `Instruction`, `InstructionSequence`, DAG reorder, linear-scan register allocator |
| `cse.jl` | 1587 | SExpr types, SymEngine CSE port, `TapeCompiler`, `compile_to_instructions` |
| `interpreter.jl` | 385 | `Interpreter{V}`, `execute!` (4 overloads), `execute_taylor!`, code-generated dispatch |
| `polynomial_input.jl` | 181 | User API, MP variable discovery, pipeline orchestrator |
| **Total** | **3243** | |

## SExpr types (`cse.jl`)

```
SExpr (abstract)
  SConst   — ComplexF64 constant
  SVar     — variable reference (1-based index)
  SParam   — parameter reference (1-based index)
  STmp     — CSE temporary (assigned by tree_cse)
  SAdd     — n-ary sum (args::Vector{SExpr})
  SMul     — n-ary product (args::Vector{SExpr})
  SPow     — integer power (base::SExpr, exp::Int)
  SNeg     — negation (arg::SExpr)
  SFuncSym — unevaluated function placeholder (only used inside opt_cse)
```

All types have custom `hash`/`==` for use as Dict/Set keys. `SAdd`/`SMul` args are canonicalized (flattened, constants collected, sorted by `_sexpr_lt = hash(a) < hash(b)`) via `_canonical_add`/`_canonical_mul`.

## CSE algorithm (`cse.jl`)

Direct port of SymEngine's `cse.cpp`, two phases:

**Phase 1 — `opt_cse`:** `FuncArgTracker` value-numbers all Add/Mul arguments, `match_common_args!` finds pairs sharing >= 2 args and factors them into `SFuncSym` placeholders. O(n^2) in Add/Mul node count. Takes ~400us for cyclic-6 jac (42 exprs). Skipping it was tested and caused 30-65% instruction count regressions — it is essential.

**Phase 2 — `tree_cse`:** `_find_repeated!` marks expressions seen 2+ times, `_rebuild` replaces them with `STmp` temporaries.

Output: `(replacements::Vector{Pair{SExpr,SExpr}}, reduced_exprs::Vector{SExpr})`.

## Compilation (`cse.jl` — TapeCompiler)

`compile_to_instructions` converts CSE output to `InstructionSequence`.

**Tape layout:** `[ constants | params | (cont_param?) | variables | scratch | assignments ]`

**Two-pass slot scheme:** During compilation, constants get temporary 1-based slots, params get negative indices `-i`, variables get `-(nparams+i)`, scratch starts at offset `10001`. After compilation, everything is remapped to the final contiguous layout.

**Instruction fusion** (`_compile_sum!`): Splits terms into positive/negative groups:
- Case 1 (1+, 1-): MULMULSUB / MULSUB / SUBMUL / SUB
- Case 2 (N+, 1-): sum_products + SUBMUL / SUB
- Case 3 (1+, N-): sum_products(neg) + MULSUB / SUB
- Case 4 (N+, N-): pairs pos/neg products for MULMULSUB, leftovers via sum_products + SUB

`_compile_mul!`: Splits num/denom, reduces via MUL3/MUL4, elides multiply-by-1/-1.

`_compile_sum_products!`: Pairs products into MULMULADD(a*b + c*d), remainders via MULADD/ADD4/ADD3/ADD.

**Assignment handling:** Scratch results get dedicated assignment slots. Non-scratch results (constants/vars/params, e.g. Jacobian entries that are 0 or 1) skip IDENTITY — assignments point directly to the input-block slot.

**Post-compilation:** `_optimize_instruction_order` (DAG topological sort), then `_reduce_space` (linear-scan register allocation).

## Interpreter (`interpreter.jl`)

`Interpreter{V}` is parameterized by tape type: `Vector{ComplexF64}`, `Vector{ComplexDF64}`, or `Vector{TruncatedTaylorSeries{N,ComplexF64}}`.

`execute_instructions!` is `@generated` — builds a nested-if dispatch chain at compile time, opcodes tested in frequency order. One level of "instruction recursion" (next instruction's dispatch inlined after each body) halves loop overhead.

`execute_taylor_instructions!` uses `@eval` instead of `@generated` due to Julia 1.12 world-age constraints (`@generated` can't call helpers defined in the same compilation unit).

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

`TruncatedTaylorSeries{N,T}` wraps `NTuple{N,T}`. All `taylor_op_*` are `@generated`, unrolling completely for compile-time `N`. Key recurrences: Cauchy product (mul), quotient rule (div), logarithmic differentiation (pow), coupled sin/cos. Higher-arity ops decompose into binary pairs (`@inline`).

`TaylorVector{N,T}` stores `n` series as `FSMat{T}` of shape `N x n`. `vectors()` is `@generated` to return a tuple of views per order.

## Performance (2026-03-26)

### Build time (CSE dominates)

| System | CSE % | poly_to_sexpr % | compile % | Total |
|--------|------:|----------------:|----------:|------:|
| cyclic6 eval (6 exprs) | 74% | 13% | 13% | 196 us |
| cyclic6 jac (42 exprs) | 73% | 13% | 14% | 810 us |
| densequad6 eval | 77% | 12% | 11% | 1384 us |
| sparse6_2 eval | 77% | 11% | 12% | 698 us |

### Runtime vs HC v2 (ratio > 1.0 = Next faster)

| Case | Eval | Jac |
|------|-----:|----:|
| cyclic6 | 1.50 | 1.19 |
| cyclic7 | 1.36 | 1.40 |
| chain6 | 0.96 | 1.17 |
| densequad6 | 1.01 | 1.01 |
| sparse6_2 | 1.02 | 1.01 |
| sparse6_4 | 1.00 | 1.22 |
| **Median** | **1.02** | **1.19** |
| **Worst** | **0.96** | **1.01** |

Execution is zero-allocation, dominated by arithmetic + `getindex`/`setindex!`. No case below 0.96x eval or 1.01x jac.

### Instruction counts (cyclic-6)

| | Next eval | HC v2 eval | Next jac | HC v2 jac |
|-|-------:|-------:|-------:|-------:|
| Instructions | 27 | 29 | 53 | 61 |
| IDENTITY ops | 0 | 0 | 0 | 9 |

## What was tried (2026-03-26)

**1. Bypassing `opt_cse` for small systems — REVERTED.** Threshold of 50 expressions tested. Instruction counts regressed 30-65% (cyclic-7: 40 -> 66). `opt_cse` is essential regardless of system size.

**2. Eliminating IDENTITY instructions — SHIPPED.** Old code forced all assignments into a contiguous tape range, emitting IDENTITY copies. Fix: non-scratch results point directly to their input-block slot. Chain-6 jac: 71 instructions (18 IDENTITY) -> 53 instructions (0 IDENTITY).

**3. MULMULSUB interleaving in `_compile_sum!` — SHIPPED.** Added Case 3 (1 pos, N neg) and Case 4 (interleave pos/neg products for MULMULSUB pairing).

## Recommended refactoring (priority order)

**1. Split `cse.jl` into three files.** Zero risk, pure cleanup:

| New file | Content | ~Lines |
|----------|---------|-------:|
| `sexpr.jl` | SExpr types, hash/==, canonicalization, `poly_to_sexpr` | 310 |
| `cse.jl` | `FuncArgTracker`, `opt_cse`, `tree_cse`, `cse` | 600 |
| `tape_compiler.jl` | `TapeCompiler`, `_compile_*`, `compile_to_instructions` | 680 |

**2. Deduplicate wrappers.** Four `build_*_interpreter` functions differ only in the type parameter — collapse to `_build_interpreter(V, polys; ...)`. Four `execute!` overloads share identical bodies — compose as `_load_inputs!` + `_extract_outputs!` + `_extract_jacobian!`.

**3. Fix `_empty_vars` double variable collection.** Default `parameters` arg calls `_empty_vars` -> `_collect_variables`, then `_effective_variables` calls `_collect_variables` again. Call it once.

**4. Replace `_sexpr_lt` hash ordering.** `_sexpr_lt = hash(a) < hash(b)` drives canonicalization throughout the pipeline. Hash ordering differs from SymEngine's structural `__cmp__`, causing suboptimal grouping on some systems. A structural comparator (type tag, then depth/size, then content) could close the remaining 0.96x-1.00x eval gaps. **Medium risk** — affects all canonicalization and CSE grouping. Use the instruction count regression test as safety net.

**5. Polynomial-specific DAG (longer term).** Replace SExpr with monomial-level value numbering. Would remove ~800 lines but must replicate `opt_cse`'s cross-expression sharing quality. Prototype separately.

**6. Replace `SFuncSym` string dispatch.** `SFuncSym("add"/mul"/pow")` in `_compile!` and `_rebuild_children` dispatches on `name::String`. Replace with `SFuncAdd`/`SFuncMul` concrete subtypes. Low risk.

## Known limitations

1. **Magic constant 10000**: Scratch slot placeholder base. Systems with >10000 constants+params+vars would collide (unlikely in practice).
2. **DynamicPolynomials introspection**: `_variable_creation_id` uses `hasfield`/`getfield` reflection into `variable_order.order.id` for deterministic ordering. Fragile across DP versions.
3. **No compiled mode**: Next is interpreter-only. HC v2 has both interpreted and compiled. The interpreter is within ~5% of compiled for most systems.
