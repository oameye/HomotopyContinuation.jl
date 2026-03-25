# TTFX Analysis for HomotopyContinuation.jl

**Date**: 2026-03-25 | **Julia**: 1.12 | **Branch**: `jet`

## Results (no precompile directives, default `:mixed`)

| Benchmark | main | jet | Change |
|-----------|------|-----|--------|
| 1st `solve(F)` | 28.5s | 25.6s | **-10%** |
| 2nd `solve(F)` same system | 0.002s | 0.002s | — |
| 3rd `solve(G)` different system | 1.9s | 2.3s | — |
| Parameter homotopy | 2.2s | 2.0s | **-9%** |
| Package load | 1.7s | 1.7s | — |

With `compile=Val(:none)` (uses `InterpretedSystem`, no unique type parameter):

| Benchmark | Val(:none) |
|-----------|------------|
| 1st `solve(F)` | **1.1s** |
| 2nd `solve(G)` different system | **0.002s** |
| Parameter homotopy | **0.3s** |

## Root Cause: `CompiledSystem{ID}`

The 30x improvement with `:none` reveals the root problem. `CompiledSystem{ID}` uses
`@generated` functions keyed on a hash type parameter. `MixedSystem{ID}` wraps it.
Every new system creates a unique type:

```
Solver{PolyhedralTracker{ToricHomotopy{MixedSystem{0xabc...}}, ...}}
```

Julia must recompile the entire pipeline (`Tracker`, `EndgameTracker`, `Solver`,
`serial_solve`, ...) from scratch each time. `InterpretedSystem` has no type parameter,
so compiled code is reused. Benchmarks show `:none` matches `:mixed` runtime
performance within 4%.

## Key Remaining Refactor: Type-erase `CompiledSystem`

Keep the compiled evaluation speed without leaking the hash into the type system:

```julia
# Current: hash in type parameter — unique type per system
struct CompiledSystem{HI} <: AbstractSystem
    ...
end

# Proposed: opaque wrapper with FunctionWrappers
struct CompiledSystem <: AbstractSystem
    ...
    _evaluate!::FunctionWrapper{Nothing, Tuple{Vector{ComplexF64}, ...}}
    _evaluate_and_jacobian!::FunctionWrapper{Nothing, Tuple{...}}
end
```

The `@generated` functions still exist but are called through `FunctionWrapper`,
which erases the type. The outer `CompiledSystem` becomes a single concrete type.
This would give `:mixed` the same TTFX as `:none` (~1.1s) while keeping compiled speed.

## Tools Used

- **SnoopCompile** (`@snoop_invalidations`, `@snoop_inference`): invalidation trees,
  inference time breakdown
- **JET** (`@report_opt`): runtime dispatch analysis — **zero dispatches** on hot path
- **Cthulhu** (`find_method_instance`, `generate_code_instance`): verified concrete return
  types: `serial_solve` → `Result`, `step!` → `Bool`, `track` → `PathResult`
- **BenchmarkTools**: StructArray vs Matrix, InterpretedSystem vs MixedSystem

## Fixes Applied

### 1. `Val` dispatch for `compile` keyword (API change)

`compile` accepts only `Val{:none}`, `Val{:all}`, `Val{:mixed}`. Each `fixed()` method
returns a concrete type. `Val(true)`/`Val(false)` removed — symbols only.

### 2. Remove `StructArray` branch from `MatrixWorkspace`

Branched on `m > 25` between `Matrix{ComplexF64}` and `StructArray{ComplexF64}`.
Benchmarks showed StructArray is slower (0.84x at n=30). Always uses `Matrix{ComplexF64}`.

### 3. Remove redundant `isbits`/`isbitstype` overloads

`DoubleF64` is naturally isbits. The overloads caused ~1,805 invalidations.

### 4. `to_number` returns `ComplexF64`

Narrows `IRStatementArg` from `Union{Nothing, Number, Symbol, IRStatementRef}` (unbounded)
to `Union{Nothing, ComplexF64, Symbol, IRStatementRef}` (4 types, union-splittable).
Also `InstructionSequence.constants`: `Vector{Number}` → `Vector{ComplexF64}`.

### 5. Typed IR construction

`IRPair` type alias, typed comprehensions in `process_sum!`, return annotation on
`reduce_to_at_most_two_multiplicants!`. Eliminates `Vector{Tuple{Any, Any}}` inference.

### 6. Typed `support_coefficients`

Returns `Vector{Vector{ComplexF64}}` for numeric systems instead of `Vector{Any}`.

### 7. Distinct variable names in pipeline

Stopped reassigning `F` through different types. Each step uses a new variable
(`F_compiled`, `F_target`, `F_chart`, `F_final`).

### 8. Eager `InterpretedSystem` fields

Removed `Union{Nothing, Interpreter{AcbRefVector}}` — always initialized.

### 9. Other

- Fixed captured variables (`prev_stmt_arg`, `found_id`)
- Type assertion on `ProgressMeter.tty_width` return
- Typed comprehension for `scaling` in `total_degree_variables`

## Invalidations

~19,400 from upstream packages. **HC contributes zero.**

| Module | Count | Root cause |
|--------|-------|------------|
| Arblib | 6,599 | `show`, `isinf`, `!=` |
| VectorizationBase | 5,077 | Broad comparison operators |
| MultivariatePolynomials | 1,488 | `==`, `isequal` |
| Static | 864 | `convert(::Type{T<:Number}, ...)` |
| Mods | 708 | `hash(::AbstractMod)` |
| Others | ~4,287 | CommonWorldInvalidations, StaticArrays, etc. |

## Further Opportunities

| Refactor | Impact | Status |
|----------|--------|--------|
| **Type-erase `CompiledSystem{ID}`** | High — `:mixed` gets same TTFX as `:none` | Planned |
| Fix SymEngine reinitialization (#643) | Enables `@compile_workload` | Blocked upstream |
| Upstream invalidation PRs | -19,400 invalidations | Out of scope |
| LoopVectorization as extension | -5,500 invalidations | Out of scope |

## Not Planned

| Refactor | Reason |
|----------|--------|
| Trait dispatch for `is_homogeneous` | Genuinely runtime-dependent, Union is correct |
| `@compile_workload` | Blocked by SymEngine reinitialization (#643) |
