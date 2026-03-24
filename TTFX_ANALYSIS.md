# TTFX Analysis for HomotopyContinuation.jl

**Date**: 2026-03-24 | **Julia**: 1.12 | **Baseline**: ~33s TTFX for first `solve()` call

## Results

| Benchmark | Before | After | Speedup |
|-----------|--------|-------|---------|
| `solve(F)` (default polyhedral) | ~33s | ~20s | **1.7x** |
| Parameter homotopy | ~33s | ~0.4s | **80x** |
| Package load | 1.8s | 1.9s | — |
| Precompilation (one-time) | 8s | 35s | +27s |

The remaining ~20s in `solve(F)` is ModelKit system construction (~70 runtime dispatches
from `to_number`) and polyhedral mixed cell computation — both outside the tracker pipeline.

## Tools Used

- **SnoopCompile** (`@snoop_invalidations`): identified 19,023 invalidations, 27 from HC
- **JET** (`@report_opt`): tracked runtime dispatches from 143 → 24 (non-ModelKit: 70 → 15)
- **Cthulhu** (`find_method_instance`, `generate_code_instance`): verified concrete return
  types for every function in the tracker pipeline (`step!`, `track`, `serial_solve`, `solve`)
- **BenchmarkTools**: benchmarked StructArray vs Matrix for MatrixWorkspace (StructArray slower)

## Fixes Applied

### 1. `Val` dispatch for `compile` keyword (API change, all files)

**Root cause**: `fixed(F; compile=:none)` branched on a runtime `Symbol`/`Bool` value,
returning `Union{InterpretedSystem, CompiledSystem, MixedSystem}`. This 3-way Union
propagated through the entire pipeline.

**Fix**: Changed `compile` to accept `Val` everywhere. `fixed()` dispatches on `Val{C}`,
each method returning a concrete type:

```julia
# Before (type-unstable)
fixed(F::System; compile = :mixed) = if compile == :none ... elseif ...

# After (type-stable via dispatch)
fixed(F::System, ::Val{:none}) = InterpretedSystem(F)
fixed(F::System, ::Val{:mixed}) = MixedSystem(F)
fixed(F::System, ::Val{:all}) = CompiledSystem(F)
```

- `COMPILE_DEFAULT` stores `Val` directly: `Ref{Val}(Val(:mixed))`
- All internal functions accept `compile::Val`
- User API: `solve(F; compile=Val(:none))`, `set_default_compile(:none)` still works
- Tests updated throughout

**JET result**: `fixed()` return type is now fully concrete — no more Union.

### 2. Remove `StructArray` branch from `MatrixWorkspace` (`src/linear_algebra.jl`)

**Root cause**: `MatrixWorkspace` branched on `m > 25` between `Matrix{ComplexF64}` and
`StructArray{ComplexF64}`, making the `M` type parameter unknown and poisoning inference
through `Jacobian{M}` → `TrackerState{M}` → `Tracker{H,M}`.

**Fix**: Benchmarks showed StructArray is slower on modern Julia (0.84x at n=30, 0.92x at
n=50). Removed the branch entirely — always uses `Matrix{ComplexF64}`.

**Cthulhu result**: `Tracker`, `EndgameTracker` now infer concrete return types.

### 3. Remove redundant `isbits`/`isbitstype` overloads (`src/DoubleDouble.jl`)

`DoubleF64` is an immutable struct of two `Float64` — naturally isbits. The overloads
added methods to compiler intrinsics called throughout Base, causing ~1,805 invalidations.

### 4. Type-stable `support_coefficients` (`src/model_kit/symbolic.jl`)

Replaced broadcasting with an explicit loop. `supports::Vector{Matrix{Int32}}` is now
concrete. Coefficients remain abstract (can be symbolic `Expression` when parameters
are present).

### 5. `to_number` returns `ComplexF64` (`src/model_kit/symengine.jl`)

Changed `to_number(x::Basic)` from returning `Union{Int32, Int64, Float64, Rational, ...}`
to returning `ComplexF64` for all numeric expressions. This narrows `IRStatementArg` from
`Union{Nothing, Number, Symbol, IRStatementRef}` (unbounded) to
`Union{Nothing, ComplexF64, Symbol, IRStatementRef}` (4 concrete types, union-splittable).

Also changed `InstructionSequence.constants` from `Vector{Number}` to `Vector{ComplexF64}`.

### 6. Distinct variable names in pipeline (`src/total_degree.jl`, `src/polyhedral.jl`, `src/solve.jl`)

Stopped reassigning `F` through different types (`System` → `InterpretedSystem` →
`FixedParameterSystem` → `AffineChartSystem` → `RandomizedSystem`). Each step uses
a new variable (`F_compiled`, `F_target`, `F_chart`, `F_final`) so the compiler sees
a single concrete type per variable.

### 7. Other fixes

- Fixed captured variable `prev_stmt_arg` (extracted to `_resolve_arg`)
- Fixed captured variable `found_id` in `UniquePoints.add!` (`src/unique_points.jl`)
- Type assertion on `ProgressMeter.tty_width` return (`src/solve.jl`)
- Typed comprehension for `scaling` in `total_degree_variables` (`src/total_degree.jl`)
- Precompile directives for tracker pipeline (`src/precompile.jl`)

## Remaining Invalidations (from dependencies)

~19,000 invalidations from upstream packages. HC itself contributes only ~27.

| Module | Count | Root cause |
|--------|-------|------------|
| Arblib | 6,599 | `show`, `isinf`, `!=` |
| VectorizationBase | 5,077 | Broad comparison operators |
| MultivariatePolynomials | 1,488 | `==`, `isequal` |
| Static | 864 | `convert(::Type{T<:Number}, ...)` |
| Mods | 708 | `hash(::AbstractMod)` |
| Others | ~4,287 | CommonWorldInvalidations, StaticArrays, FillArrays, etc. |

## Remaining JET Reports

JET `@report_opt` on `solve(solver, starts)` — the hot path — shows **10 reports**,
all outside our control:

- 4 × ProgressMeter internals (`tty_width`, `tlast`, `dt` — upstream types)
- 4 × `Result` constructor with `Union{Nothing, UInt32}` seed / `Union{Nothing, Symbol}` start_system — intentional design, Julia union-splits these efficiently
- 1 × `MultiplicityInfo` recursive closure — JET optimization limitation
- 1 × Dict assignment in multiplicity detection

**Zero dispatches in the tracker pipeline** (`step!`, `track`, `init!`, `serial_solve`).

JET crashes on paths involving `total_degree`/`polyhedral` (Taylor `@generated` functions
trigger a JET bug with Julia 1.12's Compiler internals). Cthulhu verification confirms
all tracker functions return concrete types on those paths too.

## Further Opportunities

- **Fix SymEngine reinitialization** (#643) → enables `@compile_workload`
- **File upstream PRs** for Arblib, VectorizationBase invalidations
- **LoopVectorization as extension** → -5,500 invalidations
- **Trait dispatch for `is_homogeneous`** → eliminates the last Union in homotopy construction
- **ModelKit `to_number`** → return `ComplexF64` always instead of abstract `Number`

## Reproducing

```julia
# Fresh session required for invalidation analysis
using Pkg; Pkg.activate(; temp=true)
Pkg.develop(path="."); Pkg.add(["SnoopCompileCore", "SnoopCompile", "JET", "Cthulhu"])

# SnoopCompile: invalidations
using SnoopCompileCore
invs = @snoop_invalidations begin; using HomotopyContinuation; end
using SnoopCompile
trees = invalidation_trees(invs)
trees = filter(t -> SnoopCompile.countchildren(t) > 0, trees)
sort!(trees; by=SnoopCompile.countchildren, rev=true)

# JET: runtime dispatch analysis
using JET, HomotopyContinuation
@var x y
F = InterpretedSystem(System([x^2 + y - 1, x + y^2 - 1]))
@report_opt target_modules=(HomotopyContinuation,) HomotopyContinuation.total_degree(F; compile=Val(:none))

# Cthulhu: verify concrete return types
using Cthulhu: find_method_instance, generate_code_instance, AbstractProvider
interp = Base.Compiler.NativeInterpreter()
provider = AbstractProvider(interp)
solver, starts = solver_startsolutions(F; compile=Val(:none), start_system=:total_degree)
tracker = solver.trackers[1].tracker
mi = find_method_instance(provider, HomotopyContinuation.step!, Tuple{typeof(tracker)})
ci = generate_code_instance(provider, mi)
println("step! return: ", ci.rettype)  # Should be Bool (concrete)
```
