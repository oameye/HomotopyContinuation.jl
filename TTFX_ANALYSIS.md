# TTFX Analysis for HomotopyContinuation.jl

**Date**: 2026-03-24 | **Julia**: 1.12 | **Baseline**: ~33s TTFX for first `solve()` call

## Results

| Benchmark | Before | After | Speedup |
|-----------|--------|-------|---------|
| `solve(F)` (default polyhedral) | ~33s | ~19s | **1.7x** |
| Parameter homotopy | ~33s | ~0.3s | **110x** |
| Package load | 1.8s | 1.9s | — |
| Precompilation (one-time) | 8s | 30s | +22s |

The +22s precompilation pays for itself after 1–2 sessions. The remaining ~19s in
`solve(F)` is dominated by ModelKit system construction (~10s, 70 runtime dispatches)
and polyhedral mixed cell computation (~3s) — both outside the tracker pipeline.

## Fixes Applied

### 1. Remove redundant `isbits`/`isbitstype` overloads (`src/DoubleDouble.jl`)

`DoubleF64` is an immutable struct of two `Float64` — Julia already knows it's isbits.
The overloads added methods to compiler intrinsics called pervasively in Base, causing
~1,805 unnecessary invalidations (20,828 → 19,023).

### 2. Remove `StructArray` branch from `MatrixWorkspace` (`src/linear_algebra.jl`)

`MatrixWorkspace` branched on `m > 25` between `Matrix{ComplexF64}` and
`StructArray{ComplexF64}`. This made the `M` type parameter unknown, poisoning inference
through `Jacobian{M}` → `TrackerState{M}` → `Tracker{H,M}` → the entire pipeline.

**Benchmarks showed StructArray is actually slower** (0.84x at n=30, 0.92x at n=50) on modern
Julia — the optimization is outdated. Removing it makes the entire tracker pipeline
monomorphic: `MatrixWorkspace` always returns `MatrixWorkspace{Matrix{ComplexF64}}`.

Verified with Cthulhu: `Tracker`, `EndgameTracker` now infer **concrete** return types.

### 3. Function barriers in solve pipeline (`src/solve.jl`)

**a) `parameter_homotopy` → `_parameter_homotopy`**

Split into `_parameter_homotopy` (always returns concrete `ParameterHomotopy` + homogeneity
flag) and callers that branch on the flag. Each branch calls `EndgameTracker`/`Solver` with a
concrete type. This avoids the `ishomogeneous` keyword proposed in PR #654.

**b) `solver_startsolutions` early returns**

Each branch (`start_parameters`, `polyhedral`, `total_degree`) returns immediately rather than
assigning to a shared `tracker` variable that accumulates Union types.

**c) Function barriers in `total_degree` and `polyhedral` paths**

`_make_td_tracker` and `_make_polyhedral_tracker` isolate the homotopy/tracker construction
from the type-unstable system preparation (where `fixed()` returns Union types).

### 4. ModelKit type stability improvements (`src/model_kit/instruction_sequence.jl`)

- **Fixed captured variable** `prev_stmt_arg` in instruction sequence construction
  (extracted to `_resolve_arg` / `_build_instruction_args`)
- **Changed `constants::Vector{Number}` to `Vector{ComplexF64}`** in `InstructionSequence`
  — eliminates abstract container that prevented inference

### 5. Type annotations in `total_degree` (`src/total_degree.jl`)

`support_coefficients` returns `Tuple{Any, Any}`. Added type assertions
(`::Vector{Matrix{Int32}}`) and typed comprehension for `scaling` to prevent the
abstract types from cascading.

### 3. Precompile directives (`src/precompile.jl`)

Explicit `precompile(f, types)` for the common pipeline types (`Tracker`, `EndgameTracker`,
`Solver`, `init!`, `step!`, `track`, `MatrixWorkspace`, `Jacobian`, `Result`). Uses
`precompile` (not `@compile_workload`) to avoid SymEngine reinitialization issues (#643).

## Remaining Invalidations (from dependencies)

~19,000 invalidations from upstream packages on every `using HomotopyContinuation`.
HC itself contributes only ~27.

| Module | Count | Root cause |
|--------|-------|------------|
| Arblib | 6,599 | `show`, `isinf`, `!=` |
| VectorizationBase | 5,077 | Broad comparison operators |
| MultivariatePolynomials | 1,488 | `==`, `isequal` |
| Static | 864 | `convert(::Type{T<:Number}, ...)` |
| Mods | 708 | `hash(::AbstractMod)` |
| Others | ~4,287 | CommonWorldInvalidations, StaticArrays, FillArrays, etc. |

## Further Opportunities

- **Fix SymEngine reinitialization** (#643) → enables `@compile_workload` for even better TTFX
- **File upstream PRs** for Arblib, VectorizationBase, MultivariatePolynomials invalidations
- **LoopVectorization as extension** → would remove ~5,500 invalidations
- **ModelKit type stability** → 5 remaining runtime dispatches in system construction

## Reproducing

```julia
# Fresh session required for invalidation analysis
using Pkg; Pkg.activate(; temp=true)
Pkg.develop(path="."); Pkg.add(["SnoopCompileCore", "SnoopCompile", "JET"])

using SnoopCompileCore
invs = @snoop_invalidations begin; using HomotopyContinuation; end
using SnoopCompile
trees = invalidation_trees(invs)
trees = filter(t -> SnoopCompile.countchildren(t) > 0, trees)
sort!(trees; by=SnoopCompile.countchildren, rev=true)

# JET analysis
using JET, HomotopyContinuation
@var x y
F = InterpretedSystem(System([x^2 + y - 1, x + y^2 - 1]))
@report_opt target_modules=(HomotopyContinuation,) HomotopyContinuation.total_degree(F)
```
