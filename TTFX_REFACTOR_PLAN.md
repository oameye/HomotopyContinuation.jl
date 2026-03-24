# Refactor Plan: Inherently Type-Stable HomotopyContinuation.jl

## Current State

After our fixes, the tracker hot loop (`step!`, `track`, `serial_solve`) is fully
type-stable. The remaining 20s TTFX comes from **system construction** — converting
symbolic polynomials into numerical evaluation pipelines. This document envisions
larger refactors to make the entire package inherently type-stable.

## Remaining Type Instabilities (15 non-ModelKit + 73 ModelKit)

| Source | Dispatches | Root cause |
|--------|-----------|------------|
| `is_homogeneous` branching | 3 | Returns `H` or `AffineChartHomotopy{H}` |
| `to_number` | 73 | Returns `Int32 \| Float64 \| Rational \| Complex` |
| `support_coefficients` coeffs | ~5 | Abstract element type from `to_number` |
| `F` reassignment in total_degree | ~4 | `F` changes type through pipeline |

## Refactor A: Eliminate `is_homogeneous` branching with dispatch

### Problem

Seven functions check `is_homogeneous(f)` at runtime and conditionally wrap in
`AffineChartHomotopy` / `AffineChartSystem`, producing Union return types:

```julia
# Current pattern (in parameter_homotopy, total_degree, polyhedral, etc.)
H = ParameterHomotopy(fixed(F), p0, p1)
if is_homogeneous(System(F))
    H = on_affine_chart(H)  # now Union{ParameterHomotopy, AffineChartHomotopy}
end
```

### Solution: Trait dispatch

```julia
# Define traits
struct Homogeneous end
struct Affine end

# Compute trait once
homogeneity_trait(F::System) = is_homogeneous(F) ? Homogeneous() : Affine()

# Dispatch eliminates Union — each method returns one concrete type
function _build_parameter_homotopy(::Affine, F, p0, p1, compile)
    ParameterHomotopy(fixed(F, compile), p0, p1)
end

function _build_parameter_homotopy(::Homogeneous, F, p0, p1, compile)
    on_affine_chart(ParameterHomotopy(fixed(F, compile), p0, p1))
end
```

The trait is computed once (runtime cost), but dispatch resolves the type at the
function boundary. Each method returns a concrete type — no Union.

### Where to apply

- `parameter_homotopy` in `src/solve.jl`
- `total_degree_variables` in `src/total_degree.jl`
- `total_degree_variable_groups` in `src/total_degree.jl`
- `polyhedral` in `src/polyhedral.jl`
- `linear_subspace_homotopy` in `src/solve.jl`
- `start_target_homotopy` in `src/solve.jl`
- `paths_to_track` in `src/polyhedral.jl`

### Impact

Eliminates the last 3 non-ModelKit Union dispatches in the solve pipeline.

## Refactor B: Stop reassigning `F` to different types

### Problem

In `total_degree_variables`, the variable `F` is reassigned through multiple types:

```julia
F::Union{System, AbstractSystem}  # input
F = fixed(F; compile=compile)     # now InterpretedSystem (or similar)
F = fix_parameters(F, ...)        # now FixedParameterSystem{...}
F = on_affine_chart(F)            # now AffineChartSystem{...}
F = square_up(F)                  # now RandomizedSystem{...}
```

Each reassignment changes the type. The compiler sees `F` as a Union of all possible
types it could be at any point.

### Solution: Use distinct variable names

```julia
F_fixed = fixed(F, compile)
F_param = fix_parameters(F_fixed, target_parameters)
F_chart = on_affine_chart(F_param)
F_final = square_up(F_chart)
```

Each variable has a single concrete type. No Union accumulation. This is a simple
rename refactor with no behavioral change.

### Where to apply

- `total_degree_variables` — 4 reassignments of `F`
- `total_degree_variable_groups` — 3 reassignments of `F`
- `polyhedral` — 3 reassignments of `F`
- `start_target_homotopy` — 2 reassignments each for `G` and `F`

## Refactor C: Make `to_number` return `ComplexF64`

### Problem

`to_number(x::Basic)` in `src/model_kit/symengine.jl` returns different types based
on the symbolic expression class:

```julia
if cls == :Integer → Int32 / Int64 / Int128 / BigInt
if cls == :RealDouble → Float64
if cls == :Rational → Rational{...}
if cls == :Complex → Complex{...}
```

This is the source of 73 ModelKit dispatches. Every downstream function that touches
these values (`is_one`, `is_zero`, `add_op!`, `to_smallest_eltype`) dispatches at runtime.

### Solution: Always return `ComplexF64`

```julia
function to_number(x::Basic)::ComplexF64
    cls = class(x)
    if cls == :Integer
        ComplexF64(convert(Int64, convert(BigInt, x)))
    elseif cls == :RealDouble
        ComplexF64(convert(Float64, x))
    elseif cls == :Rational
        a, b = _numer_denom(x)
        ComplexF64(convert(Float64, convert(BigInt, a)) / convert(Float64, convert(BigInt, b)))
    elseif cls == :RealMPFR
        ComplexF64(convert(Float64, convert(BigFloat, x)))
    elseif cls == :ComplexDouble || cls == :Complex
        ComplexF64(_real_part(x), _imag_part(x))
    else
        error("Cannot convert $x to ComplexF64")
    end
end
```

### Consequences

- `to_smallest_eltype` becomes trivial (always `Vector{ComplexF64}`)
- `exponents_coefficients` returns `Tuple{Matrix{Int32}, Vector{ComplexF64}}`
- `support_coefficients` returns `Tuple{Vector{Matrix{Int32}}, Vector{Vector{ComplexF64}}}`
- `InstructionSequence.constants` is already `Vector{ComplexF64}` (done)
- `IRStatementArg` can narrow `Number` to `ComplexF64`

### Risk

- Loss of exact integer arithmetic for coefficient comparisons (`is_one`, `is_zero`)
  — but these can compare with tolerance or use `== 1.0 + 0.0im`
- Loss of exact rational arithmetic — but this is already lost when converting to
  `Float64` for numerical evaluation
- `BigInt` overflow for very large integer coefficients — unlikely in practice

### Impact

Would eliminate ~73 ModelKit dispatches. The entire `System → InterpretedSystem`
construction becomes type-stable.

## Refactor D: Eager initialization of `InterpretedSystem` fields

### Problem

```julia
mutable struct InterpretedSystem <: AbstractSystem
    eval_acb::Union{Nothing, Interpreter{AcbRefVector}}  # lazy
    jac_acb::Union{Nothing, Interpreter{AcbRefVector}}    # lazy
end
```

### Solution: Always initialize, or use a type parameter

```julia
# Option 1: Always initialize (simplest)
struct InterpretedSystem <: AbstractSystem
    eval_acb::Interpreter{AcbRefVector}
    jac_acb::Interpreter{AcbRefVector}
end

# Option 2: Parameterize (zero-cost but more complex)
struct InterpretedSystem{Acb} <: AbstractSystem
    # Acb is either Interpreter{AcbRefVector} or Nothing
end
```

Option 1 is simpler. The Acb interpreters are lightweight to construct (just a tape
copy) — the cost is negligible compared to the SymEngine compilation.

## Refactor E: Typed `solver_startsolutions` via dispatch

### Problem

`solver_startsolutions` has a massive if/elseif chain that produces different tracker
types based on runtime values (`start_system`, `start_parameters`, `start_subspace`):

```julia
if start_subspace !== nothing
    tracker = EndgameTracker(linear_subspace_homotopy(...))
elseif start_parameters !== nothing
    tracker = parameter_homotopy_tracker(...)
elseif start_system == :polyhedral
    tracker, starts = polyhedral(...)
elseif start_system == :total_degree
    tracker, starts = total_degree(...)
end
Solver(tracker)  # Union of 4+ tracker types
```

### Solution: Dispatch on solver mode

```julia
# User-facing
solve(F; start_system = Val(:polyhedral), ...)

# Dispatched
function _solver_startsolutions(F, ::Val{:polyhedral}, compile; kwargs...)
    tracker, starts = polyhedral(F; compile=compile, kwargs...)
    Solver(tracker), starts  # concrete Solver type
end

function _solver_startsolutions(F, ::Val{:total_degree}, compile; kwargs...)
    tracker, starts = total_degree(F; compile=compile, kwargs...)
    Solver(tracker), starts  # concrete Solver type
end
```

Combined with the `compile::Val` refactor (already done), this makes
`solver_startsolutions` return a concrete `Solver{T}` type for each combination.

## Status

| Refactor | Impact | Effort | Status |
|----------|--------|--------|--------|
| **B** (distinct variable names) | Medium | Low | **Done** ✓ |
| **A** (trait dispatch for homogeneity) | None | Medium | Skipped — Union is genuinely runtime-dependent |
| **C** (to_number → ComplexF64) | High | Medium | **Done** ✓ |
| **D** (eager InterpretedSystem) | Low | Low | **Done** ✓ |
| **E** (Val dispatch for start_system) | Medium | Medium | **Done** ✓ |

All feasible refactors complete.

Refactors B and A are safe, non-breaking, and address the remaining non-ModelKit
dispatches. Refactor C is the highest impact but needs careful testing of the
certification/exact-arithmetic paths. Refactor E extends the `Val` pattern we
already established for `compile`.

## Expected Outcome

After all refactors, the entire `solve` pipeline from `solver_startsolutions` through
`serial_solve` would be type-stable for any given `compile` mode and `start_system`
choice. JET dispatches would drop to near-zero outside of SymEngine interop.
