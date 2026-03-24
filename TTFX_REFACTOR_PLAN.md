# Refactor Plan: Making HomotopyContinuation.jl Type-Stable

## The Core Problem

The package mixes **value-based decisions** (symbols, bools, runtime checks) with
**type-based dispatch**. Julia excels at the latter but can't optimize the former.
Every `if compile == :none ... elseif compile == :mixed ...` becomes a Union return type
that the compiler can't resolve.

## Current Architecture (simplified)

```
User: solve(System([x^2+y-1, x+y^2-1]))
  → solver_startsolutions(F; compile=:mixed, start_system=:polyhedral)
    → fixed(F; compile=:mixed)           # returns Union{Interpreted,Compiled,Mixed}
    → is_homogeneous(F) ? wrap : don't   # returns Union{H, AffineChart{H}}
    → Tracker(H)                         # M depends on MatrixWorkspace branch (FIXED)
    → EndgameTracker(tracker)
    → Solver(endgame_tracker)
  → solve(solver, starts)
    → serial_solve → track → step!       # HOT LOOP (already type-stable)
```

The hot loop is type-stable. The instability is in **system construction** — a one-time
cost per solve, but it causes JIT compilation of many Union-typed intermediate functions.

## Proposed Refactors (ordered by impact and feasibility)

### Refactor 1: Eliminate `fixed()` value dispatch → use type dispatch

**Current** (value dispatch, returns Union):
```julia
function fixed(F::System; compile = COMPILE_DEFAULT[])
    if compile == :all
        CompiledSystem(F)
    elseif compile == :none
        InterpretedSystem(F)
    elseif compile == :mixed
        MixedSystem(F)
    end
end
```

**Proposed** (type dispatch, each method returns concrete type):
```julia
# User-facing: accept symbol, dispatch to typed version
fixed(F::System; compile = COMPILE_DEFAULT[]) = _fixed(Val(compile), F)

# Type-dispatched methods — each returns a concrete type
_fixed(::Val{:all}, F::System) = CompiledSystem(F)
_fixed(::Val{:none}, F::System) = InterpretedSystem(F)
_fixed(::Val{false}, F::System) = InterpretedSystem(F)
_fixed(::Val{:mixed}, F::System) = MixedSystem(F)
_fixed(::Val{true}, F::System) = CompiledSystem(F)
```

**Impact**: Every downstream call that receives the output of `fixed()` now gets a
concrete type via dispatch. The `Val` wrapping happens once at the top; all internals
are type-stable.

**Feasibility**: High — mechanical replacement. `fixed()` is called ~30 times but always
with `compile` from a kwarg that can be Val-wrapped.

### Refactor 2: Propagate `compile` as `Val` through the pipeline

Currently `compile::Union{Bool,Symbol}` is passed as a value through the entire call
chain. Change to:

```julia
function solver_startsolutions(F; compile = COMPILE_DEFAULT[], kwargs...)
    _solver_startsolutions(Val(compile), F; kwargs...)
end

function _solver_startsolutions(::Val{C}, F; ...) where C
    # C is now a type parameter — all fixed() calls use it as Val{C}
    # and return concrete types
end
```

This propagates the compile mode as a **type parameter**, so every `fixed()` call
inside the pipeline returns a concrete type. The `Val` wrapping overhead is negligible
(once per solve call).

**Impact**: Eliminates the `Union{InterpretedSystem, CompiledSystem, MixedSystem}` that
currently flows through `ParameterHomotopy{T}`, `StraightLineHomotopy{G,F}`, etc.

**Feasibility**: Medium — requires threading `Val{C}` through ~10 functions in the solve
pipeline. But each change is mechanical.

### Refactor 3: Replace `is_homogeneous` branching with dispatch

**Current** (runtime bool check, returns Union):
```julia
H = ParameterHomotopy(fixed(F), p0, p1)
if is_homogeneous(System(F))
    H = on_affine_chart(H)  # changes type!
end
```

**Proposed** (dispatch on system trait):
```julia
# Trait types
struct Homogeneous end
struct Affine end

homogeneity(F::System) = is_homogeneous(F) ? Homogeneous() : Affine()

# Dispatch on trait
function _make_homotopy(::Homogeneous, F, p0, p1, compile)
    H = ParameterHomotopy(_fixed(compile, F), p0, p1)
    on_affine_chart(H)  # always wraps → concrete AffineChartHomotopy type
end

function _make_homotopy(::Affine, F, p0, p1, compile)
    ParameterHomotopy(_fixed(compile, F), p0, p1)  # concrete ParameterHomotopy type
end
```

The `homogeneity()` call is still runtime, but the dispatch resolves the Union at the
function boundary. Each method returns a single concrete type.

**Impact**: Eliminates `Union{ParameterHomotopy, AffineChartHomotopy}` at every call site.

**Feasibility**: Medium — the `is_homogeneous` check appears in ~7 functions, each needs
a dispatch split. But the pattern is uniform.

### Refactor 4: Concrete `InstructionSequence` constants

**Current**: `constants::Vector{Number}` (abstract element type)
**Already done**: Changed to `constants::Vector{ComplexF64}` ✓

### Refactor 5: Typed `support_coefficients`

**Current**: Returns `Tuple{Any, Any}` because `to_smallest_eltype` is type-unstable.

**Proposed**: Accept that coefficients have runtime-determined types but use concrete
containers:

```julia
function support_coefficients(F::System)
    supp_coeffs = exponents_coefficients.(F.expressions, Ref(F.variables))
    support = Matrix{Int32}[first(sc) for sc in supp_coeffs]
    coeffs = [Vector{Float64}(last(sc)) for sc in supp_coeffs]
    support, coeffs
end
```

Return `Tuple{Vector{Matrix{Int32}}, Vector{Vector{Float64}}}` — fully concrete.
Converting coefficients to `Float64` is fine since they're used for numerical
computation anyway. For integer-exact paths, add a separate
`support_coefficients_exact(F)` method.

**Impact**: Eliminates ~45 dispatches in `total_degree_variables`.

**Feasibility**: High — but need to verify no caller depends on exact integer coefficients.

### Refactor 6: Eliminate lazy `Union{Nothing, ...}` fields

**Current**: `InterpretedSystem` has `eval_acb::Union{Nothing, Interpreter{AcbRefVector}}`.

**Proposed**: Initialize all interpreters eagerly, or use a separate type:

```julia
# Option A: Always initialize (small cost)
struct InterpretedSystem <: AbstractSystem
    system::System
    eval_ComplexF64::Interpreter{Vector{ComplexF64}}
    eval_ComplexDF64::Interpreter{Vector{ComplexDF64}}
    eval_acb::Interpreter{AcbRefVector}
    taylor_ComplexF64::TaylorInterpreters{ComplexF64}
    jac_ComplexF64::Interpreter{Vector{ComplexF64}}
    jac_acb::Interpreter{AcbRefVector}
end

# Option B: Trait-based (no cost, type-stable)
struct InterpretedSystem{HasAcb} <: AbstractSystem
    # HasAcb::Bool as type parameter
end
```

**Impact**: Eliminates 3 JET reports and makes field access type-stable.

**Feasibility**: Medium — need to check if eager Acb initialization has side effects
or significant cost.

## Priority Order

1. **Refactor 1+2** (Val dispatch for `compile`) — biggest impact, mechanical change
2. **Refactor 5** (typed `support_coefficients`) — eliminates most non-ModelKit dispatches
3. **Refactor 3** (trait dispatch for homogeneity) — eliminates Union at homotopy level
4. **Refactor 6** (eliminate lazy fields) — small impact but cleaner
5. **Refactor 4** — already done ✓

## What NOT to refactor

- **ModelKit IR construction** (73 dispatches): These are from `to_number` which is
  fundamentally type-unstable (converts SymEngine expressions to Julia numbers). Would
  require redesigning the SymEngine binding layer. One-time cost, not worth the effort.
- **`to_smallest_eltype`**: Fundamentally runtime-dependent. Fix at call sites instead.
- **ProgressMeter internals**: Upstream dependency issue.

## Expected Outcome

After refactors 1-3: the entire `solver_startsolutions` → `solve` chain should be
type-stable for a given `compile` mode and system type. JET non-ModelKit dispatches
should drop from 25 to ~5 (remaining: upstream FillArrays broadcasting, ProgressMeter).
