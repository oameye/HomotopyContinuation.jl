# System Type Design

## Goal

Introduce a `System` type that caches the compiled interpreter pipeline so users pay the compilation cost once and reuse it across multiple `solve` calls. `System` replaces `PolynomialSystemInfo` and becomes the only entry point to `solve`.

## Struct Definition

```julia
struct System
    evaluator::SystemEvaluator
    degrees::Vector{Int}
    nvars::Int
    nparams::Int
    variable_groups::Vector{Vector{Int}}
    is_homogeneous::Bool
    # GC roots -- interpreters must stay alive for FunctionWrapper closures
    _seq_eval::InstructionSequence
    _seq_jac::InstructionSequence
    _interp_f64::Interpreter{Vector{ComplexF64}}
    _interp_df64::Interpreter{Vector{ComplexDF64}}
    _interp_jac::Interpreter{Vector{ComplexF64}}
    _interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2,ComplexF64}}}
    _interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3,ComplexF64}}}
    _interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4,ComplexF64}}}
end
```

## Constructor

```julia
System(polys::AbstractVector{<:MP.AbstractPolynomialLike};
       parameters::AbstractVector = <empty>,
       variables::AbstractVector = <auto-detected>)
```

Runs the full pipeline: polynomials -> SExpr -> CSE -> InstructionSequence -> Interpreters -> FunctionWrappers -> SystemEvaluator. All cached in the struct.

## User API

```julia
@polyvar x y
F = System([x^2 + y - 1, x*y - 2])

# Solving -- System is the only input
solve(F)                                          # default TotalDegree
solve(F, TotalDegree())
solve(F, Polyhedral())
solve(F, TotalDegree(; max_steps=500, seed=UInt32(42)))

# CommonSolve interface
cache = CommonSolve.init(F, TotalDegree())
result = CommonSolve.solve!(cache)

# Metadata
degrees(F)       # Vector{Int}
nvariables(F)    # Int
nparameters(F)   # Int
size(F)          # (neqs, nvars)
```

## Changes Required

### Removed
- `PolynomialSystemInfo` struct -- replaced by `System`

### Modified
- `system_eval()` -> renamed/refactored into the `System` constructor. Returns `System` instead of `(PolynomialSystemInfo, SystemEvaluator)` tuple.
- `solve(polys, alg)` -> `solve(F::System, alg)`. Raw polynomial vectors no longer accepted.
- `CommonSolve.init(polys, alg)` -> `CommonSolve.init(F::System, alg)`. Same change.
- `_total_degree_startsystem(degrees, variables)` -- currently takes `variables` to build the start system from DynamicPolynomials. Needs rethinking: the start system `x_i^d_i - 1` can be built from degrees alone, returning a `SystemEvaluator` (or `System`) without needing the original variable objects.
- `_build_parametric_system` in polyhedral.jl -- returns `System` instead of `(PolynomialSystemInfo, SystemEvaluator)`. Internal `_param_info` GC root field in `PolyhedralSolveCache` changes to hold a `System`.
- All internal code accessing `info.degrees`, `info.nvars` etc. -> `F.degrees`, `F.nvars` or accessor functions.

### Unchanged
- `SystemEvaluator` -- still exists as a field of `System`, still the FunctionWrapper firewall
- `HomotopyEvaluator`, all homotopy types, Tracker, Newton, Predictor -- untouched
- The interpreter pipeline (SExpr, InstructionSequence, Interpreter) -- untouched
- `PathResult`, `Result`, result accessors -- untouched

## Start System Handling

The total degree start system `G = [x_i^d_i - 1]` is currently built via `_total_degree_startsystem(degrees, variables)` which needs the DynamicPolynomials variable objects. Two options:

1. Build the start system `System` from polynomials inside `CommonSolve.init` before the original variables go out of scope. The start `System` is ephemeral (not cached).
2. Build the start `SystemEvaluator` directly from degrees without DynamicPolynomials (the monomial pattern is trivial: one variable raised to a power minus one). This avoids the MP roundtrip entirely.

Option 1 is simpler and sufficient. The start system compilation cost is negligible compared to tracking. Option 2 is a future optimization (already in the README TODO as "direct monomial evaluator").

Decision: **Option 1** -- build start `System` from polynomials in `init`, extract its `.evaluator`.

## Test Changes

- All tests that call `solve([x^2 - 1, y - 2])` must change to `solve(System([x^2 - 1, y - 2]))`.
- Tests that import `PolynomialSystemInfo` must be updated.
- New tests: `System` construction, metadata accessors, reuse across multiple solves.
- CheckConcreteStructs test: `System` must pass `all_concrete`.
