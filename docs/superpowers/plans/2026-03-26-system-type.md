# System Type Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a `System` type that caches the compiled interpreter pipeline, replacing `PolynomialSystemInfo` and becoming the sole entry point to `solve`.

**Architecture:** `System` merges `PolynomialSystemInfo` + `SystemEvaluator` into one struct. The `System` constructor runs the full compilation pipeline (poly -> SExpr -> CSE -> InstructionSequence -> Interpreters -> FunctionWrappers). `solve` only accepts `System`, not raw polynomials.

**Tech Stack:** Julia, MultivariatePolynomials, FunctionWrappers, CommonSolve

---

### Task 1: Define `System` struct and constructor

**Files:**
- Modify: `src/core/system_eval.jl` — replace `PolynomialSystemInfo` with `System`, refactor `system_eval` into `System` constructor
- Modify: `src/HomotopyContinuationNext.jl` — update exports

- [ ] **Step 1: Replace `PolynomialSystemInfo` with `System` struct**

In `src/core/system_eval.jl`, replace the `PolynomialSystemInfo` struct and `system_eval` function with:

```julia
## system_eval — MP polynomial input → System
#
# Converts a vector of MultivariatePolynomials into a fully-wired System
# with all FunctionWrapper closures backed by tape-based interpreters.

"""
    System

Compiled, ready-to-evaluate polynomial system. Caches the interpreter pipeline
so users pay the compilation cost once and reuse it across multiple `solve` calls.

Construct from MultivariatePolynomials:

    F = System(polys; parameters=[], variables=...)

# Examples
```julia
@polyvar x y
F = System([x^2 + y - 1, x*y - 2])
solve(F)
solve(F, Polyhedral())
```
"""
struct System
    evaluator::SystemEvaluator
    degrees::Vector{Int}
    nvars::Int
    nparams::Int
    variable_groups::Vector{Vector{Int}}
    is_homogeneous::Bool
    # GC roots — interpreters must stay alive for FunctionWrapper closures
    _seq_eval::InstructionSequence
    _seq_jac::InstructionSequence
    _interp_f64::Interpreter{Vector{ComplexF64}}
    _interp_df64::Interpreter{Vector{ComplexDF64}}
    _interp_jac::Interpreter{Vector{ComplexF64}}
    _interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}}
    _interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}}
    _interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}}
end
```

- [ ] **Step 2: Add the `System` constructor (replaces `system_eval`)**

Below the struct, add:

```julia
## ── FW-compatible wrapper functions ──────────────────────────────────────────

# (Keep _execute_eval_fw! and _execute_jac_fw! exactly as they are)

## ── System constructor ──────────────────────────────────────────────────────

"""
    System(polys; parameters=[], variables=...) -> System

Build a `System` from a vector of MultivariatePolynomials polynomials.
Compiles the full interpreter pipeline and caches everything for reuse.
"""
function System(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    )::System
    neqs = length(polys)
    nvars = length(variables)
    nparams = length(parameters)

    # ── Build interpreters ────────────────────────────────────────────────
    interp_f64 = _build_interpreter(
        Vector{ComplexF64}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )
    interp_df64 = _build_interpreter(
        Vector{ComplexDF64}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )
    interp_jac = _build_interpreter(
        Vector{ComplexF64}, polys;
        parameters = parameters, variables = variables, include_jacobian = true,
    )
    interp_t1 = _build_interpreter(
        Vector{TruncatedTaylorSeries{2, ComplexF64}}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )
    interp_t2 = _build_interpreter(
        Vector{TruncatedTaylorSeries{3, ComplexF64}}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )
    interp_t3 = _build_interpreter(
        Vector{TruncatedTaylorSeries{4, ComplexF64}}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )

    # ── Metadata ──────────────────────────────────────────────────────────
    degrees = Int[MP.maxdegree(p) for p in polys]

    is_homogeneous = try
        all(p -> MP.ishomogeneous(p), polys)
    catch
        false
    end

    # ── Wire FunctionWrappers ─────────────────────────────────────────────
    evaluator = SystemEvaluator(
        SysEvalFW((u, x, p) -> (_execute_eval_fw!(u, interp_f64, x, p); nothing)),
        SysEvalDF64FW((u, x, p) -> (_execute_eval_fw!(u, interp_df64, x, p); nothing)),
        SysEvalJacFW((u, U, x, p) -> (_execute_jac_fw!(u, U, interp_jac, x, p); nothing)),
        SysTaylor1FW((u, tx, p) -> (execute_taylor!(u, Val(1), interp_t1, tx, p); nothing)),
        SysTaylor2FW((u, tx, p) -> (execute_taylor!(u, Val(2), interp_t2, tx, p); nothing)),
        SysTaylor3FW((u, tx, p) -> (execute_taylor!(u, Val(3), interp_t3, tx, p); nothing)),
        (neqs, nvars),
        nparams,
    )

    return System(
        evaluator, degrees, nvars, nparams,
        Vector{Int}[], is_homogeneous,
        interp_f64.sequence, interp_jac.sequence,
        interp_f64, interp_df64, interp_jac,
        interp_t1, interp_t2, interp_t3,
    )
end
```

- [ ] **Step 3: Add accessor methods**

Below the constructor, add:

```julia
## ── Accessors ────────────────────────────────────────────────────────────────

Base.size(F::System)::Tuple{Int, Int} = size(F.evaluator)
degrees(F::System)::Vector{Int} = F.degrees
nvariables(F::System)::Int = F.nvars
nparameters(F::System)::Int = F.nparams
```

- [ ] **Step 4: Update exports in `src/HomotopyContinuationNext.jl`**

Add `System` to the exports:

```julia
export @polyvar, solve, System
```

- [ ] **Step 5: Remove old `system_eval` function**

Delete the old `system_eval` function from `src/core/system_eval.jl`. Keep `_execute_eval_fw!` and `_execute_jac_fw!` helper functions as they are used by the constructor.

- [ ] **Step 6: Run `make format`**

Run: `make format`

---

### Task 2: Update `solve` and `CommonSolve.init` for TotalDegree

**Files:**
- Modify: `src/solving/solve.jl` — change all signatures from polys to `System`
- Modify: `src/solving/total_degree.jl` — update `_total_degree_startsystem` to accept degrees only

- [ ] **Step 1: Update `_total_degree_startsystem` to build from degrees + variable count**

In `src/solving/total_degree.jl`, the current function takes `variables` (DynamicPolynomials variables). Change it to build polynomials internally:

```julia
function _total_degree_startsystem(degrees::Vector{Int})::System
    n = length(degrees)
    vars = [DynamicPolynomials.PolyVar{true}("_td_x$i") for i in 1:n]
    polys = [vars[i]^degrees[i] - 1 for i in 1:n]
    return System(polys; variables = vars)
end
```

This creates throwaway DynamicPolynomials variables for the start system construction. Import `DynamicPolynomials` at the top of `total_degree.jl` if not already imported (it's available via the main module).

- [ ] **Step 2: Update `CommonSolve.init` for TotalDegree**

In `src/solving/solve.jl`, change the init signature and body:

```julia
function CommonSolve.init(F::System, alg::TotalDegree)::SolveCache
    seed = alg.seed

    eval_G = _total_degree_startsystem(F.degrees)
    starts = _total_degree_solutions(F.degrees)

    rng = Random.MersenneTwister(seed)
    γ = cis(2π * rand(rng))
    H = StraightLineHomotopy(eval_G.evaluator, F.evaluator; γ = γ)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = alg.tracker_options)

    return SolveCache(tracker, starts, seed)
end
```

Note: `StraightLineHomotopy` takes `SystemEvaluator` arguments, so we pass `eval_G.evaluator` and `F.evaluator`.

- [ ] **Step 3: Update `solve` for TotalDegree**

In `src/solving/solve.jl`, change the convenience `solve` function:

```julia
"""
    solve(F::System, alg=TotalDegree())

Solve a polynomial system using homotopy continuation.

# Examples
```julia
@polyvar x y
F = System([x^2 + y - 1, x*y - 2])
result = solve(F)
solutions(result)
real_solutions(result)

# With explicit algorithm and options
result = solve(F, TotalDegree(; seed=UInt32(42)))
```
"""
function solve(F::System, alg::TotalDegree = TotalDegree())::Result
    return CommonSolve.solve!(CommonSolve.init(F, alg))
end
```

- [ ] **Step 4: Remove the `parameters` and `variables` kwargs from solve/init**

These kwargs no longer make sense — they're handled at `System` construction time. Remove them from all `solve` and `CommonSolve.init` signatures. Also remove the `_empty_vars` and `_effective_variables` helper calls.

- [ ] **Step 5: Run `make format`**

Run: `make format`

---

### Task 3: Update `solve` and `CommonSolve.init` for Polyhedral

**Files:**
- Modify: `src/solving/polyhedral.jl` — change `CommonSolve.init` to accept `System`, update `PolyhedralSolveCache` to use `System` instead of `PolynomialSystemInfo`

- [ ] **Step 1: Update `PolyhedralSolveCache`**

Change the `_param_info` field from `PolynomialSystemInfo` to `System`:

```julia
struct PolyhedralSolveCache
    toric_tracker::Tracker
    coeff_tracker::Tracker
    toric_homotopy::ToricHomotopy
    support::Vector{Matrix{Int32}}
    lifting::Vector{Vector{Int32}}
    start_solutions::Vector{Tuple{MixedSubdivisions.MixedCell, Vector{ComplexF64}}}
    seed::UInt32
    # GC roots for interpreters — must be kept alive for FunctionWrapper closures
    _param_system::System
end
```

- [ ] **Step 2: Update `CommonSolve.init` for Polyhedral**

Change the signature and update the `system_eval` call to use `System`:

```julia
function CommonSolve.init(
        F::System,
        alg::Polyhedral,
    )::PolyhedralSolveCache
```

Inside the function body:
- Remove the `parameters` and `variables` kwargs
- Get variables for support extraction: the function currently calls `support_coefficients(polys, variables)`. Since `System` doesn't store the original polynomials, we need to handle this differently.

**Key issue:** `CommonSolve.init` for Polyhedral needs the raw polynomial coefficients and support to compute mixed cells. This data isn't in `System`. Two approaches:

**Approach A:** Store support/coefficients in `System` (adds fields only polyhedral needs).
**Approach B:** Extract support from the `System`'s instruction sequence (complex, fragile).
**Approach C:** Have `Polyhedral` `init` accept both `System` (for the target evaluator) and require the user to pass raw polys for support extraction — but this defeats the purpose.
**Approach D:** Compute support and coefficients during `System` construction and store them. They're small (integer matrices + coefficient vectors) and useful metadata anyway.

**Decision: Approach D.** Add `support::Vector{Matrix{Int32}}` and `coefficients::Vector{Vector{ComplexF64}}` fields to `System`. These are computed from the polynomials during construction and are useful metadata beyond just polyhedral homotopy.

This means we need to go back and update the `System` struct (Task 1). We'll handle that in the next step.

- [ ] **Step 3: Add support and coefficients fields to `System`**

In `src/core/system_eval.jl`, add two fields to the `System` struct:

```julia
struct System
    evaluator::SystemEvaluator
    degrees::Vector{Int}
    nvars::Int
    nparams::Int
    variable_groups::Vector{Vector{Int}}
    is_homogeneous::Bool
    support::Vector{Matrix{Int32}}
    coefficients::Vector{Vector{ComplexF64}}
    # GC roots — interpreters must stay alive for FunctionWrapper closures
    _seq_eval::InstructionSequence
    _seq_jac::InstructionSequence
    _interp_f64::Interpreter{Vector{ComplexF64}}
    _interp_df64::Interpreter{Vector{ComplexDF64}}
    _interp_jac::Interpreter{Vector{ComplexF64}}
    _interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}}
    _interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}}
    _interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}}
end
```

In the `System` constructor, compute them before building interpreters (requires importing `support_coefficients` from `solving/support.jl`):

```julia
# ── Support and coefficients ──────────────────────────────────────────
supp, coeffs = if nparams == 0
    support_coefficients(polys, variables)
else
    # Parametric systems don't have fixed coefficients
    Vector{Matrix{Int32}}[], Vector{Vector{ComplexF64}}[]
end
```

And pass `supp, coeffs` to the `System` constructor call.

- [ ] **Step 4: Update the Polyhedral `init` body**

Replace the body of `CommonSolve.init(F::System, alg::Polyhedral)`:

```julia
function CommonSolve.init(F::System, alg::Polyhedral)::PolyhedralSolveCache
    seed = alg.seed
    n = F.nvars

    # 1. Get support + target coefficients from the System
    support = deepcopy(F.support)
    target_coeffs = deepcopy(F.coefficients)

    # 2. Add zero column to support for polynomials without constant term
    for (i, A) in enumerate(support)
        if !has_zero_column(A)
            support[i] = hcat(A, zeros(Int32, size(A, 1)))
            push!(target_coeffs[i], zero(ComplexF64))
        end
    end

    # 3-7: rest of the function stays the same, except:
    # - Replace `system_eval(...)` with `System(...)`:
    param_polys, param_vars, coeff_params =
        _build_parametric_system(support, _make_variables(n))
    param_system = System(
        param_polys; variables = param_vars, parameters = coeff_params,
    )

    # Use param_system.evaluator where eval_param was used before
    toric_H = ToricHomotopy(param_system.evaluator, start_coeffs)
    # ...
    coeff_H = CoefficientHomotopy(param_system.evaluator, start_coeffs_flat, target_coeffs_flat)
    # ...

    return PolyhedralSolveCache(
        toric_tracker, coeff_tracker, toric_H,
        support, lifting, all_starts, seed,
        param_system,  # was param_info
    )
end
```

Note: `_make_variables(n)` is a helper to create throwaway DynamicPolynomials variables. Add it:

```julia
function _make_variables(n::Int)
    return [DynamicPolynomials.PolyVar{true}("_hc_x$i") for i in 1:n]
end
```

- [ ] **Step 5: Update `solve` for Polyhedral**

```julia
function solve(F::System, alg::Polyhedral)::Result
    return CommonSolve.solve!(CommonSolve.init(F, alg))
end
```

- [ ] **Step 6: Run `make format`**

Run: `make format`

---

### Task 4: Clean up removed code and helpers

**Files:**
- Modify: `src/solving/solve.jl` — remove `_empty_vars`, `_effective_variables` if no longer used
- Modify: `src/HomotopyContinuationNext.jl` — verify includes are correct

- [ ] **Step 1: Check if `_empty_vars` and `_effective_variables` are still used**

Run: `grep -rn "_empty_vars\|_effective_variables" src/`

If they're only used in the old `solve`/`init` signatures (which now take `System`), delete them. If `_effective_variables` is used in the `System` constructor's default kwarg, keep it.

- [ ] **Step 2: Verify the include order in `src/HomotopyContinuationNext.jl`**

`system_eval.jl` must be included after `system_evaluator.jl` (for `SystemEvaluator`), after `support.jl` (for `support_coefficients`), and after the interpreter pipeline. Check that the include order still works. Consider renaming `system_eval.jl` to `system.jl` since it now defines `System`.

- [ ] **Step 3: Run `make format`**

Run: `make format`

---

### Task 5: Update tests

**Files:**
- Modify: `test/core_test.jl` — replace `system_eval` calls with `System` constructor
- Modify: `test/tracking_test.jl` — replace `system_eval` calls with `System` constructor
- Modify: `test/solve_test.jl` — wrap all polynomial vectors in `System(...)`
- Modify: `test/concrete_structs_test.jl` — no changes needed (System is non-parametric, will be auto-checked)

- [ ] **Step 1: Update `test/core_test.jl`**

Replace all occurrences of:
- `system_eval(F)` -> `System(F)` — returns `System` directly
- `info, seval = system_eval(F)` -> `sys = System(F)` then use `sys.evaluator` where `seval` was used, and `sys.degrees` / `sys.nvars` etc. where `info.*` was used
- `_, seval = system_eval(F)` -> `sys = System(F)` then use `sys.evaluator`
- `info isa PolynomialSystemInfo` -> `sys isa System`
- Update imports: replace `system_eval, PolynomialSystemInfo` with `System`

- [ ] **Step 2: Update `test/tracking_test.jl`**

Replace all `_, eval_G = system_eval(G)` patterns with:
```julia
sys_G = System(G)
```
Then use `sys_G.evaluator` where `eval_G` was used (in `StraightLineHomotopy` constructor etc.).

Update imports: replace `system_eval` with `System`.

- [ ] **Step 3: Update `test/solve_test.jl`**

Wrap all polynomial vectors in `System(...)`:
- `solve([x^2 - 1, y - 2])` -> `solve(System([x^2 - 1, y - 2]))`
- `solve([x^2 + y - 1, x*y - 2], Polyhedral())` -> `solve(System([x^2 + y - 1, x*y - 2]), Polyhedral())`
- `CommonSolve.init([x^2 - 1, y - 2], TotalDegree())` -> `CommonSolve.init(System([x^2 - 1, y - 2]), TotalDegree())`

Update imports: add `System` if not already exported.

- [ ] **Step 4: Run all tests**

Run: `make test`
Expected: All tests pass (1304+).

---

### Task 6: Final verification

- [ ] **Step 1: Run `make format`**

Run: `make format`

- [ ] **Step 2: Run full test suite**

Run: `make test`
Expected: All tests pass. JET reports zero issues.

- [ ] **Step 3: Verify no remaining references to old types**

Run:
```bash
grep -rn "PolynomialSystemInfo\|system_eval" src/ test/
```

Expected: No matches (except possibly comments).

- [ ] **Step 4: Update README TODO**

Remove or strike through the "Add ConcreteStructs tests" item (already done) and any items that are now resolved. The `System` type addresses the "review type system and API" TODO.
