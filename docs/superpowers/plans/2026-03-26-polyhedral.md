# Polyhedral Homotopy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the full polyhedral homotopy for `solve(F, Polyhedral())`, giving BKK-optimal path counts via MixedSubdivisions.jl, a toric first phase, and a coefficient second phase.

**Architecture:** Two-phase tracking: (1) ToricHomotopy tracks from binomial start solutions to a generic system with matching support, (2) CoefficientHomotopy deforms coefficients from the generic system to the target. Support extraction from MP polynomials feeds MixedSubdivisions to compute mixed cells. A BinomialSystemSolver (HNF-based) produces start solutions from each cell. The `Polyhedral` algorithm struct integrates into the existing CommonSolve pattern.

**Tech Stack:** MixedSubdivisions.jl (new dep), existing Tracker/HomotopyEvaluator/SystemEvaluator infrastructure, `@enumx` for return codes.

---

## Scope

This plan covers:
- Support/coefficient extraction from MP polynomials
- `BinomialSystemSolver` (Hermite Normal Form based)
- `CoefficientHomotopy <: AbstractHomotopy` (linear coefficient interpolation)
- `ToricHomotopy <: AbstractHomotopy` (weighted toric degeneration)
- `Polyhedral` algorithm struct + `CommonSolve.init` / `solve!` methods
- Integration tests comparing solution counts with HC v2

**Deferred:** Overdetermined systems, homogeneous systems, `only_non_zero` option.

## File Structure

| File | Responsibility |
|------|---------------|
| `src/solving/support.jl` | Extract exponent matrices + coefficients from MP polynomials |
| `src/solving/binomial_system.jl` | `BinomialSystemSolver` — HNF, angular part, magnitude |
| `src/core/coefficient_homotopy.jl` | `CoefficientHomotopy <: AbstractHomotopy` |
| `src/core/toric_homotopy.jl` | `ToricHomotopy <: AbstractHomotopy` |
| `src/solving/polyhedral.jl` | `Polyhedral` algorithm, `PolyhedralTracker`, two-phase tracking |
| `test/polyhedral_test.jl` | Tests |

---

### Task 1: Add MixedSubdivisions.jl Dependency + Support Extraction

**Files:**
- Modify: `Project.toml`
- Create: `src/solving/support.jl`
- Modify: `src/HomotopyContinuationNext.jl`

- [ ] **Step 1: Add MixedSubdivisions**

```bash
julia --project -e 'using Pkg; Pkg.add("MixedSubdivisions")'
make update
```

- [ ] **Step 2: Create support.jl**

Extract exponent matrices and coefficient vectors from DynamicPolynomials input. This is the bridge between MP polynomials and MixedSubdivisions.

```julia
## support.jl — extract support (exponent matrices) and coefficients from MP polynomials.

using MixedSubdivisions: MixedSubdivisions

"""
    support_coefficients(polys, variables) → (supports, coefficients)

Extract the support (exponent matrix per polynomial) and coefficient vectors.
Returns `(Vector{Matrix{Int32}}, Vector{Vector{ComplexF64}})`.

Each `supports[i]` is an `n × mᵢ` matrix where column j is the exponent vector
of the j-th term of polynomial i. `coefficients[i]` is the corresponding coefficient vector.
"""
function support_coefficients(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
    )::Tuple{Vector{Matrix{Int32}}, Vector{Vector{ComplexF64}}}
    n = length(variables)
    var_to_idx = Dict{Symbol, Int}(Symbol(v) => i for (i, v) in enumerate(variables))

    supports = Vector{Matrix{Int32}}(undef, length(polys))
    coeffs = Vector{Vector{ComplexF64}}(undef, length(polys))

    for (k, p) in enumerate(polys)
        ts = MP.terms(p)
        m = length(ts)
        S = zeros(Int32, n, m)
        c = zeros(ComplexF64, m)
        for (j, t) in enumerate(ts)
            c[j] = ComplexF64(MP.coefficient(t))
            mono = MP.monomial(t)
            for (var, exp) in zip(MP.variables(mono), MP.exponents(mono))
                idx = get(var_to_idx, Symbol(var), nothing)
                if idx !== nothing
                    S[idx, j] = Int32(exp)
                end
            end
        end
        supports[k] = S
        coeffs[k] = c
    end

    return supports, coeffs
end
```

- [ ] **Step 3: Wire in and add import**

Add to main module after `using CommonSolve`:
```julia
using MixedSubdivisions: MixedSubdivisions
```

Add include after `solving/total_degree.jl`:
```julia
include("solving/support.jl")
```

- [ ] **Step 4: Verify**

```bash
make format && make test
```

---

### Task 2: BinomialSystemSolver

**Files:**
- Create: `src/solving/binomial_system.jl`
- Modify: `src/HomotopyContinuationNext.jl`

The binomial solver takes a mixed cell and produces start solutions by:
1. Extracting the binomial system `A·log(x) = log(b)` from cell indices
2. Computing the Hermite Normal Form of A
3. Finding the angular part (roots of unity) via triangular solve
4. Finding the magnitude part via `A^T \ log|b|`

- [ ] **Step 1: Implement BinomialSystemSolver**

Port the algorithm from v2's `binomial_system.jl`. Key struct:

```julia
struct BinomialSystemSolver
    A::Matrix{Int32}
    b::Vector{ComplexF64}
    X::Matrix{ComplexF64}  # solutions, n × d_hat (resized as needed)
    H::Matrix{Int64}       # Hermite normal form
    U::Matrix{Int64}       # transformation matrix
    γ::Vector{Float64}     # angles
    μ::Vector{Float64}     # log magnitudes
    Aᵀ::Matrix{Float64}   # transposed A for magnitude solve
end
```

Key functions:
- `BinomialSystemSolver(n)` — constructor
- `init!(BSS, support, coeffs, cell)` — extract binomial system from mixed cell
- `solve!(BSS)` — compute HNF, angular part, magnitude → fills `BSS.X`
- `hnf!(H, U, A)` — Hermite Normal Form (Kannan-Bachem algorithm)

This is the most algorithmically complex task. Port faithfully from v2 — the HNF algorithm, angular part computation (with DoubleF64 precision), and magnitude solve via LU.

- [ ] **Step 2: Wire in**

Include after `support.jl`:
```julia
include("solving/binomial_system.jl")
```

- [ ] **Step 3: Verify**

```bash
make format && make test
```

---

### Task 3: CoefficientHomotopy

**Files:**
- Create: `src/core/coefficient_homotopy.jl`
- Modify: `src/HomotopyContinuationNext.jl`

Linear interpolation of coefficients: `H(x,t) = F(x; p(t))` where `p(t) = t·p_start + (1-t)·p_target`.

The system `F` is a parametric system built from the support structure, with coefficients as parameters. `CoefficientHomotopy` interpolates the parameter values.

- [ ] **Step 1: Implement CoefficientHomotopy**

```julia
struct CoefficientHomotopy <: AbstractHomotopy
    system::SystemEvaluator
    start_coeffs::FSVec{ComplexF64}
    target_coeffs::FSVec{ComplexF64}
    coeffs::FSVec{ComplexF64}       # current interpolated coefficients (mutated)
    dt_coeffs::FSVec{ComplexF64}    # start - target (constant, precomputed)
    t_cache::Base.RefValue{ComplexF64}
end
```

Key insight: the `system` is a `SystemEvaluator` that was built from a parametric polynomial system (variables + parameters = coefficients). The `coeffs` field serves as the parameter vector that gets passed to `evaluate!(u, sys, x, coeffs)`.

Interface methods:
- `evaluate!(u, H, x, t)` — compute `coeffs!(H, t)` then `evaluate!(u, H.system, x, H.coeffs)`
- `evaluate_and_jacobian!(u, U, H, x, t)` — same with jacobian
- `taylor!(u, Val(1), H, x, t)` — uses `dt_coeffs` (constant derivative)
- `taylor!(u, Val(K), H, tx, t)` — delegates to system's taylor with interpolated coeffs

For the system's `taylor!` calls, we need the system to support parameters. The existing `SystemEvaluator` already has parameter support through `evaluate!(u, sys, x, p)` where `p` is the parameter vector.

The coefficient interpolation: `p(t) = (1-t)·target + t·start` (at t=1 we're at start, at t=0 we're at target — matches v2's convention where we track from 1 to 0).

Actually, checking v2: `coeffs[i] = (1-s)*target[i] + s*start[i]` where s = real(t) for real t. So at t=1: coeffs = start, at t=0: coeffs = target. And `dt_coeffs = start - target`.

- [ ] **Step 2: Wire in**

Include after `straight_line_homotopy.jl`:
```julia
include("core/coefficient_homotopy.jl")
```

- [ ] **Step 3: Verify**

```bash
make format && make test
```

---

### Task 4: ToricHomotopy

**Files:**
- Create: `src/core/toric_homotopy.jl`
- Modify: `src/HomotopyContinuationNext.jl`

Toric homotopy with weighted coefficients: `c_j(t) = c_j · t^{w_j}` where `w_j` are weights derived from the mixed cell.

- [ ] **Step 1: Implement ToricHomotopy**

```julia
struct ToricHomotopy <: AbstractHomotopy
    system::SystemEvaluator
    system_coeffs::FSVec{ComplexF64}     # base coefficients
    weights::FSVec{Float64}              # w_j per coefficient
    t_weights::FSVec{Float64}            # cached t^w for real t
    coeffs::FSVec{ComplexF64}            # c_j * t^{w_j} (mutated)
    dt_coeffs::FSVec{ComplexF64}         # d/dt of coeffs (mutated)
    x_cache::FSVec{ComplexF64}           # scratch
    t_cache::Base.RefValue{ComplexF64}
    nparams::Int                          # total number of parameters
end
```

Key functions:
- `evaluate_weights!(t_weights, weights, t)` — compute `t^{w_j}` for each j
- `coeffs!(H, t)` — compute `c_j · t^{w_j}` for each j
- `dt_coeffs!(H, t)` — compute `w_j · c_j · t^{w_j-1}` for each j
- `update_weights!(H, support, lifting, cell; min_weight, max_weight)` — compute weights from mixed cell

The weight formula for coefficient j of polynomial i:
```
w_ij = lifting[i][j] - β_i + Σ_k support[i][k,j] * cell.normal[k]
```
where β_i = cell.β[i] and cell.normal is the outer normal.

Weights for the two cell indices (aᵢ, bᵢ) are set to 0 (these define the binomial system).

The reparameterization strategy from v2:
- If max_weight < 10: track t: 0 → 1 directly
- If max_weight ≥ 10: two-stage reparameterization

- [ ] **Step 2: Wire in**

Include after `coefficient_homotopy.jl`:
```julia
include("core/toric_homotopy.jl")
```

- [ ] **Step 3: Verify**

```bash
make format && make test
```

---

### Task 5: Polyhedral Algorithm + Integration

**Files:**
- Create: `src/solving/polyhedral.jl`
- Modify: `src/solving/solve.jl`
- Modify: `src/HomotopyContinuationNext.jl`

- [ ] **Step 1: Create Polyhedral struct and init method**

```julia
@kwdef struct Polyhedral
    tracker_options::TrackerOptions = TrackerOptions()
    seed::Union{Nothing, UInt32} = nothing
    only_non_zero::Bool = false
end
```

The `CommonSolve.init` for `Polyhedral`:
1. Extract support + coefficients from polynomials
2. Optionally add zero to support (for `only_non_zero=false`)
3. Compute mixed cells via `MixedSubdivisions.fine_mixed_cells`
4. Generate random start coefficients
5. Build parametric system from support
6. Build `ToricHomotopy` and `CoefficientHomotopy`
7. Solve binomial systems for each cell → start solutions
8. Return a `PolyhedralSolveCache`

The `solve!`:
1. For each (cell, start_solution):
   a. Phase 1: Track toric homotopy from 0 to 1 (with reparameterization if needed)
   b. Phase 2: Track coefficient homotopy from 1 to 0
2. Collect PathResults

- [ ] **Step 2: Wire into solve.jl**

Add `CommonSolve.init` and `CommonSolve.solve!` methods for `Polyhedral`.

Update the convenience `solve()` to default to `Polyhedral()` instead of `TotalDegree()`.

- [ ] **Step 3: Export Polyhedral**

Add to exports in main module:
```julia
export TotalDegree, Polyhedral, Result, PathResult
```

- [ ] **Step 4: Verify**

```bash
make format && make test
```

---

### Task 6: Tests + Comparison

**Files:**
- Create: `test/polyhedral_test.jl`
- Modify: `benchmark/compare/tracking.jl`

- [ ] **Step 1: Write polyhedral tests**

Key tests:
- Support extraction correctness (compare with manual computation)
- Binomial solver correctness (verify solutions satisfy the binomial system)
- Mixed volume matches known values (cyclic-5 = 70, katsura-3 = 8)
- `solve(F, Polyhedral())` finds correct number of solutions
- Solution count matches HC v2 for: katsura-3, katsura-4, cyclic-5
- Residuals of found solutions are small

- [ ] **Step 2: Update comparison benchmarks**

Add polyhedral tracking to `benchmark/compare/tracking.jl` so we can compare per-path times fairly (same number of paths).

- [ ] **Step 3: Full verification**

```bash
make format && make test && make compare
```

---

## Notes

### Parametric System for Polyhedral

The polyhedral approach needs a parametric system where the coefficients are parameters. In v2, this is built via `polyhedral_system(support)` which creates a `System` with symbolic coefficient parameters.

In our architecture, we use `system_eval()` which already supports parameters. The approach:
1. Build MP polynomials from support with symbolic coefficient variables as parameters
2. Call `system_eval(polys; parameters=coeff_vars, variables=vars)` to get a `SystemEvaluator` that accepts a parameter vector
3. The `CoefficientHomotopy` and `ToricHomotopy` then pass interpolated coefficient vectors as the parameter argument

### Weight Computation

The weight `w_ij` for the j-th monomial of the i-th polynomial is:
```
w_ij = lifting[i][j] - β_i + ⟨support[i][:, j], cell.normal⟩
```

For the indices (aᵢ, bᵢ) that define the binomial face, `w = 0` (these monomials are the "binomial part" that's already accounted for by the start solution).

### Two-Phase Strategy

Phase 1 (Toric): `c_j(t) = c_j^start · t^{w_j}` tracks from t=0 (degenerate binomial) to t=1 (full start system).

Phase 2 (Coefficient): `p(t) = t·p_start + (1-t)·p_target` tracks from t=1 (start system) to t=0 (target system).

The start system's coefficients are chosen randomly near unit magnitude to ensure genericity.
