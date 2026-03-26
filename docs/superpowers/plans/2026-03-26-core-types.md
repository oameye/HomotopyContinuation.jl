# Phase 3: Core Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `src/core/` layer — abstract interfaces, FunctionWrapper-based evaluators, `system_eval()` constructor, and the `StraightLineHomotopy` concrete homotopy type.

**Architecture:** `AbstractSystem`/`AbstractHomotopy` define user-facing contracts with `AbstractVector`/`AbstractMatrix` args. `SystemEvaluator`/`HomotopyEvaluator` wrap any implementation into a single concrete type via `FunctionWrapper`, so the tracker is monomorphic. Polynomial input bypasses `AbstractSystem` — `system_eval()` builds interpreters and wraps them directly into `SystemEvaluator`. `StraightLineHomotopy` is the first concrete homotopy (needed for total degree solve).

**Tech Stack:** FunctionWrappers.jl (new dep), FixedSizeArrays.jl, MultivariatePolynomials.jl, DynamicPolynomials.jl, EnumX.jl

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/core/abstract_types.jl` | `AbstractSystem`, `AbstractHomotopy`, interface methods with defaults |
| `src/core/system_evaluator.jl` | FW type aliases, `SystemEvaluator` struct, dispatch methods, constructors from `AbstractSystem` |
| `src/core/homotopy_evaluator.jl` | FW type aliases, `HomotopyEvaluator` struct, dispatch methods, constructor from `AbstractHomotopy` |
| `src/core/system_eval.jl` | `PolynomialSystemInfo`, `system_eval()` — MP polynomials to `(info, SystemEvaluator)` |
| `src/core/straight_line_homotopy.jl` | `StraightLineHomotopy <: AbstractHomotopy` with all interface methods |
| `test/core_test.jl` | Tests for all of the above |

---

### Task 1: Add FunctionWrappers.jl Dependency

**Files:**
- Modify: `Project.toml`

- [ ] **Step 1: Add FunctionWrappers to Project.toml**

```bash
cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia -e 'using Pkg; Pkg.add("FunctionWrappers")'
```

Verify it appears in `[deps]` and `[compat]` sections. The compat should be `"1"`.

- [ ] **Step 2: Add the import to the main module**

In `src/HomotopyContinuationNext.jl`, add after the `using FixedSizeArrays` line:

```julia
using FunctionWrappers: FunctionWrapper
```

- [ ] **Step 3: Verify it loads**

```bash
julia --project -e 'using HomotopyContinuationNext; println("OK")'
```

Expected: `OK`

---

### Task 2: Abstract Types and Interface Contracts

**Files:**
- Create: `src/core/abstract_types.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)

- [ ] **Step 1: Write the test for abstract interface defaults**

Add to `test/core_test.jl`:

```julia
using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: AbstractSystem, AbstractHomotopy,
    evaluate!, evaluate_and_jacobian!, taylor!,
    nparameters, set_solution!, get_solution!,
    start_parameters!, target_parameters!
using FixedSizeArrays: FixedSizeArray
const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}

# Minimal test system: F(x) = [x[1]^2 + x[2] - 1]
struct TestSystem <: AbstractSystem
    nequations::Int
    nvariables::Int
end

Base.size(F::TestSystem) = (F.nequations, F.nvariables)

function HC.evaluate!(u::AbstractVector, ::TestSystem, x::AbstractVector, p::AbstractVector)
    u[1] = x[1]^2 + x[2] - 1
    return nothing
end

function HC.evaluate_and_jacobian!(
        u::AbstractVector, U::AbstractMatrix,
        ::TestSystem, x::AbstractVector, p::AbstractVector,
    )
    u[1] = x[1]^2 + x[2] - 1
    U[1, 1] = 2x[1]
    U[1, 2] = one(eltype(x))
    return nothing
end

function HC.taylor!(
        u::AbstractVector, ::Val{1}, ::TestSystem,
        tx::HC.TaylorVector, p::AbstractVector,
    )
    # Order-1 coefficient of x[1]^2 + x[2] - 1
    # d/dt(x1^2) = 2*x1[0]*x1[1], d/dt(x2) = x2[1]
    x1 = tx[1]  # TTS{2,ComplexF64}
    x2 = tx[2]
    u[1] = 2 * x1[0] * x1[1] + x2[1]
    return nothing
end

function HC.taylor!(
        u::AbstractVector, ::Val{K}, ::TestSystem,
        tx::HC.TaylorVector, p::AbstractVector,
    ) where {K}
    u[1] = zero(eltype(u))
    return nothing
end

@testset "Core Types" begin
    @testset "AbstractSystem interface" begin
        F = TestSystem(1, 2)
        @test size(F) == (1, 2)
        @test nparameters(F) == 0

        u = zeros(ComplexF64, 1)
        x = ComplexF64[2.0, 3.0]
        p = ComplexF64[]
        evaluate!(u, F, x, p)
        @test u[1] ≈ 6.0 + 0im  # 4 + 3 - 1

        U = zeros(ComplexF64, 1, 2)
        evaluate_and_jacobian!(u, U, F, x, p)
        @test u[1] ≈ 6.0 + 0im
        @test U[1, 1] ≈ 4.0 + 0im
        @test U[1, 2] ≈ 1.0 + 0im
    end

    @testset "AbstractHomotopy defaults" begin
        x = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        y = FSVec{ComplexF64}(ComplexF64[1.0, 2.0])
        out = FSVec{ComplexF64}(zeros(ComplexF64, 2))

        # Default set_solution! just copies
        set_solution!(x, y, complex(0.5))
        @test x ≈ y

        # Default get_solution! just copies
        get_solution!(out, x, complex(0.5))
        @test out ≈ x
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

```bash
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/core_test.jl")'
```

Expected: fails because `AbstractSystem`, `evaluate!`, etc. are not defined.

- [ ] **Step 3: Implement abstract_types.jl**

Create `src/core/abstract_types.jl`:

```julia
## Abstract types and interface contracts for systems and homotopies.

abstract type AbstractSystem end
abstract type AbstractHomotopy end

# ── AbstractSystem interface ──

"""
    nparameters(F::AbstractSystem) -> Int

Number of parameters. Default: 0.
"""
nparameters(::AbstractSystem)::Int = 0

"""
    evaluate!(u, F::AbstractSystem, x, p) -> nothing

Evaluate F at (x, p), writing result into u.
"""
function evaluate! end

"""
    evaluate_and_jacobian!(u, U, F::AbstractSystem, x, p) -> nothing

Evaluate F and its Jacobian at (x, p), writing into u and U.
"""
function evaluate_and_jacobian! end

"""
    taylor!(u, ::Val{K}, F::AbstractSystem, tx, p) -> nothing

Compute order-K Taylor coefficient of F.
"""
function taylor! end

# ── AbstractHomotopy interface ──

# Required: evaluate!, evaluate_and_jacobian!, taylor! (defined above)
# Required: Base.size(H) -> (nequations, nvariables)

# Optional defaults:
"""
    set_solution!(x, y, t) -> nothing

Map y to the internal representation x at time t. Default: copy.
"""
set_solution!(x::AbstractVector, y::AbstractVector, ::ComplexF64)::Nothing =
    (copyto!(x, y); nothing)

"""
    get_solution!(out, x, t) -> nothing

Extract solution from internal representation. Default: copy.
"""
get_solution!(out::AbstractVector, x::AbstractVector, ::ComplexF64)::Nothing =
    (copyto!(out, x); nothing)

"""
    start_parameters!(H::AbstractHomotopy, p) -> H
"""
start_parameters!(H::AbstractHomotopy, ::AbstractVector) = H

"""
    target_parameters!(H::AbstractHomotopy, ::AbstractVector) -> H
"""
target_parameters!(H::AbstractHomotopy, ::AbstractVector) = H
```

- [ ] **Step 4: Wire it into the main module**

In `src/HomotopyContinuationNext.jl`, add after the model_kit includes:

```julia
include("core/abstract_types.jl")
```

Also add to the exports (or just ensure they are accessible via qualified access — ExplicitImports requires no bare `using`).

- [ ] **Step 5: Run test to verify it passes**

```bash
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/core_test.jl")'
```

Expected: PASS

- [ ] **Step 6: Run quality gates**

```bash
make test
```

Expected: all existing tests still pass.

---

### Task 3: SystemEvaluator

**Files:**
- Create: `src/core/system_evaluator.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)
- Modify: `test/core_test.jl` (add tests)

- [ ] **Step 1: Write the test for SystemEvaluator from AbstractSystem**

Append to `test/core_test.jl`:

```julia
using HomotopyContinuationNext: SystemEvaluator, TaylorVector, TruncatedTaylorSeries

@testset "SystemEvaluator" begin
    @testset "from AbstractSystem" begin
        F = TestSystem(1, 2)
        eval = SystemEvaluator(F)

        @test size(eval) == (1, 2)
        @test nparameters(eval) == 0

        u = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        x = FSVec{ComplexF64}(ComplexF64[2.0, 3.0])
        p = FSVec{ComplexF64}(ComplexF64[])

        # evaluate!
        evaluate!(u, eval, x, p)
        @test u[1] ≈ 6.0 + 0im

        # evaluate_and_jacobian!
        U = HC.FSMat{ComplexF64}(zeros(ComplexF64, 1, 2))
        evaluate_and_jacobian!(u, U, eval, x, p)
        @test u[1] ≈ 6.0 + 0im
        @test U[1, 1] ≈ 4.0 + 0im
        @test U[1, 2] ≈ 1.0 + 0im

        # taylor! order 1
        tv = TaylorVector{2, ComplexF64}(2)
        # Set x1 = 2 + 1*t, x2 = 3 + 0*t
        tv[1] = TruncatedTaylorSeries((ComplexF64(2.0), ComplexF64(1.0)))
        tv[2] = TruncatedTaylorSeries((ComplexF64(3.0), ComplexF64(0.0)))
        taylor!(u, Val(1), eval, tv, p)
        @test u[1] ≈ 4.0 + 0im  # 2*2*1 + 0
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

```bash
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/core_test.jl")'
```

Expected: fails — `SystemEvaluator` not defined.

- [ ] **Step 3: Implement system_evaluator.jl**

Create `src/core/system_evaluator.jl`:

```julia
## SystemEvaluator — FunctionWrapper-based type firewall for AbstractSystem.
#
# All dispatch through FunctionWrapper closures. The tracker sees only
# SystemEvaluator (concrete, no type parameters) regardless of the underlying system.

# ── FunctionWrapper type aliases ──

const SysEvalFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, FSVec{ComplexF64}}}
const SysEvalDF64FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexDF64}, FSVec{ComplexF64}}}
const SysEvalJacFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSMat{ComplexF64},
    FSVec{ComplexF64}, FSVec{ComplexF64}}}
const SysTaylor1FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, TaylorVector{2, ComplexF64}, FSVec{ComplexF64}}}
const SysTaylor2FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, TaylorVector{3, ComplexF64}, FSVec{ComplexF64}}}
const SysTaylor3FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, TaylorVector{4, ComplexF64}, FSVec{ComplexF64}}}

struct SystemEvaluator
    _evaluate!::SysEvalFW
    _evaluate_df64!::SysEvalDF64FW
    _evaluate_and_jacobian!::SysEvalJacFW
    _taylor_1!::SysTaylor1FW
    _taylor_2!::SysTaylor2FW
    _taylor_3!::SysTaylor3FW
    _size::Tuple{Int, Int}
    _nparameters::Int
end

# ── Dispatch methods ──

Base.size(S::SystemEvaluator) = S._size
nparameters(S::SystemEvaluator)::Int = S._nparameters

function evaluate!(
        u::FSVec{ComplexF64}, S::SystemEvaluator,
        x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._evaluate!(u, x, p)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, S::SystemEvaluator,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._evaluate_df64!(u, x, p)
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64},
        S::SystemEvaluator, x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._evaluate_and_jacobian!(u, U, x, p)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, S::SystemEvaluator,
        tx::TaylorVector{2, ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._taylor_1!(u, tx, p)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, S::SystemEvaluator,
        tx::TaylorVector{3, ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._taylor_2!(u, tx, p)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, S::SystemEvaluator,
        tx::TaylorVector{4, ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._taylor_3!(u, tx, p)
    return nothing
end

# ── Constructor from AbstractSystem ──

function SystemEvaluator(F::AbstractSystem)
    return SystemEvaluator(
        SysEvalFW((u, x, p) -> (evaluate!(u, F, x, p); nothing)),
        SysEvalDF64FW((u, x, p) -> (evaluate!(u, F, x, p); nothing)),
        SysEvalJacFW((u, U, x, p) -> (evaluate_and_jacobian!(u, U, F, x, p); nothing)),
        SysTaylor1FW((u, tx, p) -> (taylor!(u, Val(1), F, tx, p); nothing)),
        SysTaylor2FW((u, tx, p) -> (taylor!(u, Val(2), F, tx, p); nothing)),
        SysTaylor3FW((u, tx, p) -> (taylor!(u, Val(3), F, tx, p); nothing)),
        size(F),
        nparameters(F),
    )
end
```

- [ ] **Step 4: Wire it in**

In `src/HomotopyContinuationNext.jl`, add after `include("core/abstract_types.jl")`:

```julia
include("core/system_evaluator.jl")
```

- [ ] **Step 5: Run test to verify it passes**

```bash
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/core_test.jl")'
```

Expected: PASS

- [ ] **Step 6: Run quality gates**

```bash
make test
```

---

### Task 4: system_eval() — Polynomial Input to SystemEvaluator

**Files:**
- Create: `src/core/system_eval.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)
- Modify: `test/core_test.jl` (add tests)

- [ ] **Step 1: Write the test for system_eval**

Append to `test/core_test.jl`:

```julia
using HomotopyContinuationNext: system_eval, PolynomialSystemInfo
using DynamicPolynomials: @polyvar

@testset "system_eval" begin
    @testset "basic polynomial system" begin
        @polyvar x y
        polys = [x^2 + y - 1, x * y - 2]

        info, eval = system_eval(polys)

        @test info isa PolynomialSystemInfo
        @test eval isa SystemEvaluator
        @test size(eval) == (2, 2)
        @test nparameters(eval) == 0
        @test info.degrees == [2, 2]
        @test info.nvars == 2
        @test info.nparams == 0
        @test info.is_homogeneous == false

        # Evaluate at x=2, y=3: [4+3-1, 6-2] = [6, 4]
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        xv = FSVec{ComplexF64}(ComplexF64[2.0, 3.0])
        p = FSVec{ComplexF64}(ComplexF64[])
        evaluate!(u, eval, xv, p)
        @test u[1] ≈ 6.0 + 0im
        @test u[2] ≈ 4.0 + 0im

        # Jacobian: [2x 1; y x] at (2,3) = [4 1; 3 2]
        U = HC.FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        evaluate_and_jacobian!(u, U, eval, xv, p)
        @test U[1, 1] ≈ 4.0 + 0im
        @test U[1, 2] ≈ 1.0 + 0im
        @test U[2, 1] ≈ 3.0 + 0im
        @test U[2, 2] ≈ 2.0 + 0im
    end

    @testset "with parameters" begin
        @polyvar x y a b
        polys = [x^2 + a * y, x * y - b]

        info, eval = system_eval(polys; parameters = [a, b])

        @test size(eval) == (2, 2)
        @test nparameters(eval) == 2
        @test info.nparams == 2

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        xv = FSVec{ComplexF64}(ComplexF64[2.0, 3.0])
        p = FSVec{ComplexF64}(ComplexF64[1.0, 2.0])
        evaluate!(u, eval, xv, p)
        @test u[1] ≈ 7.0 + 0im  # 4 + 1*3
        @test u[2] ≈ 4.0 + 0im  # 6 - 2
    end

    @testset "DF64 evaluation" begin
        @polyvar x y
        polys = [x^2 + y - 1]

        _, eval = system_eval(polys)

        u = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        xv = FSVec{HC.ComplexDF64}(HC.ComplexDF64[HC.ComplexDF64(2.0), HC.ComplexDF64(3.0)])
        p = FSVec{ComplexF64}(ComplexF64[])
        evaluate!(u, eval, xv, p)
        @test u[1] ≈ 6.0 + 0im
    end

    @testset "Taylor evaluation" begin
        @polyvar x y
        polys = [x^2 + y - 1, x * y - 2]

        _, eval = system_eval(polys)

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        p = FSVec{ComplexF64}(ComplexF64[])

        # Taylor order 1: x(t) = [2+t, 3+0t]
        tv = TaylorVector{2, ComplexF64}(2)
        tv[1] = TruncatedTaylorSeries((ComplexF64(2.0), ComplexF64(1.0)))
        tv[2] = TruncatedTaylorSeries((ComplexF64(3.0), ComplexF64(0.0)))
        taylor!(u, Val(1), eval, tv, p)
        # d/dt(x^2+y-1) at t=0: 2*2*1 + 0 = 4
        # d/dt(x*y-2) at t=0: 1*3 + 2*0 = 3
        @test u[1] ≈ 4.0 + 0im
        @test u[2] ≈ 3.0 + 0im
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

```bash
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/core_test.jl")'
```

Expected: fails — `system_eval`, `PolynomialSystemInfo` not defined.

- [ ] **Step 3: Implement system_eval.jl**

Create `src/core/system_eval.jl`:

```julia
## system_eval — build SystemEvaluator directly from MP polynomials.
#
# No intermediate AbstractSystem subtype. Interpreters are built for each
# element type and wrapped in FunctionWrappers. PolynomialSystemInfo holds
# metadata + keeps interpreter references alive (GC roots for FW closures).

struct PolynomialSystemInfo
    degrees::Vector{Int}
    nvars::Int
    nparams::Int
    variable_groups::Union{Nothing, Vector{Vector{Int}}}
    is_homogeneous::Bool
    _seq_eval::InstructionSequence
    _seq_jac::InstructionSequence
    _interp_f64::Interpreter{Vector{ComplexF64}}
    _interp_df64::Interpreter{Vector{ComplexDF64}}
    _interp_jac::Interpreter{Vector{ComplexF64}}
    _interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}}
    _interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}}
    _interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}}
end

"""
    system_eval(polys; parameters=[], variables=...) -> (PolynomialSystemInfo, SystemEvaluator)

Build a `SystemEvaluator` directly from DynamicPolynomials input.
"""
function system_eval(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    )::Tuple{PolynomialSystemInfo, SystemEvaluator}
    nvars = length(variables)
    nparams = length(parameters)
    neqs = length(polys)

    # Build eval-only interpreter
    interp_f64 = _build_interpreter(
        Vector{ComplexF64}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )
    interp_df64 = _build_interpreter(
        Vector{ComplexDF64}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )

    # Build eval+jacobian interpreter
    interp_jac = _build_interpreter(
        Vector{ComplexF64}, polys;
        parameters = parameters, variables = variables, include_jacobian = true,
    )

    # Build Taylor interpreters (eval-only sequences, Taylor handles derivatives)
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

    # Metadata
    degrees = Int[MP.maxdegree(p) for p in polys]
    is_homogeneous = all(MP.ishomogeneous, polys)

    info = PolynomialSystemInfo(
        degrees, nvars, nparams, nothing, is_homogeneous,
        interp_f64.sequence, interp_jac.sequence,
        interp_f64, interp_df64, interp_jac,
        interp_t1, interp_t2, interp_t3,
    )

    # Wrap in FunctionWrappers
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

    return (info, evaluator)
end

# ── FW-compatible wrappers ──
# These call the interpreter's execute! but accept FSVec/FSMat (AbstractVector subtypes).

function _execute_eval_fw!(
        u::AbstractVector, interp::Interpreter, x::AbstractVector, p::AbstractVector,
    )::Nothing
    if isempty(interp.sequence.parameters_range)
        _load_inputs!(interp, x)
    else
        _load_inputs!(interp, x, p)
    end
    _execute_eval!(u, interp)
    return nothing
end

function _execute_jac_fw!(
        u::AbstractVector, U::AbstractMatrix, interp::Interpreter,
        x::AbstractVector, p::AbstractVector,
    )::Nothing
    if isempty(interp.sequence.parameters_range)
        _load_inputs!(interp, x)
    else
        _load_inputs!(interp, x, p)
    end
    _execute_jac!(u, U, interp)
    return nothing
end
```

- [ ] **Step 4: Wire it in**

In `src/HomotopyContinuationNext.jl`, add after `include("core/system_evaluator.jl")`:

```julia
include("core/system_eval.jl")
```

- [ ] **Step 5: Run test to verify it passes**

```bash
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/core_test.jl")'
```

Expected: PASS

- [ ] **Step 6: Run quality gates**

```bash
make test
```

---

### Task 5: HomotopyEvaluator

**Files:**
- Create: `src/core/homotopy_evaluator.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)
- Modify: `test/core_test.jl` (add tests)

- [ ] **Step 1: Write the test for HomotopyEvaluator**

Append to `test/core_test.jl`:

```julia
using HomotopyContinuationNext: HomotopyEvaluator

# Minimal test homotopy: H(x,t) = t*G(x) + (1-t)*F(x)
# where G = [x1-1, x2-1], F = [x1^2-1, x2^2-1]
struct TestHomotopy <: AbstractHomotopy end

Base.size(::TestHomotopy) = (2, 2)

function HC.evaluate!(u::AbstractVector, ::TestHomotopy, x::AbstractVector, t::ComplexF64)
    t1 = one(ComplexF64) - t
    u[1] = t * (x[1] - 1) + t1 * (x[1]^2 - 1)
    u[2] = t * (x[2] - 1) + t1 * (x[2]^2 - 1)
    return nothing
end

function HC.evaluate_and_jacobian!(
        u::AbstractVector, U::AbstractMatrix,
        H::TestHomotopy, x::AbstractVector, t::ComplexF64,
    )
    evaluate!(u, H, x, t)
    t1 = one(ComplexF64) - t
    U[1, 1] = t + t1 * 2x[1]
    U[1, 2] = zero(ComplexF64)
    U[2, 1] = zero(ComplexF64)
    U[2, 2] = t + t1 * 2x[2]
    return nothing
end

function HC.taylor!(
        u::AbstractVector, ::Val{1}, ::TestHomotopy,
        x::AbstractVector, t::ComplexF64,
    )
    # ∂H/∂t = G(x) - F(x) = (x[i]-1) - (x[i]^2-1) = x[i] - x[i]^2
    u[1] = x[1] - x[1]^2
    u[2] = x[2] - x[2]^2
    return nothing
end

function HC.taylor!(
        u::AbstractVector, ::Val{K}, ::TestHomotopy,
        tx::HC.TaylorVector, t::ComplexF64, incremental::Bool = false,
    ) where {K}
    fill!(u, zero(eltype(u)))
    return nothing
end

@testset "HomotopyEvaluator" begin
    @testset "from AbstractHomotopy" begin
        H = TestHomotopy()
        heval = HomotopyEvaluator(H)

        @test size(heval) == (2, 2)

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        x = FSVec{ComplexF64}(ComplexF64[2.0, 3.0])
        t = ComplexF64(0.5)

        # H(x, 0.5) = 0.5*(x-1) + 0.5*(x^2-1) = 0.5*(x-1+x^2-1) = 0.5*(x^2+x-2)
        evaluate!(u, heval, x, t)
        @test u[1] ≈ 0.5 * (4 + 2 - 2) + 0im  # 2.0
        @test u[2] ≈ 0.5 * (9 + 3 - 2) + 0im  # 5.0

        U = HC.FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        evaluate_and_jacobian!(u, U, heval, x, t)
        @test U[1, 1] ≈ 0.5 + 0.5 * 4 + 0im  # 2.5
        @test U[2, 2] ≈ 0.5 + 0.5 * 6 + 0im  # 3.5
        @test U[1, 2] ≈ 0im
        @test U[2, 1] ≈ 0im
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

```bash
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/core_test.jl")'
```

Expected: fails — `HomotopyEvaluator` not defined.

- [ ] **Step 3: Implement homotopy_evaluator.jl**

Create `src/core/homotopy_evaluator.jl`:

```julia
## HomotopyEvaluator — FunctionWrapper-based type firewall for AbstractHomotopy.

# ── FunctionWrapper type aliases ──

const HomEvalFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64}}
const HomEvalDF64FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexDF64}, ComplexF64}}
const HomEvalJacFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSMat{ComplexF64},
    FSVec{ComplexF64}, ComplexF64}}
const HomTaylor1FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64}}
const HomTaylor2FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, TaylorVector{3, ComplexF64}, ComplexF64, Bool}}
const HomTaylor3FW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, TaylorVector{4, ComplexF64}, ComplexF64, Bool}}
const HomSetSolFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64}}
const HomGetSolFW = FunctionWrapper{Nothing, Tuple{
    FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64}}
const HomParamsFW = FunctionWrapper{Nothing, Tuple{FSVec{ComplexF64}}}

struct HomotopyEvaluator
    _evaluate!::HomEvalFW
    _evaluate_df64!::HomEvalDF64FW
    _evaluate_and_jacobian!::HomEvalJacFW
    _taylor_1!::HomTaylor1FW
    _taylor_2!::HomTaylor2FW
    _taylor_3!::HomTaylor3FW
    _set_solution!::HomSetSolFW
    _get_solution!::HomGetSolFW
    _start_parameters!::HomParamsFW
    _target_parameters!::HomParamsFW
    _size::Tuple{Int, Int}
end

# ── Dispatch methods ──

Base.size(H::HomotopyEvaluator) = H._size

function evaluate!(
        u::FSVec{ComplexF64}, H::HomotopyEvaluator,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    H._evaluate!(u, x, t)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, H::HomotopyEvaluator,
        x::FSVec{ComplexDF64}, t::ComplexF64,
    )::Nothing
    H._evaluate_df64!(u, x, t)
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64},
        H::HomotopyEvaluator, x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    H._evaluate_and_jacobian!(u, U, x, t)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, H::HomotopyEvaluator,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    H._taylor_1!(u, x, t)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, H::HomotopyEvaluator,
        tx::TaylorVector{3, ComplexF64}, t::ComplexF64; incremental::Bool = false,
    )::Nothing
    H._taylor_2!(u, tx, t, incremental)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, H::HomotopyEvaluator,
        tx::TaylorVector{4, ComplexF64}, t::ComplexF64; incremental::Bool = false,
    )::Nothing
    H._taylor_3!(u, tx, t, incremental)
    return nothing
end

function set_solution!(
        x::FSVec{ComplexF64}, H::HomotopyEvaluator,
        y::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    H._set_solution!(x, y, t)
    return nothing
end

function get_solution!(
        out::FSVec{ComplexF64}, H::HomotopyEvaluator,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    H._get_solution!(out, x, t)
    return nothing
end

function start_parameters!(H::HomotopyEvaluator, p::FSVec{ComplexF64})::Nothing
    H._start_parameters!(p)
    return nothing
end

function target_parameters!(H::HomotopyEvaluator, p::FSVec{ComplexF64})::Nothing
    H._target_parameters!(p)
    return nothing
end

# ── Constructor from AbstractHomotopy ──

function HomotopyEvaluator(H::AbstractHomotopy)
    return HomotopyEvaluator(
        HomEvalFW((u, x, t) -> (evaluate!(u, H, x, t); nothing)),
        HomEvalDF64FW((u, x, t) -> (evaluate!(u, H, x, t); nothing)),
        HomEvalJacFW((u, U, x, t) -> (evaluate_and_jacobian!(u, U, H, x, t); nothing)),
        HomTaylor1FW((u, x, t) -> (taylor!(u, Val(1), H, x, t); nothing)),
        HomTaylor2FW((u, tx, t, inc) -> (taylor!(u, Val(2), H, tx, t, inc); nothing)),
        HomTaylor3FW((u, tx, t, inc) -> (taylor!(u, Val(3), H, tx, t, inc); nothing)),
        HomSetSolFW((x, y, t) -> (set_solution!(x, H, y, t); nothing)),
        HomGetSolFW((out, x, t) -> (get_solution!(out, H, x, t); nothing)),
        HomParamsFW((p) -> (start_parameters!(H, p); nothing)),
        HomParamsFW((p) -> (target_parameters!(H, p); nothing)),
        size(H),
    )
end
```

- [ ] **Step 4: Wire it in**

In `src/HomotopyContinuationNext.jl`, add after `include("core/system_evaluator.jl")`:

```julia
include("core/homotopy_evaluator.jl")
```

- [ ] **Step 5: Run test to verify it passes**

```bash
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/core_test.jl")'
```

Expected: PASS

- [ ] **Step 6: Run quality gates**

```bash
make test
```

---

### Task 6: StraightLineHomotopy

**Files:**
- Create: `src/core/straight_line_homotopy.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)
- Modify: `test/core_test.jl` (add tests)

- [ ] **Step 1: Write the test for StraightLineHomotopy**

Append to `test/core_test.jl`:

```julia
using HomotopyContinuationNext: StraightLineHomotopy

@testset "StraightLineHomotopy" begin
    @testset "construction and evaluation" begin
        @polyvar x y

        # G (start) and F (target)
        G = [x - 1, y - 1]
        F = [x^2 - 1, y^2 - 1]

        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)

        H = StraightLineHomotopy(eval_G, eval_F)
        @test size(H) == (2, 2)

        heval = HomotopyEvaluator(H)
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        xv = FSVec{ComplexF64}(ComplexF64[2.0, 3.0])

        # At t=1: H = γ*G = γ*(x-1, y-1) = γ*(1, 2)
        evaluate!(u, heval, xv, ComplexF64(1.0))
        γ = H.γ
        @test u[1] ≈ γ * 1.0
        @test u[2] ≈ γ * 2.0

        # At t=0: H = F = (x^2-1, y^2-1) = (3, 8)
        evaluate!(u, heval, xv, ComplexF64(0.0))
        @test u[1] ≈ 3.0 + 0im
        @test u[2] ≈ 8.0 + 0im
    end

    @testset "jacobian" begin
        @polyvar x y
        G = [x - 1, y - 1]
        F = [x^2 - 1, y^2 - 1]

        _, eval_G = system_eval(G)
        _, eval_F = system_eval(F)

        H = StraightLineHomotopy(eval_G, eval_F)
        heval = HomotopyEvaluator(H)

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        U = HC.FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        xv = FSVec{ComplexF64}(ComplexF64[2.0, 3.0])
        t = ComplexF64(0.5)
        γ = H.γ

        evaluate_and_jacobian!(u, U, heval, xv, t)

        # J = γ*t*J_G + (1-t)*J_F
        # J_G = [1 0; 0 1], J_F = [2x 0; 0 2y] = [4 0; 0 6]
        @test U[1, 1] ≈ γ * 0.5 * 1.0 + 0.5 * 4.0
        @test U[2, 2] ≈ γ * 0.5 * 1.0 + 0.5 * 6.0
        @test U[1, 2] ≈ 0im
        @test U[2, 1] ≈ 0im
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

```bash
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/core_test.jl")'
```

Expected: fails — `StraightLineHomotopy` not defined.

- [ ] **Step 3: Implement straight_line_homotopy.jl**

Create `src/core/straight_line_homotopy.jl`:

```julia
## StraightLineHomotopy: H(x,t) = γ·t·G(x) + (1-t)·F(x)
#
# G = start system, F = target system.
# At t=1: H = γ·G (start), at t=0: H = F (target).
# γ is a random complex number for genericity.

struct StraightLineHomotopy <: AbstractHomotopy
    start::SystemEvaluator
    target::SystemEvaluator
    γ::ComplexF64
    # Scratch buffers (all pre-allocated, contents mutated)
    u_start::FSVec{ComplexF64}
    u_target::FSVec{ComplexF64}
    ū_start::FSVec{ComplexDF64}
    ū_target::FSVec{ComplexDF64}
    U_start::FSMat{ComplexF64}
    U_target::FSMat{ComplexF64}
    dv_start::TaylorVector{4, ComplexF64}
    dv_target::TaylorVector{4, ComplexF64}
end

function StraightLineHomotopy(
        start::SystemEvaluator, target::SystemEvaluator;
        γ::ComplexF64 = cis(2π * rand()),
    )
    m, n = size(target)
    @assert size(start) == (m, n) "Start and target systems must have the same size"
    return StraightLineHomotopy(
        start, target, γ,
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, m)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, m)),
        FSMat{ComplexF64}(zeros(ComplexF64, m, n)),
        FSMat{ComplexF64}(zeros(ComplexF64, m, n)),
        TaylorVector{4, ComplexF64}(m),
        TaylorVector{4, ComplexF64}(m),
    )
end

Base.size(H::StraightLineHomotopy) = size(H.target)

# ── evaluate! ──

function evaluate!(
        u::AbstractVector, H::StraightLineHomotopy,
        x::AbstractVector{ComplexF64}, t::ComplexF64,
    )::Nothing
    p_start = FSVec{ComplexF64}(ComplexF64[])
    p_target = FSVec{ComplexF64}(ComplexF64[])
    evaluate!(H.u_start, H.start, x, p_start)
    evaluate!(H.u_target, H.target, x, p_target)
    γt = H.γ * t
    t1 = one(ComplexF64) - t
    @inbounds for i in eachindex(u)
        u[i] = γt * H.u_start[i] + t1 * H.u_target[i]
    end
    return nothing
end

# ── evaluate_and_jacobian! ──

function evaluate_and_jacobian!(
        u::AbstractVector, U::AbstractMatrix,
        H::StraightLineHomotopy, x::AbstractVector{ComplexF64}, t::ComplexF64,
    )::Nothing
    p_start = FSVec{ComplexF64}(ComplexF64[])
    p_target = FSVec{ComplexF64}(ComplexF64[])
    evaluate_and_jacobian!(H.u_start, H.U_start, H.start, x, p_start)
    evaluate_and_jacobian!(H.u_target, H.U_target, H.target, x, p_target)
    γt = H.γ * t
    t1 = one(ComplexF64) - t
    @inbounds for i in eachindex(u)
        u[i] = γt * H.u_start[i] + t1 * H.u_target[i]
    end
    @inbounds for j in axes(U, 2), i in axes(U, 1)
        U[i, j] = γt * H.U_start[i, j] + t1 * H.U_target[i, j]
    end
    return nothing
end

# ── taylor! order 1 ──
# ∂H/∂t = γ·G(x) - F(x) (for the implicit ODE)

function taylor!(
        u::AbstractVector, ::Val{1}, H::StraightLineHomotopy,
        x::AbstractVector{ComplexF64}, t::ComplexF64,
    )::Nothing
    p_start = FSVec{ComplexF64}(ComplexF64[])
    p_target = FSVec{ComplexF64}(ComplexF64[])
    evaluate!(H.u_start, H.start, x, p_start)
    evaluate!(H.u_target, H.target, x, p_target)
    @inbounds for i in eachindex(u)
        u[i] = H.γ * H.u_start[i] - H.u_target[i]
    end
    return nothing
end

# ── taylor! order K >= 2 ──
# Higher-order: H is linear in t, so ∂^K H / ∂t^K = 0 for K >= 2.
# The Taylor coefficients come from the chain rule on x(t).

function taylor!(
        u::AbstractVector, ::Val{K}, H::StraightLineHomotopy,
        tx::TaylorVector, t::ComplexF64, incremental::Bool = false,
    ) where {K}
    p_start = FSVec{ComplexF64}(ComplexF64[])
    p_target = FSVec{ComplexF64}(ComplexF64[])
    taylor!(H.u_start, Val(K), H.start, tx, p_start)
    taylor!(H.u_target, Val(K), H.target, tx, p_target)
    γt = H.γ * t
    t1 = one(ComplexF64) - t
    @inbounds for i in eachindex(u)
        u[i] = γt * H.u_start[i] + t1 * H.u_target[i]
    end
    return nothing
end
```

- [ ] **Step 4: Wire it in**

In `src/HomotopyContinuationNext.jl`, add after `include("core/homotopy_evaluator.jl")`:

```julia
include("core/straight_line_homotopy.jl")
```

- [ ] **Step 5: Run test to verify it passes**

```bash
julia --project -e 'using TestEnv; TestEnv.activate(); include("test/core_test.jl")'
```

Expected: PASS

- [ ] **Step 6: Run quality gates**

```bash
make test
```

---

### Task 7: Format and Final Verification

**Files:**
- All files created/modified above

- [ ] **Step 1: Format all code**

```bash
make format
```

- [ ] **Step 2: Run full test suite**

```bash
make test
```

Expected: all tests pass, including Aqua, JET, and ExplicitImports.

- [ ] **Step 3: Fix any JET or ExplicitImports issues**

If JET reports issues (e.g., type instabilities in FW closures) or ExplicitImports flags missing explicit imports, fix them. Common issues:
- Missing `using FunctionWrappers: FunctionWrapper` — already added in Task 1
- Need to add new symbols to explicit imports in test files
- JET may flag the FW closures — these are expected to be opaque to JET since they go through `ccall`

---

## Notes

**What's deferred to subsequent tasks:**
- `CoefficientHomotopy`, `ToricHomotopy`, `AffineChartHomotopy` — needed for polyhedral solve, not for total degree
- `evaluate!` with DF64 inputs for `StraightLineHomotopy` — add when the tracker requests extended precision
- Empty parameter vectors: the current implementation allocates `FSVec{ComplexF64}(ComplexF64[])` in hot paths of `StraightLineHomotopy`. This should be hoisted to a `const` field or module-level constant before Phase 4 (tracking), but is correct for now.

**Key design decision:** `PolynomialSystemInfo` stores separate `_seq_eval` and `_seq_jac` instruction sequences (and separate interpreters for eval vs. jac) because the existing `_build_interpreter` with `include_jacobian=true` builds a combined tape. The eval-only tape is faster for pure evaluation; the jac tape computes both. This matches how the FW closures route: `_evaluate!` uses the eval interpreter, `_evaluate_and_jacobian!` uses the jac interpreter.
