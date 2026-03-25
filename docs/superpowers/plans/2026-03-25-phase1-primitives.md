# Phase 1: Primitives — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement all numeric primitives needed by the path tracker: DoubleF64 extended precision, weighted norms, linear algebra workspace with custom LU/QR, and utility functions.

**Architecture:** Port and adapt primitives from `HomotopyContinuation/src/` (v2) into `src/primitives/` for HomotopyContinuationNext. Key changes: replace `Vector`/`Matrix` with `FSVec`/`FSMat` from FixedSizeArrays.jl, use `mutable struct` with `const` fields for workspaces, remove `AbstractNorm` hierarchy (only infinity norm used), use explicit imports everywhere.

**Tech Stack:** Julia 1.11+, FixedSizeArrays.jl, LinearAlgebra (stdlib)

**Reference code:** All v2 implementations live in `HomotopyContinuation/src/` — `DoubleDouble.jl` (1061 lines), `norm.jl` (261 lines), `linear_algebra.jl` (922 lines), `utils.jl` (601 lines). Port logic, adapt types.

---

## File Structure

```
src/
├── HomotopyContinuationNext.jl          # Modify: add includes and exports
├── utils.jl                             # Create: SegmentStepper, fast_abs, nanmin/max, nthroot
└── primitives/
    ├── double_f64.jl                    # Create: DoubleF64, ComplexDF64, arithmetic
    ├── norms.jl                         # Create: InfNorm, WeightedNorm (infinity norm only)
    └── linear_algebra.jl               # Create: MatrixWorkspace, custom LU/QR, condition est.

test/
├── utils_test.jl                        # Create: SegmentStepper tests
├── double_f64_test.jl                   # Create: DoubleF64 arithmetic tests
├── norms_test.jl                        # Create: norm/distance tests
├── linear_algebra_test.jl               # Create: LU, ldiv, condition, refinement tests
└── fixedsizearrays_compat_test.jl       # Create: verify FSMat with LAPACK ops
```

---

### Task 0: Verify FixedSizeArrays LAPACK Compatibility

**Files:**
- Create: `test/fixedsizearrays_compat_test.jl`

This is a prerequisite gate. If `lu!` or `ldiv!` fail on `FSMat`, we must adapt `MatrixWorkspace` to use `Matrix{ComplexF64}` for the factorization matrix while still using `FSVec` for scratch vectors.

- [ ] **Step 1: Write the compatibility test**

```julia
using Test
using LinearAlgebra: LinearAlgebra
using FixedSizeArrays: FixedSizeVector, FixedSizeMatrix

const LA = LinearAlgebra
const FSVec{T} = FixedSizeVector{T}
const FSMat{T} = FixedSizeMatrix{T}

@testset "FixedSizeArrays LAPACK compatibility" begin
    n = 4

    @testset "strides" begin
        A = FSMat{ComplexF64}(rand(ComplexF64, n, n))
        @test strides(A) == (1, n)
        @test stride(A, 1) == 1
        @test stride(A, 2) == n
    end

    @testset "lu! on FSMat" begin
        A = FSMat{ComplexF64}(rand(ComplexF64, n, n))
        A_copy = copy(A)
        F = LA.lu!(A_copy)
        @test F isa LA.LU
        b = FSVec{ComplexF64}(rand(ComplexF64, n))
        x = F \ Vector(b)
        @test Vector(A) * x ≈ Vector(b)
    end

    @testset "ldiv! with FSVec" begin
        A = FSMat{ComplexF64}(rand(ComplexF64, n, n))
        F = LA.lu!(copy(A))
        b = FSVec{ComplexF64}(rand(ComplexF64, n))
        x = FSVec{ComplexF64}(zeros(ComplexF64, n))
        LA.ldiv!(x, F, b)
        @test Vector(A) * Vector(x) ≈ Vector(b) atol = 1e-12
    end

    @testset "mul! with FSMat and FSVec" begin
        A = FSMat{ComplexF64}(rand(ComplexF64, n, n))
        x = FSVec{ComplexF64}(rand(ComplexF64, n))
        y = FSVec{ComplexF64}(zeros(ComplexF64, n))
        LA.mul!(y, A, x)
        @test Vector(y) ≈ Vector(A) * Vector(x)
    end

    @testset "qr! on FSMat" begin
        m, k = 6, 4
        A = FSMat{ComplexF64}(rand(ComplexF64, m, k))
        A_copy = copy(A)
        # qr! may not work on FSMat — test and document
        try
            F = LA.qr!(A_copy)
            @test F isa LA.QR
        catch e
            @info "qr! does not work on FSMat, will use Matrix for QR" exception = e
            A_mat = Matrix{ComplexF64}(A)
            F = LA.qr!(A_mat)
            @test F isa LA.QR
        end
    end

    @testset "copyto! between FSMat and Matrix" begin
        A = FSMat{ComplexF64}(rand(ComplexF64, n, n))
        B = Matrix{ComplexF64}(undef, n, n)
        copyto!(B, A)
        @test B == Matrix(A)
        C = FSMat{ComplexF64}(zeros(ComplexF64, n, n))
        copyto!(C, B)
        @test Matrix(C) == B
    end
end
```

- [ ] **Step 2: Run test to verify compatibility**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/fixedsizearrays_compat_test.jl")'`

Document results. If `lu!` on FSMat fails, the plan for MatrixWorkspace (Task 5) must use `Matrix{ComplexF64}` for `A` and `lu.factors`, while `FSVec` is used for scratch vectors only.

- [ ] **Step 3: Commit**

```bash
git add test/fixedsizearrays_compat_test.jl
git commit -m "test: verify FixedSizeArrays LAPACK compatibility"
```

---

### Task 1: Utils — fast_abs, nanmin/max, nthroot

**Files:**
- Create: `src/utils.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add `include`)
- Create: `test/utils_test.jl`

Small, self-contained utility functions used throughout the package. No dependencies beyond Base.

- [ ] **Step 1: Write failing tests for utility functions**

```julia
using Test
using HomotopyContinuationNext: fast_abs, nanmin, nanmax, nthroot

@testset "Utility functions" begin
    @testset "fast_abs" begin
        @test fast_abs(3.0 + 4.0im) ≈ 5.0
        @test fast_abs(-3.0) == 3.0
        @test fast_abs(0.0 + 0.0im) == 0.0
    end

    @testset "nanmin / nanmax" begin
        @test nanmin(1.0, 2.0) == 1.0
        @test nanmin(NaN, 2.0) == 2.0
        @test nanmin(1.0, NaN) == 1.0
        @test isnan(nanmin(NaN, NaN))
        @test nanmax(1.0, 2.0) == 2.0
        @test nanmax(NaN, 2.0) == 2.0
        @test nanmax(1.0, NaN) == 1.0
    end

    @testset "nthroot" begin
        @test nthroot(8.0, 3) ≈ 2.0
        @test nthroot(16.0, 4) ≈ 2.0
        @test nthroot(9.0, 2) ≈ 3.0
        @test nthroot(5.0, 1) == 5.0
        @test nthroot(7.0, 0) == 1.0
        @test nthroot(32.0, 5) ≈ 2.0
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/utils_test.jl")'`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement utility functions**

Create `src/utils.jl`:

```julia
"""
    fast_abs(z::Complex) -> Float64
    fast_abs(x::Real) -> Float64

Fast absolute value using `abs2` to avoid intermediate allocation for complex numbers.
"""
fast_abs(z::Complex) = sqrt(abs2(z))
fast_abs(x::Real) = abs(x)

"""
    nanmin(a, b)

Like `min(a, b)` but ignoring `NaN` values.
"""
nanmin(a, b) = isnan(a) ? b : (isnan(b) ? a : min(a, b))

"""
    nanmax(a, b)

Like `max(a, b)` but ignoring `NaN` values.
"""
nanmax(a, b) = isnan(a) ? b : (isnan(b) ? a : max(a, b))

"""
    nthroot(x::Real, n::Integer)

Compute the `n`-th root of `x`. Specializes for n = 0, 1, 2, 3, 4.
"""
function nthroot(x::Real, N::Integer)
    if N == 4
        sqrt(sqrt(x))
    elseif N == 2
        sqrt(x)
    elseif N == 3
        cbrt(x)
    elseif N == 1
        x
    elseif N == 0
        one(x)
    else
        x^(1 / N)
    end
end
```

Add to `src/HomotopyContinuationNext.jl` before `end`:

```julia
include("utils.jl")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/utils_test.jl")'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils.jl src/HomotopyContinuationNext.jl test/utils_test.jl
git commit -m "feat: add utility functions (fast_abs, nanmin/max, nthroot)"
```

---

### Task 2: Utils — SegmentStepper

**Files:**
- Modify: `src/utils.jl` (append SegmentStepper)
- Modify: `test/utils_test.jl` (append SegmentStepper tests)

SegmentStepper manages arc-length parametrization for path tracking. Mutable struct with `const` on fixed fields per design doc.

- [ ] **Step 1: Write failing tests for SegmentStepper**

Append to `test/utils_test.jl`:

```julia
using HomotopyContinuationNext: SegmentStepper, init!, propose_step!, step_success!, is_done, dist_to_target

@testset "SegmentStepper" begin
    @testset "forward stepping" begin
        S = SegmentStepper(0.0 + 0.0im, 1.0 + 0.0im)
        @test !is_done(S)
        @test S.t ≈ 0.0 + 0.0im
        @test dist_to_target(S) ≈ 1.0

        propose_step!(S, 0.3)
        @test S.s′ ≈ 0.3
        step_success!(S)
        @test S.s ≈ 0.3

        propose_step!(S, 10.0)  # clamps to target
        step_success!(S)
        @test is_done(S)
        @test S.t ≈ 1.0 + 0.0im
    end

    @testset "backward stepping" begin
        # |start| > |target| → backward
        S = SegmentStepper(1.0 + 0.0im, 0.0 + 0.0im)
        @test S.forward == false
        @test !is_done(S)

        propose_step!(S, 0.5)
        step_success!(S)
        @test !is_done(S)

        propose_step!(S, 10.0)
        step_success!(S)
        @test is_done(S)
    end

    @testset "reinit (returns new stepper)" begin
        S = SegmentStepper(0.0 + 0.0im, 1.0 + 0.0im)
        propose_step!(S, 0.5)
        step_success!(S)

        # init! returns a NEW SegmentStepper because start/target/abs_Δ/forward are const.
        # The caller (TrackerState) replaces its `segment` field with the returned value.
        S = init!(S, 0.0 + 0.0im, 2.0 + 0.0im)
        @test S.abs_Δ ≈ 2.0
        @test S.s ≈ 0.0
        @test !is_done(S)
    end

    @testset "Δs and Δt" begin
        S = SegmentStepper(0.0 + 0.0im, 1.0 + 0.0im)
        propose_step!(S, 0.25)
        @test S.Δs ≈ 0.25
        @test S.Δt ≈ 0.25 + 0.0im
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/utils_test.jl")'`
Expected: FAIL — SegmentStepper not defined.

- [ ] **Step 3: Implement SegmentStepper**

Append to `src/utils.jl`:

```julia
"""
    SegmentStepper

Manages arc-length parametrization for homotopy path tracking along a segment
in the complex plane from `start` to `target`.

## Mutable justification
Fields `s` and `s′` are advanced every tracker step. `start`, `target`, `abs_Δ`, `forward`
are fixed per segment but reset via `init!`.
"""
mutable struct SegmentStepper
    const start::ComplexF64
    const target::ComplexF64
    const abs_Δ::Float64
    const forward::Bool
    s::Float64   # current arc-length parameter
    s′::Float64  # proposed arc-length parameter
end

function SegmentStepper(start::Number, target::Number)
    s = ComplexF64(start)
    t = ComplexF64(target)
    abs_Δ = abs(t - s)
    forward = abs(s) < abs(t)
    s0 = forward ? 0.0 : abs_Δ
    SegmentStepper(s, t, abs_Δ, forward, s0, s0)
end

function init!(S::SegmentStepper, start::Number, target::Number)
    s = ComplexF64(start)
    t = ComplexF64(target)
    abs_Δ = abs(t - s)
    forward = abs(s) < abs(t)
    s0 = forward ? 0.0 : abs_Δ
    # SegmentStepper has const fields for start/target/abs_Δ/forward,
    # so we must create a new one. Return it for the caller to replace.
    SegmentStepper(s, t, abs_Δ, forward, s0, s0)
end

is_done(S::SegmentStepper) = S.forward ? S.s == S.abs_Δ : S.s == 0.0

function step_success!(S::SegmentStepper)
    S.s = S.s′
    S
end

function propose_step!(S::SegmentStepper, Δs::Real)
    if S.forward
        S.s′ = min(S.s + Δs, S.abs_Δ)
    else
        S.s′ = max(S.s - Δs, 0.0)
    end
    S
end

dist_to_target(S::SegmentStepper) = S.forward ? S.abs_Δ - S.s : S.s

function _t_helper(start::ComplexF64, target::ComplexF64, s::Float64, Δ::Float64, forward::Bool)
    if forward
        if s == 0.0
            return start
        elseif s == Δ
            return target
        else
            return start + (s / Δ) * (target - start)
        end
    else
        if s == Δ
            return start
        elseif s == 0.0
            return target
        else
            return target + (s / Δ) * (start - target)
        end
    end
end

function Base.getproperty(S::SegmentStepper, sym::Symbol)
    if sym === :Δs
        s = getfield(S, :s)
        s′ = getfield(S, :s′)
        return getfield(S, :forward) ? s′ - s : s - s′
    elseif sym === :t
        return _t_helper(
            getfield(S, :start), getfield(S, :target),
            getfield(S, :s), getfield(S, :abs_Δ), getfield(S, :forward),
        )
    elseif sym === :t′
        return _t_helper(
            getfield(S, :start), getfield(S, :target),
            getfield(S, :s′), getfield(S, :abs_Δ), getfield(S, :forward),
        )
    elseif sym === :Δt
        s = getfield(S, :s)
        s′ = getfield(S, :s′)
        start = getfield(S, :start)
        target = getfield(S, :target)
        abs_Δ = getfield(S, :abs_Δ)
        if getfield(S, :forward)
            return ((s′ - s) / abs_Δ) * (target - start)
        else
            return ((s - s′) / abs_Δ) * (target - start)
        end
    else
        return getfield(S, sym)
    end
end
```

Note: `init!` returns a new `SegmentStepper` because `start`/`target`/`abs_Δ`/`forward` are `const`. The caller (TrackerState) replaces its `segment` field. Update the test to use the return value:

```julia
S = init!(S, 0.0 + 0.0im, 2.0 + 0.0im)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/utils_test.jl")'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils.jl test/utils_test.jl
git commit -m "feat: add SegmentStepper for arc-length path parametrization"
```

---

### Task 3: DoubleF64 — Core Arithmetic

**Files:**
- Create: `src/primitives/double_f64.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)
- Create: `test/double_f64_test.jl`

Port from `HomotopyContinuation/src/DoubleDouble.jl` (1061 lines). This is a near-direct port — the arithmetic is pure scalar math with no array types to change. Key adaptations:
- Use explicit `import` instead of `import Base: +, -, *, /`
- Remove transcendental functions (sin, cos, exp, log, tan, asin, acos, atan) — only needed for non-polynomial systems which are deferred
- Keep: arithmetic (+, -, *, /, ^), sqrt, abs, comparison, conversion, promotion, constants
- Remove the `isbits` overload (causes invalidation cascades — noted in v2 code comments)

- [ ] **Step 1: Write failing tests**

```julia
using Test
using HomotopyContinuationNext: DoubleF64, ComplexDF64

@testset "DoubleF64" begin
    @testset "construction and conversion" begin
        x = DoubleF64(1.0)
        @test x.hi == 1.0
        @test x.lo == 0.0

        y = DoubleF64(big"3.141592653589793238462643383279502884197")
        @test abs(BigFloat(y) - big(π)) < 1e-30

        @test DoubleF64(1) == DoubleF64(1.0)
        @test Float64(DoubleF64(3.14)) == 3.14
    end

    @testset "arithmetic vs BigFloat" begin
        a = DoubleF64(big"1.23456789012345678901234567890")
        b = DoubleF64(big"9.87654321098765432109876543210")

        # Addition
        @test abs(BigFloat(a + b) - (big(a) + big(b))) < 1e-29

        # Subtraction
        @test abs(BigFloat(a - b) - (big(a) - big(b))) < 1e-29

        # Multiplication
        @test abs(BigFloat(a * b) - (big(a) * big(b))) < 1e-28

        # Division
        @test abs(BigFloat(a / b) - (big(a) / big(b))) < 1e-28
    end

    @testset "integer power" begin
        x = DoubleF64(2.0)
        @test abs(BigFloat(x^10) - big(2.0)^10) < 1e-25
        @test DoubleF64(3.0)^0 == DoubleF64(1.0)
        @test DoubleF64(5.0)^1 == DoubleF64(5.0)
    end

    @testset "sqrt" begin
        x = DoubleF64(2.0)
        @test abs(BigFloat(sqrt(x)) - sqrt(big(2.0))) < 1e-30
    end

    @testset "comparison" begin
        @test DoubleF64(1.0) < DoubleF64(2.0)
        @test DoubleF64(2.0) == DoubleF64(2.0)
        @test DoubleF64(3.0) <= DoubleF64(3.0)
    end

    @testset "special values" begin
        @test iszero(zero(DoubleF64))
        @test isone(one(DoubleF64))
        @test isnan(DoubleF64(NaN, NaN))
        @test isinf(DoubleF64(Inf))
        @test isfinite(DoubleF64(1.0))
    end

    @testset "isbits" begin
        @test isbits(DoubleF64(1.0))
        @test isbits(ComplexDF64(DoubleF64(1.0), DoubleF64(2.0)))
    end

    @testset "ComplexDF64" begin
        z = Complex(DoubleF64(1.0), DoubleF64(2.0))
        @test z isa ComplexDF64
        w = Complex(DoubleF64(3.0), DoubleF64(4.0))
        # complex arithmetic delegates to real DoubleF64 ops
        @test abs(BigFloat(real(z + w)) - 4.0) < 1e-30
        @test abs(BigFloat(imag(z + w)) - 6.0) < 1e-30
    end

    @testset "promotion" begin
        @test DoubleF64(1.0) + 2.0 isa DoubleF64
        @test 3 + DoubleF64(1.0) isa DoubleF64
        @test promote_type(DoubleF64, Float64) == DoubleF64
        @test promote_type(DoubleF64, Int) == DoubleF64
    end

    @testset "abs and abs2" begin
        z = Complex(DoubleF64(3.0), DoubleF64(4.0))
        @test abs(BigFloat(abs2(z)) - 25.0) < 1e-28
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/double_f64_test.jl")'`
Expected: FAIL — DoubleF64 not defined.

- [ ] **Step 3: Implement DoubleF64**

Create `src/primitives/double_f64.jl`. Port from `HomotopyContinuation/src/DoubleDouble.jl` with these changes:
- **Keep:** lines 1–530 approximately (error-free arithmetic, struct, constructors, conversions, promotion, +, -, *, /, ^, sqrt, comparison, abs, zero/one, iszero/isone/isnan/isinf/isfinite, constants, show, ldexp, nextfloat/prevfloat, eps, floatmin/floatmax, decompose, round/floor/ceil/trunc, rem/divrem/mod, isinteger)
- **Remove:** lines ~531–1061 (exp, log, log2, log10, sin, cos, sincos, tan, asin, acos, atan — transcendentals only needed for non-polynomial systems, which are deferred)
- **Change:** Replace `import Base: +, -, *, /, ^, <, ==, <=` with explicit qualified overloads `Base.:+`, `Base.:-`, etc.
- **Change:** Remove the `module DoubleDouble` wrapper — define directly in the package module.
- **Change:** Remove `export` statements — use explicit exports in main module.

The full file should be ~530 lines ported directly. The core building blocks are:

```julia
# Error-free transformations (keep exactly as-is from v2)
@inline function quick_two_sum(a::Float64, b::Float64) ... end
@inline function two_sum(a::Float64, b::Float64) ... end
@inline function two_diff(a::Float64, b::Float64) ... end
@inline function two_prod(a::Float64, b::Float64) ... end
@inline function two_square(a::Float64) ... end

struct DoubleF64 <: AbstractFloat
    hi::Float64
    lo::Float64
end
const ComplexDF64 = Complex{DoubleF64}

# Constructors, conversions, promotions — port directly from v2 lines 77-150
# Arithmetic (+, -, *, /) — port directly from v2 lines ~152-350
# wide_* variants — port directly
# square, ^, sqrt — port directly from v2 lines ~350-450
# Comparison (<, <=, ==) — port directly from v2 lines ~450-470
# Special values (iszero, isone, isnan, isinf, isfinite) — port directly
# Rounding (round, floor, ceil, trunc) — port directly
# Show, decompose, nextfloat/prevfloat, eps — port directly
```

Update `src/HomotopyContinuationNext.jl`:

```julia
include("primitives/double_f64.jl")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/double_f64_test.jl")'`
Expected: PASS

- [ ] **Step 5: Run quality gates**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && make test`
Expected: All existing tests (aqua, jet, explicit_imports) still pass alongside the new test.

- [ ] **Step 6: Commit**

```bash
git add src/primitives/double_f64.jl src/HomotopyContinuationNext.jl test/double_f64_test.jl
git commit -m "feat: add DoubleF64 extended precision arithmetic"
```

---

### Task 4: Norms — InfNorm and WeightedNorm

**Files:**
- Create: `src/primitives/norms.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)
- Create: `test/norms_test.jl`

Key design change from v2: Remove the `AbstractNorm` hierarchy and `WeightedNorm{N<:AbstractNorm}` type parameter. The design doc specifies only infinity norm is used in practice, so `WeightedNorm` is hardcoded to infinity norm. Weights use `FSVec{Float64}`.

- [ ] **Step 1: Write failing tests**

```julia
using Test
using LinearAlgebra: norm as la_norm
using FixedSizeArrays: FixedSizeVector
using HomotopyContinuationNext:
    InfNorm, WeightedNorm, WeightedNormOptions,
    inf_norm, weighted_norm, inf_distance, weighted_distance,
    init!, update!, fast_abs

const FSVec{T} = FixedSizeVector{T}

@testset "Norms" begin
    @testset "InfNorm" begin
        x = FSVec{ComplexF64}([1.0 + 2.0im, 3.0 + 4.0im, 0.5 + 0.0im])
        @test inf_norm(x) ≈ abs(3.0 + 4.0im)

        y = FSVec{ComplexF64}([2.0 + 2.0im, 1.0 + 4.0im, 0.5 + 1.0im])
        d = inf_distance(x, y)
        # max_i |x[i] - y[i]|
        expected = maximum(abs.(Vector(x) .- Vector(y)))
        @test d ≈ expected
    end

    @testset "WeightedNorm construction" begin
        n = 4
        w = WeightedNorm(n)
        @test length(w.weights) == n
        @test all(w.weights .== 1.0)
    end

    @testset "weighted_norm and weighted_distance" begin
        weights = FSVec{Float64}([2.0, 0.5, 1.0])
        w = WeightedNorm(weights)
        x = FSVec{ComplexF64}([2.0 + 0.0im, 4.0 + 0.0im, 3.0 + 0.0im])
        # ||D⁻¹x||_∞ = max(|2/2|, |4/0.5|, |3/1|) = 8.0
        @test weighted_norm(x, w) ≈ 8.0

        y = FSVec{ComplexF64}([4.0 + 0.0im, 4.0 + 0.0im, 3.0 + 0.0im])
        # ||D⁻¹(x-y)||_∞ = max(|2/2|, |0/0.5|, |0/1|) = 1.0
        @test weighted_distance(x, y, w) ≈ 1.0
    end

    @testset "init! and update!" begin
        w = WeightedNorm(3)
        x = FSVec{ComplexF64}([100.0 + 0.0im, 1e-10 + 0.0im, 50.0 + 0.0im])
        init!(w, x)
        # After init, weights should be approximately |x[i]|, clamped
        @test w.weights[1] > 0.0
        @test w.weights[2] > 0.0  # clamped to scale_min * norm
        @test w.weights[3] > 0.0

        y = FSVec{ComplexF64}([100.0 + 0.0im, 1e-10 + 0.0im, 50.0 + 0.0im])
        update!(w, y)
        # Weights should be interpolated
        @test all(w.weights .> 0.0)
    end

    @testset "weighted_norm with complex values" begin
        weights = FSVec{Float64}([1.0, 1.0])
        w = WeightedNorm(weights)
        x = FSVec{ComplexF64}([3.0 + 4.0im, 1.0 + 0.0im])
        # ||D⁻¹x||_∞ = max(|3+4im|/1, |1|/1) = max(5, 1) = 5
        @test weighted_norm(x, w) ≈ 5.0
    end

    @testset "fast_abs on ComplexDF64" begin
        using HomotopyContinuationNext: DoubleF64, ComplexDF64
        z = ComplexDF64(DoubleF64(3.0), DoubleF64(4.0))
        @test fast_abs(z) ≈ 5.0 atol = 1e-28
    end

    @testset "overflow handling" begin
        # InfNorm should handle overflow gracefully
        x = FSVec{ComplexF64}([1e300 + 1e300im, 0.0 + 0.0im])
        n = inf_norm(x)
        @test isfinite(n) || n == Inf  # should not error
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/norms_test.jl")'`
Expected: FAIL — types not defined.

- [ ] **Step 3: Implement norms**

Create `src/primitives/norms.jl` (no standalone `using` — inherits `LinearAlgebra` from parent module):

```julia
"""
    InfNorm

Singleton type representing the infinity (maximum) norm.
"""
struct InfNorm end

"""
    WeightedNormOptions

Configuration for adaptive weight scaling in [`WeightedNorm`](@ref).
"""
Base.@kwdef struct WeightedNormOptions
    scale_min::Float64 = 1e-4
    scale_abs_min::Float64 = 1e-6
    scale_max::Float64 = exp2(511)
end

"""
    WeightedNorm

Weighted infinity norm: ``||D⁻¹x||_∞`` where `D = diag(weights)`.
Weights are adaptively updated via [`init!`](@ref) and [`update!`](@ref)
to keep the norm ≈ 1.0.

Always uses infinity norm — no type parameter needed since only one norm
variant is used in practice.
"""
struct WeightedNorm
    weights::FSVec{Float64}
    options::WeightedNormOptions
end

function WeightedNorm(weights::FSVec{Float64})
    WeightedNorm(weights, WeightedNormOptions())
end

function WeightedNorm(n::Integer)
    WeightedNorm(FSVec{Float64}(ones(n)), WeightedNormOptions())
end

Base.length(w::WeightedNorm) = length(w.weights)

"""
    inf_norm(x::AbstractVector) -> Float64

Compute ``||x||_∞ = \\max_i |x_i|``. Uses `abs2` for speed with overflow fallback.
"""
function inf_norm(x::AbstractVector)
    n = length(x)
    # FSVec is always 1-based (DenseArray with Memory backing)
    @inbounds dmax = abs2(x[1])
    for i in 2:n
        @inbounds dᵢ = abs2(x[i])
        dmax = @fastmath max(dmax, dᵢ)
    end
    d = sqrt(dmax)
    if isinf(d)
        # Overflow fallback: use abs directly
        @inbounds dmax = abs(x[1])
        for i in 2:n
            @inbounds dᵢ = abs(x[i])
            dmax = max(dmax, dᵢ)
        end
        return dmax
    else
        return d
    end
end

"""
    inf_distance(x::AbstractVector, y::AbstractVector) -> Float64

Compute ``||x - y||_∞``.
"""
function inf_distance(x::AbstractVector, y::AbstractVector)
    n = length(x)
    @inbounds dmax = abs2(x[1] - y[1])
    for i in 2:n
        @inbounds dᵢ = abs2(x[i] - y[i])
        dmax = @fastmath max(dmax, dᵢ)
    end
    d = sqrt(dmax)
    if isinf(d)
        @inbounds dmax = abs(x[1] - y[1])
        for i in 2:n
            @inbounds dᵢ = abs(x[i] - y[i])
            dmax = max(dmax, dᵢ)
        end
        return dmax
    else
        return d
    end
end

"""
    weighted_norm(x::AbstractVector, w::WeightedNorm) -> Float64

Compute ``||D⁻¹x||_∞`` where `D = diag(w.weights)`.
"""
function weighted_norm(x::AbstractVector, w::WeightedNorm)
    n = length(x)
    weights = w.weights
    @inbounds dmax = abs2(x[1] / weights[1])
    for i in 2:n
        @inbounds dᵢ = abs2(x[i] / weights[i])
        dmax = @fastmath max(dmax, dᵢ)
    end
    d = sqrt(dmax)
    if isinf(d)
        @inbounds dmax = abs(x[1] / weights[1])
        for i in 2:n
            @inbounds dᵢ = abs(x[i] / weights[i])
            dmax = max(dmax, dᵢ)
        end
        return dmax
    else
        return d
    end
end

"""
    weighted_distance(x::AbstractVector, y::AbstractVector, w::WeightedNorm) -> Float64

Compute ``||D⁻¹(x - y)||_∞``.
"""
function weighted_distance(x::AbstractVector, y::AbstractVector, w::WeightedNorm)
    n = length(x)
    weights = w.weights
    @inbounds dmax = abs2((x[1] - y[1]) / weights[1])
    for i in 2:n
        @inbounds dᵢ = abs2((x[i] - y[i]) / weights[i])
        dmax = @fastmath max(dmax, dᵢ)
    end
    d = sqrt(dmax)
    if isinf(d)
        @inbounds dmax = abs((x[1] - y[1]) / weights[1])
        for i in 2:n
            @inbounds dᵢ = abs((x[i] - y[i]) / weights[i])
            dmax = max(dmax, dᵢ)
        end
        return dmax
    else
        return d
    end
end

"""
    init!(w::WeightedNorm, x::AbstractVector)

Initialize weights from `x` so that ``||D⁻¹x||_∞ ≈ 1``.
"""
function init!(w::WeightedNorm, x::AbstractVector)
    point_norm = inf_norm(x)
    opts = w.options
    for i in eachindex(w.weights)
        wᵢ = fast_abs(x[i])
        if wᵢ < opts.scale_min * point_norm
            wᵢ = opts.scale_min * point_norm
        elseif wᵢ > opts.scale_max * point_norm
            wᵢ = opts.scale_max * point_norm
        end
        w.weights[i] = max(wᵢ, opts.scale_abs_min)
    end
    w
end

"""
    update!(w::WeightedNorm, x::AbstractVector)

Update weights by interpolating between current weights and `|x[i]|`.
"""
function update!(w::WeightedNorm, x::AbstractVector)
    norm_x = weighted_norm(x, w)
    opts = w.options
    for i in eachindex(w.weights)
        wᵢ = (fast_abs(x[i]) + w.weights[i]) / 2
        if wᵢ < opts.scale_min * norm_x
            wᵢ = opts.scale_min * norm_x
        elseif wᵢ > opts.scale_max * norm_x
            wᵢ = opts.scale_max * norm_x
        end
        if isfinite(wᵢ)
            w.weights[i] = max(wᵢ, opts.scale_abs_min)
        end
    end
    w
end
```

Update `src/HomotopyContinuationNext.jl`:

```julia
include("primitives/norms.jl")
```

(Must come after `include("utils.jl")` since norms use `fast_abs`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/norms_test.jl")'`
Expected: PASS

- [ ] **Step 5: Run quality gates**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && make test`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/primitives/norms.jl src/HomotopyContinuationNext.jl test/norms_test.jl
git commit -m "feat: add InfNorm and WeightedNorm with FSVec weights"
```

---

### Task 5: Linear Algebra — MatrixWorkspace Core (LU, ldiv!)

**Files:**
- Create: `src/primitives/linear_algebra.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)
- Create: `test/linear_algebra_test.jl`

This is the largest and most critical primitive. Port from `HomotopyContinuation/src/linear_algebra.jl` (922 lines). Key changes:
- Use `mutable struct` with `const` fields per design doc
- Use `FSMat{ComplexF64}` for matrix `A` (if LAPACK compat test passes), `FSVec` for scratch vectors
- `factorized`/`scaled` are non-const `Bool` fields (not `RefValue`) since the struct is mutable
- Keep custom LU with `abs2` pivoting (avoids sqrt overhead)
- Keep custom QR with `@fastmath` division

Split into two sub-tasks: 5a (core LU + ldiv!) and 5b (scaling, refinement, condition).

- [ ] **Step 1: Write failing tests for LU + ldiv!**

```julia
using Test
using LinearAlgebra: LinearAlgebra
using FixedSizeArrays: FixedSizeVector, FixedSizeMatrix
using HomotopyContinuationNext: MatrixWorkspace, updated!, factorize!

const LA = LinearAlgebra
const FSVec{T} = FixedSizeVector{T}
const FSMat{T} = FixedSizeMatrix{T}

@testset "MatrixWorkspace" begin
    @testset "construction" begin
        WS = MatrixWorkspace(4, 4)
        @test size(WS) == (4, 4)
    end

    @testset "LU solve (square)" begin
        n = 4
        A_data = rand(ComplexF64, n, n) + 5.0 * LA.I  # well-conditioned
        b_data = rand(ComplexF64, n)

        WS = MatrixWorkspace(n, n)
        copyto!(WS.A, A_data)
        updated!(WS)

        x = FSVec{ComplexF64}(zeros(ComplexF64, n))
        b = FSVec{ComplexF64}(b_data)
        LA.ldiv!(x, WS, b)

        @test Vector(WS.A) * Vector(x) ≈ b_data atol = 1e-10
    end

    @testset "repeated solves" begin
        n = 3
        WS = MatrixWorkspace(n, n)

        for _ in 1:5
            A_data = rand(ComplexF64, n, n) + 3.0 * LA.I
            b_data = rand(ComplexF64, n)
            copyto!(WS.A, A_data)
            updated!(WS)
            x = FSVec{ComplexF64}(zeros(ComplexF64, n))
            LA.ldiv!(x, WS, FSVec{ComplexF64}(b_data))
            @test Vector(A_data) * Vector(x) ≈ b_data atol = 1e-10
        end
    end

    @testset "overdetermined (QR)" begin
        m, n = 6, 4
        A_data = rand(ComplexF64, m, n)
        b_data = A_data * rand(ComplexF64, n)  # consistent system

        WS = MatrixWorkspace(m, n)
        copyto!(WS.A, A_data)
        updated!(WS)

        x = FSVec{ComplexF64}(zeros(ComplexF64, n))
        LA.ldiv!(x, WS, FSVec{ComplexF64}(b_data))

        # Least-squares solution should recover the original x
        @test LA.norm(A_data * Vector(x) - b_data) < 1e-10
    end

    @testset "AbstractMatrix interface" begin
        WS = MatrixWorkspace(3, 3)
        WS[1, 1] = 1.0 + 0.0im
        @test WS[1, 1] == 1.0 + 0.0im
        @test WS[1] == 1.0 + 0.0im
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/linear_algebra_test.jl")'`
Expected: FAIL — MatrixWorkspace not defined.

- [ ] **Step 3: Implement MatrixWorkspace with custom LU and ldiv!**

Create `src/primitives/linear_algebra.jl`. Port the core from v2 `linear_algebra.jl`:

```julia
const LA = LinearAlgebra

"""
    MatrixWorkspace <: AbstractMatrix{ComplexF64}

Pre-allocated workspace for repeated solution of `Ax = b` (square via LU, overdetermined via QR).

## Mutable justification
`factorized` and `scaled` flags are toggled on every Jacobian update. `lu` and `qr` fields
must be reassigned after factorization (they return new wrapper objects). All buffer fields
(`A`, `row_scaling`, `x̄`, `r`, `r̄`, `δx`, work vectors) are `const` — their contents are
mutated but their references never change.
"""
mutable struct MatrixWorkspace <: AbstractMatrix{ComplexF64}
    const A::FSMat{ComplexF64}
    factorized::Bool
    lu::LA.LU{ComplexF64,FSMat{ComplexF64},Vector{LA.BlasInt}}
    # NOTE: Type signatures doc specifies FSMat here, but qr! may not work on FSMat.
    # Task 0 determines this. If FSMat works, use FSMat. Otherwise keep Matrix and
    # update 01_type_signatures.md to match.
    qr::LA.QR{ComplexF64,Matrix{ComplexF64},Vector{ComplexF64}}
    const row_scaling::FSVec{Float64}
    scaled::Bool
    const x̄::FSVec{ComplexDF64}
    const r::FSVec{ComplexF64}
    const r̄::FSVec{ComplexDF64}
    const δx::FSVec{ComplexF64}
    const inf_norm_est_work::FSVec{ComplexF64}
    const inf_norm_est_rwork::FSVec{Float64}
end
```

**Note:** If Task 0 shows `lu!` does not work on `FSMat`, change `A` to `Matrix{ComplexF64}` and adjust the LU type accordingly. The `qr` field uses `Matrix{ComplexF64}` for its factors regardless (QR is only used for overdetermined systems which are less common).

Constructor, custom LU, custom QR, ldiv!, updated! — port directly from v2 lines 1–410, replacing `Vector` with `FSVec` and `Matrix` with `FSMat` where appropriate.

Include in main module after `double_f64.jl` and `norms.jl`:

```julia
include("primitives/linear_algebra.jl")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/linear_algebra_test.jl")'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/primitives/linear_algebra.jl src/HomotopyContinuationNext.jl test/linear_algebra_test.jl
git commit -m "feat: add MatrixWorkspace with custom LU/QR and ldiv!"
```

---

### Task 6: Linear Algebra — Row Scaling, Iterative Refinement, Condition Estimation

**Files:**
- Modify: `src/primitives/linear_algebra.jl` (append functions)
- Modify: `test/linear_algebra_test.jl` (append tests)

These are the remaining MatrixWorkspace features needed by the Newton corrector and tracker.

- [ ] **Step 1: Write failing tests**

Append to `test/linear_algebra_test.jl`:

```julia
using HomotopyContinuationNext:
    skeel_row_scaling!, apply_row_scaling!,
    mixed_precision_iterative_refinement!,
    fixed_precision_iterative_refinement!,
    residual!

@testset "Row scaling" begin
    n = 4
    WS = MatrixWorkspace(n, n)
    A_data = rand(ComplexF64, n, n) + 5.0 * LA.I
    copyto!(WS.A, A_data)
    updated!(WS)

    c = FSVec{Float64}(ones(n))
    skeel_row_scaling!(WS, c)
    @test all(WS.row_scaling .> 0.0)
    @test all(isfinite.(WS.row_scaling))
end

@testset "Iterative refinement" begin
    n = 4
    A_data = rand(ComplexF64, n, n) + 5.0 * LA.I
    x_true = rand(ComplexF64, n)
    b_data = A_data * x_true

    WS = MatrixWorkspace(n, n)
    copyto!(WS.A, A_data)
    updated!(WS)

    x = FSVec{ComplexF64}(zeros(ComplexF64, n))
    LA.ldiv!(x, WS, FSVec{ComplexF64}(copy(b_data)))

    err_before = LA.norm(Vector(x) - x_true)

    # Mixed precision refinement should improve accuracy
    mixed_precision_iterative_refinement!(x, WS, FSVec{ComplexF64}(b_data))
    err_after = LA.norm(Vector(x) - x_true)
    @test err_after <= err_before + 1e-14  # should not get worse
end

@testset "Condition number estimation" begin
    using HomotopyContinuationNext: inverse_inf_norm_est

    n = 4
    A_data = rand(ComplexF64, n, n) + 5.0 * LA.I  # well-conditioned
    WS = MatrixWorkspace(n, n)
    copyto!(WS.A, A_data)
    updated!(WS)
    factorize!(WS)

    inv_norm = inverse_inf_norm_est(WS)
    @test inv_norm > 0.0
    @test isfinite(inv_norm)

    # cond should be moderate for well-conditioned matrix
    cond = LA.cond(WS)
    @test 1.0 ≤ cond < 1e6
end

@testset "Residual computation" begin
    n = 3
    A = FSMat{ComplexF64}(rand(ComplexF64, n, n))
    x = FSVec{ComplexF64}(rand(ComplexF64, n))
    b = FSVec{ComplexF64}(Vector(A) * Vector(x))
    r = FSVec{ComplexF64}(zeros(ComplexF64, n))

    residual!(r, A, x, b)
    @test LA.norm(r) < 1e-12  # Ax - b ≈ 0
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/linear_algebra_test.jl")'`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement row scaling, iterative refinement, condition estimation**

Append to `src/primitives/linear_algebra.jl`. Port from v2 `linear_algebra.jl` lines 422–922:

- `skeel_row_scaling!(d, A, c)` — Skeel's optimal scaling (port lines 434–481)
- `apply_row_scaling!(W)` — apply scaling to LU factors (port lines 491–499)
- `residual!(r, A, x, b)` — compute `Ax - b` (port lines 507–521)
- `mixed_precision_iterative_refinement!(x, M, b, norm)` — DF64 residual + solve (port lines 530–546)
- `fixed_precision_iterative_refinement!(x, M, b, norm)` — F64 residual + solve (port lines 555–569)
- `inverse_inf_norm_est(WS)` — Higham's 1-norm estimator (port lines 587–599 + the main algorithm ~600-780)
- `LinearAlgebra.cond(WS)` — condition number (port lines ~780-800)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/linear_algebra_test.jl")'`
Expected: PASS

- [ ] **Step 5: Run quality gates**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && make test`
Expected: All tests pass (aqua, jet, explicit_imports, and all new tests).

- [ ] **Step 6: Commit**

```bash
git add src/primitives/linear_algebra.jl test/linear_algebra_test.jl
git commit -m "feat: add row scaling, iterative refinement, condition estimation"
```

---

### Task 7: Jacobian Wrapper

**Files:**
- Modify: `src/primitives/linear_algebra.jl` (append Jacobian struct + methods)
- Modify: `test/linear_algebra_test.jl` (append Jacobian tests)

Wrapper over MatrixWorkspace that tracks factorization/solve counts and provides the `ldiv!` interface with automatic Skeel scaling that the Newton corrector and tracker depend on.

- [ ] **Step 1: Write failing tests**

```julia
using HomotopyContinuationNext: Jacobian

@testset "Jacobian wrapper" begin
    @testset "construction and counters" begin
        n = 3
        J = Jacobian(MatrixWorkspace(n, n))
        @test J.factorizations[] == 0
        @test J.ldivs[] == 0
        @test size(J.workspace) == (3, 3)
    end

    @testset "updated! and init!" begin
        n = 3
        J = Jacobian(MatrixWorkspace(n, n))
        A_data = rand(ComplexF64, n, n) + 5.0 * LA.I
        copyto!(J.workspace.A, A_data)
        updated!(J)
        @test J.workspace.factorized == false
    end

    @testset "ldiv! with scaling" begin
        n = 4
        A_data = rand(ComplexF64, n, n) + 5.0 * LA.I
        x_true = rand(ComplexF64, n)
        b_data = A_data * x_true

        J = Jacobian(MatrixWorkspace(n, n))
        copyto!(J.workspace.A, A_data)
        updated!(J)

        x = FSVec{ComplexF64}(zeros(ComplexF64, n))
        w = WeightedNorm(n)
        init!(w, FSVec{ComplexF64}(x_true))

        LA.ldiv!(x, J, FSVec{ComplexF64}(b_data), w)
        @test LA.norm(Vector(x) - x_true) < 1e-10
        @test J.factorizations[] >= 1
        @test J.ldivs[] >= 1
    end

    @testset "cond delegation" begin
        n = 3
        J = Jacobian(MatrixWorkspace(n, n))
        A_data = rand(ComplexF64, n, n) + 5.0 * LA.I
        copyto!(J.workspace.A, A_data)
        updated!(J)

        c = LA.cond(J)
        @test 1.0 ≤ c < 1e6
    end
end
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/linear_algebra_test.jl")'`
Expected: FAIL

- [ ] **Step 3: Implement Jacobian struct and methods**

Append to `src/primitives/linear_algebra.jl`:

```julia
"""
    Jacobian

Wraps a [`MatrixWorkspace`](@ref) with factorization/solve counters and provides
the `ldiv!` interface with automatic Skeel row scaling. Used by the Newton corrector
and tracker.
"""
struct Jacobian
    workspace::MatrixWorkspace
    factorizations::Base.RefValue{Int}
    ldivs::Base.RefValue{Int}
end

function Jacobian(workspace::MatrixWorkspace)
    Jacobian(workspace, Ref(0), Ref(0))
end

"""
    updated!(J::Jacobian)

Forward `updated!` to the workspace. Call after writing new Jacobian values into `J.workspace.A`.
"""
function updated!(J::Jacobian)
    updated!(J.workspace)
    J
end

"""
    init!(J::Jacobian)

Reset factorization and solve counters to zero.
"""
function init!(J::Jacobian)
    J.factorizations[] = 0
    J.ldivs[] = 0
    J
end

Base.size(J::Jacobian) = size(J.workspace)

"""
    LA.ldiv!(x, J::Jacobian, b)

Solve `Jx = b` via LU (square) or QR (overdetermined). Increments counters.
"""
function LA.ldiv!(x::AbstractVector, J::Jacobian, b::AbstractVector)
    LA.ldiv!(x, J.workspace, b)
    J.ldivs[] += 1
    J.factorizations[] += 1
    x
end

"""
    LA.ldiv!(x, J::Jacobian, b, w::WeightedNorm)

Solve `Jx = b` with automatic Skeel row scaling based on weighted norm `w`.
Increments counters.
"""
function LA.ldiv!(
    x::AbstractVector, J::Jacobian, b::AbstractVector, w::WeightedNorm,
)
    m, n = size(J.workspace)
    if m == n
        # Apply Skeel row scaling using weights as approximate solution scale
        skeel_row_scaling!(J.workspace, w.weights)
        apply_row_scaling!(J.workspace)
    end
    LA.ldiv!(x, J.workspace, b)
    J.ldivs[] += 1
    J.factorizations[] += 1
    x
end

"""
    LA.cond(J::Jacobian)

Condition number of the Jacobian (delegates to workspace).
"""
LA.cond(J::Jacobian) = LA.cond(J.workspace)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/linear_algebra_test.jl")'`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/primitives/linear_algebra.jl test/linear_algebra_test.jl
git commit -m "feat: add Jacobian wrapper with ldiv!, scaling, and cond"
```

---

### Task 8: Module Wiring and Exports

**Files:**
- Modify: `src/HomotopyContinuationNext.jl`

Wire all includes in correct order and add exports for the public API.

- [ ] **Step 1: Update main module file**

The final `src/HomotopyContinuationNext.jl` should be:

```julia
module HomotopyContinuationNext

using LinearAlgebra: LinearAlgebra
using Random: Random
using Printf: Printf

using MultivariatePolynomials: MultivariatePolynomials
using FixedSizeArrays: FixedSizeVector, FixedSizeMatrix

const MP = MultivariatePolynomials
const FSVec{T} = FixedSizeVector{T}
const FSMat{T} = FixedSizeMatrix{T}

# --- Primitives ---
include("utils.jl")
include("primitives/double_f64.jl")
include("primitives/norms.jl")
include("primitives/linear_algebra.jl")

end # module
```

- [ ] **Step 2: Run full quality gates**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && make test`
Expected: ALL tests pass — aqua, jet, explicit_imports, and all Phase 1 tests.

- [ ] **Step 3: Verify JET reports zero issues**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && julia --project=test -e 'include("test/jet_test.jl")'`
Expected: 0 reports.

- [ ] **Step 4: Commit**

```bash
git add src/HomotopyContinuationNext.jl
git commit -m "feat: wire Phase 1 primitives into main module"
```

---

### Task 9: Phase 1 Integration Benchmark

**Files:**
- Modify: `benchmark/benchmarks.jl`

Add basic benchmarks to establish baseline performance for primitives.

- [ ] **Step 1: Write benchmarks**

```julia
using BenchmarkTools
using HomotopyContinuationNext
using HomotopyContinuationNext:
    DoubleF64, ComplexDF64,
    MatrixWorkspace, updated!, factorize!,
    WeightedNorm, weighted_norm, init!,
    inf_norm, fast_abs
using FixedSizeArrays: FixedSizeVector, FixedSizeMatrix
using LinearAlgebra: LinearAlgebra

const FSVec{T} = FixedSizeVector{T}
const FSMat{T} = FixedSizeMatrix{T}

const SUITE = BenchmarkGroup()

# --- DoubleF64 ---
SUITE["double_f64"] = BenchmarkGroup()
let a = DoubleF64(1.234567890123456789), b = DoubleF64(9.876543210987654321)
    SUITE["double_f64"]["add"] = @benchmarkable $a + $b
    SUITE["double_f64"]["mul"] = @benchmarkable $a * $b
    SUITE["double_f64"]["div"] = @benchmarkable $a / $b
end

# --- Norms ---
SUITE["norms"] = BenchmarkGroup()
for n in [4, 16, 64]
    x = FSVec{ComplexF64}(rand(ComplexF64, n))
    SUITE["norms"]["inf_norm_$n"] = @benchmarkable inf_norm($x)
    w = WeightedNorm(n)
    init!(w, x)
    SUITE["norms"]["weighted_norm_$n"] = @benchmarkable weighted_norm($x, $w)
end

# --- Linear Algebra ---
SUITE["linear_algebra"] = BenchmarkGroup()
for n in [4, 8, 16]
    A_data = rand(ComplexF64, n, n) + 5.0 * LinearAlgebra.I
    b_data = rand(ComplexF64, n)
    WS = MatrixWorkspace(n, n)
    copyto!(WS.A, A_data)
    updated!(WS)
    x = FSVec{ComplexF64}(zeros(ComplexF64, n))
    b = FSVec{ComplexF64}(b_data)

    SUITE["linear_algebra"]["lu_ldiv_$n"] = @benchmarkable begin
        copyto!($WS.A, $A_data)
        updated!($WS)
        LinearAlgebra.ldiv!($x, $WS, $b)
    end
end

BenchmarkTools.tune!(SUITE)
results = BenchmarkTools.run(SUITE; verbose = true)
display(median(results))
BenchmarkTools.save("benchmarks_output.json", median(results))
```

- [ ] **Step 2: Run benchmark**

Run: `cd /var/home/oameye/Documents/HomotopyContinuation.jl && make benchmark`

Document baseline results. Key numbers to record:
- DoubleF64 add/mul/div (should be <10ns each)
- inf_norm for n=4 (should be <20ns)
- LU + ldiv for n=4 (should be <1μs)

- [ ] **Step 3: Commit**

```bash
git add benchmark/benchmarks.jl
git commit -m "bench: add Phase 1 primitives benchmarks"
```

---

## Dependency Order

```
Task 0 (FSA compat) ─┐
                      ├─→ Task 5 (MatrixWorkspace core)
Task 1 (utils) ──┬───┤         │
                  │   │         ├─→ Task 6 (scaling, refinement, cond)
                  │   │         │         │
Task 2 (Stepper) ┘   │         │         ├─→ Task 7 (Jacobian)
                      │         │         │
Task 3 (DoubleF64) ──┤         │         │
                      │         │         │
Task 4 (norms) ──────┘         │         │
      ↑                        │         │
      └─ needs fast_abs ───────┘         │
                                         │
Task 8 (wiring + exports) ◄─────────────┘
         │
Task 9 (benchmarks) ◄───────────────────┘
```

Parallelizable: Tasks 0, 1, 3 can run in parallel. Task 2 depends on Task 1 (appends to same files). Task 4 depends on Task 1 (fast_abs). Tasks 5-7 are sequential. Task 8 depends on all. Task 9 depends on 8.

**Deferred primitives** (needed by Phase 5, not Phase 1): VoronoiTree, UniquePoints, LinearSubspace, GroupActions. These will be implemented alongside the solve layer.
