# Phase 2: Interpreter Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **IMPORTANT:** Do NOT commit. The user controls all git operations.

**Goal:** Build the tape-based polynomial evaluator: from DynamicPolynomials input to a working `Interpreter` that can evaluate F(x), Jacobian, Taylor orders 1-3, and DF64 — all through a single `InstructionSequence`.

**Architecture:** MP polynomials → extract supports/coefficients → build shared monomial table (F + Jacobian) → compile to InstructionSequence (DAG reorder + register allocation) → execute via Interpreter{V} where V determines the arithmetic (ComplexF64, ComplexDF64, TruncatedTaylorSeries). No SymEngine — we go MP-native. The `system_eval()` FunctionWrapper integration is deferred to Phase 3.

**Tech Stack:** MultivariatePolynomials.jl (MP), DynamicPolynomials.jl (@polyvar), EnumX.jl (@enumx), Julia @generated functions

**Reference code:** `HomotopyContinuation/src/model_kit/` — `operations.jl` (213 lines), `taylor.jl` (642 lines), `intermediate_representation.jl` (529 lines), `instruction_sequence.jl` (455 lines), `instruction_interpreter.jl` (495 lines). Port logic but replace SymEngine front-end with MP-native extraction.

**Design docs:** `implementation_docs/00_design_document.md` §7 (compilation pipeline), `implementation_docs/01_type_signatures.md` §2 (model kit types).

---

## File Structure

```
src/
├── HomotopyContinuationNext.jl              # Modify: add includes, exports, EnumX dep
└── model_kit/
    ├── operations.jl                        # Create: OpType enum, arity(), op_* scalar functions
    ├── taylor.jl                            # Create: TruncatedTaylorSeries, TaylorVector, taylor_op_*
    ├── instruction_sequence.jl              # Create: Instruction, InstructionSequence, IR, DAG reorder, register alloc
    ├── interpreter.jl                       # Create: Interpreter{V}, execute!, execute_taylor!
    └── polynomial_input.jl                  # Create: MP extraction → InstructionSequence (no SymEngine)

test/
├── operations_test.jl                       # Create: op_* correctness
├── taylor_test.jl                           # Create: TaylorVector, taylor_op_* vs finite differences
├── interpreter_test.jl                      # Create: end-to-end evaluate, jacobian, taylor, DF64
└── polynomial_input_test.jl                 # Create: MP → Interpreter → correct evaluation
```

---

### Task 1: Operations — OpType Enum and Scalar Functions

**Files:**
- Create: `src/model_kit/operations.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add EnumX dep, include)
- Create: `test/operations_test.jl`

The OpType enum and scalar `op_*` functions are the foundation. Every instruction dispatches to one of these.

- [ ] **Step 1: Add EnumX dependency**

Add to `Project.toml`:
```toml
EnumX = "4e289a0a-7415-4e19-859d-a7e5c4648b56"
```
Add to main module: `using EnumX: @enumx`

- [ ] **Step 2: Write failing tests for operations**

```julia
using Test
using HomotopyContinuationNext: OpType, arity,
    op_add, op_sub, op_mul, op_div, op_neg,
    op_sqr, op_cb, op_sqrt, op_inv, op_invsqr,
    op_muladd, op_mulsub, op_submul,
    op_add3, op_add4, op_mul3, op_mul4,
    op_mulmuladd, op_mulmulsub,
    op_pow_int, op_identity, op_inv_not_zero

@testset "Operations" begin
    @testset "OpType enum" begin
        @test OpType.OP_ADD isa OpType.T
        @test OpType.OP_STOP isa OpType.T
    end

    @testset "arity" begin
        @test arity(OpType.OP_NEG) == 1
        @test arity(OpType.OP_ADD) == 2
        @test arity(OpType.OP_MULADD) == 3
        @test arity(OpType.OP_MULMULADD) == 4
    end

    @testset "scalar ops (ComplexF64)" begin
        a, b, c, d = 2.0+1.0im, 3.0-1.0im, 1.0+2.0im, 0.5+0.5im
        @test op_add(a, b) ≈ a + b
        @test op_sub(a, b) ≈ a - b
        @test op_mul(a, b) ≈ a * b
        @test op_div(a, b) ≈ a / b
        @test op_neg(a) ≈ -a
        @test op_sqr(a) ≈ a^2
        @test op_cb(a) ≈ a^3
        @test op_sqrt(a) ≈ sqrt(a)
        @test op_inv(a) ≈ 1/a
        @test op_invsqr(a) ≈ 1/a^2
        @test op_identity(a) == a
        @test op_inv_not_zero(a) ≈ 1/a
        @test op_inv_not_zero(0.0+0.0im) == 0.0+0.0im
        @test op_pow_int(a, 3) ≈ a^3
        @test op_pow_int(a, 0) ≈ 1.0
        @test op_muladd(a, b, c) ≈ a*b + c
        @test op_mulsub(a, b, c) ≈ a*b - c
        @test op_submul(a, b, c) ≈ c - a*b
        @test op_add3(a, b, c) ≈ a + b + c
        @test op_mul3(a, b, c) ≈ a * b * c
        @test op_add4(a, b, c, d) ≈ a + b + c + d
        @test op_mul4(a, b, c, d) ≈ a * b * c * d
        @test op_mulmuladd(a, b, c, d) ≈ a*b + c*d
        @test op_mulmulsub(a, b, c, d) ≈ a*b - c*d
    end

    @testset "scalar ops (Float64)" begin
        @test op_mul(2.0, 3.0) ≈ 6.0
        @test op_muladd(2.0, 3.0, 1.0) ≈ 7.0
    end
end
```

- [ ] **Step 3: Implement operations**

Create `src/model_kit/operations.jl`. Port from `HomotopyContinuation/src/model_kit/operations.jl` (213 lines):

- `@enumx OpType::Int8 begin ... end` — 24 operations per design doc
- `arity(op::OpType.T)::Int` — return 0 (STOP), 1, 2, 3, or 4
- `op_call(op::OpType.T)::Symbol` — maps OpType → function name symbol
- All `op_*` scalar functions (generic + Complex-optimized where applicable)
- `should_use_index_not_reference(op, index)` — special case for OP_POW_INT arg2

Key: use `@enumx` (not `@enum`) per CLAUDE.md, access as `OpType.OP_ADD`.

- [ ] **Step 4: Run tests**

- [ ] **Step 5: Format with `runic --inplace src/model_kit/operations.jl`**

---

### Task 2: Taylor — TruncatedTaylorSeries and TaylorVector

**Files:**
- Create: `src/model_kit/taylor.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)
- Create: `test/taylor_test.jl`

TruncatedTaylorSeries{N,T} stores N coefficients of a Taylor expansion. TaylorVector{N,T} stores a vector of such series in column-major FSMat layout. The `taylor_op_*` functions implement Cauchy product rules for each operation.

- [ ] **Step 1: Write failing tests**

```julia
using Test
using HomotopyContinuationNext: TruncatedTaylorSeries, TaylorVector

@testset "TruncatedTaylorSeries" begin
    @testset "construction and indexing" begin
        t = TruncatedTaylorSeries((1.0+0im, 2.0+0im, 3.0+0im))
        @test t[0] == 1.0+0im   # 0-indexed externally
        @test t[1] == 2.0+0im
        @test t[2] == 3.0+0im
        @test length(t) == 3
    end
end

@testset "TaylorVector" begin
    @testset "construction" begin
        # TaylorVector{3, ComplexF64} stores orders 0,1,2 for n elements
        tv = TaylorVector{3, ComplexF64}(5)  # 5 elements, 3 coefficients each
        @test size(tv) == (5,)
        @test length(tv) == 5
    end

    @testset "getindex / setindex!" begin
        tv = TaylorVector{2, ComplexF64}(3)
        tv[1] = TruncatedTaylorSeries((1.0+0im, 2.0+0im))
        t = tv[1]
        @test t[0] == 1.0+0im
        @test t[1] == 2.0+0im
    end

    @testset "vectors (split into order-k vectors)" begin
        tv = TaylorVector{2, ComplexF64}(3)
        vs = vectors(tv)
        @test length(vs) == 2
        # Each vector has length 3
        @test length(vs[1]) == 3
        @test length(vs[2]) == 3
    end
end
```

- [ ] **Step 2: Implement TruncatedTaylorSeries and TaylorVector**

Create `src/model_kit/taylor.jl`:

- `struct TruncatedTaylorSeries{N,T}` with `val::NTuple{N,T}`, 0-indexed `getindex`
- `struct TaylorVector{N,T} <: AbstractVector{TruncatedTaylorSeries{N,T}}` with `data::FSMat{T}`
  - **Matrix orientation:** `data` is `N × n` (N rows = coefficient orders, n columns = vector elements). Allocate as `FSMat{T}(zeros(T, N, n))`.
- Constructors, `getindex`, `setindex!`, `vectors()` (split into N row-views)
- Conversion: scalar → TruncatedTaylorSeries (pad with zeros)

- [ ] **Step 3: Run tests, format**

---

### Task 3: Taylor Operations — taylor_op_*

**Files:**
- Modify: `src/model_kit/taylor.jl` (append taylor_op_* functions)
- Modify: `test/taylor_test.jl` (append tests)

Each `op_*` has a corresponding `taylor_op_*` that implements the Cauchy product rule for that operation. These are the workhorses of the Taylor evaluator.

- [ ] **Step 1: Write failing tests (verify via finite differences)**

```julia
using HomotopyContinuationNext: taylor_op_add, taylor_op_mul, taylor_op_div,
    taylor_op_neg, taylor_op_sqr, taylor_op_cb, taylor_op_inv,
    taylor_op_pow_int, taylor_op_muladd, taylor_op_sqrt

@testset "taylor_op_* correctness" begin
    # For f(t) = a₀ + a₁t + a₂t², the Taylor coefficients of g(f(t))
    # should match finite-difference verification
    a = TruncatedTaylorSeries((2.0+0im, 1.0+0im, 0.5+0im))
    b = TruncatedTaylorSeries((3.0+0im, -1.0+0im, 0.0+0im))

    @testset "add" begin
        r = taylor_op_add(a, b)
        @test r[0] ≈ 5.0+0im
        @test r[1] ≈ 0.0+0im
        @test r[2] ≈ 0.5+0im
    end

    @testset "mul (Cauchy product)" begin
        r = taylor_op_mul(a, b)
        # (2+t+0.5t²)(3-t) = 6 - 2t + 3t + 1.5t² - t² = 6 + t + 0.5t²
        @test r[0] ≈ 6.0+0im
        @test r[1] ≈ a[0]*b[1] + a[1]*b[0]  # = -2+3 = 1
        @test r[2] ≈ a[0]*b[2] + a[1]*b[1] + a[2]*b[0]
    end

    @testset "neg" begin
        r = taylor_op_neg(a)
        @test r[0] ≈ -a[0]
        @test r[1] ≈ -a[1]
    end

    @testset "sqr" begin
        r = taylor_op_sqr(a)
        m = taylor_op_mul(a, a)
        @test r[0] ≈ m[0]
        @test r[1] ≈ m[1]
        @test r[2] ≈ m[2]
    end

    @testset "inv" begin
        r = taylor_op_inv(b)
        # inv(b) * b should give (1, 0, 0, ...)
        product = taylor_op_mul(r, b)
        @test product[0] ≈ 1.0+0im atol=1e-12
        @test product[1] ≈ 0.0+0im atol=1e-12
        @test product[2] ≈ 0.0+0im atol=1e-12
    end

    @testset "div" begin
        r = taylor_op_div(a, b)
        # a/b * b should give a
        product = taylor_op_mul(r, b)
        @test product[0] ≈ a[0] atol=1e-12
        @test product[1] ≈ a[1] atol=1e-12
        @test product[2] ≈ a[2] atol=1e-12
    end

    @testset "pow_int" begin
        r = taylor_op_pow_int(a, 3)
        m = taylor_op_mul(taylor_op_mul(a, a), a)
        @test r[0] ≈ m[0] atol=1e-12
        @test r[1] ≈ m[1] atol=1e-12
        @test r[2] ≈ m[2] atol=1e-12
    end

    @testset "sqrt" begin
        r = taylor_op_sqrt(a)
        # sqrt(a)^2 should give a
        product = taylor_op_sqr(r)
        @test product[0] ≈ a[0] atol=1e-12
        @test product[1] ≈ a[1] atol=1e-12
        @test product[2] ≈ a[2] atol=1e-12
    end
end
```

- [ ] **Step 2: Implement taylor_op_* functions**

Port from `HomotopyContinuation/src/model_kit/taylor.jl` (lines 230-642). Key implementations:

- **Coefficient-wise:** `taylor_op_add`, `taylor_op_sub`, `taylor_op_neg`, `taylor_op_identity`
- **Cauchy product:** `taylor_op_mul` — `c_k = Σ_{j=0}^k a_j * b_{k-j}`
- **Division:** `taylor_op_div` — `c_k = (a_k - Σ_{j=0}^{k-1} c_j * b_{k-j}) / b_0`
- **Inverse:** `taylor_op_inv` — special case of div with a=1
- **Square:** `taylor_op_sqr` — optimized self-multiply
- **Cube:** `taylor_op_cb` — triple self-multiply
- **Power:** `taylor_op_pow_int` — logarithmic differentiation recurrence
- **Square root:** `taylor_op_sqrt` — `c_k = (a_k - Σ_{j=1}^{k-1} c_j * c_{k-j}) / (2*c_0)` for k≥1 (see v2 `taylor.jl` for exact recurrence)
- **Fused ternary/quaternary:** `taylor_op_muladd`, `taylor_op_mulsub`, etc. (chain multiply + add)

Use `@generated` or manual implementations for fixed N (orders 1-3 used in practice, i.e. TTS{2}, TTS{3}, TTS{4}).

- [ ] **Step 3: Run tests, format**

---

### Task 4: Instruction Sequence — IR, DAG Reorder, Register Allocation

**Files:**
- Create: `src/model_kit/instruction_sequence.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)
- Create: `test/instruction_sequence_test.jl`

This is the compilation backend: IR data structures, the Instruction tape format, DAG-based instruction reordering, and liveness-based register allocation. This task creates the data structures and optimization passes — NOT the front-end that fills them from MP polynomials (that's Task 6).

- [ ] **Step 1: Write failing tests**

```julia
using Test
using HomotopyContinuationNext: IRStatementRef, IRStatement, IRStatementArg,
    IntermediateRepresentation, Instruction, InstructionSequence,
    OpType, build_instruction_sequence_from_ir

@testset "IR data structures" begin
    ref = IRStatementRef(1)
    @test ref.i == 1

    # IRStatement has positional-arg constructors (2, 3, 4 args) that build the NTuple
    stmt = IRStatement(OpType.OP_ADD, IRStatementRef(3),
        IRStatementRef(1), IRStatementRef(2))
    @test stmt.op == OpType.OP_ADD
end

@testset "InstructionSequence from IR" begin
    # Manual IR: f(x) = x₁ + x₂ (2 vars, no params, output_dim=1)
    stmts = [
        IRStatement(OpType.OP_ADD, IRStatementRef(3),
            IRStatementRef(1), IRStatementRef(2)),
    ]
    assignments = [(1, IRStatementRef(3))]
    ir = IntermediateRepresentation(stmts, assignments, 1)

    seq = build_instruction_sequence_from_ir(
        ir;
        nvars = 2,
        nparams = 0,
        nconstants = 0,
        constants = ComplexF64[],
    )
    @test seq.output_dim == 1
    @test seq.tape_space_needed > 0
    @test length(seq.instructions) > 0
    @test !isempty(seq.u_assignments)
end
```

- [ ] **Step 2: Implement IR types and InstructionSequence compilation**

Create `src/model_kit/instruction_sequence.jl`. Port from v2:

**Data structures (use `OpType.T` for enum field types with @enumx):**
- `IRStatementRef`, `IRStatementArg`, `IRStatement` (field `op::OpType.T`, positional-arg constructors for 1-4 args)
- `IntermediateRepresentation`
- `Instruction` (field `op::OpType.T`)
- `InstructionSequence` — include `all_u_assigned::Bool` and `all_U_assigned::Bool` fields (used to skip `zero!` at execution time)

**Compilation pipeline (IR → InstructionSequence):**
1. Allocate tape layout: constants | parameters | [continuation_param] | variables | scratch
2. Map IR statement refs → tape indices
3. `optimize_instruction_order()` — DAG topological sort for data locality
4. `index_compactification_mapping()` — liveness-based linear scan register allocation
5. Split assignments into u_assignments (outputs) and U_assignments (Jacobian)

Port from `HomotopyContinuation/src/model_kit/instruction_sequence.jl` lines 133-455. Adapt to use `@enumx` OpType.

- [ ] **Step 3: Run tests, format**

---

### Task 5: Interpreter — Tape Execution

**Files:**
- Create: `src/model_kit/interpreter.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add include)
- Create: `test/interpreter_test.jl`

The interpreter executes an InstructionSequence on a pre-allocated tape. Different tape element types give different evaluation modes. This task tests with a manually-constructed InstructionSequence.

- [ ] **Step 1: Write failing tests**

```julia
using Test
using HomotopyContinuationNext: Interpreter, InstructionSequence, Instruction, OpType,
    execute!, execute_taylor!, TruncatedTaylorSeries, TaylorVector, DoubleF64, ComplexDF64

@testset "Interpreter" begin
    @testset "execute! on manual sequence" begin
        # Build a sequence for f(x₁,x₂) = x₁*x₂ + x₁
        # Tape layout: [1]=x₁, [2]=x₂, [3]=x₁*x₂, [4]=x₁*x₂+x₁
        # ...test with manually-built InstructionSequence
    end
end
```

(Exact test code depends on InstructionSequence construction API from Task 4.)

- [ ] **Step 2: Implement Interpreter**

Create `src/model_kit/interpreter.jl`. Port from v2 `instruction_interpreter.jl` (495 lines):

- `mutable struct Interpreter{V<:AbstractVector}` with `sequence`, `tape`, `variables`, `parameters` (mutable per type signatures doc — tape contents mutated via `tape[i] = val`)
- `Interpreter(::Type{V}, seq::InstructionSequence)` — allocate tape, preload constants
- `execute!(u, I::Interpreter, x, [p])` — load inputs → execute → copy outputs
- `execute!(u, U, I::Interpreter, x, [p])` — also extract Jacobian
- `execute_taylor!(u, ::Val{K}, I::Interpreter, tx, [p])` — Taylor evaluation
- `execute_instructions!(tape, instructions)` — the inner loop (use `@generated` or manual dispatch)
- `execute_taylor_instructions!(::Val{K}, ...)` — Taylor inner loop

Key: the inner loop dispatches on `OpType.T` per instruction. Follow v2's nested-if codegen pattern (not pure `@generated`) — enumerate `instances(OpType.T)` to build the switch. With `@enumx`, use `OpType.T` (not `OpType`) for instance enumeration. This is the proven approach for TTFX-friendly dispatch.

- [ ] **Step 3: Run tests, format**

---

### Task 6: Polynomial Input — MP Extraction to InstructionSequence

**Files:**
- Create: `src/model_kit/polynomial_input.jl`
- Modify: `src/HomotopyContinuationNext.jl` (add DynamicPolynomials dep, include)
- Create: `test/polynomial_input_test.jl`

This is our front-end — replaces v2's SymEngine-based pipeline. Goes directly from MP polynomials to InstructionSequence using support extraction and monomial table construction.

- [ ] **Step 1: Add DynamicPolynomials dependency**

Add to `Project.toml`:
```toml
DynamicPolynomials = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
```

- [ ] **Step 2: Write failing tests**

```julia
using Test
using DynamicPolynomials: @polyvar
using HomotopyContinuationNext: Interpreter, execute!
using FixedSizeArrays: FixedSizeArray
const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}

@testset "Polynomial input" begin
    @testset "simple system evaluation" begin
        @polyvar x y
        F = [x^2 + y, x*y - 1]

        I = build_interpreter(F)
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        xv = FSVec{ComplexF64}([2.0+0im, 3.0+0im])
        p = FSVec{ComplexF64}(ComplexF64[])

        execute!(u, I, xv, p)
        @test u[1] ≈ 4.0 + 3.0  # x²+y = 4+3
        @test u[2] ≈ 6.0 - 1.0  # xy-1 = 6-1
    end

    @testset "Jacobian evaluation" begin
        @polyvar x y
        F = [x^2 + y, x*y - 1]

        I_jac = build_jacobian_interpreter(F)
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        xv = FSVec{ComplexF64}([2.0+0im, 3.0+0im])
        p = FSVec{ComplexF64}(ComplexF64[])

        execute!(u, U, I_jac, xv, p)
        @test u[1] ≈ 7.0+0im
        @test u[2] ≈ 5.0+0im
        @test U[1,1] ≈ 4.0+0im  # ∂(x²+y)/∂x = 2x = 4
        @test U[1,2] ≈ 1.0+0im  # ∂(x²+y)/∂y = 1
        @test U[2,1] ≈ 3.0+0im  # ∂(xy-1)/∂x = y = 3
        @test U[2,2] ≈ 2.0+0im  # ∂(xy-1)/∂y = x = 2
    end

    @testset "with parameters" begin
        @polyvar x y a b
        F = [x^2 + a*y, x*y - b]

        I = build_interpreter(F; parameters=[a, b])
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        xv = FSVec{ComplexF64}([2.0+0im, 3.0+0im])
        p = FSVec{ComplexF64}([1.0+0im, 1.0+0im])  # a=1, b=1

        execute!(u, I, xv, p)
        @test u[1] ≈ 7.0+0im  # x²+a*y = 4+3
        @test u[2] ≈ 5.0+0im  # x*y-b = 6-1
    end

    @testset "Taylor evaluation matches finite differences" begin
        @polyvar x y
        F = [x^2 + y, x*y - 1]

        I_t1 = build_taylor_interpreter(F, Val(1))
        # ... verify Taylor order 1 matches (F(x+εv) - F(x))/ε
    end

    @testset "DF64 evaluation" begin
        @polyvar x y
        F = [x^2 + y, x*y - 1]

        I_df64 = build_df64_interpreter(F)
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        xv = FSVec{ComplexDF64}([ComplexDF64(2.0+0im), ComplexDF64(3.0+0im)])
        p = FSVec{ComplexF64}(ComplexF64[])

        execute!(u, I_df64, xv, p)
        @test real(u[1]) ≈ 7.0 atol=1e-20
    end

    @testset "higher degree polynomial" begin
        @polyvar x y z
        F = [x^3*y + z^2 - 1, x*y*z + x^2 - z]

        I = build_interpreter(F)
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        xv = FSVec{ComplexF64}([1.0+0im, 2.0+0im, 3.0+0im])
        p = FSVec{ComplexF64}(ComplexF64[])

        execute!(u, I, xv, p)
        @test u[1] ≈ 1*2 + 9 - 1  # x³y + z² - 1
        @test u[2] ≈ 1*2*3 + 1 - 3  # xyz + x² - z
    end
end
```

- [ ] **Step 3: Implement MP extraction pipeline**

Create `src/model_kit/polynomial_input.jl`:

**Key functions:**
1. `extract_supports(polys, vars, params)` — using `MP.terms`, `MP.exponents`, `MP.coefficient`
2. `build_monomial_table(supports, j_supports)` — shared monomial table (CSE by construction)
3. `build_ir_from_supports(monomial_table, coeffs, ...)` — monomial → multiply tree, polynomial → sum with fusion
4. `build_instruction_sequence(polys; variables, parameters)` — full pipeline
5. `build_interpreter(polys; parameters)` → convenience wrapper around `Interpreter(Vector{ComplexF64}, seq)` constructor
6. `build_jacobian_interpreter(polys; parameters)` → includes Jacobian rows in the sequence
7. `build_taylor_interpreter(polys, ::Val{K}; parameters)` → `Interpreter(Vector{TTS{K+1, ComplexF64}}, seq)`
8. `build_df64_interpreter(polys; parameters)` → `Interpreter(Vector{ComplexDF64}, seq)`

**Note:** `build_*` functions are the public convenience API for this phase. In Phase 3, `system_eval()` will call the `Interpreter(::Type{V}, seq)` constructor directly.

The monomial compilation strategy (from design doc §7.1):
- For each monomial: power decomposition (OP_SQR for x², OP_CB for x³, OP_POW_INT for higher, OP_MUL for products)
- For each polynomial: coefficient × monomial terms, fused sums (OP_MULADD for c*m+rest, OP_ADD3 for 3-way sum, OP_MULMULADD for a*b+c*d)

- [ ] **Step 4: Run tests, format**

---

### Task 7: Integration Test and Quality Gates

**Files:**
- Modify: `test/polynomial_input_test.jl` (extend with edge cases)
- Run: `make test`

End-to-end verification that the full pipeline works and all quality gates pass.

- [ ] **Step 1: Add edge case tests**

```julia
@testset "Edge cases" begin
    @testset "constant polynomial" begin
        @polyvar x
        F = [x - x + 3.0]  # constant 3.0
        I = build_interpreter(F)
        u = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        execute!(u, I, FSVec{ComplexF64}([1.0+0im]), FSVec{ComplexF64}(ComplexF64[]))
        @test u[1] ≈ 3.0+0im
    end

    @testset "single variable" begin
        @polyvar x
        F = [x^5 - 2*x^3 + x - 7]
        I = build_interpreter(F)
        u = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        execute!(u, I, FSVec{ComplexF64}([2.0+0im]), FSVec{ComplexF64}(ComplexF64[]))
        @test u[1] ≈ 32 - 16 + 2 - 7  # = 11
    end

    @testset "linear system" begin
        @polyvar x y
        F = [2x + 3y - 1, x - y + 2]
        I = build_interpreter(F)
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        execute!(u, I, FSVec{ComplexF64}([1.0+0im, 1.0+0im]), FSVec{ComplexF64}(ComplexF64[]))
        @test u[1] ≈ 4.0+0im  # 2+3-1
        @test u[2] ≈ 2.0+0im  # 1-1+2
    end
end
```

- [ ] **Step 2: Run `make test`**

All quality gates must pass: Aqua (no new ambiguities), JET (0 reports), ExplicitImports.

- [ ] **Step 3: Format all new files**

```bash
runic --inplace src/model_kit/
```

---

## Dependency Order

```
Task 1 (operations) ──┬──→ Task 3 (taylor_op_*)
                      │         ↑
Task 2 (TTS/TV) ─────┴─────────┘──→ Task 5 (interpreter)
                                           ↑
Task 4 (InstructionSequence) ──────────────┘──→ Task 6 (polynomial input)
                                                          │
                                               Task 7 (integration) ◄──┘
```

Parallelizable: Tasks 1 and 2 are independent. Task 3 depends on both 1 and 2. Task 4 depends on Task 1. Task 5 depends on Tasks 3 and 4. Task 6 depends on Tasks 4 and 5. Task 7 depends on all.

**Exports:** Each task must add exports to `src/HomotopyContinuationNext.jl` for all names used in its tests. Follow the ExplicitImports rule — every `using HomotopyContinuationNext: foo` in a test requires `foo` to be exported.

**Note on Phase 3 boundary:** This phase produces `Interpreter` objects that can evaluate F(x), J(x), Taylor, and DF64. Phase 3 wraps these in `SystemEvaluator` via `FunctionWrapper` and builds the `system_eval()` constructor that returns `(PolynomialSystemInfo, SystemEvaluator)`.
