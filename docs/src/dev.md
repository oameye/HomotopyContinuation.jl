# Developer Documentation

HomotopyContinuationNext.jl — a ground-up rewrite of [HomotopyContinuation.jl](https://github.com/JuliaHomotopyContinuation/HomotopyContinuation.jl) for solving polynomial systems via homotopy continuation. Prioritizes type stability, minimal TTFX, and zero runtime dispatch on hot paths.

Homotopy continuation tracks paths from a known start system to the target. Each step evaluates the system and its Jacobian, solves a linear system, and computes Taylor coefficients for the predictor — thousands of times per path, across thousands of paths. The package is structured around making these operations fast and allocation-free.

## Package layout

```
src/
├── HomotopyContinuationNext.jl     # Module definition, type aliases (FSVec, FSMat)
├── utils.jl                        # SegmentStepper, fast_abs
├── primitives/                     # Numeric building blocks
│   ├── double_f64.jl               # DoubleF64, ComplexDF64
│   ├── norms.jl                    # InfNorm, WeightedNorm
│   └── linear_algebra.jl           # MatrixWorkspace, LU, QR, condition estimation
├── model_kit/                      # Polynomial → tape interpreter pipeline
│   ├── operations.jl               # OpType enum, scalar op_* functions
│   ├── taylor.jl                   # TruncatedTaylorSeries, TaylorVector, taylor_op_*
│   ├── sexpr.jl                    # SExpr IR, canonicalization, poly_to_sexpr
│   ├── cse.jl                      # Common subexpression elimination (SymEngine port)
│   ├── tape_compiler.jl            # SExpr → Instruction compilation with fusion
│   ├── instruction_sequence.jl     # Instruction type, DAG reorder, register allocator
│   ├── interpreter.jl              # Tape-based execution engine
│   └── polynomial_input.jl         # MP polynomials → Interpreter orchestration
├── core/                           # System/Homotopy types and FunctionWrapper firewall
│   ├── system.jl                   # System (compiled polynomial system, caches interpreter pipeline)
│   ├── homotopy_eval.jl            # HomotopyEvaluator (concrete, type-erased)
│   ├── straight_line_homotopy.jl   # γt·G + (1-t)·F
│   └── ...                         # CoefficientHomotopy, ToricHomotopy, AffineChart
├── tracking/                       # Path tracker and endgame
│   ├── newton_corrector.jl         # α-theory Newton
│   ├── predictor.jl                # Padé (2,1), Hermite
│   ├── tracker.jl                  # Core path tracker (monomorphic)
│   └── endgame.jl                  # Singular/infinity detection
└── solving/                        # Top-level solve orchestration
    ├── total_degree.jl             # Bézout start system
    ├── polyhedral.jl               # BKK mixed volume
    └── solve.jl                    # solve() entry point
```

---

## Primitives (`src/primitives/`)

The primitives provide the numeric foundation that the tracker calls on every step. They have no dependency on the model kit or the rest of the package.

The central type is `MatrixWorkspace` in `linear_algebra.jl`. It wraps a pre-allocated `FSMat{ComplexF64}` and provides lazy LU factorization with a custom pivot selection that uses `@fastmath` to avoid `abs` in the inner loop. On top of that sit Skeel row scaling for numerical stability, a Hager-Higham condition estimator, and mixed-precision iterative refinement that computes residuals in `ComplexDF64` to recover extra digits when the Jacobian is ill-conditioned. The entire `ldiv!` path — factorize, scale, solve, optionally refine — is zero-allocation because all scratch buffers (`x̄`, `r`, `r̄`, `δx`) are allocated once at construction. `Jacobian` wraps `MatrixWorkspace` and adds factorization/solve counters for diagnostics.

`DoubleF64` in `double_f64.jl` is what makes the mixed-precision path possible. It stores two `Float64` values (`hi`, `lo`) and provides ~31 digits of precision through error-free transformations (`two_sum`, `two_prod`) that compile to plain FP instructions. No MPFR, no BigFloat — the tracker stays on hardware floats even in extended precision. `ComplexDF64 = Complex{DoubleF64}`.

`WeightedNorm` in `norms.jl` handles step size control. It carries per-coordinate weights in an `FSVec{Float64}` that are updated every tracker step via `update!` to track the solution's scale. The norm itself is always the infinity norm — `inf_norm` uses `fast_abs` (which avoids `sqrt` via `abs2`, with an `isinf` fallback for overflow safety).

`utils.jl` contains `SegmentStepper`, a mutable struct for adaptive stepping along a complex line segment, and `fast_abs`.

---

## Model Kit (`src/model_kit/`)

The model kit is responsible for evaluating polynomial systems. Given a set of polynomials, it produces an `Interpreter` that can compute function values, Jacobians, and Taylor coefficients — all from the same compiled instruction tape, all without allocations.

The design is interpreter-first. HC v2 offers a compiled mode that generates a unique function per system via RuntimeGeneratedFunctions, giving ~5% faster execution at the cost of 28s TTFX. We use a tape-based interpreter instead: the execution loop is a `@generated` if-elseif chain over opcodes that compiles once and handles every polynomial system through the same code path. The ~5% gap disappears once O(n²) Jacobian evaluation and O(n³) LU factorization dominate, which they do for any system with five or more variables.

The compilation pipeline has four stages. First, `poly_to_sexpr` in `sexpr.jl` converts each polynomial into an S-expression tree — `SAdd`, `SMul`, `SPow`, `SConst`, `SVar`, `SParam` — mirroring SymEngine's canonical forms. If the Jacobian is requested, `MP.differentiate` produces the derivative polynomials and they enter the same representation. All compound types cache their hash at construction for fast Dict/Set operations throughout the pipeline.

Second, the combined expressions pass through `cse` in `cse.jl`, a direct port of SymEngine's two-phase CSE. The SExpr representation exists specifically to enable this: because our trees have the same canonical structure as SymEngine's, the algorithm ports one-to-one. Phase 1 (`opt_cse`) is the critical one — a `FuncArgTracker` value-numbers every argument of every Add/Mul node, then `match_common_args!` factors shared arguments into unevaluated placeholders. Skipping this phase causes 30-65% instruction count regressions even on small systems. Phase 2 (`tree_cse`) is simpler: mark subexpressions seen twice, replace with temporaries.

Third, the tape compiler in `tape_compiler.jl` converts CSE output into `Instruction` values. This is where instruction fusion happens. The compiler doesn't just emit `ADD` and `MUL` — it selects fused ops like `MULADD(a*b+c)`, `MULMULSUB(a*b-c*d)`, `MUL3`, `ADD4`. The fusion logic in `_compile_sum!` splits terms into positive and negative groups and picks the tightest op for each case. Multiplication chains and addition chains are both reduced 4/3/2-at-a-time by the same `_tree_reduce!` helper. Constants like 1, -1, and 2 are recognized so their multiplications can be elided or replaced with cheaper ops. After compilation, `_optimize_instruction_order` does a DAG-based topological sort for data locality, and `_reduce_space` runs linear-scan register allocation to minimize tape size. The 25 opcodes (arity 0-4, defined in `operations.jl`) cover all arithmetic patterns that appear in polynomial Jacobians, with Karatsuba-optimized complex specializations for `op_sqr` and `op_cb`.

Fourth, the `Interpreter{V}` in `interpreter.jl` wraps the resulting `InstructionSequence` and a pre-allocated tape. The same instruction sequence drives all element types: `Vector{ComplexF64}` for standard evaluation, `Vector{ComplexDF64}` for extended precision, and `Vector{TTS{N,ComplexF64}}` for Taylor coefficients. This works because every `op_*` and `taylor_op_*` dispatches on its argument type — no special-casing, no mode flags. The Taylor functions in `taylor.jl` are `@generated` to completely unroll for compile-time N: the Cauchy product, quotient rule, logarithmic differentiation, and coupled sin/cos recurrence all produce straight-line code with no runtime loops. The fused variants (muladd/mulsub/submul) share a single `_cauchy_product_exprs` code-gen helper.

`polynomial_input.jl` orchestrates the pipeline. Its `build_interpreter`, `build_jacobian_interpreter`, `build_taylor_interpreter`, and `build_df64_interpreter` functions are internal — the user-facing API will be `System(polys)` which wraps the resulting interpreters into a `SystemEvaluator` via FunctionWrapper.

| File | Role |
|------|------|
| `operations.jl` | `OpType` enum (25 ops), `@inline` scalar `op_*` with Karatsuba complex specializations |
| `taylor.jl` | `TTS{N,T}`, `TaylorVector{N,T}`, `@generated` Taylor recurrences |
| `sexpr.jl` | SExpr type hierarchy, hash/equality, canonicalization, `poly_to_sexpr` |
| `cse.jl` | SymEngine CSE port: `FuncArgTracker`, `opt_cse`, `tree_cse` |
| `tape_compiler.jl` | `TapeCompiler`, instruction fusion, `_tree_reduce!`, `compile_to_instructions` |
| `instruction_sequence.jl` | `Instruction`/`InstructionSequence`, DAG reorder, register allocator |
| `interpreter.jl` | `Interpreter{V}`, `@generated` dispatch loop, `execute!`/`execute_taylor!` |
| `polynomial_input.jl` | Pipeline orchestration: MP polynomials → SExpr → CSE → compile → Interpreter |

---

## Deviations from HomotopyContinuation.jl v2

This is a rewrite, not a refactor. The algorithms are the same — same predictor, same Newton corrector, same endgame, same CSE — but the engineering is different. Every change serves the same goal: eliminate the 28-second TTFX while keeping runtime performance within 5%.

The root cause of v2's TTFX is type proliferation. The tracker is parameterized on the homotopy type (`Tracker{H}`), which is parameterized on the system type, which in compiled mode is a unique type per polynomial system generated via `RuntimeGeneratedFunctions`. A new system means recompiling the entire tracking pipeline. Arrays use StaticArrays where size is a type parameter, so changing system size triggers yet more recompilation.

We break this chain at every level. Systems and homotopies are wrapped into `SystemEvaluator` / `HomotopyEvaluator` using `FunctionWrapper`, making the tracker monomorphic — it compiles once during precompilation and handles any system through the same code path. Arrays use FixedSizeArrays where size is a runtime value, so `FSVec{ComplexF64}` is the same concrete type for a 4-variable system and a 40-variable system. Compiled evaluation mode is dropped entirely: the interpreter is within 5% of compiled performance and introduces no per-system types. The `@generated` dispatch loop compiles once and is system-independent.

The symbolic backend changes too. v2 uses SymEngine, a C++ library via FFI, for differentiation, CSE, and expression canonicalization. This causes TTFX issues (FFI initialization, non-precompilable state) and was the source of bug #643. We use DynamicPolynomials for input, `MP.differentiate` for symbolic Jacobians, and a pure-Julia port of SymEngine's CSE algorithm. Same algorithm, no FFI, fully precompilable.

Smaller changes follow the same philosophy. Return codes use `@enumx` scoped enums instead of `Symbol` — type-safe and faster. `PathResult` is immutable. LoopVectorization is gone (caused ~5,000 invalidations for marginal benefit in the toric homotopy). StructArrays for Jacobian storage was benchmarked and found slower than plain `FSMat`.
