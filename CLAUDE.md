# CLAUDE.md — HomotopyContinuationNext.jl

## What is this?

A ground-up rewrite of HomotopyContinuation.jl for solving polynomial systems via homotopy continuation. The design prioritizes type stability, minimal TTFX, and zero runtime dispatch on hot paths.

## Architecture

Read `implementation_docs/00_architecture.md` for the full architecture, `implementation_docs/01_decisions.md` for pitfalls and design choices, and `implementation_docs/02_status.md` for feature/performance status.

Key design decisions:
- **Interpreter-first**: tape-based evaluator handles eval, jacobian, Taylor, DF64 — no Symbolics.jl in core
- **FunctionWrapper type firewall**: `SystemEvaluator`/`HomotopyEvaluator` wrap any system into a single concrete type — the tracker is monomorphic
- **FixedSizeArrays**: all pre-allocated scratch buffers use `FSVec`/`FSMat` (size is runtime, not a type parameter). **CRITICAL:** `FixedSizeVector{T}` is NOT concrete — the `Mem` parameter is free. Use `FixedSizeArray{T,N,Memory{T}}` for struct fields (see type aliases in main module).
- **Moshi ADTs**: `SExpr` and `ExecInstruction` use Moshi.jl `@data` for tagged unions — all variants are one concrete type, eliminating dynamic dispatch in CSE and interpreter
- **DynamicPolynomials input**: users provide polynomials via `@polyvar`, internal pipeline uses `MP.differentiate` for Jacobian
- **Immutable by default**: mutable structs require justification, use `const` fields for buffer references

## Package layout

```
src/HomotopyContinuationNext.jl     # Main module (core, Arblib-free)
src/primitives/                      # DoubleF64, norms, linear algebra
src/model_kit/                       # SExpr, CSE, tape compiler, interpreter, Taylor
src/core/                            # AbstractSystem/Homotopy, SystemEvaluator, homotopy types
src/tracking/                        # Predictor, Newton, Tracker
src/solving/                         # solve(), total degree, polyhedral, result types
src/utils.jl                         # SegmentStepper, _stable_sort!, fast_abs, etc.

lib/HomotopyContinuationNextCertification/   # Certification subpackage
  src/interval_arithmetic.jl, interval_arblib.jl   # Interval / IComplexF64 + Arb bridge
  src/acb_interpreter.jl                            # arbitrary-precision tape interpreter
  src/certification.jl, certification_arb.jl        # certify(), Krawczyk + Arb fallback
```

**Certification is a separate subpackage.** It depends on Arblib (a heavy binary
dep), so keeping it out of core is what makes core TTFX minimal (core load
dropped from ~1.6s to ~0.77s). Every certificate type embeds an `AcbMatrix`, so
a package extension is not possible (extensions cannot define/export types); a
subpackage is the correct isolation. To certify:

```julia
using HomotopyContinuationNext, HomotopyContinuationNextCertification
certify(F, solutions)
```

## Git policy

**Never commit or push.** Neither Claude nor any subagent may run `git commit`, `git push`, or any git command that modifies history. All commits are made by the user. Claude's job is to write code, run tests, and report results — the user decides when to commit.

## Development workflow

All common tasks go through the Makefile:

```sh
make test          # run core tests in parallel (ParallelTestRunner, 10 jobs), then the certification subpackage
make test-cert     # run only the certification subpackage suite (threaded)
make test-serial   # run all tests serially (for debugging)
make benchmark     # run TTFX + steady-state benchmarks
make compare       # compare primitives against HomotopyContinuation v2
make format        # format all Julia files with Runic
make deps          # instantiate all environments
make update        # update all environments
make help          # show all available targets
```

### Test structure

Tests run via ParallelTestRunner — each file is self-contained and runs in its own worker:

- `test/aqua_test.jl` — Aqua.jl: unbound args, undefined exports, stale deps, compat, piracy
- `test/jet_test.jl` — JET.jl: `report_package` for type error and optimization analysis
- `test/explicit_imports_test.jl` — ExplicitImports.jl: no implicit imports, no stale imports, qualified access

### Quick debugging with Julia MCP

Use the `julia-mcp` MCP server (tools: `julia_eval`, `julia_list_sessions`, `julia_restart`) for quick debugging and testing small snippets — e.g., checking a type, evaluating an expression, or verifying a method signature. Prefer this over spinning up a full test run when you just need a quick answer.

### Formatting

Code is formatted with [Runic.jl](https://github.com/fredrikekre/Runic.jl) (available as `runic` CLI):

```sh
make format
```

### Quality gates

Before merging any PR:
1. `make test` passes (all 3 quality test suites + any unit tests)
2. JET reports zero issues on the package
3. TTFX benchmark: first `solve(F)` < 5s
4. No `Any`-typed fields in any struct
5. Every `mutable struct` has documented justification and `const` on fixed fields

## Coding rules

### Function signatures

- **Use the most restrictive signature type possible.** This lets JET catch unintended errors. When prototyping it's fine to start loose, but committed code should have tight type declarations. When AI agents suggest code, make sure argument types are clearly specified. When in doubt, use the most restrictive type you can think of.
- **Explicit `;` for keyword arguments.** Always use an explicit semicolon before keyword arguments for clarity:
  ```julia
  # Good
  Position(; line = i - 1, character = m.match.offset - 1)
  # Bad
  Position(line = i - 1, character = m.match.offset - 1)
  ```

### Type system

- **No abstract-typed fields on hot paths.** Every struct field must be concretely typed.
- **`const` on buffer fields in mutable structs.** If a field holds a pre-allocated buffer (FSVec, FSMat, TaylorVector) that is never reassigned, mark it `const`.
- **`RefValue` for cache scalars in immutable structs.** Homotopy types are immutable; use `Base.RefValue{T}` for cached values that need mutation.
- **`NTuple{N,T}` for small fixed-size collections.** When the count is known at compile time and small (e.g., `tx_norm::NTuple{4,Float64}`).
- **Enums over Symbols.** Use `EnumX.@enumx` for return codes and state machine states — scoped (`MyEnum.Value`), type-safe, faster than Symbol comparison.
- **`FSVec{T}` / `FSMat{T}` for pre-allocated buffers.** Defined as `FixedSizeArray{T,1,Memory{T}}` / `FixedSizeArray{T,2,Memory{T}}` — same concrete type regardless of size, cannot be resized. **WARNING:** `FixedSizeVector{T}` and `FixedSizeMatrix{T}` are NOT concrete types (the `Mem` parameter is free). Always use `FSVec{T}` / `FSMat{T}` from the main module for struct fields, never `FixedSizeVector{T}` directly.
- **`AbstractVector` / `AbstractMatrix` only where truly needed.** Use them in the `AbstractSystem`/`AbstractHomotopy` interface contracts (so users don't need to import FixedSizeArrays) and in public `execute!` methods that must accept both `Vector` and `FSVec`. Prefer concrete types everywhere else.
- **Moshi `@data` for tagged unions.** Use `variant_storage(expr)` for pattern dispatch, never `isa` on the ADT module variants directly. Access the concrete type via `typeof(Module.Variant(...))` alias (e.g., `SExprT`, `ExecInstructionT`).

### Performance

For full reference, see the `julia-perf` skill (`.claude/skills/julia-perf/`) and the `julia-ttfx` skill (`.claude/skills/julia-ttfx/`) for TTFX/invalidation diagnosis.

- **Zero allocations on hot paths.** The tracker step, Newton corrector, and predictor must not allocate. Pre-allocate all buffers at construction time and mutate in-place via `!` functions.
- **Column-major access.** First index varies fastest. Inner loops over `i` (rows), outer loops over `j` (columns): `for j in 1:n, i in 1:m`.
- **No kwargs in hot paths.** Keyword arguments prevent specialization and can allocate. Expose kwargs at the API boundary (`solve(F; tol=1e-8)`), forward to positional-arg inner functions (`_solve(F, tol)`).
- **No kwargs splatting.** Never forward `kwargs...` — it blocks inference. Explicitly name and forward each keyword.
- **Fuse broadcasts.** Use `@.` or dot syntax to avoid temporary arrays. Use in-place fused assignment: `y .= @. 3x^2 + 4x`.
- **`@views` for slices.** Array slicing copies; use `@view` or `@views` to avoid allocation.
- **`@inbounds` with `eachindex`.** Use `@inbounds` only when indices are provably valid. Prefer `eachindex(x)` over `1:length(x)`.
- **`@fastmath` where safe.** Acceptable in custom LU pivot selection, norm computation, and other places where IEEE edge cases (inf/nan) are handled separately. Never in certification or interval arithmetic.
- **`abs2(z)` over `abs(z)^2`.** Avoids intermediate allocation for complex numbers. Similarly use `fld`, `cld`, `div` over `floor(x/y)` etc.
- **Avoid string interpolation in I/O.** Use `println(file, a, " ", b)` not `println(file, "$a $b")`.

### Imports and style

- **No `using X` without explicit imports.** Use `using X: func1, func2` or `import X`. ExplicitImports.jl enforces this.
- **Format with Runic.** Run `make format` before committing.

## Dependencies

| Package | Purpose |
|---------|---------|
| CommonSolve | `init`/`solve!` interface |
| DynamicPolynomials | `@polyvar`, concrete polynomial types |
| EnumX | Scoped enums (TrackerCode, PathResultCode, etc.) |
| FixedSizeArrays | Non-resizable vectors/matrices (size not in type parameter) |
| FunctionWrappers | Type-stable function erasure for SystemEvaluator/HomotopyEvaluator |
| LinearAlgebra | stdlib |
| MixedSubdivisions | BKK mixed volume computation for polyhedral start system |
| Moshi | `@data` tagged unions for SExpr and ExecInstruction |
| MultivariatePolynomials | Abstract polynomial interface, differentiation, exponent access |
