# Fast runtime and fast first call: report and plan

Measured 2026-10-19 on Julia 1.13.1, against scratch copies of main 6486b87d. Fresh-process interleaved A/B, minimum of samples. The machine was shared, so differences under ~3% are noise.

## Conclusion

No rewrite brings the first call down by more than a few percent unless the compiled code is cached. Caching it brings every workload to 0.05-0.3 s with unchanged runtime. So "low first call by design" means a closed world of package-owned specializations, a small code volume, and the precompile workload as the last step.

## Why the first call is large

- It is almost all compilation, spread thin. Total degree infers 5983 MethodInstances (3885 of them Base generics on our types). Decomposition infers 16383 (11450 Base). That is roughly 1.2-1.5 ms each, inference plus LLVM. The largest single item is 0.15 s.
- Type stability does not reduce this. Stable code compiles its whole call graph in every new session. Only code compiled during precompilation is cached.
- Cutting volume alone has a low ceiling. A sub-second uncached first call would need about 10x less code than a complete solver needs.
- Inference-time attribution overstates wall savings. Only fresh-process A/B is trustworthy.

## Prototypes

| Prototype | First call | Runtime |
|---|---|---|
| In-package `@compile_workload` over `workloads/` | total degree 0.054 s (main 9.0), polyhedral 0.10 s (16), decomposition 0.105 s (30), monodromy 0.056 s (13.6) | same |
| Same build, user scenarios outside the corpus (Expression input, ComplexF64 coefficients, new monodromy system, twisted-cubic decomposition, parameter homotopy, katsura4 default) | 0.002-0.29 s (main 5-10 s) | same |
| Same build with all 13 `Base.inferencebarrier`s removed | 0.06-0.17 s | same |
| SExpr/CSE as a plain integer node arena (no Moshi, package-owned sort) | -0 to 5% per workload; first `System` 6.3 -> 5.6 s (DP), 4.85 -> 4.34 s (Expression) | tapes bit-identical on 10 systems |
| Monodromy compiled for 2 of its 4 variants (timing bound only) | decomposition -6%, subspace trace -7% | same |
| `@noinline` on the interpreter loops | -1 to 2% | same |

Uncached A/B, arena build vs main (s):

| Workload | main | arena |
|---|---|---|
| total_degree | 9.01 | 9.00 |
| expression_frontend | 10.21 | 9.71 |
| polyhedral | 16.33 | 16.41 |
| witness_set_build | 12.78 | 12.30 |
| regeneration | 25.31 | 24.64 |
| decomposition | 30.19 | 29.64 |
| monodromy | 13.63 | 12.93 |

Runtime (interleaved, three rounds, min): katsura5, katsura6 and cyclic6 agree within noise across main, the cached build and the arena build.

Cost of the cache: precompile 8.8 -> 59 s, image 14 -> 87 MB, `using` 0.69 -> 0.92 s. Without the barriers: 65 s and 90.6 MB.

Where the cache still misses: almost entirely DynamicPolynomials' own arithmetic. Our leaks are `stable_sort!`/`unique!` called on user variables (60 + 18 ms), plus the default `solve(F)` route (threaded, with progress, 93 ms), which the corpus does not cover.

## Findings worth acting on regardless

- CSE stays. With direct lowering everywhere, Jacobians are 1.6-2.3x slower and solves 12-45% slower.
- CSE output depends on the package build. Base's `Enum` hash includes the type identity, so the order of unary and function-symbol nodes changes between builds. Hashing the enum's integer value fixes it.
- The Jacobian kernel has two speeds. On cyclic7 it runs at about 345 or about 750 ns depending on buffer layout. Main shows this too, so it is a runtime opportunity of its own.

## Plan

### Phase A: close the world

Caching becomes complete; the uncached gain is small.

1. Make `System` non-parametric. Convert user input once, at the boundary, into package-owned polynomial data, so derived systems in slicing and regeneration stop running DynamicPolynomials arithmetic. This removes the Float64/ComplexF64 `System{P}` duplicates and the sort leaks.
2. Erase the monodromy orchestration over the homotopy type and the intrinsic/projective flags (`MonodromySolver{H,P,B,UP,MO,CS}`, `MonodromyWorkerState{H,P}`, `SubspaceMonodromyBuilder{S,INTRINSIC,PROJECTIVE}`). The work is per path, not per step, so runtime is unaffected. Measured bound: 6-10%.
3. Hash enums by their integer value.

### Phase B: reduce volume

Each item is 0-5% uncached and shrinks the image.

4. Land the arena SExpr/CSE (prototype done, tapes identical).
5. One package-owned stable sort replacing the ~25 Base `sort`/`sortperm` call sites. Each distinct comparator recompiles the sort machinery, about 1.8 s of decomposition inference.
6. Setup-code discipline: few Dict key/value types (each costs about 0.04 s); no keyword arguments, Printf or broadcasting in internal setup.
7. Deduplicate the Taylor and interpreter specializations (`execute_taylor!` 6 MIs, `execute_taylor_instructions!` 3).

### Phase C: workload

8. An in-package `@compile_workload` over the existing `workloads/` corpus, plus the default `solve(F)` route.

### Phase D: delete the barriers

9. Remove the `inferencebarrier`s, the `@nospecialize` dispatchers and their entries in `test/quality/exemptions.jl`. The measurements above show the cache makes them unnecessary.

### Separately

- The two-speed Jacobian kernel.
- DynamicPolynomials' own first-call cost, upstream or by fork.
- Image size and the +0.22 s load.

Expected outcome: A and B alone cut the uncached first call by roughly 15-30%. The 30 s -> 0.1 s comes from Phase C.

## Implementation status

Branch `ttfx-first-call`, uncommitted.

| Item | State |
|---|---|
| 1. Non-parametric `System` | Done. `System` stores `Expression`s; DynamicPolynomials input is converted once at construction. `polynomials`, `variables` and `parameters` return `Expression`s (API change, approved). Slicing, regeneration, completeness checks and the parameter start pair run on `Expression`. The DP lowering pipeline (`symbolic_polynomial_compiler.jl`, `poly_to_sexpr`) is deleted; a monomial-sum lowering in `expression_compiler.jl` keeps tapes bit-identical to main with the enum fix on all 10 reference systems. |
| 2. Monodromy erasure | Done. `MonodromySolver{P,B,UP,MO,CS}`, `MonodromyWorkerState{P}` holding the homotopy as data, one `MonodromyBuilder{S,P}`. |
| 3. Enum hashing | Done. Enums hash as `Int8`. |
| 4. Arena SExpr/CSE | Done. Tapes bit-identical on 10 systems. |
| 5. Package-owned stable sort | Done. All `Float64` and comparator sorts in setup go through `stable_sort!`/`stable_sortperm`. |
| 6. Setup discipline | Done. Int32-keyed dicts, `append_unique!`; every internal keyword function is positional (tracker and endgame `init!`/`track!`, witness-set, regeneration, monodromy and progress helpers). Printf appears only in display code. |
| 7. Taylor/interpreter dedup | Done. `execute_taylor_instructions!` is `@noinline`, so each Taylor order has one native copy instead of one per caller. `execute_taylor!` keeps two MIs per order because its parameter input differs (scalars or Taylor series). The eval/Jacobian loop `execute_instructions!` stays inlined: making it `@noinline` measured about 1% slower on katsura5/6. |
| 8. Workload | Done. `@compile_workload` over `workloads/`; `:default_solve` builds `System(polys)` without keywords, so variable discovery is cached too. |
| 9. Barriers | Done. No `inferencebarrier`, no `@nospecialize` dispatcher. The lazy DF64 installer is replaced by eager wrappers, removing a FunctionWrapper nested inside a FunctionWrapper. |

Retained on purpose: the `RefValue{Any}` boxes (they keep `SolveCache{E}` and `SystemEvaluator` monomorphic and avoid the Julia 1.10 `cfunction` recursion), and `@nospecialize` on user input in `System` construction (data boundary).

Exemption table: 20 entries removed, 4 added. Removed are the dispatcher and barrier entries, `df64_evaluators`, `decompose_with_solver`, and every DynamicPolynomials site (`monomial_key`, `polynomial_instruction_compiler`, `symbol_index_map`, `support_coefficients`, `fresh_variable_name`, `hypersurface_witness_set`, `build_verification_system`). Added are dispatch the deleted barriers used to hide: `make_monodromy_progress`/`update_progress!` (ProgressMeter), `stage_equations` (FunctionWrapper), and `polynomial_variable`, the single place a user variable is named through its own `Symbol`.

Gates: core 6780/6780, certification 615/615, JET (package and certification) clean, StrictMode clean, Aqua, ExplicitImports, concrete types, inference, allocations and instruction count pass, Runic clean.

First call in a fresh process (seconds):

| Workload | main | branch |
|---|---|---|
| total_degree | 6.79 | 0.002 |
| polyhedral | 13.38 | 0.002 |
| decomposition | 10.87 | 0.046 |
| monodromy_serial | 8.90 | 0.007 |
| witness_set_build | 8.89 | 0.009 |
| regeneration | 9.98 | 0.014 |
| default `solve(F)` | 7.93 | 0.007 |

Off the workload corpus, user code the cache never saw:

| Scenario | main | branch |
|---|---|---|
| katsura4, default `solve` | 10.16 | 0.034 |
| `@var` expression input | 5.21 | 0.002 |
| integer coefficients, polyhedral | 15.96 | 0.072 |
| complex coefficients | 6.84 | 0.258 |
| rational coefficients | 7.24 | 0.275 |
| parameter homotopy | 10.37 | 0.005 |
| new monodromy system | 8.91 | 0.008 |
| twisted cubic decomposition | 9.67 | 0.017 |

The complex and rational cases compile no HomotopyContinuation method; their remaining cost is DynamicPolynomials arithmetic on the new coefficient type in the user's own expressions.

Runtime, `solve!` minimum over interleaved fresh processes (ms): katsura5 8.42 main, 8.44 branch; katsura6 23.71 main, 23.84 branch. The gap is inside the run-to-run spread on this machine. Tracking tapes are bit-identical to main with the enum-hash fix.

Cost: load 0.87 s (main 0.68 s), pkgimage 73 MB (main 14 MB), precompile about 51 s.

## Prototype diffs

In `/tmp/claude-1000/-home-oameye-Projects-HomotopyContinuation-jl/dec75d71-02e4-4c63-af30-1f1f0fa118b4/scratchpad/proto/patches/`:

- `lowering_arena_ir.diff` (arena SExpr/CSE, package-owned sort)
- `precompile_workload.diff` (in-package workload)
- `no_inferencebarrier.diff` (barrier removal)
