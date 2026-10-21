# Durable implementation decisions

This file records non-obvious constraints of the current implementation. Historical experiments, benchmark snapshots, migration notes, and future redesigns belong in Git history, CI artifacts, or roadmap issues.

## Runtime-sized buffers stay runtime-sized

System dimension is problem data, not compiler state. Hot buffers use the package aliases `FSVec{T}` and `FSMat{T}` rather than StaticArrays-style dimension parameters.

On Julia 1.10 these aliases intentionally resolve to ordinary `Vector`/`Matrix`. On Julia 1.11+ they use the concrete FixedSizeArrays defaults backed by `Memory`. In either case struct fields must be concrete; `FixedSizeVector{T}`/`FixedSizeMatrix{T}` with a free memory parameter are not valid concrete field types.

## System-specific callable types stop at the evaluator boundary

`SystemEvaluator` and `HomotopyEvaluator` are deliberate specialization firewalls. A polynomial/expression's source type or RuntimeGeneratedFunction identity must not force recompilation of the whole tracker/endgame/solver stack.

Function-wrapper indirection is therefore not automatically a defect. A dynamic boundary is acceptable when it is explicit, cold or sufficiently coarse-grained, and wins globally across runtime, first-use latency, specialization count, and native-code size. Hot package-owned kernels should still use direct concrete calls whenever measurement supports them.

The evaluator clone factory is intentionally not a zero-argument `FunctionWrapper` returning its owning evaluator type: on Julia 1.10 that creates a code-generation recursion through the containing type. Cloning is a cold worker-construction operation, so its narrow erased boundary is preferable to pushing system identity through solver types.

## Generated evaluation is an evaluator concern

RuntimeGeneratedFunctions implement the `COMPILED` and `COMPILED_ALL` evaluator modes. They may generate system-specific machine code, but downstream tracking state must remain bounded and package-owned. Adding a compiled backend must never imply specializing `Tracker` on each user system.

## Finite modes are semantic values

Finite algorithm/status choices use scoped `EnumX` enums. `Val` is reserved for small internal compile-time dispatch where measurement justifies it; it should not leak into the domain API. Booleans and Symbols should not encode a semantic state machine when a named enum is clearer.

Some finite choices are genuinely dynamic. For example a predictor or endgame can change strategy while tracking a path. Such state belongs in runtime fields even if the set of possible values is finite.

## Worker state owns mutable numerical state

Interpreter tapes, tracker buffers, predictor/Newton workspaces, and endgame state are mutable. Thread/process safety comes from constructing independent worker state, not from sharing or deep-copying live hot-path state between tasks.

Immutable instruction sequences and other data-only setup products may be shared. A worker rebuilds the mutable machinery it owns from those inputs.

## Distributed transport sends data, not process-local call pointers

Evaluator wrappers can contain process-local function pointers/closures. Distributed execution therefore serializes stable problem/lowering data and reconstructs evaluator/worker machinery on the receiving process. Serialization methods should be defined around durable data representations rather than mirroring every field of a large runtime struct positionally.

## Randomness is local and explicit

Solver routes use explicit local RNGs derived from a `UInt32` seed rather than mutating Julia's global RNG. A fixed seed must reproduce route semantics independently of the caller's global RNG state and execution mode. Parallel work gets independent derived streams rather than sharing one `MersenneTwister` across tasks.

## Certification remains a separate package

Rigorous certification depends on Arblib and defines certificate types containing Arb/Acb values. Keeping it in `lib/HomotopyContinuationCertification` prevents ordinary solving from loading that dependency and gives certification its own precision/memory contracts. It cannot be reduced to a lightweight extension while those public certificate types own Arblib values.

## Type stability is proven statically

StrictMode proves every specialization the workloads compile typestable, and the inference and concrete-layout contracts in `test/quality/` cover public constructors and result types. The package carries no runtime instrumentation such as DispatchDoctor's `@stable`: rerunning the whole suite instrumented multiplied its runtime several times over. No single analyzer is treated as the definition of quality: JET, StrictMode, concrete-layout checks, allocation checks, numerical tests, and performance/TTFX measurements cover different failure modes.

## First calls run cached native code

Nearly all of a cold first call is compiling the thousands of specializations a solve reaches, each cheap and none dominant, so trimming code moves it by a few percent. The package therefore compiles a few routes, written in `src/precompile.jl` the way a user calls them, inside a PrecompileTools `@compile_workload`, and the package image carries their native code. The file is separate from `workloads/`, which stays the analysis corpus. It calls `precompile` on each route rather than running it: running JIT-compiles the code and the image writer then compiles it again, so inference alone gives the same image in about a quarter less time. Inference stops where a type is erased, at the `RefValue{Any}` boxes and at `FunctionWrapper` targets, so every erased callable is first called where its type is still known: `solve_cache` builds the first worker from the concrete builder, not from the erased one. The evaluator cloners behind `SystemFactory` stay uncached; each costs a few milliseconds because it calls constructors the `System` already compiled. The routes are the default solve, polyhedral, parameter homotopy and monodromy. Compiling the whole corpus instead would add about 11 s of precompilation and 24 MB of image, to save up to about 1.5 s on the first call of a geometry route. The cache covers user input outside these routes too, because the solve path below the evaluator boundary specializes only on package-owned types and bounded package choices. `System` converts DynamicPolynomials input to `Expression` at construction, so a new coefficient or variable type compiles only the conversion. What still compiles for new input is the part keyed on user types: that conversion, and user callbacks, samplers and group actions in `MonodromyOptions`. A route the file misses compiles only its own part on first use. `test/quality/` analyses every workload in the corpus. Setting PrecompileTools' `precompile_workload` preference to `false` for the package skips the workload, which shortens the edit and reload loop during development.

Setup code therefore calls its routes directly. The remaining dynamic boundaries are `FunctionWrapper` construction around evaluators and `RefValue{Any}` factory boxes. `test/quality/exemptions.jl` lists every package call site allowed to dispatch at runtime, with its reason; a dispatch inside a dependency is charged to the package line that calls it. Besides the boundaries, the list holds the hot calls through a `FunctionWrapper`, whose cold `reinit_wrapper` branch dispatches, and setup code that formats messages, names user variables or reports progress.

`test/quality/jet.jl` analyses every public workload and every specialization those workloads compile, which includes the tracking code behind the boundaries where analysis from the entry point stops. It fails on errors, on runtime dispatch charged to a site outside the list, and on an entry no longer reached, so the list cannot go stale. StrictMode checks the same specializations and accepts a `:typestable` failure only when all of its dispatch is charged to the list. Every entry is a candidate for removal.

## CSE and tape compilation are shared compiler infrastructure

Both frontends converge on `Expression` and then on the same canonical S-expression/CSE/tape pipeline. Avoid frontend-specific evaluator stacks when the shared IR can express the operation. Instruction count, tape size, lowering cost, evaluator runtime, and first-use compilation are all relevant compiler metrics.

## Comments describe the current code

Permanent comments and documentation explain numerical or architectural reasons that remain true. Development history, abandoned designs, old benchmark numbers, and line-number-specific plans belong in Git history/issues. This keeps the repository readable and prevents historical implementation details from masquerading as current invariants.
