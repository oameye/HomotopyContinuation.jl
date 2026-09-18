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

## Instrumentation and allocation measurement are separate contracts

DispatchDoctor's `@stable` instrumentation introduces wrappers/closures and changes allocation behavior. The strict environment therefore uses instrumentation to prove return/type-stability behavior, while production allocation assertions are measured without that instrumentation. A test may still execute the instrumented operation; it should not interpret instrumentation allocations as package allocations.

StrictMode provides a complementary compiled-code proof. No single analyzer is treated as the definition of quality: JET, DispatchDoctor, StrictMode, concrete-layout checks, allocation checks, numerical tests, and performance/TTFX measurements cover different failure modes.

## CSE and tape compilation are shared compiler infrastructure

Both polynomial and analytic-expression frontends converge on the same canonical S-expression/CSE/tape pipeline. Avoid frontend-specific evaluator stacks when the shared IR can express the operation. Instruction count, tape size, lowering cost, evaluator runtime, and first-use compilation are all relevant compiler metrics.

## Comments describe the current code

Permanent comments and documentation explain numerical or architectural reasons that remain true. Development history, abandoned designs, old benchmark numbers, and line-number-specific plans belong in Git history/issues. This keeps the repository readable and prevents historical implementation details from masquerading as current invariants.
