# Architecture

This document describes the current implementation. Planned redesigns belong in GitHub issues so current-state documentation does not become a second roadmap.

## Design rule

**Problem data stays data.** System identity, coefficients, dimensions, path counts, seeds, tolerances, and other unbounded inputs must not propagate through the numerical stack as type parameters. Static specialization is reserved for a bounded family of package-owned algorithmic choices where it produces a measured benefit.

The corresponding hot-path goals are concrete layouts, predictable dispatch, no boxing, and zero steady-state heap allocation where the numerical contract requires it.

## Frontends and lowering

Two user-facing symbolic frontends converge on the same compiler IR:

```text
DynamicPolynomials             Expression
       │                           │
       ├─ polynomial lowering      ├─ expression lowering
       │                           │
       └──────────────┬────────────┘
                      ↓
                    SExpr
                      ↓
             canonicalization + CSE
                      ↓
              InstructionSequence
```

`SExpr` and execution instructions are Moshi tagged unions with concrete storage. Jacobians are differentiated symbolically before lowering. The tape compiler performs instruction selection/fusion, ordering, and register allocation once during setup.

## Evaluation backends

An `InstructionSequence` is data shared by the available evaluator modes:

- `CompileMode.INTERPRETED`: evaluation, Jacobian, and Taylor coefficients execute through the tape interpreter;
- `CompileMode.COMPILED`: evaluation and Jacobian use RuntimeGeneratedFunctions; Taylor remains interpreted;
- `CompileMode.COMPILED_ALL`: evaluation, Jacobian, and Taylor kernels use generated functions.

The generated function is a kernel implementation detail. System-specific generated callable types must not propagate into the tracker/solver type graph.

`SystemEvaluator` is the stable numerical interface around a system evaluator. `HomotopyEvaluator` provides the analogous boundary for homotopies. Both expose concrete package-owned call signatures to tracking even when setup originated from different symbolic representations or evaluator modes.

## Systems and homotopies

`System` owns the source representation and metadata needed by public introspection/setup operations together with its numerical evaluator. Composition, fixed-parameter systems, randomized systems, affine charts, sliced systems, and the supported homotopy types adapt that data into the same evaluator contracts.

`FSVec{T}` and `FSMat{T}` are the package's concrete runtime-sized buffer aliases. Their length/shape is deliberately not part of the Julia type.

## Tracking kernel

The tracking stack is package-owned numerical state:

```text
HomotopyEvaluator
       ↓
    Tracker
   ├─ Predictor
   ├─ NewtonCorrector
   ├─ Jacobian / MatrixWorkspace
   └─ WeightedNorm
       ↓
 EndgameTracker
   ├─ Valuation
   ├─ singular endgame
   └─ infinity detection
```

Mutable buffers are allocated at construction and reused. Predictor/Newton/endgame strategy may change during a path; genuinely dynamic finite state remains runtime state rather than being forced into type parameters.

The linear-algebra layer owns LU/QR workspaces, condition estimation, scaling, and extended-precision refinement. Numerical primitives are independent of the symbolic frontend.

## Solver orchestration and execution

Public `solve`/`CommonSolve.init` methods normalize a problem into route-specific setup state. Builders construct fresh worker state containing an evaluator and tracker. A worker owns all mutable numerical scratch storage.

`Serial`, `Threaded`, and `DistributedExecutor` determine where path work runs. Threaded execution creates worker-local state per task. The Distributed extension transports serialization-safe problem data and reconstructs process-local numerical machinery rather than treating raw function-pointer-bearing evaluator wrappers as wire data.

Higher-level algorithms—parameter sweeps, subspace continuation, monodromy, witness sets, regeneration, and numerical irreducible decomposition—reuse the same path-tracking machinery while owning their algorithm-specific orchestration/state.

## Results

Tracking produces immutable `PathResult` records. Solver routes assemble these into `Result` or route-specific result types and apply clustering, multiplicity, filtering, trace/completeness, or witness-set semantics above the tracker. Numerical result/status enums are explicit semantic values rather than Symbols.

## Certification

Rigorous certification is a separate package in `lib/HomotopyContinuationCertification`. It owns interval/Arb evaluation, arbitrary-precision tape execution, Krawczyk-style certification, and certified higher-level workflows.

The split is architectural: certification types contain Arblib values, while normal solving should not load Arblib or inherit its load-time/compiler footprint.

## Dependency direction

The intended dependency direction is:

```text
primitives
   ↑
model_kit
   ↑
core
   ↑
tracking
   ↑
solving
```

Optional extensions and the certification package may depend on the public/internal contracts they require, but the numerical tracking core must not depend on orchestration or optional integration layers.

When changing architecture, preserve this separation and keep policy at the highest layer that actually owns it.
