# HomotopyContinuation.jl

Numerical homotopy continuation for polynomial and analytic systems in Julia.

HomotopyContinuation.jl provides path tracking, total-degree and polyhedral solvers, parameter continuation, subspace methods, monodromy, witness sets, regeneration, numerical irreducible decomposition, and serial/threaded/distributed execution behind one solver interface.

## Quick start

```julia
using HomotopyContinuation

@polyvar x y
F = System([x^2 + y - 1, x * y - 2])
result = solve(F)

solutions(result)
real_solutions(result)
```

Polynomial input uses the MultivariatePolynomials ecosystem through `@polyvar`. Analytic expressions that include division, non-integer powers, or supported transcendental functions can be built with `@var` and passed through the same `System` interface.

## Solvers and continuation workflows

The package includes:

- total-degree and polyhedral start systems;
- square, overdetermined, affine/projective, and multi-homogeneous workflows;
- parameter and custom homotopies;
- intrinsic and extrinsic subspace continuation and sweeps;
- monodromy, completeness checks, witness sets, regeneration, and numerical irreducible decomposition;
- `Serial()`, `Threaded()`, and `DistributedExecutor` execution;
- interpreted and RuntimeGeneratedFunctions-backed evaluation modes through `CompileMode`.

Mutable numerical state is owned by per-worker trackers/evaluators, so the same solver semantics are available across execution modes without sharing hot-path scratch storage.

## Certification

Rigorous certification is isolated in the companion package under `lib/HomotopyContinuationCertification`, keeping Arblib out of the core package:

```julia
using HomotopyContinuation
using HomotopyContinuationCertification

certify(F, solutions(result))
```

## Development

The current architecture is documented in [`implementation_docs/00_architecture.md`](implementation_docs/00_architecture.md), with durable implementation constraints in [`implementation_docs/01_decisions.md`](implementation_docs/01_decisions.md). [`docs/src/dev.md`](docs/src/dev.md) is the short contributor guide.

Historical migration/status/debt notes are intentionally not maintained as active documentation. Any still-actionable work from them must have an owning GitHub issue before the note is removed; Git history preserves the original analysis.

Common commands:

```sh
make test           # core + certification suites
make test-strict    # DispatchDoctor-instrumented suite
make test-cert      # certification only
make test-extensive # long reference solves
make benchmark      # steady-state benchmarks
make ttfx           # fresh-process first-use workloads
make format         # Runic
make deps           # instantiate development environments
```

Run `make help` for the complete maintained command surface.