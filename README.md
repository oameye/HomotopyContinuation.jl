# HomotopyContinuation.jl v3

This branch is the v3 rewrite of HomotopyContinuation.jl for solving polynomial and analytic
systems by homotopy continuation.

**Status:** feature/capability complete relative to upstream v2.22.4. The rewrite has moved from
v2-parity work into API stabilization, release documentation and final release hardening. See
`implementation_docs/02_status.md` and `implementation_docs/09_v2_parity_freeze.md` for the
current audited state.

During development the rewrite used the temporary package name `HomotopyContinuationNext` so v2
and v3 could coexist in one Julia environment. The v3 release line restores the registered package
identity:

```text
name = HomotopyContinuation
uuid = f213a82b-91d6-5c5d-acf7-10f1c761b327
version = 3.0.0-DEV
```

## Why v3

v2's compiled evaluator architecture can propagate a system-specific type through the homotopy,
tracker, endgame and solver stack. New systems can therefore trigger substantial recompilation.

v3 inserts a monomorphic evaluator firewall:

```text
system-specific evaluation code
  -> SystemEvaluator
  -> HomotopyEvaluator
  -> Tracker
  -> EndgameTracker
```

The numerical solver types no longer depend on the identity of the polynomial system. The rewrite
also replaces the SymEngine-based symbolic path with a pure-Julia expression/tape pipeline and
keeps the heavy Arblib certification backend outside the core package.

See `implementation_docs/03_v3_vs_v2.md` for the architectural and capability comparison.

## Basic use

```julia
using HomotopyContinuation

@polyvar x y
F = System([x^2 + y - 1, x*y - 2])
result = solve(F)
solutions(result)
real_solutions(result)
```

The core package includes total-degree and polyhedral solving, projective and overdetermined
systems, parameter and custom homotopies, composition, subspace continuation and sweeps,
monodromy, witness sets/regeneration/NID, distributed execution, rational/transcendental
expression input, and the complete tracking/endgame stack.

Certification is isolated in a separate package so users who only solve systems do not load
Arblib:

```julia
using HomotopyContinuation
using HomotopyContinuationCertification

certify(F, result)
```

## Development

```sh
make test           # core + certification suites
make test-cert      # certification suite only
make test-extensive # long reference solves
make benchmark      # steady-state benchmarks
make ttfx           # fresh-process first-use workload inventory
make format         # Runic formatting
```

The historical same-process `make compare` harness is intentionally disabled after restoring the
registered package identity: v2 and v3 now share the same package name and UUID. Any live v2/v3
comparison must run the versions in isolated Julia environments/processes.

The permanent CI matrix covers core/certification on Julia 1.10 and current Julia, JET, Runic,
benchmark tracking, representative TTFX tracking, and the complete TTFX inventory on `v3` pushes.
