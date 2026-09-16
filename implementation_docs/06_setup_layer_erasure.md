# Setup-layer erasure: `SystemSpec` + `SystemHandle`

Partly superseded. Line numbers are from the tree at the time of writing.

## Status

The type-stability work took the outcome this document wanted by a shorter route,
so read the plan below against what is already true:

- `System` carries two parameters, not four. The compile mode is a field and the
  shape is read off `size`; see `01_decisions.md`, "Shape dispatch sits behind an
  inference barrier".
- The caches no longer name the builder, the worker or the excess check:
  `SolveCache{E}`, `PolyhedralSolveCache{E}`, `WorkerSolveCache{E}`, where `E` is
  the executor the caller passed. `CommonSolve.solve!`, `_solve_total_degree_threaded`
  and the `@tasks` loops therefore compile once per executor rather than once per
  input flavor, compile mode and shape. That is step 5 below, done with a
  `Base.RefValue{Any}` box rather than a `FunctionWrapper`, because a builder
  crosses the wire to a distributed worker and a `FunctionWrapper` carries a raw
  pointer into the process that made it.
- Step 2's stated goal, "`SolveCache{E,B,C}`'s `B` now collapses to one type per
  route", is therefore already met, and more than met: `B` is gone.

What the plan still buys, and what remains open: the builders are still
parameterized on the system type, so a builder and its `clone` still compile per
input flavor; the route bodies still take `SystemLike`; and step 4's
precompilation still cannot name a signature without naming a user type.
`SystemSpec` remains the way to close those.

The API unification (`00_architecture.md`, "The solving API") landed first, so the signature
list below grew by every new `solve` route: the algorithm and executor slots are the last two
positional arguments on all of them. That does not change this document's thesis — algorithm
structs never enter a cache (at the time of writing `SolveCache{E,B,C}` was parameterized by
executor, builder and excess-checker, never by the algorithm), so their type parameters do
not reach the hot loop.
The parameters this document is about are the *builder* and *system* ones, which still do.

## Why

`System{P, V, M, S}` carried four type parameters when this was written; it now carries
`P` and `V`. `01_decisions.md` ("System{P, V} type
parameters") already records that `P` and `V` "affect only the `polys`, `parameters`,
`variables` fields", yet they propagate into every route that takes a system, and from
there into the builders, the caches and the tracking loops.

Follow one parameter. All eleven builders are parameterized on the system type
(`builder.jl:9, 33, 59, 82, 108, 130, 153, 181, 210`, `monodromy.jl:911, 925`), and every
builder body does exactly one thing with the field: `_clone_system_evaluator(b.system)`.
That function (`worker_state.jl:66, 84, 103`) reads only `sys._interp_f64.sequence`,
`sys._interp_jac.sequence` and the sizes. It never touches `P` or `V`. The parameter is
threaded four layers deep and discarded at the leaf.

The cost is upward, not downward. `StraightLineBuilder{System{Polynomial{…},PolyVar,…}}`
and `StraightLineBuilder{System{Expression,Expression,…}}` are different types, so
the cache differed, so `CommonSolve.solve!`, `_solve_total_degree_threaded`, the
`@tasks` loop and the monodromy worker loops all compiled again per input flavor, compile
mode and shape. Erasing the builder fixed that half; the builder's own body still compiles
per flavor. `SystemLike = Union{System, CompositionSystem}` is one more leaf on the
same axis, doubling fifteen route bodies so compositions can reach them.

It also blocks the one TTFX lever this project has measured. `01_decisions.md`, "Tape
executors are precompiled by signature": ten declarations took
`total_degree_interpreted_serial` from 9.766s to 8.483s (-13.1%), and the stated reason it
works is that "their signatures name no user type (a tape is data, not a type parameter)".
Precompiling `solve!` today would mean enumerating the P×V×M×S cross product and naming
DynamicPolynomials-internal types in `precompile_signatures.jl`. With a data-only clone
descriptor it is one line per route, covering every input including compositions.

Outcome: the setup layer becomes concrete and non-parametric, `SystemLike` is deleted
rather than replaced, and the route bodies become precompilable.

`_SupportSystem` (`polyhedral.jl:132`) is already exactly this design,
`evaluator + eval_sequence + jacobian_sequence` with a clone that is a pure function of the
sequences and sizes. This generalizes it to every route.

## Design

Two concrete, non-parametric types. Not an abstract supertype, not a union.

```julia
@enumx StageKind::Int8 begin      # which evaluator builder rebuilds this stage
    INTERPRETED
    COMPILED
    COMPILED_ALL
    SUPPORT                       # the polyhedral coefficient-parametric evaluator
end

struct StageSpec                  # everything `clone` needs, all data
    seq_eval::InstructionSequence
    seq_jac::InstructionSequence
    size::Tuple{Int, Int}
    nparameters::Int
    kind::StageKind.T
    scales::Vector{Float64}       # undone when this stage feeds another
end

struct SystemSpec
    stages::Vector{StageSpec}     # innermost first; one stage is a plain System
end

struct SystemHandle               # what every route body takes
    evaluator::SystemEvaluator    # the primary evaluator, already built
    spec::SystemSpec              # how to rebuild it per worker
    is_homogeneous::Bool
end
Base.size(h::SystemHandle) = size(h.evaluator)
nparameters(h::SystemHandle) = nparameters(h.evaluator)
nvariables(h::SystemHandle) = size(h.evaluator)[2]
```

`clone(spec::SystemSpec)::SystemEvaluator` folds the stages, wrapping successive stages in
`_ComposedSystem` with the inner stage's `scales`, the fold `_fold_composition` already
performs. The four-way `kind` branch sits behind `Base.inferencebarrier`, the pattern
already used for shape dispatch (`system.jl:84`).

A `CompositionSystem` is a spec with more than one stage, so it stops being a second type
the routes must know about. `SystemLike`, `SystemFactory` and `_SystemCloner` are deleted,
along with the ~80ms per-stage-type `@cfunction` compile the composition clone thunk costs
today. `StageEquations` stays: `System(::CompositionSystem)` still needs the symbolic
equations, and that is the one remaining type erasure on the input side.

## Work, in testable steps

**1. Spec and clone.** New `src/core/system_spec.jl`: `StageKind`, `StageSpec`,
`SystemSpec`, `clone`. Constructors `SystemSpec(::System)` (one stage, kind from
`compile_mode`), `SystemSpec(::CompositionSystem)` (one stage per `CompositionStage`),
`SystemSpec(::_SupportSystem)` (one `SUPPORT` stage). Move the three
`_clone_system_evaluator(::System)` bodies (`worker_state.jl:66-115`) into
`clone`, and keep `_clone_system_evaluator` as a one-line shim so nothing breaks yet.
Store `spec::SystemSpec` as a field on `System` (built at construction from data it already
holds) and on `CompositionSystem`. Run `make test`.

**2. De-parameterize the builders.** All eleven builders drop `{S}` and store
`spec::SystemSpec`; bodies call `clone(b.spec)`. Update the nine construction sites
(`solve.jl:139, 157, 320`, `slice.jl:178`, `subspace_solve.jl:172, 192, 199`,
`sweep.jl:226`, `polyhedral.jl:557`) plus the two monodromy builders to pass the spec.
Same for the direct clone sites at `witness_set.jl:571` and `regeneration.jl:581`.
`SolveCache`'s builder parameter would collapse to one type per route; it has since been
erased outright, so this step is now about the builder bodies alone. Run `make test`; check
`concrete_structs_test.jl` still passes.

**3. Route bodies take `SystemHandle`.** Convert the fifteen `SystemLike` signatures
(`newton.jl:78, 137`, `solve.jl:276, 300`, `monodromy.jl:451, 487, 941, 983, 1536, 1990`,
`builder.jl:59`) to `SystemHandle`, with thin `System`/`CompositionSystem` methods at each
public entry point that build the handle and forward. Anything that needs the equations
stays on the concrete type *above* the conversion, exactly as `_lower_input` runs before
the barrier today: `find_start_pair` keeps its two concrete methods, and the `Monodromy`
route computes the start pair before erasing. `_check_square_or_overdetermined` loses its shape
dispatch on the handle path and compares sizes, as the composition method already does.
Delete `SystemLike`, `SystemFactory`, `_SystemCloner`. Run `make test`.

**4. Precompile and measure.** Add to `src/precompile_signatures.jl`: `clone(::SystemSpec)`,
each builder's call operator, `CommonSolve.solve!` for each concrete `SolveCache`
instantiation, `_solve_total_degree_serial/_threaded`, and the monodromy worker
constructors. Keep the existing file's discipline: every target `@noinline`, no signature
naming a user type.

**5. Done.** The builder is erased behind `PathBuilder{TrackingWorkerState}`, so
`SolveCache` lost `B` entirely and one compiled loop serves every route sharing a
worker-state type. One dynamic call per worker, which is once per task. The box holds the
builder itself rather than a `FunctionWrapper`, so it still serializes to a distributed
worker.

**6. Optional rename.** `AbstractSystem` -> `AbstractSystemKernel` (six subtypes) and
`_ComposedSystem` -> `_CompositionKernel`, so the kernel interface (what gets erased into a
`SystemEvaluator`) stops reading like the supertype of `System`, which it is not.

## Verification

- `make test` after each step: core 5892, certification 541, JET zero reports, Aqua,
  ExplicitImports. `test/concrete_structs_test.jl` is the direct check that no step
  reintroduces an abstract or parametric field.
- `test/alloc_check_test.jl` must stay at zero allocations. This is a setup-layer change;
  the tracker hot path is untouched.
- `make ttfx WORKLOADS="total_degree_interpreted_serial monodromy_serial"` before and after.
  Reference numbers at the time of writing: 7.665s and 12.114s. Steps 1 to 3 are expected
  to be neutral for a single-flavor session; step 4 is where the gain lands.
- A multi-flavor TTFX probe, which is where steps 1 to 3 pay: one fresh session that solves
  a `@polyvar` system and then a `@var` system, and one that solves a plain system and then
  a composition. Today each pays a second full route compilation.
- Specialization counts before and after, for `CommonSolve.solve!`,
  `_solve_total_degree_threaded` and the builder call operators:
  `length(Base.specializations(only(methods(f, sig))))`. This is the direct measurement of
  the claim above.
- `make benchmark` steady state must not move. No hot-path type changes, so any movement is
  a bug.
