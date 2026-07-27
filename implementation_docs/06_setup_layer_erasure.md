# Setup-layer erasure: `SystemSpec` + `SystemHandle`

Planned, not implemented. Line numbers are from the tree at the time of writing.

## Why

`System{P, V, M, S}` carries four type parameters. `01_decisions.md` ("System{P, V} type
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
`SolveCache{E,B,C}` differs, so `CommonSolve.solve!`, `_solve_total_degree_threaded`, the
`@tasks` loop and the monodromy worker loops all compile again per input flavor, compile
mode and shape. `SystemLike = Union{System, CompositionSystem}` is one more leaf on the
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
`_clone_system_evaluator(::System{P,V,M,S})` bodies (`worker_state.jl:66-115`) into
`clone`, and keep `_clone_system_evaluator` as a one-line shim so nothing breaks yet.
Store `spec::SystemSpec` as a field on `System` (built at construction from data it already
holds) and on `CompositionSystem`. Run `make test`.

**2. De-parameterize the builders.** All eleven builders drop `{S}` and store
`spec::SystemSpec`; bodies call `clone(b.spec)`. Update the nine construction sites
(`solve.jl:139, 157, 320`, `slice.jl:178`, `subspace_solve.jl:172, 192, 199`,
`sweep.jl:226`, `polyhedral.jl:557`) plus the two monodromy builders to pass the spec.
Same for the direct clone sites at `witness_set.jl:571` and `regeneration.jl:581`.
`SolveCache{E,B,C}`'s `B` now collapses to one type per route. Run `make test`; check
`concrete_structs_test.jl` still passes.

**3. Route bodies take `SystemHandle`.** Convert the fifteen `SystemLike` signatures
(`newton.jl:78, 137`, `solve.jl:276, 300`, `monodromy.jl:451, 487, 941, 983, 1536, 1990`,
`builder.jl:59`) to `SystemHandle`, with thin `System`/`CompositionSystem` methods at each
public entry point that build the handle and forward. Anything that needs the equations
stays on the concrete type *above* the conversion, exactly as `_lower_input` runs before
the barrier today: `find_start_pair` keeps its two concrete methods, and `monodromy_solve`
computes the start pair before erasing. `_check_square_or_overdetermined` loses its shape
dispatch on the handle path and compares sizes, as the composition method already does.
Delete `SystemLike`, `SystemFactory`, `_SystemCloner`. Run `make test`.

**4. Precompile and measure.** Add to `src/precompile_signatures.jl`: `clone(::SystemSpec)`,
each builder's call operator, `CommonSolve.solve!` for each concrete `SolveCache`
instantiation, `_solve_total_degree_serial/_threaded`, and the monodromy worker
constructors. Keep the existing file's discipline: every target `@noinline`, no signature
naming a user type.

**5. Optional, decide after measuring.** Erase the builder itself behind
`FunctionWrapper{TrackingWorkerState, Tuple{}}` so `SolveCache` loses `B` entirely and one
compiled loop serves every route sharing a worker-state type. One indirect call per worker.

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
