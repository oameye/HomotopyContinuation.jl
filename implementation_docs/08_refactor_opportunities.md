# Refactor Opportunities

Inventory of mechanical duplication left by the algorithm/executor API refactor. Everything
here works and is tested; this is debt. The fan-out half of the story is in
`07_executor_fanout_debt.md` and is not repeated.

## Interface defaults with the wrong arity

`abstract_types.jl:67,75` declare `set_solution!(x, y, t)` and `get_solution!(out, x, t)`.
Every implementation and the only call site (`homotopy_evaluator.jl:161-162`) use the
four-argument form with the homotopy in second position. The declared defaults are
unreachable, and their docstrings document a fallback that does not exist.

Because the default never applies, five homotopies hand-write the identity copy:
`straight_line_homotopy.jl:180-194`, `toric_homotopy.jl:384-398`, `affine_chart.jl:236-250`,
`linear_parameter_homotopy.jl:262-276`, `subspace_homotopies.jl:689-703` and `:840-854`.
Ten methods, all `copyto!`. Giving the default the signature
`(x, ::AbstractHomotopy, y, t)` deletes all ten; only `HomotopyEvaluator`'s forwarding pair
is real.

`start_parameters!` / `target_parameters!` have the same shape of problem. The default
(`abstract_types.jl:86,91`) returns `H` and is documented `-> H`, but all fourteen
implementations return `Nothing`, and `straight_line_homotopy.jl:198-199` /
`toric_homotopy.jl:402-403` exist only to change the return type. A `::Nothing = nothing`
default deletes those four and gives the interface one return type, which
`HomotopyEvaluator` already assumes through its FunctionWrapper.

~110 lines, pure deletion, no design decision.

## Wrapper dispatch table

161 `taylor!` and ~40 `evaluate!` definitions across 13 files, a large fraction of which are
signature enumeration rather than logic.

**24 `taylor!` stubs forwarding to one helper**, six per type (3 orders × 2 parameter forms):
`randomized_system.jl:179-206`, `start_pair_system.jl:169-194`,
`composition_system.jl:195-219`, `total_degree.jl:98-123`. `precompile_signatures.jl:19`
already generates the same enumeration with `for (K, N) in ((1, 2), (2, 3), (3, 4))`; the same
loop over `@eval` produces identical methods, so precompilation and TTFX are unaffected.

**~21 `evaluate!` methods with identical bodies.** `fixed_parameter_system.jl:121-143` and
`sliced_system.jl:78-103` are three byte-identical bodies each; `composition_system.jl:70-100`
and `randomized_system.jl:129-153` have two of three identical, the third differing only in
which scratch buffer it reads. One method over `Union{ComplexF64, ComplexDF64}` plus a
two-method `_buffer(S, x)` where the buffer varies takes ~21 methods to ~9.

`system_evaluator.jl:144-166` is *not* in this list: its three methods dispatch to different
FunctionWrappers and are the type firewall working as intended.

## Builders

`builder.jl` has 13 structs, each carrying `tracker_options` and `endgame_options` as separate
trailing fields and ending in the same `_endgame_tracker` plus worker-state tail.

`StraightLineBuilder`, `RandomizedStraightLineBuilder`, `SlicedStraightLineBuilder` and
`MultiHomogeneousBuilder` differ only in how the target evaluator is wrapped, and
`MultiHomogeneousBuilder` (`builder.jl:71-81`) already applies that wrapping conditionally. It
is the general case; the other three are specializations of it.

`SharedHomotopyBuilder` is `HomotopyBuilder(() -> H)` and `ClonedHomotopyBuilder` is
`HomotopyBuilder(() -> _clone_homotopy(H))`. Closures are concrete, so nothing is lost.

`ParameterBuilder` and `ParameterRetargetBuilder` (`builder.jl:166-228`) have identical fields
and identical bodies and differ only in the worker state returned. `AmbientWorkerState` carries
everything `TrackingWorkerState` does and `_track_path!` has a method for it, so
`ParameterBuilder` is redundant.

The `ExtrinsicSubspaceBuilder` / `ChartExtrinsicSubspaceBuilder` split is deliberate and
documented: it keeps the produced worker-state type concrete. Leave it.

## Options plumbing

Every route opens by exploding `alg` into locals and then passing eight to twelve positional
arguments down: `regeneration.jl:281-291` (ten locals), `sweep.jl:278-284` and `:305-312`
(seven each), `nid.jl:93-97`, `subspace_solve.jl:258-274`. That is why
`_init_intrinsic_subspace` and `_init_extrinsic_subspace` (`subspace_solve.jl:179-227`) take
twelve positional arguments and `_init_parameter_sweep` nine. Passing `alg` or its
`CommonOptions` removes the unpacking and halves the signatures, and removes the failure mode
where a new field reaches one unpack site but not its twin.

Related: the ten algorithm keyword constructors each restate `tracker_options`,
`endgame_options`, `seed`, `show_progress` before calling the four-positional
`CommonOptions`, and `_reseed` (six methods) and `_quiet` (three) rebuild their struct
field by field. A single `_with_common(alg, ::CommonOptions)` covers the latter two.

## Dead code

`src/precompile.jl` is not compiled in: `HomotopyContinuationNext.jl:159` is
`# include("precompile.jl")`. The file is still edited alongside the rest, and
`PrecompileTools` remains a declared dependency with a compat bound (`Project.toml:17,42`),
which makes it stale for Aqua.

`CommonOptions(; ...)` (`algorithm.jl:36-59`) has no caller in `src/`, `test/` or `lib/`. Its
eleven flattened tracker keywords are unreachable, since every algorithm uses the
four-positional inner constructor and none accepts a `CommonOptions`. Either wire it up (which
is also the fix for the repeated keyword blocks above) or delete it.

The two three-argument solution-transfer defaults, per the first section.

## Solve / init forwarding

Each route repeats five methods: `solve(…, alg, exec)`, `solve(…, exec)`,
`init(…, exec)`, and two `_bad_starts` guards. The guards are already generated by a
`for f in (:(solve), :(CommonSolve.init))` loop at `solve.jl:492`, `homotopy_solve.jl:245` and
`subspace_solve.jl:318`, which is an admission the whole group is boilerplate; the loop just
does not cover the other three. One macro per route would make adding a route a one-liner.

`_as_system` (`algorithm.jl:74-76`) has no `CompositionSystem` method, so monodromy carries
`F isa SystemLike ? F : _as_system(F)` three times (`monodromy.jl:1979,2017,2031`). One method
removes all three.

The parameter-count validation in `solve.jl:462-478` is repeated in `sweep.jl:227-241`.

## Smaller

`nsingular`, `nnonsingular` and `nreal` (`result.jl:567-595`) each rewrite the
`_solution_indices` count loop that `nresults` already implements with exactly those filters,
and `real_solutions` (`result.jl:550-561`) rewrites `solutions`. Five loops to five
delegations.

`ExtendedSolutionCertificate` (`certification.jl:76-90`) restates eight of
`SolutionCertificate`'s nine fields to add three. Composition with forwarded accessors removes
the drift risk between the two field lists.

## Not worth it

The `Val{2}` / `Val{3}` `taylor!` methods in `subspace_homotopies.jl:639-685` and `:799-838`
look duplicated but are different derivative expansions. The six `SysTaylor*FW` aliases in
`system_evaluator.jl` could be generated, but named aliases are worth more than the 30 lines.

## Priority

| | Change | Effect | Risk |
|---|---|---|---|
| 1 | Fix the four interface default signatures, delete 14 methods | ~110 lines | very low |
| 2 | Delete `precompile.jl`, the `PrecompileTools` dep, `CommonOptions(; …)` | ~85 lines | very low |
| 3 | `@eval` loop for the 24 `taylor!` stubs; merge the identical `evaluate!` triples | ~200 to ~70 | low |
| 4 | One serial and one threaded tracking loop | ~150 to ~55 | low |
| 5 | Generic progress split | ~90 to ~25 | low |
| 6 | Collapse the builder groups | ~165 to ~55 | medium |
| 7 | Pass `alg` down instead of exploding it | ~60 lines | low |
| 8 | Macro for the solve/init forwarding group | ~30 methods | medium |

Items 1 and 2 are pure deletions with no design decision in them.

Item 4 is item 1 of `07_executor_fanout_debt.md` seen from the other side. Worth recording
there: the per-cache hooks it needs already exist. `_path_result(cache, i)` and
`_excess_checker(cache)` (`result_iterator.jl`, the per-cache-kind sections) already cover all three cache
types with the right semantics, but are used only for lazy iteration, not by `solve!`. The
three serial loops (`solve.jl:277-295`, `polyhedral.jl:729-751`, `subspace_solve.jl:58-73`) and
three threaded loops (`solve.jl:312-340`, `polyhedral.jl:768-802`, `subspace_solve.jl:75-104`)
differ in nothing else. The Distributed extension already made this move: `TrackWork` and
`PolyhedralWork` (`ext/.../solve.jl:11-27`) are callables driving every route through one
`_distributed_map`.

Item 5: `_solve_X_{serial,threaded}_{with,without}_progress` plus `_dispatch_solve_policy`
appears at nine sites, five in core and four in the extension. It is one idea, keeping
`Nothing` and `Progress` out of a single inferred union, written out nine times. A generic
pair of `@noinline` helpers taking the body as a type parameter preserves the barrier and
removes 20 named functions.
