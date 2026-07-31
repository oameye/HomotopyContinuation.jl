# Executor Fan-Out Debt

Assessment of how routes consume `AbstractExecutor`, and the refactor that would fix it.
Current state works and is fully tested; this is debt, not breakage.

## The axis is right

Algorithm carries *what to compute*, executor carries *where it runs*. Builder-per-task is
correct given mutable interpreter tapes. Randomness drawn in the driver before the fan-out
makes total-degree results task-count independent.

## The executor is data, the policy is re-derived at every leaf

Six `@tasks` sites (`solve.jl`, `polyhedral.jl`, `subspace_solve.jl`, `sweep.jl`,
`witness_set.jl`, `regeneration.jl`) and five `Threads.@spawn` sites each independently decide
the task count, whether to fan out at all, and how to honour early stop. The task count alone
is spelled four ways: `cache.executor.ntasks`, `exec.ntasks`, `_local_ntasks(exec)`, and
`exec.ntasks` threaded through `_run_monodromy_loop!`.

That is a defect generator, and it has generated: `_threaded_intersection!` and
`threaded_monodromy_solve!` both ignored `exec.ntasks` and spawned `Threads.nthreads()` tasks;
`_solve_worker_distributed` dropped `early_stop` entirely; and the numbering consequence of
early-stop compaction went unnoticed on every route.

Three specific incoherences:

1. **`Threaded(1)` means two things.** `_wants_tasks(Threaded(1))` is `false`, so `membership`
   and `_threaded_intersection!` take the plain loop, but `CommonSolve.solve!` dispatches on
   `SolveCache{Threaded}` by type, so the main solve takes the `@tasks` path with one task.
2. **`_wants_tasks` and `_local_ntasks` overlap and can disagree.** They are the same predicate
   for `Serial`/`Threaded`. For `DistributedExecutor` on a single-threaded driver the first says
   "fan out" and the second says "one task", silently.
3. **`_local_ntasks(::AbstractExecutor) = Threads.nthreads()` invents a number the type does not
   carry.** `DistributedExecutor.tasks_per_process` describes the remote side and resolves
   remotely; it says nothing about a driver-local fan-out. The type is one field short and the
   fallback method papers over it.

Plus one consequence chain: `@tasks` cannot `break`, so early stop leaves holes, so every site
must call `_assigned_results`, which compacts, which breaks `path_number == position`, which
needs `_path_position`. Three derived layers from one primitive limitation, spread across five
call sites.

## Refactor, in priority order

1. **One fan-out primitive.** `_map_paths(plan, n, work, report, stop)` returning only what ran,
   with three implementations (serial loop, `@tasks`, distributed batch map). The six `@tasks`
   sites become calls, and early stop, compaction and the numbering invariant live in one place.
   Prevents the whole class above. Touches six routes, so it is a real refactor.
2. **A plan struct computed at `init`,** carrying `ntasks` and batch policy, instead of an
   executor consulted at each leaf. `_wants_tasks` and `_local_ntasks` collapse into
   `plan.ntasks`, and `Threaded(1)` gets one meaning everywhere. Cheap.
3. **Give `DistributedExecutor` an explicit local task count** so the fallback method disappears.
   One field.
4. **Factor monodromy's worker pool.** Its work is generated as solutions are found, so it cannot
   be a map, but the spawn loop, idle tracking and quiescence condition are hand-rolled twice
   (`monodromy.jl` in core and in the extension). Optional.

`Threaded(1)` currently still routes through `threaded_monodromy_solve!` rather than
`serial_monodromy_solve!`, since dispatch is on the executor type. Rerouting it would change
which per-task RNG streams exist and so change results for existing `Threaded(1)` callers,
including the v2 parity tests. Fold that into item 2 if the plan makes task count the dispatch
input.
