## solve — CommonSolve.jl integration for polynomial system solving.
#
# Pattern: solve(F, alg, exec) = solve!(init(F, alg, exec))

struct SolveCache{E <: AbstractExecutor, B, C}
    executor::E
    builder::B
    tracker::EndgameTracker
    start_solutions::Vector{Vector{ComplexF64}}
    seed::UInt32
    # ExcessSolutionChecker for overdetermined systems, Nothing for square ones
    excess_checker::C
    show_progress::Bool
    early_stop::EarlyStop
end

"""
    _check_square_or_overdetermined(F)

Throw for underdetermined input (positive-dimensional solution set).
"""
function _check_square_or_overdetermined(F::System)::Nothing
    return _check_square_or_overdetermined(system_shape(F), F)
end

_check_square_or_overdetermined(::SquareShape, ::System)::Nothing = nothing
_check_square_or_overdetermined(::OverdeterminedShape, ::System)::Nothing = nothing
function _check_square_or_overdetermined(::UnderdeterminedShape, F::System)::Nothing
    m, n = size(F.evaluator)
    throw(
        ArgumentError(
            "The system has $m equation(s) in $n variables. The solution set is " *
                "positive-dimensional; only square or overdetermined systems with " *
                "finitely many solutions are supported.",
        ),
    )
end

_check_square_or_overdetermined(F::FixedParameterSystem)::Nothing =
    _check_square_or_overdetermined(F.system)

# A composition carries no shape parameter, so the same check reads its size.
function _check_square_or_overdetermined(C::CompositionSystem)::Nothing
    m, n = size(C)
    m >= n || throw(
        ArgumentError(
            "The composition has $m equation(s) in $n variables. The solution set is " *
                "positive-dimensional; only square or overdetermined systems with " *
                "finitely many solutions are supported.",
        ),
    )
    return nothing
end

"""
    _check_projective_determined(F, route)

Throw for a homogeneous system whose projective solution set is
positive-dimensional. `route` names the algorithm in the message.
"""
function _check_projective_determined(F::CloneableSystem, route::String)::Nothing
    m, n = size(F)
    m + 1 >= n || throw(
        ArgumentError(
            "$route puts the homogeneous system on an affine chart, so $m equation(s) " *
                "in $n variables give $(m + 1). The projective solution set is " *
                "positive-dimensional; only finitely many solutions are supported.",
        ),
    )
    return nothing
end

_is_grouped(F::System)::Bool = !isempty(F.variable_groups)
_is_grouped(::CloneableSystem)::Bool = false

"""
    _check_single_group(F)

Throw for a system with more than one variable group, on a route that is about
to put it on a single affine chart. One chart leaves such a system a cone in
each of its other groups.
"""
_check_single_group(::CloneableSystem)::Nothing = nothing

function _check_single_group(F::System)::Nothing
    M = length(variable_groups(F))
    M <= 1 || throw(
        ArgumentError(
            "The system has $M variable groups, which need one affine chart per " *
                "group. Only `solve(F, TotalDegree())` charts per group; this " *
                "route charts the variables as a whole.",
        ),
    )
    return nothing
end

_polynomial_system(F::System)::System = F
_polynomial_system(C::CompositionSystem)::System = System(C)
_polynomial_system(F::FixedParameterSystem)::System =
    fix_parameters(_polynomial_system(F.system), F.parameters)

# Solved on a random affine chart: the slice by the whole ambient space, whose
# chart row makes `m = n - 1` square.
function _init_projective(
        F::CloneableSystem, alg::Union{TotalDegree, Polyhedral},
        exec::AbstractExecutor, route::String,
    )
    # Before the count check, whose message is confusing for grouped input.
    _check_single_group(F)
    _check_projective_determined(F, route)
    G = _polynomial_system(F)
    return CommonSolve.init(G, _full_subspace(nvariables(G)), alg, exec)
end

"""
    _check_parameter_free(F, route)

Throw when `F` still has parameters. `route` names the algorithm in the message.
"""
function _check_parameter_free(F::CloneableSystem, route::String)::Nothing
    np = nparameters(F)
    np == 0 || throw(
        ArgumentError(
            "$route requires a parameter-free system, but the system has $np " *
                "parameter(s). Fix them first with `fix_parameters(F, p)`, or track " *
                "from known start solutions with a parameter homotopy " *
                "(`solve(F, starts, p_start, p_target)`).",
        ),
    )
    return nothing
end

"""
    _check_polynomial(F, route)

Throw when `F` has an equation that is not polynomial in its variables. `System`
records those with a degree of `-1`.
"""
function _check_polynomial(F::System, route::String)::Nothing
    any(<(0), F.degrees) || return nothing
    throw(
        ArgumentError(
            "$route requires a system that is polynomial in its variables, but at " *
                "least one equation uses division by a variable, a negative power, " *
                "or a unary function of a variable. Clear denominators first, or " *
                "track from known start solutions with a parameter homotopy or " *
                "`Monodromy`.",
        ),
    )
end

_check_polynomial(F::FixedParameterSystem, route::String)::Nothing =
    _check_polynomial(F.system, route)

# A composed degree of `-1` comes from a stage, so the message names one.
function _check_polynomial(C::CompositionSystem, route::String)::Nothing
    any(<(0), degrees(C)) || return nothing
    throw(
        ArgumentError(
            "$route requires a composition whose stages are polynomial in their " *
                "variables, but at least one stage equation uses division by a " *
                "variable, a negative power, or a unary function of a variable. " *
                "Clear denominators first, or track from known start solutions " *
                "with a parameter homotopy or `Monodromy`.",
        ),
    )
end

# ── CommonSolve.init: System + TotalDegree ────────────────────────────────

function _solve_cache(
        exec::E,
        builder::B,
        starts::Vector{Vector{ComplexF64}},
        seed::UInt32,
        excess_checker::C,
        show_progress::Bool,
        early_stop::EarlyStop = NEVER_STOP,
    )::SolveCache{E, B, C} where {E <: AbstractExecutor, B, C}
    return SolveCache(
        exec, builder, builder().tracker, starts, seed, excess_checker, show_progress,
        early_stop,
    )
end

function CommonSolve.init(
        F::CloneableSystem, alg::TotalDegree,
        exec::AbstractExecutor = Threaded(),
    )::SolveCache
    _check_parameter_free(F, "`TotalDegree`")
    _check_polynomial(F, "`TotalDegree`")
    if _is_grouped(F)
        # First: for a grouped system homogeneity is per group. Dynamic call, as below.
        return Base.inferencebarrier(_init_multi_homogeneous)(F, alg, exec)
    end
    if is_homogeneous(F)
        # Dynamic call: inferring the projective wrapper stack from here costs the
        # common path ~3s.
        return Base.inferencebarrier(_init_projective)(F, alg, exec, "`TotalDegree`")
    end
    _check_square_or_overdetermined(F)

    rng = Random.MersenneTwister(_seed(alg))
    γ = _random_gamma(rng)
    # Dynamic call: specializes the body on the concrete `System` so that
    # `system_shape(F)` resolves statically instead of union-splitting.
    initializer = Base.inferencebarrier(_init_total_degree_shaped)
    return initializer(F, alg, exec, rng, γ)
end

_init_total_degree_shaped(
    F::CloneableSystem, alg::TotalDegree, exec::AbstractExecutor,
    rng::Random.MersenneTwister, γ::ComplexF64,
) = _init_total_degree(system_shape(F), F, alg, exec, rng, γ)

function _init_total_degree(
        ::SquareShape, F::CloneableSystem, alg::TotalDegree, exec::AbstractExecutor,
        ::Random.MersenneTwister, γ::ComplexF64,
    )
    degs = degrees(F)
    builder = StraightLineBuilder(
        degs, F, γ, _tracker_options(alg), _endgame_options(alg),
    )
    return _solve_cache(
        exec, builder, total_degree_start_solutions(degs), _seed(alg), nothing,
        _show_progress(alg), early_stop_callback(alg),
    )
end

function _init_total_degree(
        ::OverdeterminedShape, F::CloneableSystem, alg::TotalDegree,
        exec::AbstractExecutor, rng::Random.MersenneTwister, γ::ComplexF64,
    )
    n = nvariables(F)
    A, perm, excess_checker =
        _square_up(rng, F.evaluator, degrees(F), alg.excess_residual_tol)
    degs = degrees(F)[perm[1:n]]
    builder = RandomizedStraightLineBuilder(
        degs, F, A, perm, γ, _tracker_options(alg), _endgame_options(alg),
    )
    return _solve_cache(
        exec, builder, total_degree_start_solutions(degs), _seed(alg), excess_checker,
        _show_progress(alg), early_stop_callback(alg),
    )
end

# ── CommonSolve.solve!: serial ─────────────────────────────────────────────

function CommonSolve.solve!(cache::SolveCache{Serial})::Result
    solver = cache.show_progress ?
        _solve_total_degree_serial_with_progress :
        _solve_total_degree_serial_without_progress
    solver = Base.inferencebarrier(solver)
    return _dispatch_solve_policy(solver, cache)
end

@noinline function _dispatch_solve_policy(solver::Function, cache)::Result
    Base.@nospecialize solver cache
    return solver(cache)
end

@noinline _solve_total_degree_serial_without_progress(cache::SolveCache{Serial}) =
    _solve_total_degree_serial(cache, nothing)
@noinline _solve_total_degree_serial_with_progress(cache::SolveCache{Serial}) =
    _solve_total_degree_serial(cache, make_progress(length(cache.start_solutions), true))

# `start_solution` is copied: the caller's `starts` vector outlives the result.
function _track_path!(
        eg::EndgameTracker, x₀::Vector{ComplexF64}, k::Int,
    )::PathResult
    track!(eg, x₀)
    return PathResult(eg; path_number = k, start_solution = copy(x₀))
end

_track_path!(ws::TrackingWorkerState, x₀::Vector{ComplexF64}, k::Int)::PathResult =
    _track_path!(ws.tracker, x₀, k)

function _solve_total_degree_serial(cache::SolveCache{Serial}, progress)::Result
    eg = cache.tracker
    n_paths = length(cache.start_solutions)
    path_results = PathResult[]
    sizehint!(path_results, n_paths)

    stats = ProgressStats()
    stop = cache.early_stop
    for (k, x₀) in enumerate(cache.start_solutions)
        pr = _track_path!(eg, x₀, k)
        push!(path_results, pr)
        update_progress!(progress, k, stats, pr)
        is_success(pr) && stop(pr) && break
    end

    return _finalize_result(
        path_results, length(path_results), cache.seed, cache.excess_checker,
    )
end

# ── CommonSolve.solve!: threaded ───────────────────────────────────────────

function CommonSolve.solve!(cache::SolveCache{Threaded})::Result
    solver = cache.show_progress ?
        _solve_total_degree_threaded_with_progress :
        _solve_total_degree_threaded_without_progress
    solver = Base.inferencebarrier(solver)
    return _dispatch_solve_policy(solver, cache)
end

@noinline _solve_total_degree_threaded_without_progress(cache::SolveCache{Threaded}) =
    _solve_total_degree_threaded(cache, nothing)
@noinline _solve_total_degree_threaded_with_progress(cache::SolveCache{Threaded}) =
    _solve_total_degree_threaded(cache, make_progress(length(cache.start_solutions), true))

function _solve_total_degree_threaded(cache::SolveCache{Threaded}, progress)::Result
    nt = cache.executor.ntasks
    starts = cache.start_solutions
    n_paths = length(starts)
    results = Vector{PathResult}(undef, n_paths)

    stats = ProgressStats()
    counter = Threads.Atomic{Int}(0)
    plock = ReentrantLock()
    stop = cache.early_stop
    stopped = Threads.Atomic{Bool}(false)

    @tasks for i in eachindex(starts)
        @set ntasks = nt
        @local ws = cache.builder()
        if !stopped[]
            pr = _track_path!(ws, starts[i], i)
            results[i] = pr
            if progress !== nothing
                k = Threads.atomic_add!(counter, 1) + 1
                @lock plock update_progress!(progress, k, stats, pr)
            end
            is_success(pr) && stop(pr) && (stopped[] = true)
        end
    end

    tracked = _assigned_results(results)
    return _finalize_result(tracked, length(tracked), cache.seed, cache.excess_checker)
end

# ── CommonSolve.solve!: distributed (extension) ────────────────────────────

CommonSolve.solve!(cache::SolveCache{DistributedExecutor})::Result =
    _distributed_solve!(cache)

# ── Convenience: solve(F, alg, exec) ─────────────────────────────────────

"""
    solve(F::System, alg = TotalDegree(), exec = Threaded())

Solve a polynomial system using homotopy continuation.

`F` must be parameter-free; for a parametric system fix the values first with
[`fix_parameters`](@ref).

A homogeneous `F` is solved projectively, on a random affine chart drawn from the
algorithm's seed. One equation fewer than there are variables is therefore square,
and the returned solutions are representatives of projective points, so any two of
them that agree up to a complex scaling are the same solution.

A [`CompositionSystem`](@ref) is accepted too: the total-degree start system
needs only the composed degrees, which are folded from the stages, so the
equations are never rebuilt.
"""
function solve(
        F::CloneableSystem,
        alg::TotalDegree = TotalDegree(),
        exec::AbstractExecutor = Threaded(),
    )::Result
    return CommonSolve.solve!(CommonSolve.init(F, alg, exec))
end

solve(F::CloneableSystem, exec::AbstractExecutor)::Result =
    solve(F, TotalDegree(), exec)

"""
    paths_to_track(F::System, alg = TotalDegree()) -> Int

Number of paths [`solve`](@ref) would track for `F` under `alg`, without tracking
any of them. Throws whatever `solve` would throw for input it cannot handle.

For a system built with `variable_groups` this is the multi-homogeneous Bezout
number, which is at most the total Bezout number and usually far below it.

# Example
```julia
@polyvar x y
paths_to_track(System([x * y - 2, x^2 - 4]))                                # 4
paths_to_track(System([x * y - 2, x^2 - 4]; variable_groups = [[x], [y]]))  # 2
```
"""
paths_to_track(F::CloneableSystem, alg::TotalDegree = TotalDegree())::Int =
    length(CommonSolve.init(F, _quiet(alg), Serial()).start_solutions)

# The BKK bound of the prepared support, not a solve: building the start solutions
# means a binomial solve per mixed cell and both trackers, all of it discarded.
paths_to_track(F::Union{System, CompositionSystem}, alg::Polyhedral)::Int =
    _mixed_volume(_polyhedral_support(_polyhedral_system(F, alg), alg))

"""
    mixed_volume(F::System) -> Int

Mixed volume (BKK bound) of the Newton polytopes of `F`: the number of solutions
with all coordinates non-zero, counted with multiplicity, for a generic system
with `F`'s support. This is the number of paths
`solve(F, Polyhedral(; only_torus = true))` tracks.

An overdetermined `F` is squared up and a homogeneous one is put on an affine
chart first, as [`Polyhedral`](@ref) does.
"""
mixed_volume(F::System)::Int = paths_to_track(F, Polyhedral(; only_torus = true))

"""
    SemialgebraicSetsHCSolver(; algorithm, executor, real_tol, compile)

A `SemialgebraicSets.AbstractAlgebraicSolver` backed by homotopy continuation,
for use with `SemialgebraicSets.@set` and `algebraicset`. Load SemialgebraicSets
to make it available.

`algorithm` (a [`TotalDegree`](@ref) or [`Polyhedral`](@ref)) and `executor` are
forwarded to [`solve`](@ref), `compile` to [`System`](@ref), and `real_tol` to
[`real_solutions`](@ref), which is what selects the returned points.

With SemialgebraicSets loaded, [`solve`](@ref) also takes an algebraic set
directly and returns the full [`Result`](@ref), and
[`real_solutions`](@ref) takes a set and a solver.

```julia
using HomotopyContinuationNext, SemialgebraicSets
@polyvar x y
collect(SemialgebraicSets.@set x^2 == 1 && y^2 == 2 SemialgebraicSetsHCSolver())
```
"""
function SemialgebraicSetsHCSolver(args...; kwargs...)
    Base.@nospecialize args kwargs
    # The extension's zero-argument method is more specific than this `Vararg` one,
    # so it takes over once SemialgebraicSets is loaded.
    throw(
        ArgumentError(
            "`SemialgebraicSetsHCSolver` needs the SemialgebraicSets extension. " *
                "Run `using SemialgebraicSets`.",
        ),
    )
end

solve(
    F::System, alg::Polyhedral, exec::AbstractExecutor = Threaded(),
)::Result = CommonSolve.solve!(CommonSolve.init(F, alg, exec))

# The polyhedral start system is built from the composed monomials, which only
# the substituted equations carry.
solve(
    C::CompositionSystem, alg::Polyhedral, exec::AbstractExecutor = Threaded(),
)::Result = solve(System(C), alg, exec)

CommonSolve.init(
    C::CompositionSystem, alg::Polyhedral, exec::AbstractExecutor = Threaded(),
) = CommonSolve.init(System(C), alg, exec)

# ── Parameter homotopy ─────────────────────────────────────────────────────

"""
    solve(F::System, starts, p_start, p_target, alg = Continuation(), exec = Threaded())

Track the solutions `starts` of `F(x; p_start)` to `F(x; p_target)` along a
parameter homotopy, moving the parameters linearly and leaving the equations
untouched.

`starts` may be a vector of solution vectors, a [`Result`](@ref), or a
[`ResultIterator`](@ref), as on every route that takes start solutions.

# Example
```julia
@polyvar x y a
F = System([x^2 - a, y^2 - a]; variables = [x, y], parameters = [a])
r₁ = solve(fix_parameters(F, [1.0]))
solve(F, r₁, [1.0], [4.0])
```
"""
function solve(
        F::SystemLike,
        starts::StartsLike,
        p_start::AbstractVector{<:Number},
        p_target::AbstractVector{<:Number},
        alg::Continuation = Continuation(),
        exec::AbstractExecutor = Threaded(),
    )::Result
    return CommonSolve.solve!(
        CommonSolve.init(F, starts, p_start, p_target, alg, exec),
    )
end

solve(
    F::SystemLike, starts::StartsLike, p_start::AbstractVector{<:Number},
    p_target::AbstractVector{<:Number}, exec::AbstractExecutor,
)::Result = solve(F, starts, p_start, p_target, Continuation(), exec)

CommonSolve.init(
    F::SystemLike, starts::StartsLike, p_start::AbstractVector{<:Number},
    p_target::AbstractVector{<:Number}, exec::AbstractExecutor,
) = CommonSolve.init(F, starts, p_start, p_target, Continuation(), exec)

function CommonSolve.init(
        F::SystemLike,
        starts::StartsLike,
        p_start::AbstractVector{<:Number},
        p_target::AbstractVector{<:Number},
        alg::Continuation = Continuation(),
        exec::AbstractExecutor = Threaded(),
    )
    _check_square_or_overdetermined(F)
    np = nparameters(F)
    np > 0 || throw(
        ArgumentError(
            "a parameter homotopy requires a parametric system, but the system has " *
                "no parameters.",
        ),
    )
    length(p_start) == np || throw(
        ArgumentError(
            "p_start has length $(length(p_start)), but the system has $np parameter(s).",
        ),
    )
    length(p_target) == np || throw(
        ArgumentError(
            "p_target has length $(length(p_target)), but the system has $np parameter(s).",
        ),
    )

    sp = ComplexF64.(p_start)
    tp = ComplexF64.(p_target)
    builder = ParameterBuilder(
        F, sp, tp, _tracker_options(alg), _endgame_options(alg),
    )

    return _solve_cache(
        exec, builder, _start_points(starts), _seed(alg), nothing,
        _show_progress(alg), early_stop_callback(alg),
    )
end

for f in (:(solve), :(CommonSolve.init))
    @eval begin
        $f(
            ::SystemLike, starts, ::AbstractVector{<:Number},
            ::AbstractVector{<:Number}, ::Continuation = Continuation(),
            ::AbstractExecutor = Threaded(),
        ) = _bad_starts(starts)

        $f(
            ::SystemLike, starts, ::AbstractVector{<:Number},
            ::AbstractVector{<:Number}, ::AbstractExecutor,
        ) = _bad_starts(starts)
    end
end
