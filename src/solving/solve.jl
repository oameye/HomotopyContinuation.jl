## solve — CommonSolve.jl integration for polynomial system solving.
#
# Pattern: solve(F, alg, exec) = solve!(init(F, alg, exec))

struct SolveCache{E <: AbstractExecutor}
    executor::E
    builder::TrackingBuilder
    worker::TrackingWorkerState
    start_solutions::Vector{Vector{ComplexF64}}
    seed::UInt32
    # One checker when the system was squared up, none when it was already square.
    excess_checkers::ExcessCheckers
    show_progress::Bool
    early_stop::EarlyStop
end

"""
    check_square_or_overdetermined(F)

Throw for underdetermined input (positive-dimensional solution set).
"""
function check_square_or_overdetermined(F::System)::Nothing
    return with_system_shape(F) do shape
        check_square_or_overdetermined(shape, F)
    end
end

check_square_or_overdetermined(::SquareShape, ::System)::Nothing = nothing
check_square_or_overdetermined(::OverdeterminedShape, ::System)::Nothing = nothing
function check_square_or_overdetermined(::UnderdeterminedShape, F::System)::Nothing
    m, n = size(F.evaluator)
    throw(
        ArgumentError(
            "The system has $m equation(s) in $n variables. The solution set is " *
                "positive-dimensional; only square or overdetermined systems with " *
                "finitely many solutions are supported.",
        ),
    )
end

check_square_or_overdetermined(F::FixedParameterSystem)::Nothing =
    check_square_or_overdetermined(F.system)

# A composition carries no shape parameter, so the same check reads its size.
function check_square_or_overdetermined(C::CompositionSystem)::Nothing
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
    check_projective_determined(F, route)

Throw for a homogeneous system whose projective solution set is
positive-dimensional. `route` names the algorithm in the message.
"""
function check_projective_determined(F::CloneableSystem, route::String)::Nothing
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

is_grouped(F::System)::Bool = !isempty(F.variable_groups)
is_grouped(::CloneableSystem)::Bool = false

"""
    check_single_group(F)

Throw for a system with more than one variable group, on a route that is about
to put it on a single affine chart. One chart leaves such a system a cone in
each of its other groups.
"""
check_single_group(::CloneableSystem)::Nothing = nothing

function check_single_group(F::System)::Nothing
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

as_polynomial_system(F::System)::System = F
as_polynomial_system(C::CompositionSystem)::System = System(C)
as_polynomial_system(F::FixedParameterSystem)::System =
    fix_parameters(as_polynomial_system(F.system), F.parameters)

# Solved on a random affine chart: the slice by the whole ambient space, whose
# chart row makes `m = n - 1` square.
function init_projective(
        F::CloneableSystem, alg::TotalDegree, exec::E, route::String,
    )::SolveCache{E} where {E <: AbstractExecutor}
    # Before the count check, whose message is confusing for grouped input.
    check_single_group(F)
    check_projective_determined(F, route)
    G = as_polynomial_system(F)
    return CommonSolve.init(G, full_subspace(nvariables(G)), alg, exec)
end

"""
    check_parameter_free(F, route)

Throw when `F` still has parameters. `route` names the algorithm in the message.
"""
function check_parameter_free(F::CloneableSystem, route::String)::Nothing
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
    check_polynomial(F, route)

Throw when `F` has an equation that is not polynomial in its variables. `System`
records those with a degree of `-1`.
"""
function check_polynomial(F::System, route::String)::Nothing
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

check_polynomial(F::FixedParameterSystem, route::String)::Nothing =
    check_polynomial(F.system, route)

# A composed degree of `-1` comes from a stage, so the message names one.
function check_polynomial(C::CompositionSystem, route::String)::Nothing
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

function solve_cache(
        exec::E,
        builder::AbstractPathBuilder,
        starts::Vector{Vector{ComplexF64}},
        seed::UInt32,
        excess_checkers::ExcessCheckers,
        show_progress::Bool,
        early_stop::EarlyStop = NEVER_STOP,
    )::SolveCache{E} where {E <: AbstractExecutor}
    erased = TrackingBuilder(builder)
    return SolveCache(
        exec, erased, builder(), starts, seed, excess_checkers, show_progress,
        early_stop,
    )
end

function CommonSolve.init(
        F::CloneableSystem, alg::TotalDegree,
        exec::E = Threaded(),
    )::SolveCache{E} where {E <: AbstractExecutor}
    check_parameter_free(F, "`TotalDegree`")
    check_polynomial(F, "`TotalDegree`")
    # A grouped system is homogeneous per group, so grouping is decided first.
    is_grouped(F) && return init_multi_homogeneous(F, alg, exec)
    is_homogeneous(F) && return init_projective(F, alg, exec, "`TotalDegree`")
    check_square_or_overdetermined(F)

    rng = Random.MersenneTwister(algorithm_seed(alg))
    γ = random_gamma(rng)
    return init_total_degree_shaped(F, alg, exec, rng, γ)
end

function init_total_degree_shaped(
        F::CloneableSystem, alg::TotalDegree, exec::E,
        rng::Random.MersenneTwister, γ::ComplexF64,
    )::SolveCache{E} where {E <: AbstractExecutor}
    return with_system_shape(F) do shape
        init_total_degree(shape, F, alg, exec, rng, γ)
    end
end

function init_total_degree(
        shape::UnderdeterminedShape, F::System, ::TotalDegree, ::E,
        ::Random.MersenneTwister, ::ComplexF64,
    )::SolveCache{E} where {E <: AbstractExecutor}
    return check_square_or_overdetermined(shape, F)
end

function init_total_degree(
        ::SquareShape, F::CloneableSystem, alg::TotalDegree, exec::E,
        ::Random.MersenneTwister, γ::ComplexF64,
    )::SolveCache{E} where {E <: AbstractExecutor}
    degs = degrees(F)
    builder = StraightLineBuilder(
        degs, F, γ, algorithm_tracker_options(alg), algorithm_endgame_options(alg),
    )
    return solve_cache(
        exec, builder, total_degree_start_solutions(degs), algorithm_seed(alg),
        ExcessCheckers(), algorithm_show_progress(alg), early_stop_callback(alg),
    )
end

function init_total_degree(
        ::OverdeterminedShape, F::CloneableSystem, alg::TotalDegree,
        exec::E, rng::Random.MersenneTwister, γ::ComplexF64,
    )::SolveCache{E} where {E <: AbstractExecutor}
    n = nvariables(F)
    A, perm, excess_checker =
        square_up(rng, F.evaluator, degrees(F), alg.excess_residual_tol)
    degs = degrees(F)[perm[1:n]]
    builder = RandomizedStraightLineBuilder(
        degs, F, A, perm, γ, algorithm_tracker_options(alg), algorithm_endgame_options(alg),
    )
    return solve_cache(
        exec, builder, total_degree_start_solutions(degs), algorithm_seed(alg),
        ExcessCheckers([excess_checker]), algorithm_show_progress(alg), early_stop_callback(alg),
    )
end

# `start_solution` is copied: the caller's `starts` vector outlives the result.
function track_path!(
        eg::EndgameTracker, x₀::Vector{ComplexF64}, k::Int,
    )::PathResult
    track!(eg, x₀)
    return PathResult(eg; path_number = k, start_solution = copy(x₀))
end

track_path!(ws::TrackingWorkerState, x₀::Vector{ComplexF64}, k::Int)::PathResult =
    track_path!(ws.tracker, x₀, k)

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
        exec::E = Threaded(),
    )::Result where {E <: AbstractExecutor}
    cache::PathSolveCache = CommonSolve.init(F, alg, exec)
    return CommonSolve.solve!(cache)
end

function solve(F::CloneableSystem, exec::E)::Result where {E <: AbstractExecutor}
    return solve(F, TotalDegree(), exec)
end

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
    length(CommonSolve.init(F, without_progress(alg), Serial()).start_solutions)

# The BKK bound of the prepared support, not a solve: building the start solutions
# means a binomial solve per mixed cell and both trackers, all of it discarded.
paths_to_track(F::Union{System, CompositionSystem}, alg::Polyhedral)::Int =
    with_polyhedral_system(F, alg) do G
    support_mixed_volume(polyhedral_support(G, alg))
end

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
using HomotopyContinuation, SemialgebraicSets
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

function solve(
        F::Union{System, CompositionSystem}, alg::Polyhedral, exec::E = Threaded(),
    )::Result where {E <: AbstractExecutor}
    cache::PathSolveCache = CommonSolve.init(F, alg, exec)
    return CommonSolve.solve!(cache)
end

# The polyhedral start system is built from the composed monomials, which only
# the substituted equations carry.
function CommonSolve.init(
        C::CompositionSystem, alg::Polyhedral, exec::E = Threaded(),
    )::PolyhedralSolveCache{E} where {E <: AbstractExecutor}
    return CommonSolve.init(System(C), alg, exec)
end

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
        exec::E = Threaded(),
    )::Result where {E <: AbstractExecutor}
    return CommonSolve.solve!(
        CommonSolve.init(F, starts, p_start, p_target, alg, exec),
    )
end

function CommonSolve.init(
        F::SystemLike,
        starts::StartsLike,
        p_start::AbstractVector{<:Number},
        p_target::AbstractVector{<:Number},
        alg::Continuation = Continuation(),
        exec::E = Threaded(),
    )::SolveCache{E} where {E <: AbstractExecutor}
    check_square_or_overdetermined(F)
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
        F, sp, tp, algorithm_tracker_options(alg), algorithm_endgame_options(alg),
    )

    return solve_cache(
        exec, builder, start_points(starts), algorithm_seed(alg), ExcessCheckers(),
        algorithm_show_progress(alg), early_stop_callback(alg),
    )
end

for f in (:(solve), :(CommonSolve.init))
    @eval begin
        $f(
            F::SystemLike, starts::StartsLike, p_start::AbstractVector{<:Number},
            p_target::AbstractVector{<:Number}, exec::E,
        ) where {E <: AbstractExecutor} =
            $f(F, starts, p_start, p_target, Continuation(), exec)

        $f(
            ::SystemLike, starts, ::AbstractVector{<:Number},
            ::AbstractVector{<:Number}, ::Continuation = Continuation(),
            ::AbstractExecutor = Threaded(),
        ) = bad_starts(starts)

        $f(
            ::SystemLike, starts, ::AbstractVector{<:Number},
            ::AbstractVector{<:Number}, ::AbstractExecutor,
        ) = bad_starts(starts)
    end
end
