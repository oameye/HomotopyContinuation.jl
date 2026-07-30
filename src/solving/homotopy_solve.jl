## Tracking along a homotopy the caller supplies: the straight line between two
## parameter-free systems, or an explicit homotopy object. Both fill the ordinary
## `SolveCache`, so every `solve!` method applies unchanged.

# ── solve(G, F, starts) ────────────────────────────────────────────────────

function _check_start_target(G::CloneableSystem, F::CloneableSystem)::Nothing
    _check_parameter_free(G, "`solve(G, F, starts)`")
    _check_parameter_free(F, "`solve(G, F, starts)`")
    size(G) == size(F) || throw(
        ArgumentError(
            "the start system has size $(size(G)), but the target system has size " *
                "$(size(F)); a homotopy needs both to agree.",
        ),
    )
    is_homogeneous(G) == is_homogeneous(F) || throw(
        ArgumentError(
            "the target system is $(is_homogeneous(F) ? "" : "not ")homogeneous and " *
                "the start system is $(is_homogeneous(G) ? "" : "not ")homogeneous; a " *
                "homotopy between them would be tracked in the wrong coordinates.",
        ),
    )
    if is_homogeneous(F)
        _check_projective_determined(F, "`solve(G, F, starts)`")
    else
        _check_square_or_overdetermined(F)
    end
    return nothing
end

function _check_start_length(points::Vector{Vector{ComplexF64}}, n::Int)::Nothing
    for x in points
        length(x) == n || throw(
            ArgumentError(
                "a start solution has length $(length(x)), but $n variable(s) are " *
                    "tracked.",
            ),
        )
    end
    return nothing
end

"""
    solve(G, F, starts, exec = Threaded(); options...)

Track the solutions `starts` of `G` to `F` along `γ·t·G(x) + (1 - t)·F(x)`,
with `γ` drawn from `seed`.

Both systems must be parameter-free and of the same size; fix a parametric one
with [`fix_parameters`](@ref). A homogeneous pair is tracked on a random affine
chart, so `starts` may be any projective representatives.

`starts` may be a vector of solution vectors, a [`Result`](@ref), or a
[`ResultIterator`](@ref).

# Example
```julia
@polyvar x y a b
F = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
solve(fix_parameters(F, [1, 0]), fix_parameters(F, [2, 4]), [[1, 1]])
```
"""
function solve(
        G::CloneableSystem,
        F::CloneableSystem,
        starts,
        exec::AbstractExecutor = Threaded();
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = true,
    )::Result
    return CommonSolve.solve!(
        CommonSolve.init(
            G, F, starts, exec;
            seed = seed,
            tracker_options = tracker_options,
            endgame_options = endgame_options,
            show_progress = show_progress,
        ),
    )
end

function CommonSolve.init(
        G::CloneableSystem,
        F::CloneableSystem,
        starts,
        exec::AbstractExecutor = Threaded();
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = true,
    )::SolveCache
    _check_start_target(G, F)

    rng = Random.MersenneTwister(seed)
    γ = _random_gamma(rng)
    chart = is_homogeneous(F) ? _affine_chart(rng, F) : ComplexF64[]

    points = _start_points(starts)
    _check_start_length(points, nvariables(F))
    isempty(chart) || _place_on_chart!(points, chart)

    builder = StartTargetBuilder(
        G, F, chart, γ, tracker_options, endgame_options,
    )
    return _solve_cache(exec, builder, points, seed, nothing, show_progress)
end

# ── solve(H, starts) ───────────────────────────────────────────────────────

# A chart row makes the start points projective representatives.
_place_on_chart!(::Vector{Vector{ComplexF64}}, ::AbstractHomotopy)::Nothing = nothing

_place_on_chart!(
    points::Vector{Vector{ComplexF64}}, H::AffineChartHomotopy,
)::Nothing = _place_on_chart!(points, H.chart)

function _place_on_chart!(
        points::Vector{Vector{ComplexF64}}, chart::Vector{ComplexF64},
    )::Nothing
    for x in points
        on_chart!(x, chart)
    end
    return nothing
end

function _homotopy_cache(
        exec::AbstractExecutor, builder, H::AbstractHomotopy, starts,
        seed::UInt32, show_progress::Bool,
    )::SolveCache
    m, n = size(H)
    m >= n || throw(
        ArgumentError(
            "the homotopy has $m equation(s) in $n variables. The solution set is " *
                "positive-dimensional; only square or overdetermined homotopies with " *
                "finitely many solutions are supported.",
        ),
    )
    points = _start_points(starts)
    _check_start_length(points, n)
    _place_on_chart!(points, H)
    return _solve_cache(exec, builder, points, seed, nothing, show_progress)
end

"""
    solve(H::AbstractHomotopy, starts, exec = Serial(); options...)

Track the solutions `starts` of `H(x, 1)` to `t = 0`.

Start points are in `H`'s own coordinates, except for an
[`AffineChartHomotopy`](@ref), which takes projective representatives.

Anything but `Serial()` rebuilds `H` per task, since it owns the buffers it
evaluates through. Any homotopy can be rebuilt; a `_clone_homotopy` method for
your own type makes it cheaper by sharing its read-only data.

# Example
```julia
@polyvar x y a b
F = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
solve(ParameterHomotopy(F, [1, 0], [2, 4]), [[1, 1]])
```
"""
function solve(
        H::AbstractHomotopy,
        starts,
        exec::AbstractExecutor = Serial();
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = true,
    )::Result
    return CommonSolve.solve!(
        CommonSolve.init(
            H, starts, exec;
            seed = seed,
            tracker_options = tracker_options,
            endgame_options = endgame_options,
            show_progress = show_progress,
        ),
    )
end

function CommonSolve.init(
        H::AbstractHomotopy,
        starts,
        exec::AbstractExecutor = Serial();
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = true,
    )::SolveCache
    builder = _homotopy_builder(exec, H, tracker_options, endgame_options)
    return _homotopy_cache(exec, builder, H, starts, seed, show_progress)
end

_homotopy_builder(
    ::Serial, H::AbstractHomotopy, tracker_options::TrackerOptions,
    endgame_options::EndgameOptions,
) = SharedHomotopyBuilder(H, tracker_options, endgame_options)

function _homotopy_builder(
        ::AbstractExecutor, H::AbstractHomotopy, tracker_options::TrackerOptions,
        endgame_options::EndgameOptions,
    )
    _clone_homotopy(H)   # a homotopy that cannot be rebuilt fails here, not in a task
    return ClonedHomotopyBuilder(H, tracker_options, endgame_options)
end

"""
    solve(build_homotopy::Function, starts, exec; options...)

Track the solutions `starts` of `H(x, 1)` to `t = 0`, where
`H = build_homotopy()`, one homotopy per task.

Every call must allocate a homotopy sharing nothing mutable with the others. One
built around an existing system's evaluator does *not* qualify: the evaluator
carries the interpreter tapes.
"""
function solve(
        build_homotopy::Function,
        starts,
        exec::AbstractExecutor;
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = true,
    )::Result
    return CommonSolve.solve!(
        CommonSolve.init(
            build_homotopy, starts, exec;
            seed = seed,
            tracker_options = tracker_options,
            endgame_options = endgame_options,
            show_progress = show_progress,
        ),
    )
end

function CommonSolve.init(
        build_homotopy::Function,
        starts,
        exec::AbstractExecutor;
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = true,
    )::SolveCache
    H = build_homotopy()
    H isa AbstractHomotopy || throw(
        ArgumentError(
            "the first argument returned a $(typeof(H)); `solve(build_homotopy, " *
                "starts, exec)` needs a function returning an `AbstractHomotopy`.",
        ),
    )
    builder = HomotopyBuilder(build_homotopy, tracker_options, endgame_options)
    return _homotopy_cache(exec, builder, H, starts, seed, show_progress)
end
