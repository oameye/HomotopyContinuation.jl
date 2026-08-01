## Algorithm structs: what to compute and how, one per route family.
#
# Two rules split the option surface. The executor carries where the work runs
# (`ntasks`, `pids`, `tasks_per_process`, `batch_size`) and nothing else. The
# algorithm carries everything else, reporting included, so `solve` takes no
# keyword arguments and no keyword list is re-declared per route.

"""
    AbstractAlgorithm

Supertype of the algorithms [`solve`](@ref) dispatches on. Every subtype accepts
the [`CommonOptions`](@ref) keywords alongside its own.
"""
abstract type AbstractAlgorithm end

"""
    CommonOptions(; tracker_options, endgame_options, seed, show_progress, tracker keywords...)

The settings every algorithm carries: path-tracker and endgame options, the seed
every random choice descends from, and whether to draw a progress bar.

The [`TrackerOptions`](@ref) fields may be given directly instead of a pre-built
`tracker_options`, so `CommonOptions(; max_steps = 500)` and
`CommonOptions(; tracker_options = TrackerOptions(; max_steps = 500))` agree.
"""
struct CommonOptions
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
    seed::UInt32
    show_progress::Bool
end

# The only place the flattened `TrackerOptions` keywords are spelled out. Every
# algorithm takes a pre-built `tracker_options`, so none of them repeats this.
function CommonOptions(;
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        show_progress::Bool = true,
        max_steps::Int = tracker_options.max_steps,
        max_step_size::Float64 = tracker_options.max_step_size,
        max_initial_step_size::Float64 = tracker_options.max_initial_step_size,
        extended_precision::Bool = tracker_options.extended_precision,
        min_step_size::Float64 = tracker_options.min_step_size,
        terminate_cond::Float64 = tracker_options.terminate_cond,
        a::Float64 = tracker_options.a,
        β_a::Float64 = tracker_options.β_a,
        β_ω::Float64 = tracker_options.β_ω,
        β_τ::Float64 = tracker_options.β_τ,
        strict_β_τ::Float64 = tracker_options.strict_β_τ,
    )::CommonOptions
    opts = TrackerOptions(;
        max_steps, max_step_size, max_initial_step_size,
        extended_precision, min_step_size, terminate_cond,
        a, β_a, β_ω, β_τ, strict_β_τ,
    )
    return CommonOptions(opts, endgame_options, seed, show_progress)
end

"""
    PolynomialInput

Polynomial input a route accepts in place of a [`System`](@ref): a vector of
polynomials, or a single polynomial.
"""
const PolynomialInput = Union{
    AbstractVector{<:MP.AbstractPolynomialLike}, MP.AbstractPolynomialLike,
}

# One normalizer, so no route carries a duplicate method per input form. The
# identity method is unannotated on purpose: `::System` would assert the UnionAll
# and lose the concrete parameters that `WitnessSet{S}` return types depend on.
_as_system(F::System) = F
_as_system(F::AbstractVector{<:MP.AbstractPolynomialLike}) = System(collect(F))
_as_system(f::MP.AbstractPolynomialLike) = System([f])

_tracker_options(alg::AbstractAlgorithm)::TrackerOptions = alg.common.tracker_options
_endgame_options(alg::AbstractAlgorithm)::EndgameOptions = alg.common.endgame_options
_seed(alg::AbstractAlgorithm)::UInt32 = alg.common.seed
_show_progress(alg::AbstractAlgorithm)::Bool = alg.common.show_progress

"""
    seed(alg::AbstractAlgorithm) -> UInt32

The seed every random choice of `alg` descends from, so the same seed reproduces
the same run regardless of the state of the global random number generator.
"""
seed(alg::AbstractAlgorithm)::UInt32 = alg.common.seed

_with_seed(o::CommonOptions, seed::UInt32)::CommonOptions =
    CommonOptions(o.tracker_options, o.endgame_options, seed, o.show_progress)

# Routes that track nothing (`paths_to_track`) or track lazily
# (`result_iterator`) have no progress to report, so they silence the algorithm
# they were handed rather than rejecting one that asks for a bar.
_quiet(o::CommonOptions)::CommonOptions =
    CommonOptions(o.tracker_options, o.endgame_options, o.seed, false)

# ── Early stop ──────────────────────────────────────────────────────────────

"""
    EarlyStop

A callback deciding whether to stop a solve once a path has succeeded. Wrapped so
the algorithm carrying it stays a single concrete type whatever the callback is.
"""
const EarlyStop = FunctionWrapper{Bool, Tuple{PathResult}}

_never_stop(::PathResult)::Bool = false

_early_stop(f::EarlyStop)::EarlyStop = f
_early_stop(f)::EarlyStop = EarlyStop(f)

# `false` for every path, so a solve with no callback runs every start solution.
const NEVER_STOP = EarlyStop(_never_stop)

"""
    early_stop_callback(alg) -> EarlyStop

The early-stop callback of a tracking algorithm. Algorithms that track no paths of
their own report a callback that stops nothing.
"""
early_stop_callback(alg::AbstractAlgorithm)::EarlyStop = NEVER_STOP

# ── Excess-solution tolerance ───────────────────────────────────────────────

"""
    excess_residual_tol(alg) -> Float64

The bound on `‖F(x)‖∞` at which an endpoint that solves only the squared-up
system is still reported as a solution, read on the equations [`evaluate`](@ref)
gives rather than on the squared-up system the [`residual`](@ref) of a
[`PathResult`](@ref) is measured on. `0.0` for a strict check, and for every
algorithm that does not square a system up.
"""
excess_residual_tol(::AbstractAlgorithm)::Float64 = 0.0

function _checked_excess_residual_tol(tol::Float64)::Float64
    tol >= 0 || throw(
        ArgumentError("`excess_residual_tol` must be non-negative, got $tol"),
    )
    return tol
end

# ── Continuation: the homotopy is already determined by the arguments ────────

"""
    Continuation(; intrinsic, early_stop_callback, tracker_options, endgame_options, seed, show_progress)

Track given start solutions along the homotopy the remaining arguments determine:
a parameter homotopy, a subspace homotopy, a start/target system pair, or a
homotopy supplied directly. Nothing is constructed, unlike [`TotalDegree`](@ref)
and [`Polyhedral`](@ref).

`intrinsic` applies to subspace endpoints only and chooses the tracking
coordinates; it defaults to `dim(L_start) <= codim(L_start)`.

`early_stop_callback` is called with each successful [`PathResult`](@ref); return
`true` to stop. Paths already running still finish, so which extra results appear
is not reproducible under [`Threaded`](@ref) or [`DistributedExecutor`](@ref).
"""
struct Continuation{I <: Union{Nothing, Bool}} <: AbstractAlgorithm
    common::CommonOptions
    # `nothing` means "decide from the start subspace", which is not known here.
    intrinsic::I
    early_stop::EarlyStop
end

Continuation(;
    intrinsic::Union{Nothing, Bool} = nothing,
    early_stop_callback = _never_stop,
    tracker_options::TrackerOptions = TrackerOptions(),
    endgame_options::EndgameOptions = EndgameOptions(),
    seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    show_progress::Bool = true,
) = Continuation(
    CommonOptions(tracker_options, endgame_options, seed, show_progress),
    intrinsic,
    _early_stop(early_stop_callback),
)

early_stop_callback(alg::Continuation)::EarlyStop = alg.early_stop

_quiet(alg::Continuation)::Continuation =
    Continuation(_quiet(alg.common), alg.intrinsic, alg.early_stop)

# ── Sweep: one homotopy retargeted over many targets ────────────────────────

"""
    Sweep(; transform_result, transform_parameters, flatten, intrinsic, options...)

Track given start solutions to every target in a vector of targets, retargeting a
single homotopy per target, and return one entry per target.

`transform_result(result, target)` builds each entry, `flatten = true` concatenates
array-valued entries, and `transform_parameters(target)` maps each element of the
target vector to the actual target, so the vector may hold indices or other
metadata. The transforms exist so a long sweep need not retain every
[`Result`](@ref).
"""
struct Sweep{I <: Union{Nothing, Bool}, TR, TP} <: AbstractAlgorithm
    common::CommonOptions
    intrinsic::I
    transform_result::TR
    transform_parameters::TP
    flatten::Bool
end

Sweep(;
    transform_result = tuple,
    transform_parameters = identity,
    flatten::Bool = false,
    intrinsic::Union{Nothing, Bool} = nothing,
    tracker_options::TrackerOptions = TrackerOptions(),
    endgame_options::EndgameOptions = EndgameOptions(),
    seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    show_progress::Bool = true,
) = Sweep(
    CommonOptions(tracker_options, endgame_options, seed, show_progress),
    intrinsic, transform_result, transform_parameters, flatten,
)
