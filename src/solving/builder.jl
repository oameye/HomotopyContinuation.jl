## Builder structs: concrete callable types that produce fresh WorkerState from immutable data.

"""
    AbstractPathBuilder

Supertype of the callable builders a solve cache holds. Calling one returns a
worker state for one task, built from immutable data the builder stores.

Every subtype must return worker states sharing nothing mutable, so that any number
of them can track concurrently. A subtype that cannot must say so by defining
`_builds_independent_workers`; `SharedHomotopyBuilder` is the only one that does.
"""
abstract type AbstractPathBuilder end

# Whether repeated calls hand out worker states sharing nothing mutable, which is
# what lets the same paths be tracked on several tasks at once.
_builds_independent_workers(::AbstractPathBuilder)::Bool = true

"""
    StraightLineBuilder

Builder for TotalDegree homotopy. Stores immutable reconstruction data; each call
produces a fresh `TrackingWorkerState` with independent mutable state.
"""
struct StraightLineBuilder{S <: CloneableSystem} <: AbstractPathBuilder
    degrees::Vector{Int}
    target_system::S
    γ::ComplexF64
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::StraightLineBuilder)()::TrackingWorkerState
    start_eval = _total_degree_startevaluator(b.degrees)
    target_eval = _clone_system_evaluator(b.target_system)
    H = StraightLineHomotopy(start_eval, target_eval; γ = b.γ)
    eg = _endgame_tracker(H, b.tracker_options, b.endgame_options)
    return TrackingWorkerState(eg)
end

"""
    RandomizedStraightLineBuilder

Builder for TotalDegree homotopy against a squared-up overdetermined target.
The randomization block `A` and permutation are shared read-only across workers;
each worker gets a fresh `RandomizedSystem` (independent scratch buffers) around
a fresh clone of the target evaluator.
"""
struct RandomizedStraightLineBuilder{S <: CloneableSystem} <: AbstractPathBuilder
    degrees::Vector{Int}          # squared-up degrees (length n)
    target_system::S
    A::FSMat{ComplexF64}
    perm::Vector{Int}
    γ::ComplexF64
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::RandomizedStraightLineBuilder)()::TrackingWorkerState
    start_eval = _total_degree_startevaluator(b.degrees)
    inner_eval = _clone_system_evaluator(b.target_system)
    target_eval = _randomized_evaluator(inner_eval, b.A, b.perm)
    H = StraightLineHomotopy(start_eval, target_eval; γ = b.γ)
    eg = _endgame_tracker(H, b.tracker_options, b.endgame_options)
    return TrackingWorkerState(eg)
end

"""
    MultiHomogeneousBuilder

Builder for the multi-homogeneous total-degree homotopy. The start system is a
`System` like the target, so both are cloned per worker. `subspace` holds the
chart rows, one per variable group, and is codimension zero when the system is
not homogeneous; `perm` is empty unless the charted target is squared up.
"""
struct MultiHomogeneousBuilder{G <: System, S <: System} <: AbstractPathBuilder
    start_system::G
    target_system::S
    A::FSMat{ComplexF64}
    perm::Vector{Int}
    subspace::LinearSubspace{ComplexF64}
    γ::ComplexF64
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::MultiHomogeneousBuilder)()::TrackingWorkerState
    start_eval = _clone_system_evaluator(b.start_system)
    target_eval = _clone_system_evaluator(b.target_system)
    codim(b.subspace) == 0 ||
        (target_eval = _sliced_evaluator(target_eval, b.subspace, ComplexF64[]))
    isempty(b.perm) ||
        (target_eval = _randomized_evaluator(target_eval, b.A, b.perm))
    H = StraightLineHomotopy(start_eval, target_eval; γ = b.γ)
    eg = _endgame_tracker(H, b.tracker_options, b.endgame_options)
    return TrackingWorkerState(eg)
end

"""
    StartTargetBuilder

Builder for the straight-line homotopy between two parameter-free systems.
`chart` is empty unless they are homogeneous, and goes on the homotopy.
"""
struct StartTargetBuilder{G <: CloneableSystem, S <: CloneableSystem} <: AbstractPathBuilder
    start_system::G
    target_system::S
    chart::Vector{ComplexF64}
    γ::ComplexF64
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::StartTargetBuilder)()::TrackingWorkerState
    H = StraightLineHomotopy(
        _clone_system_evaluator(b.start_system),
        _clone_system_evaluator(b.target_system);
        γ = b.γ,
    )
    eg = isempty(b.chart) ?
        _endgame_tracker(H, b.tracker_options, b.endgame_options) :
        _endgame_tracker(
            AffineChartHomotopy(H, b.chart), b.tracker_options, b.endgame_options,
        )
    return TrackingWorkerState(eg)
end

"""
    SharedHomotopyBuilder

Builder wrapping a caller's homotopy as it stands, so a cache holding one runs
on one task. [`ClonedHomotopyBuilder`](@ref) is what the other executors get.
"""
struct SharedHomotopyBuilder{H <: AbstractHomotopy} <: AbstractPathBuilder
    homotopy::H
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

(b::SharedHomotopyBuilder)()::TrackingWorkerState =
    TrackingWorkerState(_endgame_tracker(b.homotopy, b.tracker_options, b.endgame_options))

# The one exception to the `AbstractPathBuilder` contract: this builder wraps the
# caller's homotopy as it stands, so two of its workers hold one evaluator's tapes.
_builds_independent_workers(::SharedHomotopyBuilder)::Bool = false

"""
    ClonedHomotopyBuilder

Builder rebuilding a caller's homotopy per task, so the caller's own is never
tracked.
"""
struct ClonedHomotopyBuilder{H <: AbstractHomotopy} <: AbstractPathBuilder
    homotopy::H
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

(b::ClonedHomotopyBuilder)()::TrackingWorkerState = TrackingWorkerState(
    _endgame_tracker(
        _clone_homotopy(b.homotopy), b.tracker_options, b.endgame_options,
    ),
)

"""
    HomotopyBuilder

Builder calling `build()` per task, which must return a fresh homotopy.
"""
struct HomotopyBuilder{F <: Function} <: AbstractPathBuilder
    build::F
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

(b::HomotopyBuilder)()::TrackingWorkerState =
    TrackingWorkerState(_endgame_tracker(b.build(), b.tracker_options, b.endgame_options))

"""
    ParameterBuilder

Builder for parameter homotopy via ParameterHomotopy (general parameter
dependence; CoefficientHomotopy's tangent shortcut is only valid for systems
linear and homogeneous in the parameters).
"""
struct ParameterBuilder{S <: SystemLike} <: AbstractPathBuilder
    param_system::S
    start_parameters::Vector{ComplexF64}
    target_parameters::Vector{ComplexF64}
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::ParameterBuilder)()::TrackingWorkerState
    sys_eval = _clone_system_evaluator(b.param_system)
    H = ParameterHomotopy(sys_eval, b.start_parameters, b.target_parameters)
    eg = _endgame_tracker(H, b.tracker_options, b.endgame_options)
    return TrackingWorkerState(eg)
end

"""
    SlicedStraightLineBuilder

Builder for the total-degree homotopy against `[F; A x − b]`, where the sliced
target is a [`SlicedSystem`](@ref) wrapping a fresh clone of `F`'s evaluator.
Each worker gets independent interpreter tapes and its own copy of the linear
block; the polynomials are never rebuilt.
"""
struct SlicedStraightLineBuilder{S <: System} <: AbstractPathBuilder
    degrees::Vector{Int}
    target_system::S
    subspace::LinearSubspace{ComplexF64}
    chart::Vector{ComplexF64}
    γ::ComplexF64
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::SlicedStraightLineBuilder)()::TrackingWorkerState
    start_eval = _total_degree_startevaluator(b.degrees)
    inner_eval = _clone_system_evaluator(b.target_system)
    target_eval = _sliced_evaluator(inner_eval, b.subspace, b.chart)
    H = StraightLineHomotopy(start_eval, target_eval; γ = b.γ)
    eg = _endgame_tracker(H, b.tracker_options, b.endgame_options)
    return TrackingWorkerState(eg)
end

"""
    ParameterRetargetBuilder

Builder for a parameter homotopy whose target is retargeted between solves.
Unlike [`ParameterBuilder`](@ref) it returns an [`AmbientWorkerState`](@ref), so
the caller keeps the concrete `ParameterHomotopy` handle.
"""
struct ParameterRetargetBuilder{S <: System} <: AbstractPathBuilder
    param_system::S
    start_parameters::Vector{ComplexF64}
    target_parameters::Vector{ComplexF64}
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::ParameterRetargetBuilder)()::AmbientWorkerState{ParameterHomotopy}
    sys_eval = _clone_system_evaluator(b.param_system)
    H = ParameterHomotopy(sys_eval, b.start_parameters, b.target_parameters)
    eg = _endgame_tracker(H, b.tracker_options, b.endgame_options)
    return AmbientWorkerState(H, eg)
end

"""
    ExtrinsicSubspaceBuilder

Builder for moving ambient points from one linear subspace to another via
`ExtrinsicSubspaceHomotopy`. `gamma` is shared across workers so every path
traces the same perturbed homotopy.
"""
struct ExtrinsicSubspaceBuilder{S <: System} <: AbstractPathBuilder
    system::S
    start::LinearSubspace{ComplexF64}
    target::LinearSubspace{ComplexF64}
    gamma::ComplexF64
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::ExtrinsicSubspaceBuilder)()::AmbientWorkerState{ExtrinsicSubspaceHomotopy}
    ev = _clone_system_evaluator(b.system)
    H = ExtrinsicSubspaceHomotopy(ev, b.start, b.target; gamma = b.gamma)
    eg = _endgame_tracker(H, b.tracker_options, b.endgame_options)
    return AmbientWorkerState(H, eg)
end

"""
    ChartExtrinsicSubspaceBuilder

`ExtrinsicSubspaceBuilder` for a projective problem: the ambient system is
positive-dimensional, so the homotopy is wrapped in an `AffineChartHomotopy`.
A separate type keeps the produced worker-state type concrete.
"""
struct ChartExtrinsicSubspaceBuilder{S <: System} <: AbstractPathBuilder
    system::S
    start::LinearSubspace{ComplexF64}
    target::LinearSubspace{ComplexF64}
    chart::Vector{ComplexF64}
    gamma::ComplexF64
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::ChartExtrinsicSubspaceBuilder)()::AmbientWorkerState{
        AffineChartHomotopy{ExtrinsicSubspaceHomotopy},
    }
    ev = _clone_system_evaluator(b.system)
    H = AffineChartHomotopy(
        ExtrinsicSubspaceHomotopy(ev, b.start, b.target; gamma = b.gamma), b.chart,
    )
    eg = _endgame_tracker(H, b.tracker_options, b.endgame_options)
    return AmbientWorkerState(H, eg)
end

"""
    IntrinsicSubspaceBuilder

Builder for tracking inside a linear subspace via `IntrinsicSubspaceHomotopy`.
For a projective problem `chart` is nonempty and the chart row is appended to
the *system*, which leaves the homotopy type unchanged.
"""
struct IntrinsicSubspaceBuilder{S <: System} <: AbstractPathBuilder
    system::S
    start::LinearSubspace{ComplexF64}
    target::LinearSubspace{ComplexF64}
    chart::Vector{ComplexF64}
    gamma::ComplexF64
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::IntrinsicSubspaceBuilder)()::IntrinsicWorkerState
    ev = _clone_system_evaluator(b.system)
    inner = isempty(b.chart) ? ev : SystemEvaluator(AffineChartSystem(ev, b.chart))
    H = IntrinsicSubspaceHomotopy(inner, b.start, b.target; gamma = b.gamma)
    eg = _endgame_tracker(H, b.tracker_options, b.endgame_options)
    return IntrinsicWorkerState(
        H, eg, FSVec{ComplexF64}(zeros(ComplexF64, size(H)[2])),
    )
end

"""
    PolyhedralBuilder

Builder for two-phase polyhedral homotopy. Returns `PolyhedralWorkerState` with
coupled toric homotopy + tracker.

The coefficient vectors are shared read-only across workers. Thread safety relies on
`ToricHomotopy` and `CoefficientHomotopy` constructors copying into independent buffers.
"""
struct PolyhedralBuilder{S} <: AbstractPathBuilder
    support_system::S
    start_coeffs::Vector{Vector{ComplexF64}}
    flat_start::Vector{ComplexF64}
    flat_target::Vector{ComplexF64}
    toric_options::TrackerOptions
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::PolyhedralBuilder)()::PolyhedralWorkerState
    # Phase 1: toric — homotopy and tracker are coupled
    toric_eval = _clone_system_evaluator(b.support_system)
    toric_H = ToricHomotopy(toric_eval, b.start_coeffs)
    toric_tracker = Tracker(HomotopyEvaluator(toric_H); options = b.toric_options)

    # Phase 2: coefficient
    coeff_eval = _clone_system_evaluator(b.support_system)
    coeff_H = CoefficientHomotopy(coeff_eval, b.flat_start, b.flat_target)
    coeff_tracker = _endgame_tracker(coeff_H, b.tracker_options, b.endgame_options)

    n = size(toric_eval)[2]
    return PolyhedralWorkerState(
        toric_H, toric_tracker, coeff_tracker, Vector{ComplexF64}(undef, n),
    )
end
