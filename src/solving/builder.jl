## Builder structs: concrete callable types that produce fresh WorkerState from immutable data.

"""
    StraightLineBuilder

Builder for TotalDegree homotopy. Stores immutable reconstruction data; each call
produces a fresh `TrackingWorkerState` with independent mutable state.
"""
struct StraightLineBuilder{S <: CloneableSystem}
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
struct RandomizedStraightLineBuilder{S <: CloneableSystem}
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
    ParameterBuilder

Builder for parameter homotopy via ParameterHomotopy (general parameter
dependence; CoefficientHomotopy's tangent shortcut is only valid for systems
linear and homogeneous in the parameters).
"""
struct ParameterBuilder{S <: SystemLike}
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
struct SlicedStraightLineBuilder{S <: System}
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
struct ParameterRetargetBuilder{S <: System}
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
struct ExtrinsicSubspaceBuilder{S <: System}
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
struct ChartExtrinsicSubspaceBuilder{S <: System}
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
struct IntrinsicSubspaceBuilder{S <: System}
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
struct PolyhedralBuilder{S}
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
