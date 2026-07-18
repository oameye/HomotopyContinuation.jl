## Builder structs — concrete callable types that produce fresh WorkerState from immutable data.

"""
    StraightLineBuilder

Builder for TotalDegree homotopy. Stores immutable reconstruction data; each call
produces a fresh `TrackingWorkerState` with independent mutable state.
"""
struct StraightLineBuilder{S <: System}
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
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = b.tracker_options)
    eg = EndgameTracker(tracker, b.endgame_options)
    return TrackingWorkerState(eg)
end

"""
    RandomizedStraightLineBuilder

Builder for TotalDegree homotopy against a squared-up overdetermined target.
The randomization block `A` and permutation are shared read-only across workers;
each worker gets a fresh `RandomizedSystem` (independent scratch buffers) around
a fresh clone of the target evaluator.
"""
struct RandomizedStraightLineBuilder{S <: System}
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
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = b.tracker_options)
    eg = EndgameTracker(tracker, b.endgame_options)
    return TrackingWorkerState(eg)
end

"""
    ParameterBuilder

Builder for parameter homotopy via ParameterHomotopy (general parameter
dependence; CoefficientHomotopy's tangent shortcut is only valid for systems
linear and homogeneous in the parameters).
"""
struct ParameterBuilder{S <: System}
    param_system::S
    start_parameters::Vector{ComplexF64}
    target_parameters::Vector{ComplexF64}
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::ParameterBuilder)()::TrackingWorkerState
    sys_eval = _clone_system_evaluator(b.param_system)
    H = ParameterHomotopy(sys_eval, b.start_parameters, b.target_parameters)
    heval = HomotopyEvaluator(H)
    tracker = Tracker(heval; options = b.tracker_options)
    eg = EndgameTracker(tracker, b.endgame_options)
    return TrackingWorkerState(eg)
end

"""
    PolyhedralBuilder

Builder for two-phase polyhedral homotopy. Returns `PolyhedralWorkerState` with
coupled toric homotopy + tracker.

The coefficient vectors are shared read-only across workers. Thread safety relies on
`ToricHomotopy` and `CoefficientHomotopy` constructors copying into independent buffers.
"""
struct PolyhedralBuilder{S <: System}
    param_system::S
    start_coeffs::Vector{Vector{ComplexF64}}
    flat_start::Vector{ComplexF64}
    flat_target::Vector{ComplexF64}
    toric_options::TrackerOptions
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

function (b::PolyhedralBuilder)()::PolyhedralWorkerState
    # Phase 1: toric — homotopy and tracker are coupled
    toric_eval = _clone_system_evaluator(b.param_system)
    toric_H = ToricHomotopy(toric_eval, b.start_coeffs)
    toric_tracker = Tracker(HomotopyEvaluator(toric_H); options = b.toric_options)

    # Phase 2: coefficient
    coeff_eval = _clone_system_evaluator(b.param_system)
    coeff_H = CoefficientHomotopy(coeff_eval, b.flat_start, b.flat_target)
    coeff_tracker = EndgameTracker(
        Tracker(HomotopyEvaluator(coeff_H); options = b.tracker_options),
        b.endgame_options,
    )

    n = size(toric_eval)[2]
    return PolyhedralWorkerState(
        toric_H, toric_tracker, coeff_tracker, Vector{ComplexF64}(undef, n),
    )
end
