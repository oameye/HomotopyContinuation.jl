# EndgameTracker — wraps Tracker with endgame detection and singular endpoint handling.

using LinearAlgebra: LinearAlgebra as LA

@kwdef struct EndgameOptions
    endgame_start::Float64 = 0.1
    max_endgame_steps::Int = 2000
    max_endgame_extended_steps::Int = 400
    only_nonsingular::Bool = false
    at_infinity_check::Bool = true
    zero_is_at_infinity::Bool = false
    max_winding_number::Int = 6
    val_finite_tol::Float64 = 0.05
    val_at_infinity_tol::Float64 = 0.01
    singular_min_accuracy::Float64 = 1.0e-6
    min_cond::Float64 = 1.0e6
    min_cond_growth::Float64 = 1.0e4
    min_coord_growth::Float64 = 100.0
    sing_cond::Float64 = inv(eps(Float64))
    sing_accuracy::Float64 = 1.0e-12
    scaling_threshold::Float64 = -30.0
    max_residual::Float64 = 1.0e-3
    refine_steps::Int = 3
    lambda::Float64 = 0.25
end

@enumx EndgameCode::Int8 begin
    TRACKING
    SUCCESS
    AT_INFINITY
    AT_ZERO
    TERMINATED_MAX_STEPS
    TERMINATED_MAX_EXTENDED_STEPS
    TERMINATED_MAX_WINDING_NUMBER
    TERMINATED_ACCURACY_LIMIT
    TERMINATED_ILL_CONDITIONED
    TERMINATED_INVALID_STARTVALUE
    TERMINATED_INVALID_STARTVALUE_SINGULAR_JACOBIAN
    TERMINATED_STEP_SIZE_TOO_SMALL
end

"""
    EndgameState

Mutable endgame-specific state. Wraps all per-path endgame tracking data.
"""
mutable struct EndgameState
    code::EndgameCode.T
    in_endgame::Bool
    in_singular_endgame::Bool
    winding_number::Int
    # Solution
    const solution::FSVec{ComplexF64}
    accuracy::Float64
    cond::Float64
    singular::Bool
    # Step counters
    steps_eg::Int
    ext_steps_eg_start::Int
    # Jump-to-zero tracking: (prev_prev, prev) history of whether the tracker
    # proposed a step reaching t=0. Used to gate singular endgame entry for
    # m=1 paths — prevents spurious singular endgame entry for regular paths.
    jump_to_zero_attempted::Tuple{Bool, Bool}
    # At-infinity per-coordinate tracking
    const at_inf_starts::FSVec{Float64}
    const at_inf_abs_coords::FSVec{Float64}
    const at_inf_conds::FSVec{Float64}
    const at_inf_active::Vector{Bool}
    # Singular endgame
    const samples::Vector{TaylorVector{2, ComplexF64}}
    const sample_times::FSVec{Float64}
    const sample_conds::FSVec{Float64}
    singular_start::Float64
    singular_steps::Int
    const prediction::FSVec{ComplexF64}
    const prev_prediction::FSVec{ComplexF64}
    prev_accuracy::Float64
    # Best singular prediction handed back to the tracker, so a later regular
    # endpoint cannot replace it with a worse one.
    const best_singular::FSVec{ComplexF64}
    best_singular_accuracy::Float64
    best_singular_winding::Int
    # Scaling for condition number. Every endgame decision reads the condition
    # number of the column-scaled Jacobian; `unit_scaling` is the all-ones row
    # scaling that says so. Skeel row scaling normalizes away the spread between
    # the row norms, which is the divergence signal a path running to infinity
    # leaves behind.
    const col_scaling::FSVec{Float64}
    const unit_scaling::FSVec{Float64}
end

function EndgameState(n::Int)
    return EndgameState(
        EndgameCode.TRACKING,
        false, false, 0,                                # code, in_endgame, in_singular, winding
        FSVec{ComplexF64}(zeros(ComplexF64, n)),         # solution
        NaN, 1.0, false,                                 # accuracy, cond, singular
        0, typemax(Int),                                 # steps_eg, ext_steps_eg_start
        (false, false),                                  # jump_to_zero_attempted
        FSVec{Float64}(fill(NaN, n)),                    # at_inf_starts
        FSVec{Float64}(fill(NaN, n)),                    # at_inf_abs_coords
        FSVec{Float64}(fill(NaN, n)),                    # at_inf_conds
        fill(false, n),                                  # at_inf_active
        [TaylorVector{2, ComplexF64}(n) for _ in 1:3],   # samples
        FSVec{Float64}(zeros(3)),                         # sample_times
        FSVec{Float64}(zeros(3)),                         # sample_conds
        NaN, 0,                                           # singular_start, singular_steps
        FSVec{ComplexF64}(zeros(ComplexF64, n)),          # prediction
        FSVec{ComplexF64}(zeros(ComplexF64, n)),          # prev_prediction
        Inf,                                              # prev_accuracy
        FSVec{ComplexF64}(zeros(ComplexF64, n)),          # best_singular
        Inf, 0,                                           # best_singular_accuracy, _winding
        FSVec{Float64}(zeros(n)),                         # col_scaling
        FSVec{Float64}(ones(n)),                          # unit_scaling
    )
end

struct EndgameTracker
    tracker::Tracker
    state::EndgameState
    val::Valuation
    options::EndgameOptions
end

function EndgameTracker(
        tracker::Tracker,
        options::EndgameOptions = EndgameOptions(),
    )
    n = length(tracker.state.x)
    return EndgameTracker(tracker, EndgameState(n), Valuation(n), options)
end

function endgame_tracker(
        H::AbstractHomotopy, tracker_options::TrackerOptions,
        endgame_options::EndgameOptions,
    )::EndgameTracker
    return EndgameTracker(
        Tracker(HomotopyEvaluator(H); options = tracker_options), endgame_options,
    )
end
