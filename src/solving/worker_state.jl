## Worker state types and system evaluator cloning for thread-safe path tracking.

# ── Worker state bundles ─────────────────────────────────────────────────────

"""
    TrackingWorkerState

Worker-local state for single-phase homotopy tracking (TotalDegree, parameter homotopy).
"""
struct TrackingWorkerState
    tracker::EndgameTracker
end

"""
    AmbientWorkerState{H}

Worker-local state for routes that track in ambient coordinates and keep the
concrete homotopy beside the tracker, so `target_parameters!` can retarget the
exact homotopy captured inside the tracker's `HomotopyEvaluator` closures.
"""
struct AmbientWorkerState{H}
    homotopy::H
    tracker::EndgameTracker
end

"""
    IntrinsicWorkerState

Worker-local state for tracking inside a linear subspace. Paths run in intrinsic
coordinates; `u` holds the converted start point, and endpoints are converted
back to ambient coordinates through `homotopy`.
"""
struct IntrinsicWorkerState
    homotopy::IntrinsicSubspaceHomotopy
    tracker::EndgameTracker
    u::FSVec{ComplexF64}
end

const RetargetWorkerState = Union{AmbientWorkerState, IntrinsicWorkerState}

"""
    PolyhedralWorkerState

Worker-local state for two-phase polyhedral homotopy tracking.
The `toric_homotopy` and `toric_tracker` are coupled — `update_weights!` must mutate
the exact `ToricHomotopy` captured inside the tracker's `HomotopyEvaluator` closures.
"""
struct PolyhedralWorkerState
    toric_homotopy::ToricHomotopy
    toric_tracker::Tracker
    coeff_tracker::EndgameTracker
    x_buffer::Vector{ComplexF64}
end

# ── System evaluator cloning ─────────────────────────────────────────────────

# The evaluator carries the thunk that rebuilds it, so the compile mode is
# already accounted for and a system only has to hand over its evaluator.
_clone_system_evaluator(sys::System)::SystemEvaluator =
    _clone_system_evaluator(sys.evaluator)
