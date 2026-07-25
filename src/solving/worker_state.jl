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

"""
    _clone_system_evaluator(sys::System) -> SystemEvaluator

Create a fresh `SystemEvaluator` from `sys`'s instruction sequences. The new evaluator
has independent interpreter tapes (mutable) but shares the instruction sequences (immutable).
Preserves the compile mode: INTERPRETED rebuilds all 6 interpreters, COMPILED rebuilds
4 interpreters + re-generates @RuntimeGeneratedFunctions for eval/jac, COMPILED_ALL
re-generates everything including Taylor via RuntimeGeneratedFunctions.
"""
function _clone_system_evaluator(
        sys::System{P, V, CompileMode.INTERPRETED, S},
    )::SystemEvaluator where {P, V, S}
    seq_eval = sys._interp_f64.sequence
    seq_jac = sys._interp_jac.sequence
    m, n = size(sys.evaluator)
    np = nparameters(sys.evaluator)
    return _build_system_evaluator(
        Interpreter(Vector{ComplexF64}, seq_eval),
        Interpreter(Vector{ComplexDF64}, seq_eval),
        Interpreter(Vector{ComplexF64}, seq_jac),
        Interpreter(Vector{TruncatedTaylorSeries{2, ComplexF64}}, seq_eval),
        Interpreter(Vector{TruncatedTaylorSeries{3, ComplexF64}}, seq_eval),
        Interpreter(Vector{TruncatedTaylorSeries{4, ComplexF64}}, seq_eval),
        m, n, np,
    )
end

function _clone_system_evaluator(
        sys::System{P, V, CompileMode.COMPILED, S},
    )::SystemEvaluator where {P, V, S}
    seq_eval = sys._interp_f64.sequence
    seq_jac = sys._interp_jac.sequence
    m, n = size(sys.evaluator)
    np = nparameters(sys.evaluator)
    return _build_compiled_evaluator(
        seq_eval, seq_jac,
        Interpreter(Vector{ComplexDF64}, seq_eval),
        _build_taylor_fws(
            Interpreter(Vector{TruncatedTaylorSeries{2, ComplexF64}}, seq_eval),
            Interpreter(Vector{TruncatedTaylorSeries{3, ComplexF64}}, seq_eval),
            Interpreter(Vector{TruncatedTaylorSeries{4, ComplexF64}}, seq_eval),
        ),
        m, n, np,
    )
end

function _clone_system_evaluator(
        sys::System{P, V, CompileMode.COMPILED_ALL, S},
    )::SystemEvaluator where {P, V, S}
    seq_eval = sys._interp_f64.sequence
    seq_jac = sys._interp_jac.sequence
    m, n = size(sys.evaluator)
    np = nparameters(sys.evaluator)
    return _build_compiled_evaluator(
        seq_eval, seq_jac,
        Interpreter(Vector{ComplexDF64}, seq_eval),
        _build_taylor_fws(seq_eval),
        m, n, np,
    )
end
