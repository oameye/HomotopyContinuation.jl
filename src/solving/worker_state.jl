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
function _clone_system_evaluator(sys::System)::SystemEvaluator
    seq_eval = sys._interp_f64.sequence
    seq_jac = sys._interp_jac.sequence
    m, n = size(sys.evaluator)
    np = nparameters(sys.evaluator)

    if sys.compile_mode == CompileMode.INTERPRETED
        return _build_system_evaluator(
            Interpreter(Vector{ComplexF64}, seq_eval),
            Interpreter(Vector{ComplexDF64}, seq_eval),
            Interpreter(Vector{ComplexF64}, seq_jac),
            Interpreter(Vector{TruncatedTaylorSeries{2, ComplexF64}}, seq_eval),
            Interpreter(Vector{TruncatedTaylorSeries{3, ComplexF64}}, seq_eval),
            Interpreter(Vector{TruncatedTaylorSeries{4, ComplexF64}}, seq_eval),
            m, n, np,
        )
    elseif sys.compile_mode == CompileMode.COMPILED
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
    else  # CompileMode.COMPILED_ALL
        return _build_compiled_evaluator(
            seq_eval, seq_jac,
            Interpreter(Vector{ComplexDF64}, seq_eval),
            _build_taylor_fws(seq_eval),
            m, n, np,
        )
    end
end
