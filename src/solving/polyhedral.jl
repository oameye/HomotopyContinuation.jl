## Polyhedral — BKK-optimal start system via mixed subdivisions.
#
# Two-phase polyhedral homotopy:
#   Phase 1 (toric):       Track from binomial start solutions through ToricHomotopy (t: 0 -> 1)
#   Phase 2 (coefficient): Track from generic system to target through CoefficientHomotopy (t: 1 -> 0)

"""
    Polyhedral(; seed, max_steps, extended_precision, ...)

Algorithm that constructs a polyhedral (BKK-optimal) start system using mixed subdivisions.
The number of paths tracked equals the mixed volume, which is at most the Bezout bound.

Accepts all `TrackerOptions` fields as keyword arguments, or a pre-built
`tracker_options` object.

# Examples
```julia
@polyvar x y
result = solve(System([x^2 + y - 1, x*y - 2]), Polyhedral())

# Tune tracker options directly
result = solve(F, Polyhedral(; max_steps=500, extended_precision=false))
```
"""
struct Polyhedral
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
    seed::UInt32
end

function Polyhedral(;
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
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
    )
    opts = TrackerOptions(;
        max_steps, max_step_size, max_initial_step_size,
        extended_precision, min_step_size, terminate_cond,
        a, β_a, β_ω, β_τ, strict_β_τ,
    )
    return Polyhedral(opts, endgame_options, seed)
end

"""
    PolyhedralSolveCache

Holds pre-built trackers and start solutions for the two-phase polyhedral homotopy.
Created by `CommonSolve.init`.
"""
struct PolyhedralSolveCache{S <: System}
    toric_tracker::Tracker
    coeff_tracker::EndgameTracker
    toric_homotopy::ToricHomotopy
    support::Vector{Matrix{Int32}}
    lifting::Vector{Vector{Int32}}
    start_solutions::Vector{Tuple{MixedSubdivisions.MixedCell, Vector{ComplexF64}}}
    seed::UInt32
    # GC roots for interpreters — must be kept alive for FunctionWrapper closures
    _param_system::S
end

# ── Helper: build parametric system from support ────────────────────────────

"""
    _build_parametric_system(support, variables) -> (polys, variables, coeff_params)

Build MP polynomials with symbolic coefficient parameters from the given support matrices.
Returns `(polys, variables, coeff_params)` where `coeff_params` is a flat vector of
parameter variables `[c_1_1, c_1_2, ..., c_n_m_n]`.
"""
function _build_parametric_system(
        support::Vector{Matrix{Int32}},
        variables::AbstractVector,
    )
    n = length(support)
    n_coeffs = sum(size(A, 2) for A in support)

    # Create parameter variables programmatically using DynamicPolynomials
    VarType = typeof(variables[1])
    coeff_vars = VarType[
        VarType("_hc_c$(i)_$(j)") for i in 1:n for j in 1:size(support[i], 2)
    ]

    # Build polynomials: F_i = sum_j coeff_vars[offset+j] * prod(x_k^A[k,j])
    # Build first polynomial to infer the concrete Polynomial type
    m_1 = size(support[1], 2)
    p1 = coeff_vars[1] * prod(variables[k]^Int(support[1][k, 1]) for k in 1:n)
    for j in 2:m_1
        monomial = prod(variables[k]^Int(support[1][k, j]) for k in 1:n)
        p1 = p1 + coeff_vars[j] * monomial
    end
    PolyType = typeof(p1)
    polys = PolyType[p1]
    offset = m_1
    for i in 2:n
        A = support[i]
        m_i = size(A, 2)
        p = coeff_vars[offset + 1] * prod(variables[k]^Int(A[k, 1]) for k in 1:n)
        for j in 2:m_i
            monomial = prod(variables[k]^Int(A[k, j]) for k in 1:n)
            p = p + coeff_vars[offset + j] * monomial
        end
        push!(polys, p)
        offset += m_i
    end

    return polys, collect(variables), coeff_vars
end

# ── CommonSolve.init: polys + Polyhedral ────────────────────────────────────

function CommonSolve.init(F::System, alg::Polyhedral)::PolyhedralSolveCache
    seed = alg.seed
    n = F.nvars

    # Seed the global RNG so that coefficient generation and MixedSubdivisions
    # produce reproducible results for a given seed.
    Random.seed!(seed)

    # 1. Get support + target coefficients from the System
    source_support, source_coeffs = support_coefficients(F)

    # 2. Generate start coefficients for ORIGINAL support FIRST.
    #    Coefficients must be generated before zero column addition so the RNG
    #    stream is deterministic. cospi/sinpi give exact values at rational π.
    start_coeffs_orig = Vector{Vector{ComplexF64}}(undef, length(source_coeffs))
    for i in eachindex(source_coeffs)
        c = source_coeffs[i]
        nrm = LinearAlgebra.norm(c, Inf)
        start_coeffs_orig[i] = ComplexF64[
            let r = rand(), φ = rand()
                    (0.9 + 0.2 * r) * complex(cospi(2φ), sinpi(2φ)) * nrm
            end
                for _ in 1:length(c)
        ]
    end

    # 3. Add zero columns to support and extend coefficients.
    #    Zero-column extensions use randn(ComplexF64) for start, 0.0 for target.
    support = Vector{Matrix{Int32}}(undef, length(source_support))
    target_coeffs = Vector{Vector{ComplexF64}}(undef, length(source_coeffs))
    start_coeffs = Vector{Vector{ComplexF64}}(undef, length(source_coeffs))
    for (i, A) in enumerate(source_support)
        if has_zero_column(A)
            support[i] = A
            target_coeffs[i] = source_coeffs[i]
            start_coeffs[i] = start_coeffs_orig[i]
        else
            support[i] = hcat(A, zeros(Int32, size(A, 1)))
            target_coeffs[i] = push!(copy(source_coeffs[i]), zero(ComplexF64))
            start_coeffs[i] = vcat(start_coeffs_orig[i], randn(ComplexF64))
        end
    end

    # 4. Compute mixed cells via MixedSubdivisions
    result = MixedSubdivisions.fine_mixed_cells(support; show_progress = false)
    if result === nothing
        error("MixedSubdivisions.fine_mixed_cells returned nothing — could not compute mixed cells")
    end
    mixed_cells, lifting = result

    if isempty(mixed_cells)
        error("No mixed cells found — the system may have no isolated solutions")
    end

    # 5. Solve binomial systems for each mixed cell
    max_d_hat = maximum(c.volume for c in mixed_cells)
    BSS = BinomialSystemSolver(n; max_d_hat = max_d_hat)
    X = Matrix{ComplexF64}(undef, n, max_d_hat)

    all_starts = Tuple{MixedSubdivisions.MixedCell, Vector{ComplexF64}}[]
    for cell in mixed_cells
        d_hat = solve_binomial!(X, BSS, support, start_coeffs, cell)
        for j in 1:d_hat
            x = ComplexF64[X[i, j] for i in 1:n]
            push!(all_starts, (cell, x))
        end
    end

    # 6. Build parametric system from support
    @polyvar _hc_x[1:n]
    param_polys, param_vars, coeff_params =
        _build_parametric_system(support, collect(_hc_x))
    param_system = System(
        param_polys; variables = param_vars, parameters = coeff_params,
        compile = CompileMode.COMPILED,
    )

    # 7. Build toric homotopy (phase 1: t goes from 0 to 1)
    #    Toric tracker uses conservative max_initial_step_size=0.2
    toric_H = ToricHomotopy(param_system.evaluator, start_coeffs)
    toric_heval = HomotopyEvaluator(toric_H)
    toric_opts = TrackerOptions(;
        max_steps = alg.tracker_options.max_steps,
        max_step_size = alg.tracker_options.max_step_size,
        max_initial_step_size = min(alg.tracker_options.max_initial_step_size, 0.2),
        extended_precision = alg.tracker_options.extended_precision,
        min_step_size = alg.tracker_options.min_step_size,
        terminate_cond = alg.tracker_options.terminate_cond,
        a = alg.tracker_options.a,
        β_a = alg.tracker_options.β_a,
        β_ω = alg.tracker_options.β_ω,
        β_τ = alg.tracker_options.β_τ,
        strict_β_τ = alg.tracker_options.strict_β_τ,
    )
    toric_tracker = Tracker(toric_heval; options = toric_opts)

    # 8. Build coefficient homotopy (phase 2: t goes from 1 to 0)
    flat_start = reduce(vcat, start_coeffs)
    flat_target = reduce(vcat, target_coeffs)
    coeff_H = CoefficientHomotopy(param_system.evaluator, flat_start, flat_target)
    coeff_heval = HomotopyEvaluator(coeff_H)
    coeff_tracker = EndgameTracker(
        Tracker(coeff_heval; options = alg.tracker_options),
        alg.endgame_options,
    )

    return PolyhedralSolveCache(
        toric_tracker, coeff_tracker, toric_H,
        support, lifting,
        all_starts, seed,
        param_system,
    )
end

# ── Toric phase tracking with two-stage reparameterization ─────────────────

"""
    _init_toric!(tracker, x₀, t_start, t_end)

Initialize the toric tracker with v2-tuned initial parameters:
ω=20 (optimistic initial Lipschitz), μ=1e-12, max_initial_step_size=0.2.
"""
function _init_toric!(
        tracker::Tracker,
        x₀::AbstractVector{<:Number},
        t_start::ComplexF64,
        t_end::ComplexF64,
    )::TrackerCode.T
    # Pass ω/μ directly so init! skips init_newton — the toric phase uses
    # empirically-tuned parameters since exact start solutions are known
    return init!(
        tracker, x₀, t_start, t_end;
        ω = 20.0, μ = 1.0e-12, max_initial_step_size = 0.2,
    )
end

"""
    _track_toric_phase!(tracker, H, x₀, min_weight, max_weight, support, lifting, cell)

Track the toric phase with two-stage reparameterization for large weights.
When max_weight >= 10, the path is split into two stages to avoid numerical
issues from t^w where w is large and t is near 1.
"""
function _track_toric_phase!(
        tracker::Tracker,
        H::ToricHomotopy,
        x₀::AbstractVector{ComplexF64},
        min_weight::Float64,
        max_weight::Float64,
        support::Vector{Matrix{Int32}},
        lifting::Vector{Vector{Int32}},
        cell::MixedSubdivisions.MixedCell,
    )::TrackerCode.T

    if max_weight < 10.0
        # Simple case: track directly from 0 to 1
        code = _init_toric!(tracker, x₀, complex(0.0), complex(1.0))
        if code != TrackerCode.TRACKING
            return code
        end
        while tracker.state.code == TrackerCode.TRACKING
            step!(tracker)
        end
        return tracker.state.code
    end

    # Two-stage reparameterization for large weights:
    # Stage 1: track from 0 to t₀
    t₀ = clamp(0.1^(10.0 / max_weight), 0.9, 1.0 - 1.0e-6)
    code = _init_toric!(tracker, x₀, complex(0.0), complex(t₀))
    if code != TrackerCode.TRACKING
        return code
    end
    while tracker.state.code == TrackerCode.TRACKING
        step!(tracker)
    end
    if tracker.state.code != TrackerCode.TRACKER_SUCCESS
        return tracker.state.code
    end

    # Stage 2: renormalize weights with max_weight=10, track from t_restart to 1
    saved_ω = tracker.state.ω
    saved_μ = tracker.state.μ

    new_min_w, _ = update_weights!(H, support, lifting, cell; max_weight = 10.0)
    t_restart = t₀^(1.0 / new_min_w)

    # Re-init tracker from current solution at t_restart to 1.0
    # Pass ω/μ from stage 1, keep step counters
    code = init!(
        tracker, tracker.state.x, complex(t_restart), complex(1.0);
        ω = saved_ω, μ = saved_μ,
        keep_steps = true,
    )

    if code != TrackerCode.TRACKING
        return code
    end
    while tracker.state.code == TrackerCode.TRACKING
        step!(tracker)
    end
    return tracker.state.code
end

# ── CommonSolve.solve!: two-phase path tracking ────────────────────────────

function CommonSolve.solve!(cache::PolyhedralSolveCache)::Result
    toric_tracker = cache.toric_tracker
    coeff_tracker = cache.coeff_tracker
    toric_H = cache.toric_homotopy
    support = cache.support
    lifting = cache.lifting

    n_paths = length(cache.start_solutions)
    path_results = PathResult[]
    sizehint!(path_results, n_paths)

    # Pre-allocate buffer for passing solutions between phases
    n = size(support[1], 1)
    x_buffer = Vector{ComplexF64}(undef, n)

    # Phase 1 + Phase 2 for each start solution
    for (cell, x₀) in cache.start_solutions
        # Phase 1: Toric homotopy — track from t=0 to t=1
        #
        # Strategy:
        #   a) Normalize weights so min non-zero weight = 1.
        #   b) If max_weight < 10: track directly from 0 to 1.
        #   c) If max_weight >= 10: two-stage reparameterization to avoid
        #      t^w precision loss for large w (see _track_toric_phase!).
        min_w, max_w = update_weights!(toric_H, support, lifting, cell; min_weight = 1.0)

        code = _track_toric_phase!(
            toric_tracker, toric_H, x₀, min_w, max_w,
            support, lifting, cell
        )

        if code != TrackerCode.TRACKER_SUCCESS
            # Toric phase failed — record failure and skip coefficient phase
            push!(path_results, PathResult(toric_tracker))
            continue
        end

        # Save toric-phase state before it's lost to the coefficient phase
        toric_accepted = toric_tracker.state.accepted_steps
        toric_rejected = toric_tracker.state.rejected_steps

        # Extract solution from toric phase into pre-allocated buffer
        copyto!(x_buffer, toric_tracker.state.x)

        # Phase 2: Coefficient homotopy — track from t=1 to t=0
        # Carry over the toric-phase accuracy estimate to the coefficient phase.
        init!(coeff_tracker, x_buffer; μ = toric_tracker.state.μ)
        while coeff_tracker.state.code == EndgameCode.TRACKING
            step!(coeff_tracker)
        end

        # Accumulate toric-phase steps into the final PathResult
        push!(path_results, _add_steps(PathResult(coeff_tracker), toric_accepted, toric_rejected))
    end

    return Result(path_results, n_paths, cache.seed)
end
