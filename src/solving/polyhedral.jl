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

    # 1. Get support + target coefficients from the System
    #    Copy only entries that need modification (zero column addition).
    source_support, source_coeffs = support_coefficients(F)
    support = Vector{Matrix{Int32}}(undef, length(source_support))
    target_coeffs = Vector{Vector{ComplexF64}}(undef, length(source_coeffs))
    for (i, A) in enumerate(source_support)
        if has_zero_column(A)
            support[i] = A
            target_coeffs[i] = source_coeffs[i]
        else
            support[i] = hcat(A, zeros(Int32, size(A, 1)))
            target_coeffs[i] = push!(copy(source_coeffs[i]), zero(ComplexF64))
        end
    end

    # 3. Compute mixed cells via MixedSubdivisions
    result = MixedSubdivisions.fine_mixed_cells(support)
    if result === nothing
        error("MixedSubdivisions.fine_mixed_cells returned nothing — could not compute mixed cells")
    end
    mixed_cells, lifting = result

    if isempty(mixed_cells)
        error("No mixed cells found — the system may have no isolated solutions")
    end

    # 4. Generate random start coefficients (near unit magnitude, random phase)
    rng = Random.MersenneTwister(seed)
    start_coeffs = Vector{Vector{ComplexF64}}(undef, length(target_coeffs))
    for i in eachindex(target_coeffs)
        c_target = target_coeffs[i]
        nrm = LinearAlgebra.norm(c_target, Inf)
        scale = max(nrm, 1.0)
        start_coeffs[i] = ComplexF64[
            (0.9 + 0.2 * rand(rng)) * cis(2π * rand(rng)) * scale
                for _ in 1:length(c_target)
        ]
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
    )

    # 7. Build toric homotopy (phase 1: t goes from 0 to 1)
    toric_H = ToricHomotopy(param_system.evaluator, start_coeffs)
    toric_heval = HomotopyEvaluator(toric_H)
    toric_tracker = Tracker(toric_heval; options = alg.tracker_options)

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
        # Update weights for this mixed cell (normalize so min non-zero weight = 1)
        update_weights!(toric_H, support, lifting, cell; min_weight = 1.0)

        code = track!(toric_tracker, x₀; t₁ = complex(0.0), t₀ = complex(1.0))

        if code != TrackerCode.TRACKER_SUCCESS
            # Toric phase failed — record failure and skip coefficient phase
            push!(path_results, PathResult(toric_tracker))
            continue
        end

        # Extract solution from toric phase into pre-allocated buffer
        copyto!(x_buffer, toric_tracker.state.x)

        # Phase 2: Coefficient homotopy — track from t=1 to t=0
        track!(coeff_tracker, x_buffer)
        push!(path_results, PathResult(coeff_tracker))
    end

    return Result(path_results, n_paths, cache.seed)
end
