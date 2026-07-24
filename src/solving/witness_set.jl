## Witness sets.
#
# A witness set for a `d`-dimensional component of `V(F)` is the finite set
# `V(F) ∩ L`, where `L` is a generic affine linear subspace of dimension `d`
# (codimension `n - d`). Its cardinality is the degree of the component.
#
# Design: stay in AMBIENT coordinates and append the linear equations
# `A·x − b` of `L` to the system (rather than slicing `F(x)` into intrinsic
# coordinates `F(A·v + b)`). This reuses the existing machinery directly:
#   * initialization  → build the square system `[F; A·x − b]` and reuse the
#                        ordinary total-degree `solve`.
#   * moving a set     → `ExtrinsicSubspaceHomotopy` (ambient coordinates, so
#                        witness points need no coordinate conversion).
# The linear block `A` is orthonormal (guaranteed by `ExtrinsicDescription`),
# which keeps the combined Jacobian `[J_F; A]` well conditioned.

# ── WitnessSet type ──────────────────────────────────────────────────────────

"""
    WitnessSet(F, L, points)

Store the points `points` of `V(F) ∩ L` as a witness set. `F` is the system,
`L` the (affine) linear subspace, and `points` the witness points (only the
nonsingular solutions are kept).
"""
struct WitnessSet{S <: System}
    F::S
    L::LinearSubspace{ComplexF64}
    # only non-singular witness points
    R::Vector{Vector{ComplexF64}}
    projective::Bool
    # set by `decompose`; nothing until decided
    is_irreducible::Union{Nothing, Bool}
end

function WitnessSet(
        F::System,
        L::LinearSubspace,
        R::Vector{Vector{ComplexF64}};
        projective::Bool = is_linear(L) && is_homogeneous(F),
        is_irreducible::Union{Nothing, Bool} = nothing,
    )
    # A parametric system would fail deep inside membership/moves (the
    # evaluator is called with an empty parameter vector); fail fast here.
    nparameters(F) == 0 || throw(
        ArgumentError(
            "a WitnessSet requires a parameter-free system; substitute the " *
                "parameter values first (cf. `witness_set(F; target_parameters)`).",
        ),
    )
    return WitnessSet(
        F, convert(LinearSubspace{ComplexF64}, L), R, projective, is_irreducible,
    )
end

# ── Accessors ────────────────────────────────────────────────────────────────

"""
    system(W::WitnessSet)

Return the system stored in `W`.
"""
system(W::WitnessSet) = W.F

"""
    linear_subspace(W::WitnessSet)

Return the linear subspace stored in `W`.
"""
linear_subspace(W::WitnessSet) = W.L

"""
    solutions(W::WitnessSet)

Return the witness points stored in `W`.
"""
solutions(W::WitnessSet) = W.R
points(W::WitnessSet) = W.R

"""
    degree(W::WitnessSet)

Return the degree of the witness set, i.e. the number of witness points.
"""
degree(W::WitnessSet)::Int = length(W.R)

"""
    dim(W::WitnessSet)

The dimension of the algebraic set encoded by the witness set.
"""
dim(W::WitnessSet)::Int = codim(W.L)

"""
    codim(W::WitnessSet)

The codimension of the algebraic set encoded by the witness set.
"""
codim(W::WitnessSet)::Int = dim(W.L)

"""
    is_irreducible(W::WitnessSet)

Return `true` if `W` was computed to be irreducible, `false` if reducible, and
`:undecided` otherwise.
"""
function is_irreducible(W::WitnessSet)
    return W.is_irreducible === nothing ? :undecided : W.is_irreducible
end

function Base.show(io::IO, W::WitnessSet)
    print(io, "Witness set for dimension $(dim(W)) of degree $(degree(W))")
    return
end

# ── rank / corank ────────────────────────────────────────────────────────────

"""
    rank(F::System)

Numerically estimate the rank of the Jacobian of `F` at a random point.
"""
function LA.rank(F::System)::Int
    m, n = size(F)
    u = FSVec{ComplexF64}(zeros(ComplexF64, m))
    U = FSMat{ComplexF64}(zeros(ComplexF64, m, n))
    x = FSVec{ComplexF64}(randn(ComplexF64, n))
    p = FSVec{ComplexF64}(ComplexF64[])
    evaluate_and_jacobian!(u, U, F.evaluator, x, p)
    return LA.rank(Matrix(U))
end

"""
    corank(F::System)

Numerically estimate the corank `nvariables(F) - rank(F)` of `F`, an upper
bound on the dimension of `V(F)`.
"""
corank(F::System)::Int = nvariables(F) - LA.rank(F)

# ── Internal primitives ──────────────────────────────────────────────────────

# Substitute the parameter values `p` into `F`, returning a parameter-free
# system in the same variables. The compile mode is preserved. Throws when `p`
# does not match the parameters.
function _fix_parameters(
        F::System{P, V, M},
        p::Union{Nothing, AbstractVector{<:Number}},
    )::System where {P, V, M}
    np = nparameters(F)
    p === nothing && throw(
        ArgumentError(
            "The system has $np parameters; pass their values via " *
                "`target_parameters` to compute a witness set.",
        ),
    )
    np == 0 && throw(
        ArgumentError(
            "`target_parameters` was given, but the system has no parameters.",
        ),
    )
    length(p) == np || throw(
        ArgumentError(
            "The number of parameter values ($(length(p))) does not match the " *
                "number of parameters ($np).",
        ),
    )
    params = collect(parameters(F))
    vars = collect(variables(F))
    pc = Vector{ComplexF64}(p)
    fixed = [MP.polynomial(MP.subs(f, params => pc)) for f in polynomials(F)]
    return System(fixed; variables = vars, compile = M)
end

# The full ambient space as a codim-0 subspace (`A` is `0 × n`): witness sets
# of zero-dimensional varieties slice with the whole space, so the sliced
# system is `F` itself (plus a chart row in the projective case).
_full_subspace(n::Int)::LinearSubspace{ComplexF64} =
    LinearSubspace(zeros(ComplexF64, 0, n), ComplexF64[])

# Build the degree-1 polynomials `A[i,:]·vars - b[i]` describing the extrinsic
# subspace `L = {x | A x = b}`.
function _linear_equations(L::LinearSubspace, vars)
    E = extrinsic(L)
    A, b = E.A, E.b
    n = length(vars)
    return [
        sum(A[i, j] * vars[j] for j in 1:n) - b[i] for i in 1:size(A, 1)
    ]
end

# Build the chart equation `c·x − 1` fixing the projective scaling on the chart
# `c`. Homogeneous systems are positive-dimensional in ambient coordinates; the
# chart row makes the sliced system square.
function _chart_equation(chart::AbstractVector, vars::AbstractVector)
    n = length(vars)
    return sum(chart[j] * vars[j] for j in 1:n) - 1
end

# Build the sliced system `[polys; A x − b]` in ambient coordinates, i.e. the
# zero set `V(polys) ∩ L`. For a projective (homogeneous) problem a chart
# equation `c·x − 1` is appended so the system is square.
function _sliced_system(
        polys::AbstractVector, vars::AbstractVector, L::LinearSubspace;
        chart::Union{Nothing, AbstractVector} = nothing,
    )::System
    lin = _linear_equations(L, vars)
    eqs = vcat(collect(polys), lin)
    chart !== nothing && push!(eqs, _chart_equation(chart, vars))
    return System(eqs; parameters = empty(vars), variables = vars)
end

@noinline function _solve_witness_system(
        combined::S, algorithm::TotalDegree, executor::E, show_progress::Bool,
    )::Result where {S <: System, E <: AbstractExecutor}
    return solve(combined, algorithm, executor; show_progress = show_progress)
end

# Initialize witness points: solve `V(F) ∩ L` by appending the linear equations
# of `L` to `F` and running the ordinary total-degree solve. Returns the
# nonsingular witness points. For projective problems a random affine chart is
# appended so the combined system is square.
function _witness_init(
        F::System,
        L::LinearSubspace;
        projective::Bool = is_linear(L) && is_homogeneous(F),
        show_progress::Bool = false,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::UInt32 = rand(UInt32),
    )::Vector{Vector{ComplexF64}}
    vars = collect(variables(F))
    combined = if projective
        chart = randn(ComplexF64, length(vars))
        _sliced_system(polynomials(F), vars, L; chart = chart)
    else
        _sliced_system(polynomials(F), vars, L)
    end
    alg = TotalDegree(;
        seed = seed,
        tracker_options = tracker_options,
        endgame_options = endgame_options,
    )
    executor = threading ? Threaded() : Serial()
    res = _solve_witness_system(combined, alg, executor, show_progress)
    return [solution(pr) for pr in results(res; only_nonsingular = true)]
end

# Move a set of witness points from the subspace `L_start` to `L_target`, in
# ambient coordinates, via `ExtrinsicSubspaceHomotopy`. Returns the nonsingular
# endpoints. The gamma perturbation only rescales the linear equations by a unit
# complex, which preserves the start solution set (the start points stay exact).
function _move_witness_points(
        F::System,
        starts::AbstractVector{<:AbstractVector},
        L_start::LinearSubspace,
        L_target::LinearSubspace;
        projective::Bool = false,
        chart::Union{Nothing, Vector{ComplexF64}} = nothing,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
    )::Vector{Vector{ComplexF64}}
    base = ExtrinsicSubspaceHomotopy(F.evaluator, L_start, L_target)
    # In the projective setting the ambient system is positive-dimensional, so
    # wrap it in an affine chart `c·x = 1`. Start points are re-scaled onto the
    # same chart before tracking (they are projective representatives). The
    # `HomotopyEvaluator` type firewall erases the homotopy type, so both
    # branches build the SAME concrete `EndgameTracker` (no Union) and `c` is
    # always bound (no possibly-undefined binding for JET).
    c = projective ?
        (chart === nothing ? randn(ComplexF64, nvariables(F)) : chart) :
        ComplexF64[]
    eg = if projective
        EndgameTracker(
            Tracker(
                HomotopyEvaluator(AffineChartHomotopy(base, c));
                options = tracker_options,
            ),
            endgame_options,
        )
    else
        EndgameTracker(
            Tracker(HomotopyEvaluator(base); options = tracker_options),
            endgame_options,
        )
    end
    out = Vector{Vector{ComplexF64}}()
    for s in starts
        x₀ = FSVec{ComplexF64}(ComplexF64.(s))
        projective && on_chart!(x₀, c)
        track!(eg, x₀)
        pr = PathResult(eg; path_number = 0, start_solution = Vector{ComplexF64}(x₀))
        if is_success(pr) && is_nonsingular(pr)
            push!(out, solution(pr))
        end
    end
    return out
end

# ── Constructing witness sets ────────────────────────────────────────────────

"""
    witness_set(F::System; dim = nothing, codim = nothing, options...)

Compute a [`WitnessSet`](@ref) for `F` in the given dimension (resp.
codimension) by sampling a random affine linear subspace and solving
`V(F) ∩ L`.

    witness_set(F::System, L::LinearSubspace; options...)

Compute a [`WitnessSet`](@ref) for `F` and the given (affine) linear subspace
`L`.

    witness_set(W::WitnessSet, L::LinearSubspace; options...)

Move the witness set `W` to the new linear subspace `L`.

`F` may also be given as a single polynomial or a vector of polynomials.

For a parametric system pass the parameter values via `target_parameters`;
the parameters are substituted into `F` and the returned witness set stores
the resulting parameter-free system.

# Example
```julia
@polyvar x y
F = System([x^2 + y^2 - 5])
W = witness_set(F)   # Witness set for dimension 1 of degree 2
```
"""
function witness_set(
        F::System;
        dim::Union{Nothing, Int} = nothing,
        codim::Union{Nothing, Int} = nothing,
        target_parameters::Union{Nothing, AbstractVector{<:Number}} = nothing,
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::Integer = rand(UInt32),
    )
    if nparameters(F) > 0 || target_parameters !== nothing
        G = _fix_parameters(F, target_parameters)
        return witness_set(
            G;
            dim = dim, codim = codim,
            show_progress = show_progress, threading = threading,
            tracker_options = tracker_options, endgame_options = endgame_options,
            seed = seed,
        )
    end
    Random.seed!(seed)
    seed32 = UInt32(seed % UInt32)
    n = nvariables(F)
    projective = is_homogeneous(F)
    if dim === nothing && codim === nothing
        variety_dim = corank(F) - (projective ? 1 : 0)
    elseif codim !== nothing
        variety_dim = n - codim - (projective ? 1 : 0)
    else
        variety_dim = something(dim)   # dim !== nothing on this branch; strip the Union
    end
    variety_dim < 0 && throw(
        ArgumentError(
            "The variety has negative dimension $variety_dim (V(F) is empty).",
        ),
    )
    # dim(L) = n - variety_dim, i.e. codim(L) = variety_dim. A zero-dimensional
    # variety is sliced with the whole space (codim-0 subspace).
    L = variety_dim == 0 ? _full_subspace(n) :
        rand_subspace(n; codim = variety_dim, affine = !projective)
    return witness_set(
        F, L;
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
        seed = seed32,
    )
end

function witness_set(
        F::System,
        L::LinearSubspace;
        target_parameters::Union{Nothing, AbstractVector{<:Number}} = nothing,
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::Integer = rand(UInt32),
    )
    if nparameters(F) > 0 || target_parameters !== nothing
        G = _fix_parameters(F, target_parameters)
        return witness_set(
            G, L;
            show_progress = show_progress, threading = threading,
            tracker_options = tracker_options, endgame_options = endgame_options,
            seed = seed,
        )
    end
    seed32 = UInt32(seed % UInt32)
    R = _witness_init(
        F, L;
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
        seed = seed32,
    )
    return WitnessSet(F, L, R)
end

# Polynomial input forms: a single polynomial or a vector of polynomials,
# with or without an explicit subspace.
function witness_set(
        F::AbstractVector{<:MP.AbstractPolynomialLike};
        dim::Union{Nothing, Int} = nothing,
        codim::Union{Nothing, Int} = nothing,
        target_parameters::Union{Nothing, AbstractVector{<:Number}} = nothing,
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::Integer = rand(UInt32),
    )
    return witness_set(
        System(collect(F));
        dim = dim, codim = codim, target_parameters = target_parameters,
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
        seed = seed,
    )
end

function witness_set(
        F::AbstractVector{<:MP.AbstractPolynomialLike},
        L::LinearSubspace;
        target_parameters::Union{Nothing, AbstractVector{<:Number}} = nothing,
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::Integer = rand(UInt32),
    )
    return witness_set(
        System(collect(F)), L;
        target_parameters = target_parameters,
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
        seed = seed,
    )
end

function witness_set(
        f::MP.AbstractPolynomialLike;
        dim::Union{Nothing, Int} = nothing,
        codim::Union{Nothing, Int} = nothing,
        target_parameters::Union{Nothing, AbstractVector{<:Number}} = nothing,
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::Integer = rand(UInt32),
    )
    return witness_set(
        [f];
        dim = dim, codim = codim, target_parameters = target_parameters,
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
        seed = seed,
    )
end

function witness_set(
        f::MP.AbstractPolynomialLike,
        L::LinearSubspace;
        target_parameters::Union{Nothing, AbstractVector{<:Number}} = nothing,
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::Integer = rand(UInt32),
    )
    return witness_set(
        [f], L;
        target_parameters = target_parameters,
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
        seed = seed,
    )
end

function witness_set(
        W::WitnessSet,
        L::LinearSubspace;
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
    )
    if W.projective && !is_linear(L)
        error(
            "The given space is an affine linear subspace (b ≠ 0). Expected a " *
                "linear subspace since the given witness set is projective.",
        )
    end
    R = _move_witness_points(
        W.F, W.R, W.L, L;
        projective = W.projective,
        tracker_options = tracker_options, endgame_options = endgame_options,
    )
    return WitnessSet(
        W.F, convert(LinearSubspace{ComplexF64}, L), R;
        projective = is_linear(L) && W.projective,
    )
end

# ── Trace test ───────────────────────────────────────────────────────────────

"""
    trace_test(W::WitnessSet; options...)

Perform a trace test to verify whether the witness set `W` is complete. Returns
the (normalized) trace, which is theoretically `0` for a complete witness set.
Due to floating-point arithmetic the value is small but nonzero and must be
compared against a tolerance. Returns `nothing` if a path-tracking failure
prevented the test.

[^LRS18]: Leykin, Rodriguez, Sottile. "Trace test." Arnold Math. J. 4.1 (2018).
"""
function trace_test(
        W::WitnessSet;
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
    )
    L₀ = W.L
    F = W.F
    S₀ = W.R
    isempty(S₀) && return nothing

    # In the projective setting we fix a single affine chart `c` and place all
    # witness points (and both translated sets) on it, so the barycenters are
    # comparable representatives.
    # Assign `chart` inside the branch so that within the projective branch its
    # type narrows to `Vector{ComplexF64}` and `on_chart!(y, chart)` resolves to
    # a concrete method (no `on_chart!(..., ::Nothing)` for JET).
    if W.projective
        chart = randn(ComplexF64, nvariables(F))
        S₀c = map(S₀) do s
            y = ComplexF64.(s)
            on_chart!(y, chart)
            return y
        end
    else
        chart = nothing
        S₀c = S₀
    end

    s₀ = sum(S₀c)
    v = randn(ComplexF64, codim(L₀))
    L₁ = translate(L₀, v)
    L₋₁ = translate(L₀, -v)

    R₁ = _move_witness_points(
        F, S₀, L₀, L₁;
        projective = W.projective, chart = chart,
        tracker_options = tracker_options, endgame_options = endgame_options,
    )
    length(R₁) == degree(W) || return nothing
    R₋₁ = _move_witness_points(
        F, S₀, L₀, L₋₁;
        projective = W.projective, chart = chart,
        tracker_options = tracker_options, endgame_options = endgame_options,
    )
    length(R₋₁) == degree(W) || return nothing

    s₁ = sum(R₁)
    s₋₁ = sum(R₋₁)

    # Columns are the barycenters at t = -1, 0, 1; the top n rows should be
    # collinear (constant "velocity"), so the smallest singular value vanishes.
    M = [reshape(s₋₁, :, 1) reshape(s₀, :, 1) reshape(s₁, :, 1); ones(ComplexF64, 1, 3)]
    singvals = LA.svdvals(M)
    return singvals[3] / singvals[1]
end

# ── Membership ───────────────────────────────────────────────────────────────

"""
    membership(p, W::WitnessSet; options...)
    membership(P::AbstractVector{<:AbstractVector}, W::WitnessSet; options...)

Test whether the point `p` (resp. each point in `P`) lies on the algebraic set
encoded by the witness set `W`. Returns a `Bool` (resp. a `Vector{Bool}`).
For a projective witness set the query points are projective representatives
(any scaling).

# Options
* `atol = 1e-14`, `rtol = sqrt(eps())`: a point `y` equals `x` when their
  distance is below `max(atol, norm(x, Inf) * rtol)`.
* `endgame_options = EndgameOptions(; max_endgame_steps = 100,
  max_endgame_extended_steps = 100, sing_cond = 1e12)`: capped so witness
  moves that degenerate (the query point is not on the component) fail fast
  instead of grinding through the full endgame.
* `show_progress = true`: display a progress bar.
* `threading = Threads.nthreads() > 1`: query the points in parallel.
"""
function membership(
        p::AbstractVector{<:Number}, W::WitnessSet;
        atol::Float64 = 1.0e-14,
        rtol::Float64 = sqrt(eps()),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(;
            max_endgame_steps = 100, max_endgame_extended_steps = 100,
            sing_cond = 1.0e12,
        ),
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
    )::Bool
    return first(
        membership(
            [Vector{ComplexF64}(p)], W;
            atol = atol, rtol = rtol,
            tracker_options = tracker_options, endgame_options = endgame_options,
            show_progress = show_progress, threading = threading,
        ),
    )
end

# Per-task membership state: a cloned evaluator for the cheap residual reject
# plus an IntrinsicSubspaceHomotopy/EndgameTracker pair for the witness-point
# moves. Intrinsic unconditionally (one concrete type, no runtime dispatch):
# the extrinsic system `[F; A x − b]` is singular at the start points when
# `dim(W.L) <= codim(W.L)` (the junk-removal regime), so its tracks die at the
# start and report false negatives. The intrinsic form `x = A(t) v + a(t)` is
# well conditioned in every dim/codim regime. For a projective witness set the
# homotopy runs the system on the affine chart `chart` (the sliced projective
# system is square only on a chart).
struct MembershipState
    evaluator::SystemEvaluator
    homotopy::IntrinsicSubspaceHomotopy
    tracker::EndgameTracker
    y0::FSVec{ComplexF64}
    y::FSVec{ComplexF64}
    u_buf::FSVec{ComplexF64}
    amb_buf::FSVec{ComplexF64}
    # per-query scratch: the query point, its FSVec view for the evaluator,
    # the scale-gauge point, an empty parameter vector, and (projective only)
    # a witness-point chart-representative buffer
    xc::Vector{ComplexF64}
    xc_fs::FSVec{ComplexF64}
    gauge::FSVec{ComplexF64}
    p_empty::FSVec{ComplexF64}
    q_chart::Vector{ComplexF64}
end

function MembershipState(
        F::System, L::LinearSubspace{ComplexF64}, projective::Bool,
        chart::Vector{ComplexF64}, gamma::ComplexF64,
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
    )
    m, n = size(F)
    ev = _clone_system_evaluator(F)
    hom_ev = projective ? SystemEvaluator(AffineChartSystem(ev, chart)) : ev
    homotopy = IntrinsicSubspaceHomotopy(hom_ev, L, L; gamma = gamma)
    tracker = EndgameTracker(
        Tracker(HomotopyEvaluator(homotopy); options = tracker_options),
        endgame_options,
    )
    return MembershipState(
        ev, homotopy, tracker,
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexF64}(zeros(ComplexF64, size(homotopy)[2])),
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        zeros(ComplexF64, n),
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexF64}(ComplexF64[]),
        zeros(ComplexF64, n),
    )
end

# One membership query: cheap residual reject, then build a generic subspace
# L_x through x with the same dimension as W.L, move the witness points there,
# and check whether x is among the endpoints. In the projective case L_x is
# linear through the ray of x, and all comparisons happen on the chart.
# `R` carries this query's pre-drawn randomness (drawn in the driver from the
# global RNG, so serial and threaded runs are bit-identical); it is consumed
# (mutated) here. `A`/`b` are allocated fresh per query on purpose:
# `LinearSubspace` keeps them by reference and the homotopy reads the stored
# subspace during tracking, so they must not be buffer-reused.
function _membership_query(
        st::MembershipState, W::WitnessSet, chart::Vector{ComplexF64},
        x0::Vector{ComplexF64}, x::AbstractVector{<:Number},
        R::Matrix{ComplexF64}, atol::Float64, rtol::Float64,
    )::Bool
    n = length(st.xc)
    length(x) == n || throw(
        DimensionMismatch("query point has length $(length(x)), expected $n"),
    )
    copyto!(st.xc, x)
    k = codim(W.L)

    # First check: is x approximately on V(F)? (`x0` gauges the scale of F.)
    scale = LA.norm(st.xc, Inf)
    @. st.gauge = scale * x0
    evaluate!(st.y0, st.evaluator, st.gauge, st.p_empty)
    copyto!(st.xc_fs, st.xc)
    evaluate!(st.y, st.evaluator, st.xc_fs, st.p_empty)
    LA.norm(st.y, Inf) > 1.0e-2 * LA.norm(st.y0, Inf) && return false

    if W.projective
        on_chart!(st.xc, chart)   # compare chart representatives
        A = _orthonormal_rows_through!(R, st.xc)
        b = zeros(ComplexF64, k)
    else
        A = _orthonormal_rows(R)
        b = A * st.xc
    end
    L_x = LinearSubspace(ExtrinsicDescription(A, b; orthonormal = true))
    set_subspaces!(st.homotopy, W.L, L_x)

    rad = max(atol, LA.norm(st.xc, Inf) * rtol)
    for q0 in W.R
        # Affine: q0 is used read-only, no copy. Projective: rescale a copy
        # onto the chart. `q` may alias `st.q_chart`; the PathResult below
        # only stores it as start_solution and is discarded before the next
        # iteration mutates the buffer.
        q = if W.projective
            copyto!(st.q_chart, q0)
            on_chart!(st.q_chart, chart)
            st.q_chart
        else
            q0
        end
        intrinsic_coordinates!(st.u_buf, st.homotopy, q, complex(1.0))
        track!(st.tracker, st.u_buf)
        pr = PathResult(st.tracker; path_number = 0, start_solution = q)
        is_success(pr) || continue
        ambient_coordinates!(st.amb_buf, st.homotopy, solution(pr), complex(0.0))
        _vt_distance(InfNorm(), st.amb_buf, st.xc) < rad && return true
    end
    return false
end

function membership(
        P::AbstractVector{<:AbstractVector},
        W::WitnessSet{S};
        atol::Float64 = 1.0e-14,
        rtol::Float64 = sqrt(eps()),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(;
            max_endgame_steps = 100, max_endgame_extended_steps = 100,
            sing_cond = 1.0e12,
        ),
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
    )::Vector{Bool} where {S <: System}
    F = W.F
    n = size(F)[2]
    # A single chart shared by all queries so the compared representatives are
    # consistent (only used when W is projective).
    chart = W.projective ? randn(ComplexF64, n) : ComplexF64[]
    # Random point used to gauge the scale of F for the cheap first reject.
    x0 = LA.normalize!(randn(ComplexF64, n))
    # One genericity perturbation shared by all queries and tasks, and one
    # pre-drawn random matrix per query (the only per-query randomness). Both
    # come from the global RNG here in the driver, so threaded and serial runs
    # consume the same stream and produce bit-identical results.
    gamma = cis(2 * pi * rand())
    k = codim(W.L)
    Rs = [randn(ComplexF64, k, n) for _ in eachindex(P)]

    out = Vector{Bool}(undef, length(P))
    progress = make_progress(
        length(P), show_progress; desc = "Testing membership: ",
    )
    if threading && length(P) > 1
        nt = Threads.nthreads()
        plock = ReentrantLock()
        @tasks for i in eachindex(P)
            @set ntasks = nt
            @local st = MembershipState(
                F, W.L, W.projective, chart, gamma,
                tracker_options, endgame_options,
            )
            out[i] = _membership_query(st, W, chart, x0, P[i], Rs[i], atol, rtol)
            if progress !== nothing
                @lock plock ProgressMeter.next!(progress)
            end
        end
    else
        st = MembershipState(
            F, W.L, W.projective, chart, gamma, tracker_options, endgame_options,
        )
        for (i, x) in enumerate(P)
            out[i] = _membership_query(st, W, chart, x0, x, Rs[i], atol, rtol)
            progress !== nothing && ProgressMeter.next!(progress)
        end
    end
    return out
end

# Orthonormalize the rows of `R` in place of a fresh matrix (the rowspace is
# preserved by the SVD).
function _orthonormal_rows(R::Matrix{ComplexF64})::Matrix{ComplexF64}
    size(R, 1) == 0 && return R
    return Matrix(LA.svd(R).Vt)
end

# Project the rows of `R` off `x` (mutating `R`) and orthonormalize, giving
# `k` orthonormal rows annihilating `x` (`A x = 0`), so the linear subspace
# `{y | A y = 0}` passes through the ray of `x`. Requires `k < n`: the
# complement of `x` has dimension `n - 1`, so `k = n` rows would be rank
# deficient and the orthonormalized rows would no longer annihilate `x`.
function _orthonormal_rows_through!(
        R::Matrix{ComplexF64}, x::Vector{ComplexF64},
    )::Matrix{ComplexF64}
    k, n = size(R)
    k == 0 && return R
    k < n || throw(
        ArgumentError(
            "cannot build $k orthonormal rows orthogonal to a point in dimension $n",
        ),
    )
    x̂ = x ./ LA.norm(x)
    R .-= (R * x̂) * x̂'
    return Matrix(LA.svd(R).Vt)
end
