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
    Irreducibility

Whether a [`WitnessSet`](@ref) has been shown to be irreducible:
`Irreducibility.IRREDUCIBLE`, `Irreducibility.REDUCIBLE`, or
`Irreducibility.UNKNOWN` before [`decompose`](@ref) has decided.
"""
@enumx Irreducibility::Int8 begin
    UNKNOWN
    IRREDUCIBLE
    REDUCIBLE
end

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
    # set by `decompose`; UNKNOWN until decided
    irreducibility::Irreducibility.T
end

function WitnessSet(
        F::System,
        L::LinearSubspace,
        R::Vector{Vector{ComplexF64}};
        projective::Bool = is_linear(L) && is_homogeneous(F),
        irreducibility::Irreducibility.T = Irreducibility.UNKNOWN,
    )
    # A parametric system would fail deep inside membership/moves (the
    # evaluator is called with an empty parameter vector); fail fast here.
    nparameters(F) == 0 || throw(
        ArgumentError(
            "a `WitnessSet` requires a parameter-free system, but the system has " *
                "$(nparameters(F)) parameter(s). Fix them first with " *
                "`fix_parameters(F, p)`.",
        ),
    )
    return WitnessSet(
        F, convert(LinearSubspace{ComplexF64}, L), R, projective, irreducibility,
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
    is_irreducible(W::WitnessSet) -> Irreducibility.T

`Irreducibility.IRREDUCIBLE` if `W` was computed to be irreducible,
`Irreducibility.REDUCIBLE` if reducible, and `Irreducibility.UNKNOWN` if
[`decompose`](@ref) has not decided.
"""
is_irreducible(W::WitnessSet)::Irreducibility.T = W.irreducibility

function Base.show(io::IO, W::WitnessSet)
    print(io, "Witness set for dimension $(dim(W)) of degree $(degree(W))")
    return
end

# ── rank / corank ────────────────────────────────────────────────────────────

"""
    rank(F::System; rng = Random.default_rng())

Numerically estimate the rank of the Jacobian of `F` at a random point drawn
from `rng`.
"""
function LA.rank(F::System; rng::Random.AbstractRNG = Random.default_rng())::Int
    m, n = size(F)
    u = FSVec{ComplexF64}(zeros(ComplexF64, m))
    U = FSMat{ComplexF64}(zeros(ComplexF64, m, n))
    x = FSVec{ComplexF64}(randn(rng, ComplexF64, n))
    p = FSVec{ComplexF64}(ComplexF64[])
    evaluate_and_jacobian!(u, U, F.evaluator, x, p)
    return LA.rank(Matrix(U))
end

"""
    corank(F::System; rng = Random.default_rng())

Numerically estimate the corank `nvariables(F) - rank(F)` of `F`, an upper
bound on the dimension of `V(F)`.
"""
corank(F::System; rng::Random.AbstractRNG = Random.default_rng())::Int =
    nvariables(F) - LA.rank(F; rng = rng)

# ── Internal primitives ──────────────────────────────────────────────────────

@noinline _solve_witness_cache(cache)::Result = CommonSolve.solve!(cache)

# Initialize witness points: solve `V(F) ∩ L` by appending the linear equations
# of `L` to `F` and running the ordinary total-degree solve. Returns the
# nonsingular witness points. For projective problems a random affine chart is
# appended so the combined system is square.
function _witness_init(
        F::System,
        L::LinearSubspace,
        rng::Random.MersenneTwister;
        projective::Bool = is_linear(L) && is_homogeneous(F),
        show_progress::Bool = false,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
    )::Vector{Vector{ComplexF64}}
    chart = projective ? _affine_chart(rng, F) : ComplexF64[]
    alg = TotalDegree(;
        seed = rand(rng, UInt32),
        tracker_options = tracker_options,
        endgame_options = endgame_options,
    )
    executor = threading ? Threaded() : Serial()
    res = _solve_witness_cache(
        _init_sliced_total_degree(F, L, chart, alg, executor, show_progress),
    )
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
        L_target::LinearSubspace,
        rng::Random.MersenneTwister;
        projective::Bool = false,
        chart::Union{Nothing, Vector{ComplexF64}} = nothing,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
    )::Vector{Vector{ComplexF64}}
    base = ExtrinsicSubspaceHomotopy(
        F.evaluator, L_start, L_target; gamma = _random_gamma(rng),
    )
    # In the projective setting the ambient system is positive-dimensional, so
    # wrap it in an affine chart `c·x = 1`. Start points are re-scaled onto the
    # same chart before tracking (they are projective representatives). The
    # `HomotopyEvaluator` type firewall erases the homotopy type, so both
    # branches build the SAME concrete `EndgameTracker` (no Union) and `c` is
    # always bound (no possibly-undefined binding for JET).
    c = projective ?
        (chart === nothing ? randn(rng, ComplexF64, nvariables(F)) : chart) :
        ComplexF64[]
    eg = if projective
        _endgame_tracker(AffineChartHomotopy(base, c), tracker_options, endgame_options)
    else
        _endgame_tracker(base, tracker_options, endgame_options)
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

`F` must be parameter-free; fix the values first with [`fix_parameters`](@ref):
the parameters are substituted into `F` and the returned witness set stores
the resulting parameter-free system.

Every random choice descends from `seed`, so passing the same `seed` reproduces
the same witness set regardless of the state of the global random number
generator.

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
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    )
    _check_parameter_free(F, "`witness_set`")
    rng = Random.MersenneTwister(seed)
    n = nvariables(F)
    projective = is_homogeneous(F)
    if dim === nothing && codim === nothing
        variety_dim = corank(F; rng = rng) - (projective ? 1 : 0)
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
        rand_subspace(rng, n; codim = variety_dim, affine = !projective)
    # `rng` and not a fresh one from `seed`: the chart drawn downstream must be
    # independent of the `L` just drawn from the same stream.
    return _witness_set(
        F, L, rng, show_progress, threading, tracker_options, endgame_options,
    )
end

function witness_set(
        F::System,
        L::LinearSubspace;
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    )
    _check_parameter_free(F, "`witness_set`")
    return _witness_set(
        F, L, Random.MersenneTwister(seed),
        show_progress, threading, tracker_options, endgame_options,
    )
end

function _witness_set(
        F::System, L::LinearSubspace, rng::Random.MersenneTwister,
        show_progress::Bool, threading::Bool,
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
    )::WitnessSet
    R = _witness_init(
        F, L, rng;
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
    )
    return WitnessSet(F, L, R)
end

# Polynomial input forms: a single polynomial or a vector of polynomials,
# with or without an explicit subspace.
function witness_set(
        F::AbstractVector{<:MP.AbstractPolynomialLike};
        dim::Union{Nothing, Int} = nothing,
        codim::Union{Nothing, Int} = nothing,
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    )
    return witness_set(
        System(collect(F));
        dim = dim, codim = codim,
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
        seed = seed,
    )
end

function witness_set(
        F::AbstractVector{<:MP.AbstractPolynomialLike},
        L::LinearSubspace;
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    )
    return witness_set(
        System(collect(F)), L;
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
        seed = seed,
    )
end

function witness_set(
        f::MP.AbstractPolynomialLike;
        dim::Union{Nothing, Int} = nothing,
        codim::Union{Nothing, Int} = nothing,
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    )
    return witness_set(
        [f];
        dim = dim, codim = codim,
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
        seed = seed,
    )
end

function witness_set(
        f::MP.AbstractPolynomialLike,
        L::LinearSubspace;
        show_progress::Bool = true,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    )
    return witness_set(
        [f], L;
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
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    )
    if W.projective && !is_linear(L)
        error(
            "The given space is an affine linear subspace (b ≠ 0). Expected a " *
                "linear subspace since the given witness set is projective.",
        )
    end
    R = _move_witness_points(
        W.F, W.R, W.L, L, Random.MersenneTwister(seed);
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

Every random choice descends from `seed`, so passing the same `seed` reproduces
the same trace regardless of the state of the global random number generator.

[^LRS18]: Leykin, Rodriguez, Sottile. "Trace test." Arnold Math. J. 4.1 (2018).
"""
function trace_test(
        W::WitnessSet;
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    )
    L₀ = W.L
    F = W.F
    S₀ = W.R
    isempty(S₀) && return nothing
    rng = Random.MersenneTwister(seed)

    # In the projective setting we fix a single affine chart `c` and place all
    # witness points (and both translated sets) on it, so the barycenters are
    # comparable representatives.
    # Assign `chart` inside the branch so that within the projective branch its
    # type narrows to `Vector{ComplexF64}` and `on_chart!(y, chart)` resolves to
    # a concrete method (no `on_chart!(..., ::Nothing)` for JET).
    if W.projective
        chart = randn(rng, ComplexF64, nvariables(F))
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
    v = randn(rng, ComplexF64, codim(L₀))
    L₁ = translate(L₀, v)
    L₋₁ = translate(L₀, -v)

    R₁ = _move_witness_points(
        F, S₀, L₀, L₁, rng;
        projective = W.projective, chart = chart,
        tracker_options = tracker_options, endgame_options = endgame_options,
    )
    length(R₁) == degree(W) || return nothing
    R₋₁ = _move_witness_points(
        F, S₀, L₀, L₋₁, rng;
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
* `seed`: every random choice descends from it, so the same `seed` gives the
  same answers regardless of the state of the global random number generator.
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
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    )::Bool
    return first(
        membership(
            [Vector{ComplexF64}(p)], W;
            atol = atol, rtol = rtol,
            tracker_options = tracker_options, endgame_options = endgame_options,
            show_progress = show_progress, threading = threading, seed = seed,
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
    tracker = _endgame_tracker(homotopy, tracker_options, endgame_options)
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
# `R` carries this query's pre-drawn randomness (drawn in the driver off the
# seeded rng, so serial and threaded runs are bit-identical); it is consumed
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
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    )::Vector{Bool} where {S <: System}
    rng = Random.MersenneTwister(seed)
    F = W.F
    n = size(F)[2]
    # A single chart shared by all queries so the compared representatives are
    # consistent (only used when W is projective).
    chart = W.projective ? randn(rng, ComplexF64, n) : ComplexF64[]
    # Random point used to gauge the scale of F for the cheap first reject.
    x0 = LA.normalize!(randn(rng, ComplexF64, n))
    # One genericity perturbation shared by all queries and tasks, and one
    # pre-drawn random matrix per query (the only per-query randomness). Both
    # are drawn here in the driver, off `rng`, so threaded and serial runs
    # consume the same stream and produce bit-identical results.
    gamma = _random_gamma(rng)
    k = codim(W.L)
    Rs = [randn(rng, ComplexF64, k, n) for _ in eachindex(P)]

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
