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
        show_progress::Bool = false,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::Integer = rand(UInt32),
    )
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
    variety_codim = n - variety_dim
    # dim(L) = variety_codim, i.e. codim(L) = variety_dim
    L = rand_subspace(n; codim = variety_dim, affine = !projective)
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
        show_progress::Bool = false,
        threading::Bool = Threads.nthreads() > 1,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        seed::Integer = rand(UInt32),
    )
    seed32 = UInt32(seed % UInt32)
    R = _witness_init(
        F, L;
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
        seed = seed32,
    )
    return WitnessSet(F, L, R)
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

# Options
* `atol = 1e-14`, `rtol = sqrt(eps())`: a point `y` equals `x` when their
  distance is below `max(atol, norm(x, Inf) * rtol)`.
"""
function membership(
        p::AbstractVector{<:Number}, W::WitnessSet;
        atol::Float64 = 1.0e-14,
        rtol::Float64 = sqrt(eps()),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = false,
    )::Bool
    return first(
        membership(
            [Vector{ComplexF64}(p)], W;
            atol = atol, rtol = rtol,
            tracker_options = tracker_options, endgame_options = endgame_options,
            show_progress = show_progress,
        ),
    )
end

function membership(
        P::AbstractVector{<:AbstractVector},
        W::WitnessSet;
        atol::Float64 = 1.0e-14,
        rtol::Float64 = sqrt(eps()),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(),
        show_progress::Bool = false,
    )::Vector{Bool}
    # Projective membership needs a chart-consistent subspace move; the intrinsic
    # homotopy below has no chart, so on a homogeneous system the sliced system is
    # not square and the move is ill-posed. Projective witness sets are not yet
    # supported: fail loudly here rather than with a cryptic dimension error deep
    # in the tracker.
    W.projective && throw(
        ArgumentError(
            "membership is not yet implemented for projective (homogeneous) " *
                "witness sets; only affine witness sets are supported.",
        ),
    )
    F = W.F
    m, n = size(F)
    i = codim(W)               # = codim of the witness subspace
    L_dim = dim(W.L)           # dimension of the linear subspace to build

    # Random point used to gauge the scale of F for the cheap first reject.
    x0 = LA.normalize!(randn(ComplexF64, n))
    y0 = FSVec{ComplexF64}(zeros(ComplexF64, m))
    y = FSVec{ComplexF64}(zeros(ComplexF64, m))
    p_empty = FSVec{ComplexF64}(ComplexF64[])

    # One IntrinsicSubspaceHomotopy, retargeted per query point via
    # set_subspaces!. Intrinsic unconditionally (one concrete type, no runtime
    # dispatch): the extrinsic system `[F; A x − b]` is singular at the start
    # points when `dim(W.L) <= codim(W.L)` (the junk-removal regime), so its
    # tracks die at the start and report false negatives. The intrinsic form
    # `x = A(t) v + a(t)` is well conditioned in every dim/codim regime. Start
    # points are intrinsic, so ambient witness points convert in and out.
    Hom = IntrinsicSubspaceHomotopy(F.evaluator, W.L, W.L)
    eg = EndgameTracker(
        Tracker(HomotopyEvaluator(Hom); options = tracker_options), endgame_options,
    )
    u_buf = FSVec{ComplexF64}(zeros(ComplexF64, size(Hom)[2]))
    amb_buf = FSVec{ComplexF64}(zeros(ComplexF64, n))

    out = Vector{Bool}(undef, length(P))
    progress = show_progress ?
        ProgressMeter.Progress(
            length(P); desc = "Testing membership: ", enabled = true,
        ) : nothing
    for (idx, x) in enumerate(P)
        xc = ComplexF64.(x)
        # First check: is x approximately on V(F)?
        evaluate!(y0, F.evaluator, FSVec{ComplexF64}(LA.norm(xc, Inf) .* x0), p_empty)
        evaluate!(y, F.evaluator, FSVec{ComplexF64}(xc), p_empty)
        if LA.norm(y, Inf) > 1.0e-2 * LA.norm(y0, Inf)
            out[idx] = false
            progress !== nothing && ProgressMeter.next!(progress)
            continue
        end

        # Second check: build a generic subspace L_x of the same dimension as
        # W.L passing through x, move the witness points there, and check if x
        # is among them.
        A = _random_orthonormal(n - L_dim, n)
        b = A * xc
        L_x = LinearSubspace(ExtrinsicDescription(A, b; orthonormal = true))
        set_subspaces!(Hom, W.L, L_x)

        rad = max(atol, LA.norm(xc, Inf) * rtol)
        contained = false
        for q0 in W.R
            intrinsic_coordinates!(u_buf, Hom, ComplexF64.(q0), complex(1.0))
            track!(eg, u_buf)
            pr = PathResult(eg; path_number = 0, start_solution = Vector{ComplexF64}(q0))
            if is_success(pr)
                ambient_coordinates!(amb_buf, Hom, solution(pr), complex(0.0))
                if _vt_distance(InfNorm(), amb_buf, xc) < rad
                    contained = true
                    break
                end
            end
        end
        out[idx] = contained
        progress !== nothing && ProgressMeter.next!(progress)
    end
    return out
end

# A random matrix with `k` orthonormal rows in `n`-space.
function _random_orthonormal(k::Int, n::Int)::Matrix{ComplexF64}
    k == 0 && return zeros(ComplexF64, 0, n)
    return Matrix(LA.svd(randn(ComplexF64, k, n)).Vt)
end
