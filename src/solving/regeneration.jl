## u-regeneration.
#
# Computes witness supersets for `V(F)` equation-by-equation, following the
# u-regeneration algorithm of Duff, Leykin and Rodriguez
# (https://arxiv.org/abs/2206.02869). Returns one `WitnessSet` per codimension
# WITHOUT decomposing into irreducible components (a witness *superset*); use
# `decompose` for the irreducible decomposition.
#
# Notes:
# * The deformation equation is `u^d - 1`, divided by the denominator of the
#   equation it deforms into so both endpoint systems have the same poles.
# * Every `slice(System, L)` is the ambient `_sliced_system` (`[F; A x − b]`).
# * The intersection u-homotopy is a `StraightLineHomotopy` between two sliced
#   ambient systems; the fill-up and membership steps reuse the existing
#   `ExtrinsicSubspaceHomotopy` and `MonodromySolver`.

# ── WitnessPoints: an undecomposed witness superset during regeneration ───────

"""
    WitnessPoints(L, Lᵤ, R)

Internal container used by [`regeneration`](@ref). Stores witness points `R`
together with the two flag subspaces used by u-regeneration: `L` sets `u = c`
(type 2), `Lᵤ` leaves `u` free (type 1).
"""
mutable struct WitnessPoints
    L::LinearSubspace{ComplexF64}
    Lᵤ::LinearSubspace{ComplexF64}
    R::Vector{Vector{ComplexF64}}
end

linear_subspace(W::WitnessPoints) = W.L
linear_subspace_u(W::WitnessPoints) = W.Lᵤ
points(W::WitnessPoints) = W.R
codim(W::WitnessPoints)::Int = dim(W.L)
dim(W::WitnessPoints)::Int = codim(W.L)
degree(W::WitnessPoints)::Int = length(W.R)
Base.push!(W::WitnessPoints, p::Vector{ComplexF64}) = push!(W.R, p)

# Strip the u coordinate from the points and the u-column from the subspace,
# projecting a WitnessPoints in (n+1)-space back to an n-space WitnessSet.
function u_transform(W::WitnessPoints)
    L = W.Lᵤ
    E = extrinsic(L)
    A, b = E.A, E.b
    E2 = ExtrinsicDescription(A[:, 1:(end - 1)], b; orthonormal = false)
    P = [p[1:(end - 1)] for p in W.R]
    return P, LinearSubspace(E2)
end

# ── Flag construction ────────────────────────────────────────────────────────

# Build the flag of (type-2, type-1) subspace pairs from a dimension-1 base
# subspace `L₀` in n-space. For index `i` this drops the first `i-1` rows of the
# base and, for type 2, prepends the equation `u = c`.
function get_flag(iter, L₀::LinearSubspace)
    A₀ = extrinsic(L₀).A          # orthonormal rows, size (n-1) × n
    b₀ = extrinsic(L₀).b
    n = size(A₀, 1) + 1
    m = size(A₀, 2)

    # type 1: no condition on u (last column zero)
    Aᵤ = [A₀ zeros(ComplexF64, n - 1)]
    bᵤ = b₀
    # type 2: prepend the equation u = c
    c = randn(ComplexF64)
    A = [zeros(ComplexF64, 1, m) one(ComplexF64); A₀ zeros(ComplexF64, n - 1)]
    b = [c; b₀]

    return map(iter) do i
        j = i + 1
        Eᵤ = ExtrinsicDescription(Aᵤ[i:end, :], bᵤ[i:end]; orthonormal = true)
        Lᵤ = LinearSubspace(Eᵤ)
        E = ExtrinsicDescription(A[[1; j:n], :], b[[1; j:n]]; orthonormal = true)
        L = LinearSubspace(E)
        return (L, Lᵤ)
    end
end

function initialize_witness_sets(codim::Int, n::Int)::Vector{WitnessPoints}
    L₀ = rand_subspace(n; dim = 1)
    flag = get_flag(1:codim, L₀)
    out = Vector{WitnessPoints}(undef, length(flag))
    for (i, (L, Lᵤ)) in enumerate(flag)
        out[i] = WitnessPoints(L, Lᵤ, Vector{Vector{ComplexF64}}())
    end
    return out
end

# Witness set of each hypersurface `f_i = 0` on the seed subspace `L`. A rational
# equation goes through its numerator, minus the zeros that are poles of it.
function initialize_hypersurfaces(
        F::System{P, V}, vars::Vector{V}, L::LinearSubspace;
        threading::Bool, tracker_options::TrackerOptions,
        endgame_options::EndgameOptions,
    ) where {P, V}
    fs = polynomials(F)
    HS = System{P, V, CompileMode.INTERPRETED, UnderdeterminedShape}
    out = Vector{WitnessSet{HS}}(undef, length(fs))
    for i in eachindex(fs)
        h = System([fs[i]]; parameters = empty(vars), variables = vars)::HS
        G, Q = _numerator_system(fs[i], h, vars)
        R = _witness_init(
            G, L;
            threading = threading, tracker_options = tracker_options,
            endgame_options = endgame_options,
        )
        out[i] = WitnessSet(h, L, Q === nothing ? R : _drop_poles(Q, R))
    end
    return out
end

# ── Regeneration state ───────────────────────────────────────────────────────

# `SystemShape` changes as equations accumulate, so one mutable state cannot
# carry a fixed concrete `Fᵢ` type across the whole loop. Instead, construct a
# fresh immutable state at each phase boundary. The outer loop pays one dynamic
# dispatch after `System(...)` chooses its runtime shape; homotopy construction
# and all tracking then specialize on a fully concrete state.
struct RegenerationState{P, V, S <: System}
    eqs::Vector{P}            # sorted equations (in vars incl. u)
    vars::Vector{V}
    u::V
    i::Int
    codim::Int
    Fᵢ::S
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
end

@noinline function _intersect_regeneration_phase!(
        out::Vector{WitnessPoints}, H::Vector{W},
        eqs::Vector{P}, vars::Vector{V}, u::V,
        i::Int, codim::Int, F_prev::S,
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
        threading::Bool, atol::Float64, rtol::Float64,
    )::Nothing where {W <: WitnessSet, P, V, S <: System}
    state = RegenerationState(
        eqs, vars, u, i, codim, F_prev, tracker_options, endgame_options,
    )
    intersect_all!(
        out, H, state; threading = threading, atol = atol, rtol = rtol,
    )
    return nothing
end

@noinline function _fill_regeneration_phase!(
        out::Vector{WitnessPoints}, monodromy_options::MonodromyOptions,
        eqs::Vector{P}, vars::Vector{V}, u::V,
        i::Int, codim::Int, Fᵢ::S,
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
        show_monodromy_progress::Bool, threading::Bool,
    )::Nothing where {P, V, S <: System}
    state = RegenerationState(
        eqs, vars, u, i, codim, Fᵢ, tracker_options, endgame_options,
    )
    fill_up!(out, monodromy_options, state, show_monodromy_progress, threading)
    return nothing
end

# ── regeneration ─────────────────────────────────────────────────────────────

"""
    regeneration(F::System; options...)

Solve `F = 0` equation-by-equation and return a [`WitnessSet`](@ref) for every
dimension without decomposing into irreducible components (a witness
*superset*). Based on the u-regeneration algorithm of Duff, Leykin and
Rodriguez (https://arxiv.org/abs/2206.02869).

Every equation must be polynomial or rational in the variables of `F`. A rational
equation is handled through its numerator, and the zeros of the numerator that
are poles of the equation are dropped.

# Options
* `sorted = true`: sort the polynomials of `F` by decreasing degree.
* `max_codim`: maximal codimension of witness supersets to compute.
* `tracker_options`, `endgame_options`, `monodromy_options`.
* `threading = true`: enable multi-threading.
* `seed`: random seed.
"""
function regeneration(
        F::AbstractVector{<:MP.AbstractPolynomialLike};
        sorted::Bool = true,
        max_codim::Union{Int, Nothing} = nothing,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(;
            max_endgame_steps = 100, max_endgame_extended_steps = 100,
            sing_cond = 1.0e12,
        ),
        monodromy_options::MonodromyOptions = MonodromyOptions(;
            trace_test = true, parameter_sampler = weighted_normal,
        ),
        show_progress::Bool = true,
        show_monodromy_progress::Bool = false,
        threading::Bool = Threads.nthreads() > 1,
        seed::Union{Nothing, Integer} = nothing,
        atol::Float64 = 1.0e-14,
        rtol::Float64 = sqrt(eps()),
    )
    return regeneration(
        System(F);
        sorted = sorted, max_codim = max_codim,
        tracker_options = tracker_options, endgame_options = endgame_options,
        monodromy_options = monodromy_options,
        show_progress = show_progress,
        show_monodromy_progress = show_monodromy_progress,
        threading = threading, seed = seed, atol = atol, rtol = rtol,
    )
end

function regeneration(
        F::S;
        sorted::Bool = true,
        max_codim::Union{Int, Nothing} = nothing,
        tracker_options::TrackerOptions = TrackerOptions(),
        # A tighter singular-condition threshold (1e12 vs the 1e14 global
        # default) so points lying on a higher-dimensional component, which are
        # ill-conditioned at t = 0, are rejected as singular by the u-homotopy
        # instead of being collected as spurious isolated points.
        endgame_options::EndgameOptions = EndgameOptions(;
            max_endgame_steps = 100, max_endgame_extended_steps = 100,
            sing_cond = 1.0e12,
        ),
        monodromy_options::MonodromyOptions = MonodromyOptions(;
            trace_test = true, parameter_sampler = weighted_normal,
        ),
        show_progress::Bool = true,
        show_monodromy_progress::Bool = false,
        threading::Bool = Threads.nthreads() > 1,
        seed::Union{Nothing, Integer} = nothing,
        atol::Float64 = 1.0e-14,
        rtol::Float64 = sqrt(eps()),
    )::Vector{WitnessSet{S}} where {S <: System}
    nparameters(F) == 0 || throw(
        ArgumentError(
            "regeneration does not support parametric systems; substitute the " *
                "parameter values first (cf. `witness_set(F; target_parameters)`).",
        ),
    )
    _check_regeneration_input(F, "`regeneration`")
    seed !== nothing && Random.seed!(seed)

    vars = collect(variables(F))
    n = nvariables(F)         # ambient dimension
    c = size(F)[1]            # number of equations
    expected_max_codim = min(c, n)
    codim = if max_codim !== nothing && max_codim < expected_max_codim
        # compute one extra codim so spurious points can be removed
        max_codim + 1
    else
        expected_max_codim
    end

    # u-regeneration adds a fresh variable u to F
    u = _fresh_variable(vars)
    push!(vars, u)

    # witness supersets, out[k] for codimension k
    out = initialize_witness_sets(codim, n)

    # witness sets for each hypersurface f_i = 0 on the seed subspace
    H = initialize_hypersurfaces(
        F, vars, linear_subspace(out[1]);
        threading = threading, tracker_options = tracker_options,
        endgame_options = endgame_options,
    )

    # sort equations by decreasing degree
    eqs = _regeneration_equations(F)
    if sorted
        σ = sortperm(H; by = degree, rev = true)
        eqs = eqs[σ]
        H = H[σ]
    end

    # core loop: intersect all current witness sets with each hypersurface
    progress = show_progress ?
        ProgressMeter.Progress(
            c; desc = "Regenerating witness sets: ", enabled = true,
        ) : nothing
    for i in 1:c
        if i == 1
            for p in solutions(H[1])
                push!(out[1], p)
            end
        else
            F_prev = System(
                eqs[1:(i - 1)]; parameters = empty(vars), variables = vars,
            )
            _intersect_regeneration_phase!(
                out, H, eqs, vars, u, i, codim, F_prev,
                tracker_options, endgame_options, threading, atol, rtol,
            )
            Fᵢ = System(eqs[1:i]; parameters = empty(vars), variables = vars)
            _fill_regeneration_phase!(
                out, monodromy_options, eqs, vars, u, i, codim, Fᵢ,
                tracker_options, endgame_options, show_monodromy_progress, threading,
            )
        end
        progress !== nothing && ProgressMeter.next!(progress)
    end
    pop!(vars)

    if max_codim !== nothing && max_codim < expected_max_codim
        pop!(out)   # drop the extra codim used to remove spurious points
    end

    filter!(W -> degree(W) > 0, out)
    isempty(out) && return WitnessSet{S}[]

    result = Vector{WitnessSet{S}}(undef, length(out))
    for i in eachindex(out)
        W = out[i]
        P, L = u_transform(W)
        result[i] = WitnessSet(F, L, P)
    end
    # Junk removal: a codim-k witness superset can pick up points where its
    # slice crosses a higher-dimensional component; those points lie on that
    # higher-dimensional witness set and are removed here.
    _remove_contained_points!(result; atol = atol, rtol = rtol)
    filter!(W -> degree(W) > 0, result)
    return result
end

# ── Front-end plumbing ───────────────────────────────────────────────────────
#
# Regeneration rebuilds its equations instead of only evaluating them, so each
# front-end supplies the equations, the degree that sets the number of roots of
# unity, the deformation start equation and the hypersurface to seed from.

_regeneration_equations(
    F::System{<:MP.AbstractPolynomialLike},
)::Vector{<:MP.AbstractPolynomialLike} = [MP.polynomial(p) for p in polynomials(F)]

_regeneration_equations(F::System{Expression})::Vector{Expression} =
    collect(polynomials(F))

# Every equation must be a polynomial or a ratio of polynomials in the variables.
_check_regeneration_input(
    ::System{<:MP.AbstractPolynomialLike}, ::String,
)::Nothing = nothing

function _check_regeneration_input(F::System{Expression}, route::String)::Nothing
    vars = collect(variables(F))
    for f in polynomials(F)
        (p, q) = num_den(f)
        (degree(p, vars) < 0 || degree(q, vars) < 0) && throw(
            ArgumentError(
                "$route needs equations that are polynomial or rational in the " *
                    "variables, but `$(f)` is neither. Clear the denominators " *
                    "first, or use a route that only evaluates the system.",
            ),
        )
    end
    return nothing
end

# `intersect` rebuilds all its input into a single system, so everything has to
# come from the same front-end.
_is_expression_front_end(::System{Expression})::Bool = true
_is_expression_front_end(::System)::Bool = false

_check_front_end(F::System, expression::Bool)::Nothing =
    _is_expression_front_end(F) == expression ? nothing : throw(
        ArgumentError(
            "the input mixes the polynomial and `Expression` front-ends; build the " *
            "witness sets and the hypersurface from the same kind of input.",
        ),
    )

# Number of roots of unity the u-homotopy starts from.
_u_degree(h::MP.AbstractPolynomialLike, ::AbstractVector)::Int = MP.maxdegree(h)
_u_degree(h::Expression, vars::AbstractVector{Expression})::Int =
    degree(first(num_den(h)), vars)

# Start equation of the u-homotopy. Carrying the denominator of the equation it
# deforms into makes both endpoint systems singular on the same set.
_u_start_equation(::MP.AbstractPolynomialLike, d::Int, u) = u^d - 1
_u_start_equation(h::Expression, d::Int, u::Expression)::Expression =
    (u^d - 1) / last(num_den(h))

# The hypersurface whose witness set gives the witness set of `f = 0`, and the
# denominator of `f`, or `nothing` when it has none.
_numerator_system(::MP.AbstractPolynomialLike, h::System, ::Vector) = (h, nothing)

function _numerator_system(f::Expression, h::System, vars::Vector{Expression})
    (p, q) = num_den(f)
    degree(q, vars) <= 0 && return (h, nothing)
    return (
        System([p]; parameters = Expression[], variables = vars),
        System([q]; parameters = Expression[], variables = vars),
    )
end

# Relative distance to `V(Q)` below which a witness point counts as lying on it.
const POLE_DISTANCE_TOL = 1.0e-10

# Drop the points on the denominator variety `V(Q)`: the numerator vanishes there too,
# so the point is a pole of the equation and not a zero of it. Tested by the
# first-order distance `|q(r)| / ‖∇q(r)‖` relative to `‖r‖`, which is invariant under
# rescaling the equation or its denominator and, unlike the magnitude of `q`, keeps a
# zero that merely sits near a pole.
function _drop_poles(
        Q::System, R::Vector{Vector{ComplexF64}},
    )::Vector{Vector{ComplexF64}}
    isempty(R) && return R
    m, n = size(Q)
    p_empty = FSVec{ComplexF64}(ComplexF64[])
    y = FSVec{ComplexF64}(zeros(ComplexF64, m))
    J = FSMat{ComplexF64}(zeros(ComplexF64, m, n))
    x = FSVec{ComplexF64}(zeros(ComplexF64, n))
    out = Vector{Vector{ComplexF64}}()
    for r in R
        x .= r
        evaluate_and_jacobian!(y, J, Q.evaluator, x, p_empty)
        v = LA.norm(y, Inf)
        isfinite(v) || continue
        grad = 0.0
        for j in 1:n
            grad += abs2(J[1, j])
        end
        grad = sqrt(grad)
        # Without a gradient there is no distance estimate, so only an exactly
        # vanishing denominator counts.
        d = grad > 0.0 ? v / grad : (iszero(v) ? 0.0 : Inf)
        d > POLE_DISTANCE_TOL * max(LA.norm(r, Inf), 1.0) && push!(out, r)
    end
    return out
end

# Express `eqs`, given in the variables `from`, in the variables `to`.
_rename_variables(eqs::Vector{<:MP.AbstractPolynomialLike}, from, to) =
    [MP.polynomial(MP.subs(f, from => to)) for f in eqs]

_rename_variables(eqs::Vector{Expression}, from, to)::Vector{Expression} =
    [subs(f, from => to) for f in eqs]

# Mint a fresh variable not colliding with any name in `vars`.
function _fresh_variable_name(vars::AbstractVector)::String
    names = Set(string(v) for v in vars)
    base = "u"
    name = base
    k = 0
    while name in names
        k += 1
        name = string("##", base, "_", k)
    end
    return name
end

_fresh_variable(vars::Vector{V}) where {V} = V(_fresh_variable_name(vars))
# An `Expression` is a tagged union, so its constructor does not take a name.
_fresh_variable(vars::Vector{Expression})::Expression =
    variable(_fresh_variable_name(vars))

# ── Intersection with a hypersurface ─────────────────────────────────────────

function intersect_all!(
        out, H, state::RegenerationState;
        threading::Bool, atol::Float64, rtol::Float64,
    )
    i = state.i
    codim = state.codim
    Hᵢ = H[i]

    # enumerate reversely, so points can be added to already-processed sets
    for k in length(out):-1:1     # k = codim(out[k])
        Wₖ = out[k]
        if k < i
            Wₖ₊₁ = k < codim ? out[k + 1] : nothing
            intersect_with_hypersurface!(
                Wₖ, Hᵢ, Wₖ₊₁, state;
                threading = threading, atol = atol, rtol = rtol,
            )
        end
    end
    return nothing
end

function intersect_with_hypersurface!(
        W::WitnessPoints, H::WitnessSet, X, state::RegenerationState;
        threading::Bool, atol::Float64, rtol::Float64,
    )
    F = state.Fᵢ
    h = state.eqs[state.i]
    u = state.u
    vars = state.vars
    P = points(W)

    # Step 1: points of W already contained in H stay; the rest go up a dim.
    # The containment test must move H's own witness points along `system(H)` and
    # check whether x lies on V(H). Passing the accumulated `state.Fᵢ` is wrong:
    # H's points are not on V(Fᵢ), so the move starts off-variety and can land
    # spuriously near x (false positive), mis-routing a point and dropping a
    # whole component.
    m = .!(
        _is_contained(
            W, H, system(H);
            atol = atol, rtol = rtol,
            tracker_options = state.tracker_options,
            endgame_options = state.endgame_options,
        )
    )
    P_next = _manage_initial_points!(P, m)
    isempty(P_next) && return nothing
    X === nothing && return nothing

    # Step 2: track P_next × (d-th roots of unity) through the u-homotopy.
    F₀, G₀, d = _u_homotopy_systems(W, F, X, h, vars, u)
    γ = cis(2π * rand())
    roots = ComplexF64[cis(2π * k / d) for k in 0:(d - 1)]
    if threading && length(P_next) * d > 1
        _threaded_intersection!(
            X, P_next, roots, F₀, G₀, γ,
            state.tracker_options, state.endgame_options,
        )
    else
        Hom = StraightLineHomotopy(F₀.evaluator, G₀.evaluator; γ = γ)
        eg = _endgame_tracker(Hom, state.tracker_options, state.endgame_options)
        _serial_intersection!(X, P_next, roots, eg)
    end
    return nothing
end

function _manage_initial_points!(P, m)
    P_next = P[m]
    deleteat!(P, m)
    return P_next
end

# Track one (point, root-of-unity) start through the u-homotopy. Returns the
# endpoint when it is a valid isolated intersection point, else `nothing`.
function _track_u_root(
        eg::EndgameTracker, q0::Vector{ComplexF64},
    )::Union{Nothing, Vector{ComplexF64}}
    track!(eg, FSVec{ComplexF64}(q0))
    pr = PathResult(eg; path_number = 0, start_solution = q0)
    if is_success(pr) && is_finite(pr) && is_nonsingular(pr)
        q = solution(pr)
        # sanity: q must be trackable slightly back from t = 0
        code = track!(
            eg.tracker, FSVec{ComplexF64}(q);
            t₁ = complex(0.0), t₀ = complex(0.1),
        )
        code == TrackerCode.TRACKER_SUCCESS && return q
    end
    return nothing
end

function _serial_intersection!(X::WitnessPoints, P, roots, eg::EndgameTracker)
    for p in P
        for ζ in roots
            q0 = copy(p)
            q0[end] = ζ    # replace the u-placeholder with a d-th root of unity
            q = _track_u_root(eg, q0)
            q === nothing || push!(X, q)
        end
    end
    return nothing
end

# Threaded variant: one task per (point, root) pair, each with its own
# u-homotopy built from cloned evaluators (interpreter tapes are mutable, so
# tasks must not share them). All tasks use the same γ, since they track paths
# of the SAME homotopy. Endpoints are collected per job index and pushed in the
# serial order.
function _threaded_intersection!(
        X::WitnessPoints, P::Vector{Vector{ComplexF64}}, roots::Vector{ComplexF64},
        F₀::S1, G₀::S2, γ::ComplexF64,
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
    )::Nothing where {S1 <: System, S2 <: System}
    nroots = length(roots)
    njobs = length(P) * nroots
    results = [ComplexF64[] for _ in 1:njobs]
    nt = Threads.nthreads()
    @tasks for k in 1:njobs
        @set ntasks = nt
        @local eg = _endgame_tracker(
            StraightLineHomotopy(
                _clone_system_evaluator(F₀), _clone_system_evaluator(G₀); γ = γ,
            ),
            tracker_options, endgame_options,
        )
        i, j = fldmod1(k, nroots)
        q0 = copy(P[i])
        q0[end] = roots[j]
        q = _track_u_root(eg, q0)
        q === nothing || (results[k] = q)
    end
    for r in results
        isempty(r) || push!(X, r)
    end
    return nothing
end

# Deform `u^d − 1` into `h`, moving from W's type-1 subspace to X's type-2.
# Returns the two sliced ambient endpoint systems and the degree `d`.
function _u_homotopy_systems(
        W::WitnessPoints, F::System, X::WitnessPoints, h, vars, u,
    )
    d = _u_degree(h, vars)
    h0 = _u_start_equation(h, d, u)
    eqs = polynomials(F)

    L = linear_subspace_u(W)      # start: u free
    K = linear_subspace(X)        # target: u = c

    F₀ = _sliced_system([eqs; h0], collect(vars), L)
    G₀ = _sliced_system([eqs; h], collect(vars), K)
    return F₀, G₀, d
end

# ── Membership within regeneration (structured subspace) ─────────────────────

# Whether each point of X lies on Y's variety. Builds a query subspace reusing
# X's linear equations (plus random rows), passing through each point, then
# moves Y's witness points there and checks proximity.
function _is_contained(
        X::WitnessPoints, Y::WitnessSet, F::System;
        atol::Float64, rtol::Float64,
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
    )::BitVector
    LX = linear_subspace(X)
    LY = linear_subspace(Y)
    (isempty(points(Y)) || isempty(points(X))) && return falses(length(points(X)))

    A = _membership_matrix(LX, LY)
    m, n = size(F)
    x0 = LA.normalize!(randn(ComplexF64, n))
    y0 = FSVec{ComplexF64}(zeros(ComplexF64, m))
    y = FSVec{ComplexF64}(zeros(ComplexF64, m))
    p_empty = FSVec{ComplexF64}(ComplexF64[])

    # Move Y's witness points with an IntrinsicSubspaceHomotopy, unconditionally
    # (one concrete type, no runtime dispatch). The extrinsic system `[F; A x − b]`
    # is singular at the start points whenever `dim(LY) <= codim(LY)`, so tracks
    # die with PATH_TERMINATED_INVALID_START and a whole component gets dropped.
    # The intrinsic form `x = A(t) v + a(t)` is well conditioned in every
    # dim/codim regime and tracks in only `dim(LY) <= n` coordinates.
    Hom = IntrinsicSubspaceHomotopy(F.evaluator, LY, LY)
    eg = _endgame_tracker(Hom, tracker_options, endgame_options)
    u_buf = FSVec{ComplexF64}(zeros(ComplexF64, size(Hom)[2]))
    amb_buf = FSVec{ComplexF64}(zeros(ComplexF64, n))

    out = falses(length(points(X)))
    for (idx, x) in enumerate(points(X))
        xc = ComplexF64.(x)
        evaluate!(y0, F.evaluator, FSVec{ComplexF64}(LA.norm(xc, Inf) .* x0), p_empty)
        evaluate!(y, F.evaluator, FSVec{ComplexF64}(xc), p_empty)
        LA.norm(y, Inf) > 1.0e-2 * LA.norm(y0, Inf) && continue

        bx = A * xc
        L_x = LinearSubspace(ExtrinsicDescription(A, bx; orthonormal = true))
        set_subspaces!(Hom, LY, L_x)
        rad = max(atol, LA.norm(xc, Inf) * rtol)
        for q0 in points(Y)
            intrinsic_coordinates!(u_buf, Hom, ComplexF64.(q0), complex(1.0))
            track!(eg, u_buf)
            pr = PathResult(eg; path_number = 0, start_solution = Vector{ComplexF64}(q0))
            is_success(pr) || continue
            ambient_coordinates!(amb_buf, Hom, solution(pr), complex(0.0))
            if _vt_distance(InfNorm(), amb_buf, xc) < rad
                out[idx] = true
                break
            end
        end
    end
    return out
end

# Query-subspace matrix reusing X's rows: row 1 fixes u, the next `k` rows are
# random, the rest reuse X's linear equations. b is set per point as A·x.
function _membership_matrix(LX::LinearSubspace, LY::LinearSubspace)::Matrix{ComplexF64}
    n = ambient_dim(LY)
    cX = codim(LX)
    cY = codim(LY)
    k = cY - cX
    A = zeros(ComplexF64, cY, n)
    AX = extrinsic(LX).A
    A[1, n] = one(ComplexF64)
    for i in 2:(k + 1), j in 1:n
        A[i, j] = randn(ComplexF64)
    end
    for i in 2:cX
        ℓ = k + i
        for j in 1:n
            A[ℓ, j] = AX[i, j]
        end
    end
    return A
end

# ── Fill up witness supersets via monodromy ──────────────────────────────────

# Run monodromy with a full MonodromyOptions object (the public entrypoint only
# accepts explicit kwargs, so go through MonodromySolver + _monodromy_solve!).
function _monodromy_with_options(
        F::System, X::AbstractVector{<:AbstractVector}, L::LinearSubspace,
        opts::MonodromyOptions;
        tracker_options::TrackerOptions = TrackerOptions(),
        threading::Bool, show_progress::Bool,
        seed::UInt32 = rand(UInt32),
    )::MonodromyResult
    cp = convert(LinearSubspace{ComplexF64}, L)
    MS = MonodromySolver(F, cp; options = opts, tracker_options = tracker_options)
    return _monodromy_solve!(
        MS, X, cp, seed;
        show_progress = show_progress, threading = threading,
        catch_interrupt = true, warning = false,
    )
end

function fill_up!(
        out, monodromy_options::MonodromyOptions, state::RegenerationState,
        show_monodromy_progress::Bool, threading::Bool,
    )
    Fᵢ = state.Fᵢ
    for W in out
        if W !== nothing && dim(W) > 0 && degree(W) > 0
            opts = _regeneration_monodromy_options(monodromy_options, W)
            res = _monodromy_with_options(
                Fᵢ, W.R, linear_subspace(W), opts;
                tracker_options = state.tracker_options,
                threading = threading, show_progress = show_monodromy_progress,
            )
            W.R = nsolutions(res) == 0 ? Vector{Vector{ComplexF64}}() :
                unique_points(solutions(res))
        end
    end
    return nothing
end

# ── Intersecting witness sets (u-homotopy) ───────────────────────────────────

# The u-value `c` such that the appended points `[x; c]` lie on the type-2
# subspace `flag[1][1]`.
_get_c(flag) = extrinsic(flag[1][1]).b[1]

"""
    intersect(W::WitnessSet, H::WitnessSet; options...)

Intersect the witness set `W` with the witness set `H` of a single hypersurface,
returning the witness set(s) of `V(system(W)) ∩ V(system(H))` obtained by one
u-regeneration step. Returns a single `WitnessSet` when the result has one
dimension, otherwise a `Vector{WitnessSet}`.

    intersect(W::WitnessSet, f; options...)

Compute a witness set `H` for the hypersurface `f` and return `intersect(W, H)`.
"""
function Base.intersect(
        W::WitnessSet,
        H::WitnessSet;
        # `show_progress` is accepted for API consistency with `regeneration` /
        # `nid` (both take it) and to keep unknown-keyword typos loud rather than
        # silently swallowed. `intersect` performs a single u-regeneration step
        # and renders no step-level bar; the monodromy fill-up sub-progress is
        # controlled by `show_monodromy_progress`.
        show_progress::Bool = false,
        show_monodromy_progress::Bool = false,
        tracker_options::TrackerOptions = TrackerOptions(),
        # Endgame options (see `regeneration` above): the tighter sing_cond
        # rejects near-singular points on higher-dimensional components that
        # would otherwise over-collect the u-homotopy output.
        endgame_options::EndgameOptions = EndgameOptions(;
            max_endgame_steps = 100, max_endgame_extended_steps = 100,
            sing_cond = 1.0e12,
        ),
        monodromy_options::MonodromyOptions = MonodromyOptions(;
            trace_test = true, parameter_sampler = weighted_normal,
        ),
        threading::Bool = Threads.nthreads() > 1,
        atol::Float64 = 1.0e-14,
        rtol::Float64 = sqrt(eps()),
    )
    size(system(H))[1] == 1 ||
        throw(ArgumentError("The second argument must be defined by a single equation."))
    size(system(W))[2] == size(system(H))[2] ||
        throw(ArgumentError("Witness sets must be in the same ambient space."))
    _check_front_end(system(H), _is_expression_front_end(system(W)))
    _check_regeneration_input(system(W), "`intersect`")
    _check_regeneration_input(system(H), "`intersect`")

    vars = collect(variables(system(W)))
    u = _fresh_variable(vars)
    vars_u = [vars; u]

    # W's and H's equations, both expressed in W's variables `vars`.
    eqs = _regeneration_equations(system(W))
    heqs = _rename_variables(
        _regeneration_equations(system(H)), variables(system(H)), vars,
    )
    h = heqs[1]

    # Flags in (n+1)-space; the u-value fixes the appended coordinate.
    flagW = get_flag(1:2, linear_subspace(W))
    flagH = get_flag(1:1, linear_subspace(H))
    cW = _get_c(flagW)
    cH = _get_c(flagH)

    W₁ = WitnessPoints(
        flagW[1][1], flagW[1][2], [ComplexF64[x; cW] for x in solutions(W)],
    )
    W₂ = dim(W) == 0 ? nothing :
        WitnessPoints(flagW[2][1], flagW[2][2], Vector{Vector{ComplexF64}}())
    Hᵤ = WitnessSet(
        System(heqs; parameters = empty(vars_u), variables = vars_u), flagH[1][1],
        [ComplexF64[x; cH] for x in solutions(H)],
    )

    intersect_state = RegenerationState(
        [eqs; h], vars_u, u, length(eqs) + 1, 2,
        System(eqs; parameters = empty(vars_u), variables = vars_u),
        tracker_options, endgame_options,
    )

    intersect_with_hypersurface!(
        W₁, Hᵤ, W₂, intersect_state;
        threading = threading, atol = atol, rtol = rtol,
    )
    fill_state = RegenerationState(
        [eqs; h], vars_u, u, length(eqs) + 1, 2,
        System([eqs; h]; parameters = empty(vars_u), variables = vars_u),
        tracker_options, endgame_options,
    )
    Ws = W₂ === nothing ? WitnessPoints[W₁] : WitnessPoints[W₁, W₂]
    # The d-th-root tracking can reach the same solution more than once; dedupe
    # the start points so the monodromy fill-up is not seeded with duplicates.
    for Wi in Ws
        isempty(Wi.R) || (Wi.R = unique_points(Wi.R))
    end
    fill_up!(Ws, monodromy_options, fill_state, show_monodromy_progress, threading)

    G = System([eqs; h]; parameters = empty(vars), variables = vars)
    out = WitnessSet[]
    for Wi in Ws
        P, L = u_transform(Wi)
        push!(out, WitnessSet(G, L, P))
    end
    # Remove spurious witness points of a lower-dimensional set that actually
    # lie on a higher-dimensional component (they are junk from the u-homotopy).
    _remove_contained_points!(out; atol = atol, rtol = rtol)
    filter!(X -> degree(X) > 0, out)
    return length(out) == 1 ? first(out) : out
end

# For witness sets sorted by decreasing dimension, drop from each set the points
# that are contained in any higher-dimensional set (junk points).
function _remove_contained_points!(
        out::Vector{<:WitnessSet}; atol::Float64, rtol::Float64,
    )
    sort!(out; by = dim, rev = true)
    for i in eachindex(out)
        Wi = out[i]
        isempty(Wi.R) && continue
        keep = trues(length(Wi.R))
        for j in 1:(i - 1)
            dim(out[j]) > dim(Wi) || continue
            for (idx, p) in enumerate(Wi.R)
                keep[idx] || continue
                membership(
                    p, out[j];
                    atol = atol, rtol = rtol, show_progress = false,
                ) && (keep[idx] = false)
            end
        end
        out[i] = WitnessSet(
            Wi.F, Wi.L, Wi.R[keep]; projective = Wi.projective,
        )
    end
    return nothing
end

function Base.intersect(
        W::WitnessSet,
        f::MP.AbstractPolynomialLike;
        show_progress::Bool = false,
        show_monodromy_progress::Bool = false,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(;
            max_endgame_steps = 100, max_endgame_extended_steps = 100,
            sing_cond = 1.0e12,
        ),
        monodromy_options::MonodromyOptions = MonodromyOptions(;
            trace_test = true, parameter_sampler = weighted_normal,
        ),
        threading::Bool = Threads.nthreads() > 1,
        atol::Float64 = 1.0e-14,
        rtol::Float64 = sqrt(eps()),
    )
    _check_front_end(system(W), false)
    H = _hypersurface_witness_set(
        f, collect(variables(system(W)));
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
    )
    return intersect(
        W, H;
        show_progress = show_progress,
        show_monodromy_progress = show_monodromy_progress,
        tracker_options = tracker_options, endgame_options = endgame_options,
        monodromy_options = monodromy_options, threading = threading,
        atol = atol, rtol = rtol,
    )
end

function Base.intersect(
        W::WitnessSet,
        f::Expression;
        show_progress::Bool = false,
        show_monodromy_progress::Bool = false,
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = EndgameOptions(;
            max_endgame_steps = 100, max_endgame_extended_steps = 100,
            sing_cond = 1.0e12,
        ),
        monodromy_options::MonodromyOptions = MonodromyOptions(;
            trace_test = true, parameter_sampler = weighted_normal,
        ),
        threading::Bool = Threads.nthreads() > 1,
        atol::Float64 = 1.0e-14,
        rtol::Float64 = sqrt(eps()),
    )
    _check_front_end(system(W), true)
    H = _hypersurface_witness_set(
        f, _as_variables(collect(variables(system(W))));
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
    )
    return intersect(
        W, H;
        show_progress = show_progress,
        show_monodromy_progress = show_monodromy_progress,
        tracker_options = tracker_options, endgame_options = endgame_options,
        monodromy_options = monodromy_options, threading = threading,
        atol = atol, rtol = rtol,
    )
end

# Witness set of `V(f)` in the ambient space of `vars`, which may hold variables `f`
# does not use. The slice is the affine line the flag is built from, also for a
# homogeneous `f`, whose projective slice has one dimension too many for it.
function _hypersurface_witness_set(
        f::MP.AbstractPolynomialLike, vars::Vector;
        show_progress::Bool, threading::Bool,
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
    )::WitnessSet
    extra = setdiff(MP.effective_variables(f), vars)
    isempty(extra) || throw(
        ArgumentError(
            "the hypersurface must be given in the variables of the witness set, " *
                "but `$(f)` also uses $(join(extra, ", ")).",
        ),
    )
    return witness_set(
        System([f]; parameters = empty(vars), variables = vars),
        rand_subspace(length(vars); dim = 1);
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
    )
end

# A rational `f` is solved through its numerator, with its poles dropped.
function _hypersurface_witness_set(
        f::Expression, vars::Vector{Expression};
        show_progress::Bool, threading::Bool,
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
    )::WitnessSet
    (p, q) = num_den(f)
    (degree(p, vars) < 0 || degree(q, vars) < 0) && throw(
        ArgumentError(
            "`intersect` needs a hypersurface that is polynomial or rational in " *
                "the variables, but `$(f)` is neither.",
        ),
    )
    h = System([f]; parameters = Expression[], variables = vars)
    G, Q = _numerator_system(f, h, vars)
    Wp = witness_set(
        G, rand_subspace(length(vars); dim = 1);
        show_progress = show_progress, threading = threading,
        tracker_options = tracker_options, endgame_options = endgame_options,
    )
    R = Q === nothing ? solutions(Wp) : _drop_poles(Q, solutions(Wp))
    return WitnessSet(h, linear_subspace(Wp), R)
end

function _regeneration_monodromy_options(M::MonodromyOptions, W)
    return MonodromyOptions(;
        permutations = false,
        trace_test = true,
        single_loop_per_start_solution = M.single_loop_per_start_solution,
        check_startsolutions = M.check_startsolutions,
        group_actions = M.group_actions,
        loop_finished_callback = M.loop_finished_callback,
        parameter_sampler = M.parameter_sampler,
        equivalence_classes = M.equivalence_classes,
        trace_test_tol = M.trace_test_tol,
        # allow a little slack in case a singular solution slips through
        target_solutions_count = Int(floor(1.5 * degree(W))),
        timeout = M.timeout,
        min_solutions = M.min_solutions,
        max_loops_no_progress = M.max_loops_no_progress,
        reuse_loops = M.reuse_loops,
        distance = M.distance,
        triangle_inequality = M.triangle_inequality,
        unique_points_atol = M.unique_points_atol,
        unique_points_rtol = M.unique_points_rtol,
    )
end
