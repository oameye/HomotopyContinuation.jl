## u-regeneration.
#
# Computes witness supersets for `V(F)` equation-by-equation, following the
# u-regeneration algorithm of Duff, Leykin and Rodriguez
# (https://arxiv.org/abs/2206.02869). Returns one `WitnessSet` per codimension
# WITHOUT decomposing into irreducible components (a witness *superset*); use
# `decompose` for the irreducible decomposition.
#
# Notes:
# * The deformation equation is `u^d - 1` (`u^d - ℓ^d` for homogeneous input, so
#   that it stays homogeneous), divided by the denominator of the equation it
#   deforms into so both endpoint systems have the same poles.
# * Every `slice(System, L)` is the ambient `_sliced_system` (`[F; A x − b]`).
# * The intersection u-homotopy is a `StraightLineHomotopy` between two sliced
#   ambient systems; the fill-up and membership steps reuse the existing
#   `ExtrinsicSubspaceHomotopy` and `MonodromySolver`.

# ── WitnessPoints: an undecomposed witness superset during regeneration ───────

"""
    WitnessPoints(L, Lᵤ, R)

Internal container used by [`Regeneration`](@ref). Stores witness points `R`
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

# Build the flag of (type-2, type-1) subspace pairs from a base subspace `L₀` in
# n-space. For index `i` this drops the first `i-1` rows of the base and, for
# type 2, prepends the equation `u = c`. A linear `L₀` takes `c = 0`, so the whole
# flag stays linear and the u-regeneration runs projectively.
function get_flag(iter, L₀::LinearSubspace, rng::Random.MersenneTwister)
    A₀ = extrinsic(L₀).A          # orthonormal rows
    b₀ = extrinsic(L₀).b
    n = size(A₀, 1) + 1
    m = size(A₀, 2)

    # type 1: no condition on u (last column zero)
    Aᵤ = [A₀ zeros(ComplexF64, n - 1)]
    bᵤ = b₀
    # type 2: prepend the equation u = c
    c = is_linear(L₀) ? zero(ComplexF64) : randn(rng, ComplexF64)
    A = [zeros(ComplexF64, 1, m) one(ComplexF64); A₀ zeros(ComplexF64, n - 1)]
    b = [c; b₀]

    return map(iter) do i
        # `i > n - 1` leaves no rows: an empty range, not a wrapped one.
        idx = i <= n - 1 ? (i:(n - 1)) : (n:(n - 1))
        Eᵤ = ExtrinsicDescription(Aᵤ[idx, :], bᵤ[idx]; orthonormal = true)
        Lᵤ = LinearSubspace(Eᵤ)
        rows = [1; idx .+ 1]
        E = ExtrinsicDescription(A[rows, :], b[rows]; orthonormal = true)
        L = LinearSubspace(E)
        return (L, Lᵤ)
    end
end

function initialize_witness_sets(
        codim::Int, n::Int, rng::Random.MersenneTwister; affine::Bool = true,
    )::Vector{WitnessPoints}
    # A projective slice needs one more ambient dimension than an affine one to
    # cut out the same projective dimension.
    d = affine ? 1 : 2
    L₀ = d == n ? LinearSubspace(zeros(ComplexF64, 0, n)) :
        rand_subspace(rng, n; dim = d, affine = affine)
    flag = get_flag(1:codim, L₀, rng)
    out = Vector{WitnessPoints}(undef, length(flag))
    for (i, (L, Lᵤ)) in enumerate(flag)
        out[i] = WitnessPoints(L, Lᵤ, Vector{Vector{ComplexF64}}())
    end
    return out
end

# Witness set of each hypersurface `f_i = 0` on the seed subspace `L`. A rational
# equation goes through its numerator, minus the zeros that are poles of it.
function initialize_hypersurfaces(
        F::System{P, V}, vars::Vector{V}, L::LinearSubspace,
        rng::Random.MersenneTwister, exec::AbstractExecutor,
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
    ) where {P, V}
    fs = polynomials(F)
    hyper_alg = Witness(;
        solver = TotalDegree(;
            tracker_options = tracker_options, endgame_options = endgame_options,
        ),
        show_progress = false,
    )
    HS = System{P, V, CompileMode.INTERPRETED, UnderdeterminedShape}
    out = Vector{WitnessSet{HS}}(undef, length(fs))
    for i in eachindex(fs)
        h = System([fs[i]]; parameters = empty(vars), variables = vars)::HS
        G, Q = _numerator_system(fs[i], h, vars)
        R = _witness_init(G, L, rng, hyper_alg, exec)
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
    # Homogeneous input is regenerated projectively: the flag stays linear, the
    # deformation equation is `u^d − ℓ^d` for the linear form `ℓ` (so it is
    # homogeneous too), and every homotopy runs on an affine chart.
    projective::Bool
    ℓ::P
    ℓ_coeffs::Vector{Float64}
    tracker_options::TrackerOptions
    endgame_options::EndgameOptions
    # Every draw below a regeneration route comes off this one stream, so the
    # route's `seed` alone determines the result.
    rng::Random.MersenneTwister
end

@noinline function _intersect_regeneration_phase!(
        out::Vector{WitnessPoints}, H::Vector{W},
        eqs::Vector{P}, vars::Vector{V}, u::V,
        i::Int, codim::Int, F_prev::S,
        projective::Bool, ℓ::P, ℓ_coeffs::Vector{Float64},
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
        rng::Random.MersenneTwister,
        exec::AbstractExecutor, atol::Float64, rtol::Float64,
    )::Nothing where {W <: WitnessSet, P, V, S <: System}
    state = RegenerationState(
        eqs, vars, u, i, codim, F_prev, projective, ℓ, ℓ_coeffs,
        tracker_options, endgame_options, rng,
    )
    intersect_all!(out, H, state, exec; atol = atol, rtol = rtol)
    return nothing
end

@noinline function _fill_regeneration_phase!(
        out::Vector{WitnessPoints}, monodromy_options::MonodromyOptions,
        eqs::Vector{P}, vars::Vector{V}, u::V,
        i::Int, codim::Int, Fᵢ::S,
        projective::Bool, ℓ::P, ℓ_coeffs::Vector{Float64},
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
        rng::Random.MersenneTwister,
        show_monodromy_progress::Bool, exec::AbstractExecutor,
    )::Nothing where {P, V, S <: System}
    state = RegenerationState(
        eqs, vars, u, i, codim, Fᵢ, projective, ℓ, ℓ_coeffs,
        tracker_options, endgame_options, rng,
    )
    fill_up!(out, monodromy_options, state, show_monodromy_progress, exec)
    return nothing
end

# A tighter singular-condition threshold than the global default, so points lying
# on a higher-dimensional component (ill-conditioned at t = 0) are rejected as
# singular by the u-homotopy instead of collected as spurious isolated points.
const _REGENERATION_ENDGAME = EndgameOptions(;
    max_endgame_steps = 100, max_endgame_extended_steps = 100, sing_cond = 1.0e12,
)

_equation_by_equation_monodromy() =
    MonodromyOptions(; trace_test = true, parameter_sampler = weighted_normal)

"""
    EquationSorting

How [`Regeneration`](@ref) orders the equations before regenerating:
`EquationSorting.UNSORTED` keeps the given order, `EquationSorting.BY_DEGREE`
sorts by decreasing degree, and `EquationSorting.RANDOMIZED` additionally
replaces them by a random upper-triangular combination of the sorted equations.
"""
@enumx EquationSorting::Int8 begin
    UNSORTED
    BY_DEGREE
    RANDOMIZED
end

"""
    Regeneration(; sorted, max_codim, monodromy, atol, rtol, options...)

Solve a system equation by equation and return a [`WitnessSet`](@ref) for every
dimension, without decomposing into irreducible components (a witness *superset*).

Every equation must be polynomial or rational in the variables. A rational
equation is handled through its numerator, and numerator zeros that are poles of
the equation are dropped.

`sorted` is an [`EquationSorting`](@ref) choosing how the equations are ordered;
`EquationSorting.RANDOMIZED` is not available for homogeneous input.
`max_codim` bounds the codimension computed. `show_progress` draws the
codimension bar, `show_monodromy_progress` the bar of every monodromy fill-up
underneath it.
"""
struct Regeneration{MO <: MonodromyOptions} <: AbstractAlgorithm
    common::CommonOptions
    monodromy::MO
    show_monodromy_progress::Bool
    sorted::EquationSorting.T
    max_codim::Union{Nothing, Int}
    atol::Float64
    rtol::Float64
end

Regeneration(;
    sorted::EquationSorting.T = EquationSorting.BY_DEGREE,
    max_codim::Union{Nothing, Int} = nothing,
    monodromy::MonodromyOptions = _equation_by_equation_monodromy(),
    atol::Float64 = 1.0e-14,
    rtol::Float64 = sqrt(eps()),
    tracker_options::TrackerOptions = TrackerOptions(),
    endgame_options::EndgameOptions = _REGENERATION_ENDGAME,
    seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    show_progress::Bool = true,
    show_monodromy_progress::Bool = false,
) = Regeneration(
    CommonOptions(tracker_options, endgame_options, seed, show_progress),
    monodromy, show_monodromy_progress, sorted, max_codim, atol, rtol,
)

"""
    Intersection(; monodromy, atol, rtol, options...)

Intersect a [`WitnessSet`](@ref) with another witness set or with a hypersurface,
by one regeneration step followed by a monodromy fill-up.

One regeneration step renders no bar of its own: `show_progress` applies to the
witness set computed for a hypersurface argument, and `show_monodromy_progress`
to the fill-up.
"""
struct Intersection{MO <: MonodromyOptions} <: AbstractAlgorithm
    common::CommonOptions
    monodromy::MO
    show_monodromy_progress::Bool
    atol::Float64
    rtol::Float64
end

Intersection(;
    monodromy::MonodromyOptions = _equation_by_equation_monodromy(),
    atol::Float64 = 1.0e-14,
    rtol::Float64 = sqrt(eps()),
    tracker_options::TrackerOptions = TrackerOptions(),
    endgame_options::EndgameOptions = _REGENERATION_ENDGAME,
    seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    show_progress::Bool = false,
    show_monodromy_progress::Bool = false,
) = Intersection(
    CommonOptions(tracker_options, endgame_options, seed, show_progress),
    monodromy, show_monodromy_progress, atol, rtol,
)

_reseed(alg::Regeneration, seed::UInt32) = Regeneration(
    _with_seed(alg.common, seed), alg.monodromy, alg.show_monodromy_progress,
    alg.sorted, alg.max_codim, alg.atol, alg.rtol,
)

_reseed(alg::Intersection, seed::UInt32) = Intersection(
    _with_seed(alg.common, seed), alg.monodromy, alg.show_monodromy_progress,
    alg.atol, alg.rtol,
)

# ── regeneration ─────────────────────────────────────────────────────────────

"""
    solve(F, alg::Regeneration, exec = Threaded())

Solve `F = 0` equation-by-equation and return a [`WitnessSet`](@ref) for every
dimension without decomposing into irreducible components (a witness
*superset*). Based on the u-regeneration algorithm of Duff, Leykin and
Rodriguez (https://arxiv.org/abs/2206.02869).

`F` may be a [`System`](@ref), a single polynomial or a vector of polynomials,
and must be parameter-free; fix the values first with [`fix_parameters`](@ref).
Every equation must be polynomial or rational in the variables of `F`. A rational
equation is handled through its numerator, and the zeros of the numerator that
are poles of the equation are dropped.

See [`Regeneration`](@ref) for the options. Every random choice descends from its
`seed`, so the same `seed` gives the same witness supersets regardless of the
state of the global random number generator.
"""
function solve(
        F::S,
        alg::Regeneration,
        exec::AbstractExecutor = Threaded(),
    )::Vector{WitnessSet{S}} where {S <: System}
    sorted = alg.sorted
    max_codim = alg.max_codim
    tracker_options = _tracker_options(alg)
    endgame_options = _endgame_options(alg)
    monodromy_options = alg.monodromy
    show_progress = _show_progress(alg)
    show_monodromy_progress = alg.show_monodromy_progress
    seed = _seed(alg)
    atol = alg.atol
    rtol = alg.rtol
    nparameters(F) == 0 || throw(
        ArgumentError(
            "`regeneration` requires a parameter-free system, but the system has " *
                "$(nparameters(F)) parameter(s). Fix them first with " *
                "`fix_parameters(F, p)`.",
        ),
    )
    _check_regeneration_input(F, "`regeneration`")
    rng = Random.MersenneTwister(seed)

    xvars = collect(variables(F))
    vars = copy(xvars)
    n = nvariables(F)         # ambient dimension
    c = size(F)[1]            # number of equations
    projective = is_homogeneous(F)
    _check_regeneration_sorted(sorted, projective)
    # A projective variety of codimension k lives in the ambient space with one
    # dimension spent on the cone direction.
    expected_max_codim = min(c, n - projective)
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
    out = initialize_witness_sets(codim, n, rng; affine = !projective)

    # witness sets for each hypersurface f_i = 0 on the seed subspace
    H = initialize_hypersurfaces(
        F, vars, linear_subspace(out[1]), rng, exec,
        tracker_options, endgame_options,
    )

    # sort equations by decreasing degree
    eqs = _regeneration_equations(F)
    if sorted !== EquationSorting.UNSORTED
        σ = sortperm(H; by = degree, rev = true)
        eqs = eqs[σ]
        H = H[σ]
    end
    if sorted === EquationSorting.RANDOMIZED
        eqs = _randomize_equations(eqs, rng)
        # The combination is upper triangular with generic coefficients, so
        # `V(eqs[i:end])` is unchanged and the sorted degrees still hold; the
        # hypersurface witness sets that seeded the sort are not.
        H = initialize_hypersurfaces(
            System(eqs; parameters = empty(vars), variables = vars),
            vars, linear_subspace(out[1]), rng, exec,
            tracker_options, endgame_options,
        )
    end

    # The deformation equation `u^d − ℓ^d` must be homogeneous in the projective
    # case, so `1` is replaced by a generic linear form.
    ℓ_coeffs = projective ? randn(rng, n) : Float64[]
    ℓ = _linear_form(eqs, xvars, ℓ_coeffs)

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
                projective, ℓ, ℓ_coeffs,
                tracker_options, endgame_options, rng, exec, atol, rtol,
            )
            Fᵢ = System(eqs[1:i]; parameters = empty(vars), variables = vars)
            _fill_regeneration_phase!(
                out, monodromy_options, eqs, vars, u, i, codim, Fᵢ,
                projective, ℓ, ℓ_coeffs,
                tracker_options, endgame_options, rng,
                show_monodromy_progress, exec,
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
        result[i] = WitnessSet(F, L, P; projective = projective)
    end
    # Junk removal: a codim-k witness superset can pick up points where its
    # slice crosses a higher-dimensional component; those points lie on that
    # higher-dimensional witness set and are removed here.
    _remove_contained_points!(result, rng, exec; atol = atol, rtol = rtol)
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

# Start equation of the u-homotopy: `u^d − 1` affinely, `u^d − ℓ^d` projectively,
# where `ℓ` keeps it homogeneous. Carrying the denominator of the equation it
# deforms into makes both endpoint systems singular on the same set.
_u_start_equation(::MP.AbstractPolynomialLike, d::Int, u, one_or_ℓ) =
    u^d - one_or_ℓ^d
_u_start_equation(h::Expression, d::Int, u::Expression, one_or_ℓ)::Expression =
    (u^d - one_or_ℓ^d) / last(num_den(h))

# What the start equation `u^d − ℓ^d` subtracts, typed like the equations it
# joins: the linear form `Σ coeffs[j] xvars[j]` projectively, and `1` affinely,
# which is the plain `u^d − 1`. Real coefficients: the locus they must avoid is a
# proper subvariety, which meets `ℝⁿ` in measure zero.
function _linear_form(
        eqs::Vector{P}, xvars::Vector{V}, coeffs::Vector{Float64},
    )::P where {P <: MP.AbstractPolynomialLike, V}
    isempty(coeffs) && return one(P)
    return convert(P, sum(coeffs[j] * xvars[j] for j in eachindex(coeffs)))
end

function _linear_form(
        ::Vector{Expression}, xvars::Vector{Expression}, coeffs::Vector{Float64},
    )::Expression
    isempty(coeffs) && return Expression(1)
    return sum(coeffs[j] * xvars[j] for j in eachindex(coeffs))
end

# `u` starts on the `d`-th roots of `ℓ(p)^d` rather than of 1.
_u_scaling(p::Vector{ComplexF64}, ℓ_coeffs::Vector{Float64})::ComplexF64 =
    isempty(ℓ_coeffs) ? one(ComplexF64) :
    sum(ℓ_coeffs[j] * p[j] for j in eachindex(ℓ_coeffs))

# Replace the sorted equations by a random upper-triangular combination of
# themselves. `V(eqs[i:end])` is preserved for every `i`, so the regeneration
# still sees the same flag of varieties, but each equation now has the top degree.
function _randomize_equations(
        eqs::Vector{P}, rng::Random.MersenneTwister,
    )::Vector{P} where {P}
    c = length(eqs)
    return P[
        sum(randn(rng) * eqs[j] for j in i:c) for i in 1:c
    ]
end

function _check_regeneration_sorted(
        sorted::EquationSorting.T, projective::Bool,
    )::Nothing
    (sorted === EquationSorting.RANDOMIZED && projective) && throw(
        ArgumentError(
            "`EquationSorting.RANDOMIZED` is not available for a homogeneous " *
                "system: a random combination of the equations is not homogeneous " *
                "unless they all have the same degree.",
        ),
    )
    return nothing
end

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

solve(
    F::PolynomialInput, alg::Regeneration, exec::AbstractExecutor = Threaded(),
) = solve(_as_system(F), alg, exec)

# ── Intersection with a hypersurface ─────────────────────────────────────────

function intersect_all!(
        out, H, state::RegenerationState, exec::AbstractExecutor;
        atol::Float64, rtol::Float64,
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
                Wₖ, Hᵢ, Wₖ₊₁, state, exec; atol = atol, rtol = rtol,
            )
        end
    end
    return nothing
end

function intersect_with_hypersurface!(
        W::WitnessPoints, H::WitnessSet, X, state::RegenerationState,
        exec::AbstractExecutor; atol::Float64, rtol::Float64,
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
            W, H, system(H), state.rng, state.projective;
            atol = atol, rtol = rtol,
            tracker_options = state.tracker_options,
            endgame_options = state.endgame_options,
        )
    )
    P_next = _manage_initial_points!(P, m)
    isempty(P_next) && return nothing
    X === nothing && return nothing

    # Step 2: track P_next × (d-th roots of unity) through the u-homotopy.
    F₀, G₀, d = _u_homotopy_systems(W, F, X, h, vars, u, state.ℓ)
    γ = _random_gamma(state.rng)
    roots = ComplexF64[cis(2π * k / d) for k in 0:(d - 1)]
    # One chart for both endpoint systems and every task, so all the tracked
    # representatives are comparable. Empty means the route is affine.
    chart = state.projective ?
        randn(state.rng, ComplexF64, size(F₀)[2]) : ComplexF64[]
    ntasks = _wants_tasks(exec) && length(P_next) * d > 1 ? _local_ntasks(exec) : 0
    if state.projective
        _run_intersection!(
            X, P_next, roots, F₀, G₀, γ, chart, state.ℓ_coeffs, Val(true),
            state.tracker_options, state.endgame_options, ntasks,
        )
    else
        _run_intersection!(
            X, P_next, roots, F₀, G₀, γ, chart, state.ℓ_coeffs, Val(false),
            state.tracker_options, state.endgame_options, ntasks,
        )
    end
    return nothing
end

# `ntasks == 0` runs serially. Both branches are concretely typed in the homotopy.
function _run_intersection!(
        X::WitnessPoints, P::Vector{Vector{ComplexF64}}, roots::Vector{ComplexF64},
        F₀::S1, G₀::S2, γ::ComplexF64, chart::Vector{ComplexF64},
        ℓ_coeffs::Vector{Float64}, projective::Val,
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
        ntasks::Int,
    )::Nothing where {S1 <: System, S2 <: System}
    if ntasks > 0
        _threaded_intersection!(
            X, P, roots, F₀, G₀, γ, chart, ℓ_coeffs, projective,
            tracker_options, endgame_options, ntasks,
        )
    else
        eg = _endgame_tracker(
            _u_homotopy(F₀.evaluator, G₀.evaluator, γ, chart, projective),
            tracker_options, endgame_options,
        )
        _serial_intersection!(X, P, roots, eg, chart, ℓ_coeffs)
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

# The u-placeholder becomes a d-th root of 1 affinely, of `ℓ(p)^d` projectively.
# Charting afterwards rescales x and u together, which leaves `u^d = ℓ(x)^d` and
# the linear equations (all with `b = 0` here) satisfied.
function _u_start_point(
        p::Vector{ComplexF64}, ζ::ComplexF64, chart::Vector{ComplexF64},
        ℓ_coeffs::Vector{Float64},
    )::Vector{ComplexF64}
    q0 = copy(p)
    q0[end] = _u_scaling(p, ℓ_coeffs) * ζ
    isempty(chart) || on_chart!(q0, chart)
    return q0
end

function _serial_intersection!(
        X::WitnessPoints, P, roots, eg::EndgameTracker,
        chart::Vector{ComplexF64}, ℓ_coeffs::Vector{Float64},
    )
    for p in P
        for ζ in roots
            q = _track_u_root(eg, _u_start_point(p, ζ, chart, ℓ_coeffs))
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
        F₀::S1, G₀::S2, γ::ComplexF64, chart::Vector{ComplexF64},
        ℓ_coeffs::Vector{Float64}, projective::Val,
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
        nt::Int,
    )::Nothing where {S1 <: System, S2 <: System}
    nroots = length(roots)
    njobs = length(P) * nroots
    results = [ComplexF64[] for _ in 1:njobs]
    @tasks for k in 1:njobs
        @set ntasks = nt
        @local eg = _endgame_tracker(
            _u_homotopy(
                _clone_system_evaluator(F₀), _clone_system_evaluator(G₀),
                γ, chart, projective,
            ),
            tracker_options, endgame_options,
        )
        i, j = fldmod1(k, nroots)
        q = _track_u_root(eg, _u_start_point(P[i], roots[j], chart, ℓ_coeffs))
        q === nothing || (results[k] = q)
    end
    for r in results
        isempty(r) || push!(X, r)
    end
    return nothing
end

# Deform the start equation into `h`, moving from W's type-1 subspace to X's
# type-2. Returns the two sliced ambient endpoint systems and the degree `d`.
function _u_homotopy_systems(
        W::WitnessPoints, F::System, X::WitnessPoints, h, vars, u, ℓ,
    )
    d = _u_degree(h, vars)
    h0 = _u_start_equation(h, d, u, ℓ)
    eqs = polynomials(F)

    L = linear_subspace_u(W)      # start: u free
    K = linear_subspace(X)        # target: u = c

    F₀ = _sliced_system([eqs; h0], collect(vars), L)
    G₀ = _sliced_system([eqs; h], collect(vars), K)
    return F₀, G₀, d
end

# The u-homotopy. Projectively, both sliced endpoint systems are homogeneous and
# one row short of square, and the chart row completes them. The `Val` keeps each
# branch's homotopy type concrete instead of returning a union.
_u_homotopy(
    F₀::SystemEvaluator, G₀::SystemEvaluator, γ::ComplexF64,
    ::Vector{ComplexF64}, ::Val{false},
) = StraightLineHomotopy(F₀, G₀; γ = γ)

_u_homotopy(
    F₀::SystemEvaluator, G₀::SystemEvaluator, γ::ComplexF64,
    chart::Vector{ComplexF64}, ::Val{true},
) = on_affine_chart(StraightLineHomotopy(F₀, G₀; γ = γ), copy(chart))

# ── Membership within regeneration (structured subspace) ─────────────────────

# Whether each point of X lies on Y's variety. Builds a query subspace reusing
# X's linear equations (plus random rows), passing through each point, then
# moves Y's witness points there and checks proximity.
function _is_contained(
        X::WitnessPoints, Y::WitnessSet, F::System, rng::Random.MersenneTwister,
        projective::Bool;
        atol::Float64, rtol::Float64,
        tracker_options::TrackerOptions, endgame_options::EndgameOptions,
    )::BitVector
    LX = linear_subspace(X)
    LY = linear_subspace(Y)
    (isempty(points(Y)) || isempty(points(X))) && return falses(length(points(X)))

    m, n = size(F)
    cY = codim(LY)
    # Affinely the query rows reuse X's, which keeps `L_x` structured and well
    # conditioned. Projectively they must annihilate `x` instead, so they are
    # drawn fresh per point and projected off it.
    A = projective ? zeros(ComplexF64, 0, 0) : _membership_matrix(LX, LY, rng)
    Rs = projective ? [randn(rng, ComplexF64, cY, n) for _ in points(X)] :
        Matrix{ComplexF64}[]
    chart = projective ? randn(rng, ComplexF64, n) : ComplexF64[]
    x0 = LA.normalize!(randn(rng, ComplexF64, n))
    y0 = FSVec{ComplexF64}(zeros(ComplexF64, m))
    y = FSVec{ComplexF64}(zeros(ComplexF64, m))
    p_empty = FSVec{ComplexF64}(ComplexF64[])

    # Move Y's witness points with an IntrinsicSubspaceHomotopy, unconditionally
    # (one concrete type, no runtime dispatch). The extrinsic system `[F; A x − b]`
    # is singular at the start points whenever `dim(LY) <= codim(LY)`, so tracks
    # die with PATH_TERMINATED_INVALID_START and a whole component gets dropped.
    # The intrinsic form `x = A(t) v + a(t)` is well conditioned in every
    # dim/codim regime and tracks in only `dim(LY) <= n` coordinates. Projectively
    # the sliced system is square only on a chart, so the chart row joins it.
    hom_ev = projective ?
        SystemEvaluator(AffineChartSystem(F.evaluator, chart)) : F.evaluator
    Hom = IntrinsicSubspaceHomotopy(hom_ev, LY, LY; gamma = _random_gamma(rng))
    eg = _endgame_tracker(Hom, tracker_options, endgame_options)
    u_buf = FSVec{ComplexF64}(zeros(ComplexF64, size(Hom)[2]))
    amb_buf = FSVec{ComplexF64}(zeros(ComplexF64, n))
    q_chart = zeros(ComplexF64, n)

    out = falses(length(points(X)))
    for (idx, x) in enumerate(points(X))
        xc = ComplexF64.(x)
        evaluate!(y0, F.evaluator, FSVec{ComplexF64}(LA.norm(xc, Inf) .* x0), p_empty)
        evaluate!(y, F.evaluator, FSVec{ComplexF64}(xc), p_empty)
        LA.norm(y, Inf) > 1.0e-2 * LA.norm(y0, Inf) && continue

        if projective
            on_chart!(xc, chart)      # compare chart representatives
            Ax = _orthonormal_rows_through!(Rs[idx], xc)
            L_x = LinearSubspace(
                ExtrinsicDescription(Ax, zeros(ComplexF64, cY); orthonormal = true),
            )
        else
            L_x = LinearSubspace(
                ExtrinsicDescription(A, A * xc; orthonormal = true),
            )
        end
        set_subspaces!(Hom, LY, L_x)
        rad = max(atol, LA.norm(xc, Inf) * rtol)
        for q0 in points(Y)
            q = if projective
                copyto!(q_chart, q0)
                on_chart!(q_chart, chart)
                q_chart
            else
                ComplexF64.(q0)
            end
            intrinsic_coordinates!(u_buf, Hom, q, complex(1.0))
            track!(eg, u_buf)
            pr = PathResult(eg; path_number = 0, start_solution = copy(q))
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
function _membership_matrix(
        LX::LinearSubspace, LY::LinearSubspace, rng::Random.MersenneTwister,
    )::Matrix{ComplexF64}
    n = ambient_dim(LY)
    cX = codim(LX)
    cY = codim(LY)
    k = cY - cX
    A = zeros(ComplexF64, cY, n)
    AX = extrinsic(LX).A
    A[1, n] = one(ComplexF64)
    for i in 2:(k + 1), j in 1:n
        A[i, j] = randn(rng, ComplexF64)
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
        exec::AbstractExecutor, tracker_options::TrackerOptions = TrackerOptions(),
        show_progress::Bool = false,
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    )::MonodromyResult
    cp = convert(LinearSubspace{ComplexF64}, L)
    # `_monodromy_solve!` seeds its loops from `seed`; the solver's chart takes a
    # tagged stream off the same seed so the two do not share draws.
    MS = MonodromySolver(
        F, cp;
        options = opts, tracker_options = tracker_options,
        rng = _tagged_rng(seed, 0x0000_0001), start_solutions = X,
    )
    return _monodromy_solve!(MS, X, cp, seed, show_progress, exec)
end

function fill_up!(
        out, monodromy_options::MonodromyOptions, state::RegenerationState,
        show_monodromy_progress::Bool, exec::AbstractExecutor,
    )
    Fᵢ = state.Fᵢ
    for W in out
        if W !== nothing && dim(W) > 0 && degree(W) > 0
            opts = _regeneration_monodromy_options(monodromy_options, W)
            res = _monodromy_with_options(
                Fᵢ, W.R, linear_subspace(W), opts;
                exec = exec, tracker_options = state.tracker_options,
                show_progress = show_monodromy_progress,
                seed = rand(state.rng, UInt32),
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
    intersect(W::WitnessSet, H::WitnessSet, alg = Intersection(), exec = Threaded())

Intersect the witness set `W` with the witness set `H` of a single hypersurface,
returning the witness set(s) of `V(system(W)) ∩ V(system(H))` obtained by one
u-regeneration step. Returns a single `WitnessSet` when the result has one
dimension, otherwise a `Vector{WitnessSet}`.

    intersect(W::WitnessSet, f, alg = Intersection(), exec = Threaded())

Compute a witness set `H` for the hypersurface `f` and return `intersect(W, H)`.
"""
function Base.intersect(
        W::WitnessSet,
        H::WitnessSet,
        alg::Intersection = Intersection(),
        exec::AbstractExecutor = Threaded(),
    )
    show_monodromy_progress = alg.show_monodromy_progress
    tracker_options = _tracker_options(alg)
    endgame_options = _endgame_options(alg)
    monodromy_options = alg.monodromy
    atol = alg.atol
    rtol = alg.rtol
    rng = Random.MersenneTwister(_seed(alg))
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
    # Both inputs must agree: a projective step keeps the whole flag linear, and
    # mixing the two would slice one of the varieties in the wrong ambient space.
    W.projective == H.projective || throw(
        ArgumentError(
            "cannot intersect a projective witness set with an affine one; both " *
                "must be projective or both affine.",
        ),
    )
    projective = W.projective

    # W's and H's equations, both expressed in W's variables `vars`.
    eqs = _regeneration_equations(system(W))
    heqs = _rename_variables(
        _regeneration_equations(system(H)), variables(system(H)), vars,
    )
    h = heqs[1]
    ℓ_coeffs = projective ? randn(rng, length(vars)) : Float64[]
    ℓ = _linear_form([eqs; h], vars, ℓ_coeffs)

    # Flags in (n+1)-space; the u-value fixes the appended coordinate.
    flagW = get_flag(1:2, linear_subspace(W), rng)
    flagH = get_flag(1:1, linear_subspace(H), rng)
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
        projective, ℓ, ℓ_coeffs,
        tracker_options, endgame_options, rng,
    )

    intersect_with_hypersurface!(
        W₁, Hᵤ, W₂, intersect_state, exec; atol = atol, rtol = rtol,
    )
    fill_state = RegenerationState(
        [eqs; h], vars_u, u, length(eqs) + 1, 2,
        System([eqs; h]; parameters = empty(vars_u), variables = vars_u),
        projective, ℓ, ℓ_coeffs,
        tracker_options, endgame_options, rng,
    )
    Ws = W₂ === nothing ? WitnessPoints[W₁] : WitnessPoints[W₁, W₂]
    # The d-th-root tracking can reach the same solution more than once; dedupe
    # the start points so the monodromy fill-up is not seeded with duplicates.
    for Wi in Ws
        isempty(Wi.R) || (Wi.R = unique_points(Wi.R))
    end
    fill_up!(Ws, monodromy_options, fill_state, show_monodromy_progress, exec)

    G = System([eqs; h]; parameters = empty(vars), variables = vars)
    out = WitnessSet[]
    for Wi in Ws
        P, L = u_transform(Wi)
        push!(out, WitnessSet(G, L, P; projective = projective))
    end
    # Remove spurious witness points of a lower-dimensional set that actually
    # lie on a higher-dimensional component (they are junk from the u-homotopy).
    _remove_contained_points!(out, rng, exec; atol = atol, rtol = rtol)
    filter!(X -> degree(X) > 0, out)
    return length(out) == 1 ? first(out) : out
end

# For witness sets sorted by decreasing dimension, drop from each set the points
# that are contained in any higher-dimensional set (junk points).
function _remove_contained_points!(
        out::Vector{<:WitnessSet}, rng::Random.MersenneTwister,
        exec::AbstractExecutor; atol::Float64, rtol::Float64,
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
                    p, out[j],
                    Membership(; show_progress = false, seed = rand(rng, UInt32)),
                    exec; atol = atol, rtol = rtol,
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
        f::MP.AbstractPolynomialLike,
        alg::Intersection = Intersection(),
        exec::AbstractExecutor = Threaded(),
    )
    _check_front_end(system(W), false)
    rng = Random.MersenneTwister(_seed(alg))
    H = _hypersurface_witness_set(
        f, collect(variables(system(W))), rng, alg, exec, W.projective,
    )
    # A derived seed, so the regeneration step does not replay the stream that
    # produced H's slice.
    return intersect(W, H, _reseed(alg, rand(rng, UInt32)), exec)
end

function Base.intersect(
        W::WitnessSet,
        f::Expression,
        alg::Intersection = Intersection(),
        exec::AbstractExecutor = Threaded(),
    )
    _check_front_end(system(W), true)
    rng = Random.MersenneTwister(_seed(alg))
    H = _hypersurface_witness_set(
        f, _as_variables(collect(variables(system(W)))), rng, alg, exec, W.projective,
    )
    return intersect(W, H, _reseed(alg, rand(rng, UInt32)), exec)
end

# Witness set of `V(f)` in the ambient space of `vars`, which may hold variables
# `f` does not use. The slice matches the flag the regeneration step builds: an
# affine line, or a linear plane when the step runs projectively.
_hypersurface_witness(alg::Intersection, seed::UInt32)::Witness{TotalDegree} =
    Witness(
    _with_seed(alg.common, seed), nothing, nothing,
    TotalDegree(;
        tracker_options = _tracker_options(alg),
        endgame_options = _endgame_options(alg),
        seed = seed, show_progress = _show_progress(alg),
    ),
)

function _hypersurface_witness_set(
        f::MP.AbstractPolynomialLike, vars::Vector, rng::Random.MersenneTwister,
        alg::Intersection, exec::AbstractExecutor, projective::Bool,
    )::WitnessSet
    extra = setdiff(MP.effective_variables(f), vars)
    isempty(extra) || throw(
        ArgumentError(
            "the hypersurface must be given in the variables of the witness set, " *
                "but `$(f)` also uses $(join(extra, ", ")).",
        ),
    )
    return solve(
        System([f]; parameters = empty(vars), variables = vars),
        _hypersurface_slice(rng, length(vars), projective),
        _hypersurface_witness(alg, rand(rng, UInt32)),
        exec,
    )
end

_hypersurface_slice(rng::Random.MersenneTwister, n::Int, projective::Bool) =
    projective ? rand_subspace(rng, n; dim = 2, affine = false) :
    rand_subspace(rng, n; dim = 1)

# A rational `f` is solved through its numerator, with its poles dropped.
function _hypersurface_witness_set(
        f::Expression, vars::Vector{Expression}, rng::Random.MersenneTwister,
        alg::Intersection, exec::AbstractExecutor, projective::Bool,
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
    Wp = solve(
        G, _hypersurface_slice(rng, length(vars), projective),
        _hypersurface_witness(alg, rand(rng, UInt32)), exec,
    )
    R = Q === nothing ? solutions(Wp) : _drop_poles(Q, solutions(Wp))
    return WitnessSet(h, linear_subspace(Wp), R; projective = projective)
end

# The witness points of one component are a trace-tested loop with no permutations;
# every other option is the caller's. The target count allows a little slack in case
# a singular solution slips through.
_regeneration_monodromy_options(M::MonodromyOptions, W) = _with_fields(
    M,
    (
        permutations = false, trace_test = true,
        target_solutions_count = Int(floor(1.5 * degree(W))),
    ),
)
