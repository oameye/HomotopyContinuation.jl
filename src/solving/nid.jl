## Numerical irreducible decomposition (NID).
#
# `Decomposition` splits each witness superset of the `Regeneration` stage into
# irreducible components using monodromy permutations plus the trace test, and
# collects them in a `NumericalIrreducibleDecomposition`.

# A tighter singular-accuracy threshold than `_REGENERATION_ENDGAME`, so the
# regeneration u-homotopy rejects points on higher-dimensional components as
# singular rather than over-collecting them.
const _DECOMPOSITION_ENDGAME = EndgameOptions(;
    max_endgame_steps = 100, max_endgame_extended_steps = 100, sing_accuracy = 1.0e-10,
)

_decompose_stage_monodromy() =
    MonodromyOptions(; trace_test_tol = 1.0e-10, parameter_sampler = weighted_normal)

"""
    Decomposition(; regeneration, monodromy, max_iters, warning, options...)

Compute the numerical irreducible decomposition of a variety: run the
equation-by-equation stage to obtain witness supersets, then split each into
irreducible components.

Applied to a [`WitnessSet`](@ref) it runs only the splitting stage.

`regeneration` configures the first stage and `monodromy` the second; the two use
different trace-test tolerances, so they are separate. `max_iters` bounds the
splitting iterations. The remaining options configure the first stage, and a
`regeneration` given explicitly carries its own; `endgame_options` applies there
too, since the splitting stage runs no endgame. `show_monodromy_progress` draws
the bar of every monodromy run of the splitting stage.
"""
struct Decomposition{R <: Regeneration, MO <: MonodromyOptions} <: AbstractAlgorithm
    common::CommonOptions
    regeneration::R
    monodromy::MO
    show_monodromy_progress::Bool
    max_iters::Int
    warning::Bool
    atol::Float64
    rtol::Float64
end

function Decomposition(;
        regeneration::Union{Nothing, Regeneration} = nothing,
        monodromy::MonodromyOptions = _decompose_stage_monodromy(),
        max_iters::Int = 50,
        warning::Bool = true,
        sorted::EquationSorting.T = EquationSorting.BY_DEGREE,
        max_codim::Union{Nothing, Int} = nothing,
        atol::Float64 = 1.0e-14,
        rtol::Float64 = sqrt(eps()),
        tracker_options::TrackerOptions = TrackerOptions(),
        endgame_options::EndgameOptions = _DECOMPOSITION_ENDGAME,
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        show_progress::Bool = true,
        show_monodromy_progress::Bool = false,
    )
    regen = if regeneration !== nothing
        regeneration
    else
        Regeneration(;
            sorted = sorted, max_codim = max_codim, atol = atol, rtol = rtol,
            tracker_options = tracker_options,
            endgame_options = endgame_options,
            seed = seed, show_progress = show_progress,
            show_monodromy_progress = show_monodromy_progress,
        )
    end
    return Decomposition(
        CommonOptions(tracker_options, endgame_options, seed, show_progress),
        regen, monodromy, show_monodromy_progress, max_iters, warning, atol, rtol,
    )
end

_reseed(alg::Decomposition, seed::UInt32) = Decomposition(
    _with_seed(alg.common, seed), alg.regeneration, alg.monodromy,
    alg.show_monodromy_progress, alg.max_iters, alg.warning, alg.atol, alg.rtol,
)

# ── decompose ────────────────────────────────────────────────────────────────

"""
    solve(W::WitnessSet, alg::Decomposition = Decomposition(), exec = Threaded())
    solve(Ws::Vector{<:WitnessSet}, alg::Decomposition, exec = Threaded())

Decompose a witness set (or witness superset) into irreducible components. See
[`Decomposition`](@ref) for the options.
"""
function solve(
        Ws::Vector{WT},
        alg::Decomposition,
        exec::AbstractExecutor = Threaded(),
    )::Vector{WT} where {WT <: WitnessSet}
    monodromy_options = alg.monodromy
    max_iters = alg.max_iters
    warning = alg.warning
    show_monodromy_progress = alg.show_monodromy_progress
    tracker_options = _tracker_options(alg)
    atol = alg.atol
    rtol = alg.rtol
    rng = Random.MersenneTwister(_seed(alg))

    Ws = sort(Ws; by = dim, rev = true)
    options = _decompose_monodromy_options(monodromy_options, atol, rtol)
    out = WT[]
    isempty(Ws) && return out

    for witness in Ws
        if degree(witness) > 0
            dec = _decompose_with_monodromy(
                witness, options, max_iters, warning, rng, exec,
                show_monodromy_progress, tracker_options, atol, rtol,
            )
            append!(out, dec)
        end
    end
    return out
end

solve(
    W::WitnessSet, alg::Decomposition = Decomposition(),
    exec::AbstractExecutor = Threaded(),
) = solve([W], alg, exec)

solve(
    F::PolynomialInput, alg::Decomposition, exec::AbstractExecutor = Threaded(),
) = solve(_as_system(F), alg, exec)

# Persistent point identities across repeated monodromy calls. The index owns a
# separate `UniquePoints` with the solver's distance and triangle-inequality
# policy, but never a symmetry quotient. Tolerances are fixed by the outer
# `Decomposition`, so every loop and the persistent union-find share one notion
# of witness-point identity.
struct DecompositionPointIdentity{UP <: UniquePoints}
    unique_points::UP
    master::Vector{Vector{ComplexF64}}
    parent::Vector{Int}
    done::BitVector
end

function DecompositionPointIdentity(MS::MonodromySolver, n::Int)
    solver_points = MS.unique_points
    identity_points = UniquePoints(
        n;
        distance = solver_points.tree.distance,
        group_actions = solver_points.group_actions,
        triangle_inequality = solver_points.tree.triangle_inequality,
    )
    return DecompositionPointIdentity(
        identity_points, Vector{Vector{ComplexF64}}(), Int[], BitVector(),
    )
end

function _identity_root!(identity::DecompositionPointIdentity, i::Int)::Int
    parent = identity.parent
    root = i
    while parent[root] != root
        root = parent[root]
    end
    while parent[i] != root
        parent[i], i = root, parent[i]
    end
    return root
end

function _union_identities!(
        identity::DecompositionPointIdentity, a::Int, b::Int,
    )::Nothing
    root_a = _identity_root!(identity, a)
    root_b = _identity_root!(identity, b)
    root_a != root_b && (identity.parent[root_a] = root_b)
    return nothing
end

function _point_identity!(
        identity::DecompositionPointIdentity,
        point::Vector{ComplexF64},
        atol::Float64,
        rtol::Float64,
    )::Int
    candidate = length(identity.master) + 1
    id, is_new = add!(
        identity.unique_points, point, candidate;
        atol = atol, rtol = rtol,
    )
    if is_new
        push!(identity.master, point)
        push!(identity.parent, candidate)
        push!(identity.done, false)
    end
    return id
end

function _point_identity!(
        identity::DecompositionPointIdentity,
        path_result::PathResult,
        atol::Float64,
        rtol::Float64,
    )::Int
    return _point_identity!(identity, solution(path_result), atol, rtol)
end

function _absorb_monodromy_result!(
        identity::DecompositionPointIdentity,
        result::MonodromyResult,
        atol::Float64,
        rtol::Float64,
    )::Nothing
    path_results = results(result)
    indices = Int[_point_identity!(identity, r, atol, rtol) for r in path_results]
    Π = permutations(result)
    for column in eachcol(Π), (i, j) in enumerate(column)
        (j == 0 || i > length(indices) || j > length(indices)) && continue
        _union_identities!(identity, indices[i], indices[j])
    end
    return nothing
end

function _decompose_with_monodromy(
        W::WT, options::MonodromyOptions, max_iters::Int,
        warning::Bool, rng::Random.MersenneTwister, exec::AbstractExecutor,
        show_monodromy_progress::Bool, tracker_options::TrackerOptions,
        atol::Float64, rtol::Float64,
    )::Vector{WT} where {WT <: WitnessSet}
    P = points(W)
    L = linear_subspace(W)
    G = system(W)
    n = ambient_dim(L)
    decomposition = WT[]

    # zero-dimensional: every point is its own irreducible component
    if dim(L) >= n
        for p in P
            push!(decomposition, WitnessSet(G, L, [p]; irreducibility = Irreducibility.IRREDUCIBLE))
        end
        return decomposition
    end

    cp = convert(LinearSubspace{ComplexF64}, L)
    MS = MonodromySolver(
        G, cp; options = options, tracker_options = tracker_options, rng = rng,
        start_solutions = P,
    )

    res = _monodromy_solve!(
        MS, P, cp, rand(rng, UInt32), show_monodromy_progress, exec,
    )

    if warning && (something(trace(res), Inf) > options.trace_test_tol)
        if trace_complete(MS)
            @warn "Trying to decompose a non-complete set of witness points for " *
                "codimension $(dim(L)) (trace test failed). Output contains all " *
                "components for which the trace test succeeded."
        else
            @warn "The trace test for codimension $(dim(L)) is inconclusive: " *
                "$(MS.trace_dropped) of $(MS.trace_dropped + MS.trace_paths) paths " *
                "failed to track around the trace loop, so the trace is summed over " *
                "fewer points than the witness set holds. Output contains all " *
                "components for which the trace test succeeded."
        end
    end

    # Accumulate orbit connectivity across iterations: a single loop fragments a
    # degree-d component into partial orbits (often singletons), only merging
    # permutations across loops assembles the full orbit. Since `_monodromy_solve!`
    # can grow the set and drop start points (shifting indices), connectivity is
    # tracked by POINT IDENTITY over a growing master list plus a union-find.
    identity = DecompositionPointIdentity(MS, n)
    _absorb_monodromy_result!(identity, res, atol, rtol)
    master = identity.master
    done = identity.done
    d = length(master)                            # running total degree

    iter = 0
    inconclusive = 0
    while any(!, done)
        iter += 1
        iter > max_iters && break

        # Each call needs a fresh seed off `rng`; reusing one rebuilds the
        # identical loops and the iteration never converges.
        if iter > 1
            active = [master[k] for k in eachindex(master) if !done[k]]
            n_before = length(master)
            res = _monodromy_solve!(
                MS, active, cp, rand(rng, UInt32), show_monodromy_progress, exec,
            )
            _absorb_monodromy_result!(identity, res, atol, rtol)
            d += length(master) - n_before      # new points grow the total degree
        end

        # Group the not-yet-emitted master points into orbits (union-find roots).
        orbit_of = Dict{Int, Vector{Int}}()
        for k in eachindex(master)
            done[k] && continue
            push!(get!(orbit_of, _identity_root!(identity, k), Int[]), k)
        end

        for orbit in values(orbit_of)
            # An earlier trace run in this pass may already have merged and
            # certified a snapshot orbit. Only process identities still live.
            live_orbit = [k for k in orbit if !done[k]]
            isempty(live_orbit) && continue

            P_orbit = master[live_orbit]
            n_before = length(master)
            res_orbit = _monodromy_solve!(
                MS, P_orbit, cp, rand(rng, UInt32), show_monodromy_progress, exec,
            )

            # Trace-test monodromy is itself allowed to discover points and orbit
            # connections. Fold those into the persistent identity before deciding
            # what the certified orbit actually contains.
            _absorb_monodromy_result!(identity, res_orbit, atol, rtol)
            d += length(master) - n_before
            root = _identity_root!(identity, first(live_orbit))
            grown_orbit = [
                k for k in eachindex(master)
                if !done[k] && _identity_root!(identity, k) == root
            ]

            if something(trace(res_orbit), Inf) >= options.trace_test_tol
                trace_complete(MS) || (inconclusive += 1)
                continue
            end

            # Singleton gate: a point of a degree > 1 component often passes the
            # trace test alone before accumulation connects it to its siblings.
            (length(grown_orbit) > 1 || iter >= 5) || continue

            push!(
                decomposition,
                WitnessSet(
                    G, L, copy(master[grown_orbit]);
                    irreducibility = Irreducibility.IRREDUCIBLE,
                ),
            )
            for k in grown_orbit
                done[k] = true
            end
        end
        # done?
        if sum(degree, decomposition; init = 0) == d
            break
        end
    end

    unresolved = Dict{Int, Vector{Int}}()
    for k in eachindex(master)
        done[k] && continue
        push!(get!(unresolved, _identity_root!(identity, k), Int[]), k)
    end
    unresolved_degree = sum(length, values(unresolved); init = 0)
    for orbit in values(unresolved)
        push!(
            decomposition,
            WitnessSet(
                G, L, copy(master[orbit]); irreducibility = Irreducibility.UNKNOWN,
            ),
        )
    end

    if warning && unresolved_degree > 0
        detail = inconclusive > 0 ?
            "; $inconclusive orbit trace test(s) were inconclusive because paths failed to track" :
            ""
        @warn "Decomposition stopped after $max_iters iteration(s) with " *
            "$unresolved_degree of $d witness point(s) still unresolved$detail. " *
            "They are returned with irreducibility UNKNOWN."
    end

    return decomposition
end

# Decomposition needs the permutations of a trace-tested single loop; every other
# option is the caller's.
_decompose_monodromy_options(
    M::MonodromyOptions, atol::Float64, rtol::Float64,
) = _with_fields(
    M,
    (
        permutations = true, trace_test = true,
        single_loop_per_start_solution = true,
        # Monodromy is an internal orbit-discovery engine here. Decomposition
        # owns witness cardinality, so neither endpoint-adaptive tolerances nor
        # a symmetry quotient may redefine point identity underneath it.
        unique_points_atol = atol,
        unique_points_rtol = rtol,
        equivalence_classes = false,
    ),
)

# ── NumericalIrreducibleDecomposition ────────────────────────────────────────

"""
    NumericalIrreducibleDecomposition

Stores the irreducible components of `V(F)` as witness sets grouped by
dimension. Construct with [`Decomposition`](@ref).
"""
struct NumericalIrreducibleDecomposition{W <: WitnessSet} <: AbstractResult
    Witness_Sets::Dict{Int, Vector{W}}
    seed::UInt32
end

function NumericalIrreducibleDecomposition(
        Ws::Vector{WT}, seed::UInt32,
    ) where {WT <: WitnessSet}
    D = Dict{Int, Vector{WT}}()
    for witness in Ws
        push!(get!(D, dim(witness), WT[]), witness)
    end
    return NumericalIrreducibleDecomposition(D, seed)
end

"""
    witness_sets(N::NumericalIrreducibleDecomposition; dims = nothing, irreducibility = nothing)

Return the witness sets stored in `N` as a `Dict` keyed by dimension. By default
this includes unresolved witness sets: incomplete numerical work is never hidden.
Set `irreducibility` to an [`Irreducibility`](@ref) value to filter by status.
"""
function witness_sets(
        N::NumericalIrreducibleDecomposition;
        dims::Union{Vector{Int}, Nothing} = nothing,
        irreducibility::Union{Irreducibility.T, Nothing} = nothing,
    )
    D = N.Witness_Sets
    selected_dims = dims === nothing ? keys(D) : dims
    out = empty(D)
    for k in selected_dims
        haskey(D, k) || continue
        Ws = irreducibility === nothing ? D[k] :
            filter(W -> is_irreducible(W) == irreducibility, D[k])
        isempty(Ws) || (out[k] = Ws)
    end
    return out
end
witness_sets(
    N::NumericalIrreducibleDecomposition, dim::Int;
    irreducibility::Union{Irreducibility.T, Nothing} = nothing,
) = witness_sets(N; dims = [dim], irreducibility = irreducibility)
seed(N::NumericalIrreducibleDecomposition) = N.seed

"""Return only witness sets proven irreducible by the decomposition trace tests."""
irreducible_components(
    N::NumericalIrreducibleDecomposition; dims::Union{Vector{Int}, Nothing} = nothing,
) = witness_sets(N; dims = dims, irreducibility = Irreducibility.IRREDUCIBLE)
irreducible_components(N::NumericalIrreducibleDecomposition, dim::Int) =
    irreducible_components(N; dims = [dim])

"""Return witness sets whose irreducibility has not yet been decided."""
unresolved_witness_sets(
    N::NumericalIrreducibleDecomposition; dims::Union{Vector{Int}, Nothing} = nothing,
) = witness_sets(N; dims = dims, irreducibility = Irreducibility.UNKNOWN)
unresolved_witness_sets(N::NumericalIrreducibleDecomposition, dim::Int) =
    unresolved_witness_sets(N; dims = [dim])

"""Return the total witness degree whose irreducibility is still unresolved."""
function unresolved_degree(
        N::NumericalIrreducibleDecomposition;
        dims::Union{Vector{Int}, Nothing} = nothing,
    )::Int
    D = unresolved_witness_sets(N; dims = dims)
    return sum((degree(W) for Ws in values(D) for W in Ws); init = 0)
end
unresolved_degree(N::NumericalIrreducibleDecomposition, dim::Int)::Int =
    unresolved_degree(N; dims = [dim])

"""
    ncomponents(N::NumericalIrreducibleDecomposition; dims = nothing)

Return the number of *proven irreducible* components. Unresolved witness sets are
available through [`unresolved_witness_sets`](@ref) and are not counted as
components until the trace test proves irreducibility.
"""
function ncomponents(
        N::NumericalIrreducibleDecomposition;
        dims::Union{Vector{Int}, Nothing} = nothing,
    )::Int
    D = irreducible_components(N; dims = dims)
    return sum(length, values(D); init = 0)
end
ncomponents(N::NumericalIrreducibleDecomposition, dim::Int) = ncomponents(N; dims = [dim])
n_components(N::NumericalIrreducibleDecomposition; dims = nothing) = ncomponents(N; dims = dims)
n_components(N::NumericalIrreducibleDecomposition, dim::Int) = ncomponents(N; dims = [dim])

"""
    degrees(N::NumericalIrreducibleDecomposition; dims = nothing)

Return a `Dict` mapping each dimension to the degrees of its proven irreducible
components. Unresolved witness degree is reported separately by
[`unresolved_degree`](@ref).
"""
function degrees(
        N::NumericalIrreducibleDecomposition;
        dims::Union{Vector{Int}, Nothing} = nothing,
    )
    D = irreducible_components(N; dims = dims)
    return Dict(k => [degree(W) for W in Ws] for (k, Ws) in D)
end

function _max_dim(N::NumericalIrreducibleDecomposition)::Int
    ks = keys(irreducible_components(N))
    return isempty(ks) ? -1 : maximum(ks)
end

function Base.show(io::IO, N::NumericalIrreducibleDecomposition)
    D = irreducible_components(N)
    U = unresolved_witness_sets(N)
    total = sum(length, values(D); init = 0)
    unresolved_sets = sum(length, values(U); init = 0)
    unresolved_deg = unresolved_degree(N)
    s = total == 1 ? "component" : "components"
    header = "Numerical irreducible decomposition with $total proven $s"
    println(io, header)
    println(io, "="^length(header))
    if unresolved_sets > 0
        println(io, "• $unresolved_sets unresolved witness set(s), total degree $unresolved_deg.")
    end
    mdim = _max_dim(N)
    if mdim >= 0
        for d in mdim:-1:0
            if haskey(D, d)
                n = length(D[d])
                n > 0 && println(io, "• $n proven component(s) of dimension $d.")
            end
        end
        println(io, "\n degree table of proven components:")
        _degree_table(io, N)
    end
    return
end

# Hand-rolled unicode degree table (avoids a PrettyTables dependency).
function _degree_table(io::IO, N::NumericalIrreducibleDecomposition)
    D = irreducible_components(N)
    ks = sort(collect(keys(D)); rev = true)

    rows = Vector{Tuple{String, String}}()
    for k in ks
        comps = sort([degree(W) for W in D[k]]; rev = true)
        deg_str = if length(comps) == 1
            string(first(comps))
        elseif length(comps) <= 10
            string("(", join(comps, ", "), ")")
        else
            string("(", join(comps[1:10], ", "), ", ...)")
        end
        push!(rows, (string(k), deg_str))
    end

    h1, h2 = "dimension", "degrees of components"
    w1 = maximum(length, [h1; [r[1] for r in rows]])
    w2 = maximum(length, [h2; [r[2] for r in rows]])

    center(s, w) = begin
        pad = w - length(s)
        l = pad ÷ 2
        string(" "^l, s, " "^(pad - l))
    end
    top = string("╭─", "─"^w1, "─┬─", "─"^w2, "─╮")
    mid = string("├─", "─"^w1, "─┼─", "─"^w2, "─┤")
    bot = string("╰─", "─"^w1, "─┴─", "─"^w2, "─╯")

    println(io, top)
    println(io, "│ ", center(h1, w1), " │ ", center(h2, w2), " │")
    println(io, mid)
    for (a, b) in rows
        println(io, "│ ", center(a, w1), " │ ", center(b, w2), " │")
    end
    print(io, bot)
    return
end

# ── Top-level entry points ───────────────────────────────────────────────────

"""
    solve(F, alg::Decomposition, exec = Threaded())

Compute the numerical irreducible decomposition of `V(F)`, returned as a
[`NumericalIrreducibleDecomposition`](@ref): run the [`Regeneration`](@ref) stage
to obtain witness supersets, then split each into irreducible components.

`F` may be a [`System`](@ref), a single polynomial or a vector of polynomials. See
[`Decomposition`](@ref) for the options. Both stages draw from `seed`, so the same
`seed` gives the same decomposition regardless of the state of the global random
number generator.
"""
function solve(
        F::S,
        alg::Decomposition,
        exec::AbstractExecutor = Threaded(),
    )::NumericalIrreducibleDecomposition{WitnessSet{S}} where {S <: System}
    seed = _seed(alg)
    rng = Random.MersenneTwister(seed)

    # Each stage gets a seed derived from this one, so a single `seed` reproduces
    # the whole decomposition.
    Ws = solve(F, _reseed(alg.regeneration, rand(rng, UInt32)), exec)
    dec = solve(Ws, _reseed(alg, rand(rng, UInt32)), exec)
    return NumericalIrreducibleDecomposition(dec, seed)
end
