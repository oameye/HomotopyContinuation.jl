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
        regen, monodromy, show_monodromy_progress, max_iters, warning,
    )
end

_reseed(alg::Decomposition, seed::UInt32) = Decomposition(
    _with_seed(alg.common, seed), alg.regeneration, alg.monodromy,
    alg.show_monodromy_progress, alg.max_iters, alg.warning,
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
    rng = Random.MersenneTwister(_seed(alg))

    Ws = sort(Ws; by = dim, rev = true)
    options = _decompose_monodromy_options(monodromy_options)
    out = WT[]
    isempty(Ws) && return out

    for witness in Ws
        if degree(witness) > 0
            dec = _decompose_with_monodromy(
                witness, options, max_iters, warning, rng, exec,
                show_monodromy_progress, tracker_options,
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
# separate `UniquePoints` with exactly the solver's distance, group-action, and
# triangle-inequality policy; each insertion uses the same endpoint-dependent
# tolerances as `MonodromySolver.add!`.
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
        options::MonodromyOptions,
    )::Int
    atol, rtol = _dedup_tolerances(options, path_result)
    return _point_identity!(identity, solution(path_result), atol, rtol)
end

function _absorb_monodromy_result!(
        identity::DecompositionPointIdentity,
        result::MonodromyResult,
        options::MonodromyOptions,
    )::Nothing
    path_results = results(result)
    indices = Int[_point_identity!(identity, r, options) for r in path_results]
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
    _absorb_monodromy_result!(identity, res, options)
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
            _absorb_monodromy_result!(identity, res, options)
            d += length(master) - n_before      # new points grow the total degree
        end

        # Group the not-yet-emitted master points into orbits (union-find roots).
        orbit_of = Dict{Int, Vector{Int}}()
        for k in eachindex(master)
            done[k] && continue
            push!(get!(orbit_of, _identity_root!(identity, k), Int[]), k)
        end

        for orbit in values(orbit_of)
            P_orbit = master[orbit]
            res_orbit = _monodromy_solve!(
                MS, P_orbit, cp, rand(rng, UInt32), show_monodromy_progress, exec,
            )
            if something(trace(res_orbit), Inf) >= options.trace_test_tol
                # A dropped column is why the trace failed here; the orbit is
                # retried on the next iteration and only silently lost if the
                # iterations run out.
                trace_complete(MS) || (inconclusive += 1)
                continue
            end

            # Singleton gate: a point of a degree > 1 component often passes the
            # trace test alone before accumulation connects it to its siblings,
            # so trust a singleton as a genuine degree-1 component only at
            # `iter >= 5`.
            (length(orbit) > 1 || iter >= 5) || continue

            push!(
                decomposition,
                WitnessSet(G, L, copy(P_orbit); irreducibility = Irreducibility.IRREDUCIBLE),
            )
            for k in orbit
                done[k] = true
            end
        end

        # done?
        if sum(degree, decomposition; init = 0) == d
            break
        end
    end

    if warning && inconclusive > 0 && sum(degree, decomposition; init = 0) < d
        @warn "Codimension $(dim(L)) is missing " *
            "$(d - sum(degree, decomposition; init = 0)) of its $d witness points, " *
            "and $inconclusive orbit trace tests were inconclusive because paths " *
            "failed to track around the trace loop. The components below are the " *
            "ones the trace test could confirm."
    end

    return decomposition
end

function _decompose_monodromy_options(M::MonodromyOptions)
    return MonodromyOptions(;
        permutations = true,
        trace_test = true,
        single_loop_per_start_solution = true,
        check_startsolutions = M.check_startsolutions,
        group_actions = M.group_actions,
        loop_finished_callback = M.loop_finished_callback,
        parameter_sampler = M.parameter_sampler,
        equivalence_classes = M.equivalence_classes,
        trace_test_tol = M.trace_test_tol,
        target_solutions_count = M.target_solutions_count,
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
    witness_sets(N::NumericalIrreducibleDecomposition; dims = nothing)

Return the witness sets in `N` as a `Dict` keyed by dimension. `dims` restricts
to the given dimensions.
"""
function witness_sets(
        N::NumericalIrreducibleDecomposition;
        dims::Union{Vector{Int}, Nothing} = nothing,
    )
    D = N.Witness_Sets
    dims === nothing && return D
    out = empty(D)
    for k in dims
        haskey(D, k) && (out[k] = D[k])
    end
    return out
end
witness_sets(N::NumericalIrreducibleDecomposition, dim::Int) = witness_sets(N; dims = [dim])
seed(N::NumericalIrreducibleDecomposition) = N.seed

"""
    ncomponents(N::NumericalIrreducibleDecomposition; dims = nothing)

Return the total number of irreducible components (optionally restricted to
`dims`).
"""
function ncomponents(
        N::NumericalIrreducibleDecomposition;
        dims::Union{Vector{Int}, Nothing} = nothing,
    )::Int
    D = N.Witness_Sets
    isempty(D) && return 0
    if dims === nothing
        return sum(length(Ws) for Ws in values(D))
    end
    return sum(haskey(D, d) ? length(D[d]) : 0 for d in dims; init = 0)
end
ncomponents(N::NumericalIrreducibleDecomposition, dim::Int) = ncomponents(N; dims = [dim])
n_components(N::NumericalIrreducibleDecomposition; dims = nothing) = ncomponents(N; dims = dims)
n_components(N::NumericalIrreducibleDecomposition, dim::Int) = ncomponents(N; dims = [dim])

"""
    degrees(N::NumericalIrreducibleDecomposition; dims = nothing)

Return a `Dict` mapping each dimension to the degrees of its components.
"""
function degrees(
        N::NumericalIrreducibleDecomposition;
        dims::Union{Vector{Int}, Nothing} = nothing,
    )
    D = N.Witness_Sets
    out = Dict{Int, Vector{Int}}()
    ks = dims === nothing ? collect(keys(D)) : dims
    for k in ks
        haskey(D, k) && (out[k] = [degree(W) for W in D[k]])
    end
    return out
end

function _max_dim(N::NumericalIrreducibleDecomposition)::Int
    ks = keys(N.Witness_Sets)
    return isempty(ks) ? -1 : maximum(ks)
end

function Base.show(io::IO, N::NumericalIrreducibleDecomposition)
    D = N.Witness_Sets
    total = isempty(D) ? 0 : sum(length(Ws) for Ws in values(D))
    s = total == 1 ? "component" : "components"
    header = "Numerical irreducible decomposition with $total $s"
    println(io, header)
    println(io, "="^length(header))
    mdim = _max_dim(N)
    mdim < 0 && return
    for d in mdim:-1:0
        if haskey(D, d)
            ℓ = length(D[d])
            ℓ > 0 && println(io, "• $ℓ component(s) of dimension $d.")
        end
    end
    println(io, "\n degree table of components:")
    _degree_table(io, N)
    return
end

# Hand-rolled unicode degree table (avoids a PrettyTables dependency).
function _degree_table(io::IO, N::NumericalIrreducibleDecomposition)
    D = N.Witness_Sets
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
