# Trace test (del Campo & Rodriguez 2017; Leykin, Rodriguez & Sottile 2018).
# Cold diagnostic path.

# Augmented system `[F(x, p + λv); (Σᵢ aᵢxᵢ - 1)λ + t]` used by the trace test.
# DynamicPolynomials variables are identity distinct, so the fresh variables
# below cannot collide with user variables of the same name.
function _build_verification_system(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        x::AbstractVector, p::AbstractVector, n::Int, m::Int,
    )
    @polyvar t v[1:m] a[1:n] λ
    # `@polyvar v[1:m]` hands back a `Vector` with no element type on Julia 1.11,
    # which loses it for everything built from the array variables.
    VT = typeof(λ)
    vs = Vector{VT}(v)
    as = Vector{VT}(a)
    # `MP.subs` does not say what it returns, and `System` reads its first
    # parameter off the element type it is given, so the equations go into a
    # vector of one declared polynomial type, filled in place: a comprehension
    # would take its element type from `subs` and lose it.
    PT = MP.polynomial_type(eltype(polys), Float64)
    eqs = Vector{PT}(undef, length(polys) + 1)
    for (i, f) in enumerate(polys)
        eqs[i] = MP.subs(f, p => p .+ λ .* vs)
    end
    eqs[end] = (sum(as .* x) - 1) * λ + t
    return System(eqs; variables = [x; λ], parameters = [t; p; vs; as])
end

# `Expression` variables are keyed by name, so the fresh ones are renamed until
# they no longer clash.
function _build_verification_system(
        polys::AbstractVector{Expression},
        x::AbstractVector, p::AbstractVector, n::Int, m::Int,
    )::System{Expression, Expression}
    taken = Expression[x; p]
    fresh(name)::Expression = (w = unique_variable(name, taken, Expression[]); push!(taken, w); w)

    t = fresh(:t)
    λ = fresh(:λ)
    v = Expression[fresh(Symbol("v", map_subscripts(i))) for i in 1:m]
    a = Expression[fresh(Symbol("a", map_subscripts(i))) for i in 1:n]

    exprs = Expression[subs(f, p => p .+ λ .* v) for f in polys]
    push!(exprs, (sum(a .* x) - 1) * λ + t)
    return System(
        exprs; variables = Expression[x; λ], parameters = Expression[t; p; v; a],
    )
end

# Parameter sampler for the auxiliary monodromy computation: keeps the first
# parameter (the trace variable t) fixed at zero. The LinearSubspace method
# exists only to keep the sampler total over both MonodromyLoop branches; the
# auxiliary system always has vector parameters, so it is unreachable.
function _zero_first_parameter_sampler(
        rng::Random.AbstractRNG, pp::AbstractVector,
    )::Vector{ComplexF64}
    return [0.0 + 0.0im; randn(rng, ComplexF64, length(pp) - 1)]
end

function _zero_first_parameter_sampler(::Random.AbstractRNG, ::LinearSubspace)
    throw(ArgumentError("the completeness verification sampler only supports vector parameters"))
end

"""
    verify_solution_completeness(F::System, R::MonodromyResult, alg = Monodromy(), exec = Threaded(); options...)
    verify_solution_completeness(F::System, sols, p, alg = Monodromy(), exec = Threaded(); options...)

Verify that a monodromy computation found all solutions of the polynomial
system `F(x; p) = 0` on the fiber over the parameters `p` using the trace
test. The correctness of this verification procedure requires that the
parametrized family is irreducible and that the given solutions are correct.

The algorithm constructs the augmented system
`[F(x, p + λv); (Σᵢ aᵢxᵢ - 1)λ + t]` in the variables `[x; λ]` with
parameters `[t; p; v; a]`, computes additional witnesses on the `λ ≠ 0`
component via monodromy (with the first parameter `t` fixed to zero), and
then performs two parameter homotopies moving `t` along a random complex
direction. The combined witness set is complete if and only if the traces of
the three witness sets are colinear; the deviation from colinearity is
measured by the relative third singular value of the trace matrix and
compared against `trace_tol`.

Returns `true` (complete), `false` (incomplete), or `nothing` when a
parameter homotopy lost solutions so no verdict is possible.

`alg` supplies the seed, the tracker options, the progress setting and
`max_loops_no_progress` for the auxiliary monodromy computation.

## Options

* `trace_tol = 1e-14`: tolerance for the trace colinearity test.
* `endgame_options`: forwarded to the two trace parameter homotopies.
"""
function verify_solution_completeness(
        F::System,
        mres::MonodromyResult,
        alg::Monodromy = Monodromy(),
        exec::E = Threaded();
        trace_tol::Float64 = 1.0e-14,
        endgame_options::EndgameOptions = EndgameOptions(),
    )::Completeness.T where {E <: AbstractExecutor}
    return verify_solution_completeness(
        F, solutions(mres), Vector(parameters(mres)), alg, exec;
        trace_tol = trace_tol, endgame_options = endgame_options,
    )
end

"""
    Completeness

Verdict of [`verify_solution_completeness`](@ref).

- `COMPLETE`: the given solutions are all of them.
- `INCOMPLETE`: a further solution exists.
- `INCONCLUSIVE`: a solution was lost during the parameter homotopy, so the
  check could not decide.
"""
@enumx Completeness::Int8 begin
    COMPLETE
    INCOMPLETE
    INCONCLUSIVE
end

function verify_solution_completeness(
        F::System,
        sols::AbstractVector{<:AbstractVector},
        q::AbstractVector,
        alg::Monodromy = Monodromy(),
        exec::E = Threaded();
        trace_tol::Float64 = 1.0e-14,
        endgame_options::EndgameOptions = EndgameOptions(),
    )::Completeness.T where {E <: AbstractExecutor}
    show_progress = _show_progress(alg)
    seed = _seed(alg)
    tracker_options = _tracker_options(alg)
    n = nvariables(F)
    m = nparameters(F)

    verify_system = _build_verification_system(
        F.polys, collect(F.variables), collect(F.parameters), n, m,
    )

    # Monodromy computation for the additional witnesses: use verify_system
    # but enforce t = 0 and start with λ ≠ 0 so we stay on a different
    # irreducible component.
    if show_progress
        @info "Compute additional witnesses for completeness check..."
    end

    rng = Random.MersenneTwister(seed)
    q0 = convert(Vector{ComplexF64}, q)

    # Start solutions: sample random parameters qq to set v = qq - q. More
    # than one start solution is good; construct up to n by a parameter
    # homotopy to qq, then compute an `a` such that those solutions lie on
    # the linear space a⋅x - 1 = 0.
    qq = randn(rng, ComplexF64, m)
    qq_res = solve(
        F, sols[1:min(n, length(sols))], q0, qq,
        Continuation(;
            seed = rand(rng, UInt32), tracker_options = tracker_options,
            endgame_options = endgame_options, show_progress = show_progress,
        ),
        Serial(),
    )
    a0 = reduce(vcat, transpose.(solutions(qq_res))) \ ones(nsolutions(qq_res))
    Y = map(s -> [s; 1], solutions(qq_res))
    base_params = [q0; qq .- q0; a0]

    additional_mres = solve(
        verify_system, Y, [0.0; base_params],
        Monodromy(;
            parameter_sampler = _zero_first_parameter_sampler,
            seed = rand(rng, UInt32), show_progress = show_progress,
            tracker_options = tracker_options,
            max_loops_no_progress = alg.options.max_loops_no_progress,
        ),
        exec,
    )
    additional_sols = solutions(additional_mres)
    if show_progress
        @info additional_mres
        @info "Computed $(length(additional_sols)) additional witnesses"
        @info "Compute trace using two parameter homotopies..."
    end

    # Parameter homotopies for the trace: move t along a random direction γ.
    S = [map(s -> [s; 0], sols); additional_sols]
    γ = randn(rng, ComplexF64)
    res1 = solve(
        verify_system, S, [0.0; base_params], [0.5 * γ; base_params],
        Continuation(;
            seed = rand(rng, UInt32), tracker_options = tracker_options,
            endgame_options = endgame_options, show_progress = show_progress,
        ),
        Serial(),
    )
    S1 = solutions(res1)
    if length(S1) != length(S)
        if show_progress
            @warn "Lost solution during parameter homotopy. Abort."
        end
        return Completeness.INCONCLUSIVE
    end

    res2 = solve(
        verify_system, S1, [0.5 * γ; base_params], [1.0 * γ; base_params],
        Continuation(;
            seed = rand(rng, UInt32), tracker_options = tracker_options,
            endgame_options = endgame_options, show_progress = show_progress,
        ),
        Serial(),
    )
    S2 = solutions(res2)
    if length(S2) != length(S)
        if show_progress
            @warn "Lost solution during parameter homotopy. Abort."
        end
        return Completeness.INCONCLUSIVE
    end

    T = sum(S)
    T1 = sum(S1)
    T2 = sum(S2)

    M = [T T1 T2; 1 1 1]
    singvals = LA.svdvals(M)
    trace_norm = singvals[3] / singvals[1]

    if show_progress
        @info "Norm of trace: $trace_norm"
    end

    return trace_norm < trace_tol ? Completeness.COMPLETE : Completeness.INCOMPLETE
end
