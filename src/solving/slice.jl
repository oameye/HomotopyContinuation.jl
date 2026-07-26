## Slicing a system with a linear subspace, and the solve routes built on it.
#
# `V(F) ∩ L` is expressed in AMBIENT coordinates: the linear equations `A x − b`
# of `L` are appended to `F` rather than substituting intrinsic coordinates
# `F(A v + b)`. Solutions are therefore ambient points and the ordinary
# total-degree / polyhedral machinery applies unchanged.

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
                "`target_parameters`.",
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
    return System(_substitute(polynomials(F), params, pc); variables = vars, compile = M)
end

# Substitution runs through whichever front-end built the system.
_substitute(
    polys::FSVec{<:MP.AbstractPolynomialLike}, params::Vector, pc::Vector{ComplexF64},
) = [MP.polynomial(MP.subs(f, params => pc)) for f in polys]

_substitute(
    polys::FSVec{Expression}, params::Vector{Expression}, pc::Vector{ComplexF64},
)::Vector{Expression} = [subs(f, params => pc) for f in polys]

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
        params::AbstractVector = empty(vars),
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::System
    lin = _linear_equations(L, vars)
    eqs = vcat(collect(polys), lin)
    chart !== nothing && push!(eqs, _chart_equation(chart, vars))
    return System(eqs; parameters = params, variables = vars, compile = compile)
end

"""
    slice(F::System, L::LinearSubspace; chart = nothing) -> System

Return the system whose zero set is `V(F) ∩ L`, namely `F` with the linear
equations `A x − b` of `L` appended. The variables, parameters and compile mode
of `F` are preserved, so the result is in the same ambient coordinates as `F`.

Pass `chart = c` to additionally append the affine chart equation `c·x − 1`;
this is required for a homogeneous `F`, which is positive-dimensional in ambient
coordinates.

# Example
```julia
@polyvar x y z
F = System([x^2 + y^2 + z^2 - 1])
L = rand_subspace(3; codim = 2)
G = slice(F, L)   # square system with the two linear equations appended
```
"""
function slice(
        F::System{P, V, M}, L::LinearSubspace;
        chart::Union{Nothing, AbstractVector} = nothing,
    )::System where {P, V, M}
    ambient_dim(L) == nvariables(F) || throw(
        ArgumentError(
            "The subspace lives in dimension $(ambient_dim(L)), but the system " *
                "has $(nvariables(F)) variables.",
        ),
    )
    return _sliced_system(
        polynomials(F), collect(variables(F)), L;
        chart = chart, params = collect(parameters(F)), compile = M,
    )
end

# ── solve(F, L) ──────────────────────────────────────────────────────────────

# Fix the parameters (if any) and draw the affine chart for a projective
# problem. `seed` makes the chart reproducible.
function _sliced_solve_setup(
        F::System, L::LinearSubspace, seed::UInt32,
        target_parameters::Union{Nothing, AbstractVector{<:Number}},
    )
    G = if nparameters(F) > 0 || target_parameters !== nothing
        _fix_parameters(F, target_parameters)
    else
        F
    end
    chart = if is_linear(L) && is_homogeneous(G)
        randn(Random.MersenneTwister(seed), ComplexF64, nvariables(G))
    else
        ComplexF64[]
    end
    return G, chart
end

_rebuild_sliced(G::System, L::LinearSubspace, chart::Vector{ComplexF64})::System =
    isempty(chart) ? slice(G, L) : slice(G, L; chart = chart)

# The polynomial sliced system a solve route would track, parameters fixed and
# the projective chart row appended.
function _sliced_solve_system(
        F::System, L::LinearSubspace, seed::UInt32,
        target_parameters::Union{Nothing, AbstractVector{<:Number}},
    )::System
    G, chart = _sliced_solve_setup(F, L, seed, target_parameters)
    return _rebuild_sliced(G, L, chart)
end

# Total degree over `[G; A x − b]`. When the sliced system is square the linear
# rows are appended by wrapping `G`'s evaluator (`SlicedSystem`), which skips CSE
# and tape construction entirely. An under- or overdetermined slice falls back to
# rebuilding the polynomial system, so the shape check and the square-up
# machinery (excess-solution checker) apply unchanged.
function _init_sliced_total_degree(
        G::System, L::LinearSubspace, chart::Vector{ComplexF64},
        alg::TotalDegree, exec::AbstractExecutor, show_progress::Bool,
    )
    # The square branch below reads `G.degrees` directly, so guard first: a
    # degree of -1 would size the start-solution array negatively.
    _check_polynomial(G, "`TotalDegree`")
    m = size(G)[1]
    nrows = m + codim(L) + (isempty(chart) ? 0 : 1)
    nrows == nvariables(G) || return CommonSolve.init(
        _rebuild_sliced(G, L, chart), alg, exec; show_progress = show_progress,
    )

    degrees = [G.degrees; ones(Int, nrows - m)]
    subspace = convert(LinearSubspace{ComplexF64}, L)
    γ = _random_gamma(alg.seed)
    builder = SlicedStraightLineBuilder(
        degrees, G, subspace, chart, γ, alg.tracker_options, alg.endgame_options,
    )
    return _total_degree_solve_cache(
        exec, builder, _sliced_evaluator(G.evaluator, subspace, chart), degrees,
        alg.seed, nothing, show_progress, alg.tracker_options, alg.endgame_options, γ,
    )
end

function CommonSolve.init(
        F::System, L::LinearSubspace, alg::TotalDegree,
        exec::AbstractExecutor = Threaded();
        target_parameters::Union{Nothing, AbstractVector{<:Number}} = nothing,
        show_progress::Bool = true,
    )
    G, chart = _sliced_solve_setup(F, L, alg.seed, target_parameters)
    return _init_sliced_total_degree(G, L, chart, alg, exec, show_progress)
end

function CommonSolve.init(
        F::System, L::LinearSubspace, alg::Polyhedral,
        exec::AbstractExecutor = Threaded();
        target_parameters::Union{Nothing, AbstractVector{<:Number}} = nothing,
        show_progress::Bool = true,
    )::PolyhedralSolveCache
    return CommonSolve.init(
        _sliced_solve_system(F, L, alg.seed, target_parameters), alg, exec;
        show_progress = show_progress,
    )
end

"""
    solve(F::System, L::LinearSubspace, alg = TotalDegree(), exec = Threaded())

Solve `V(F) ∩ L` for the (affine) linear subspace `L`. The returned solutions
are ambient points, i.e. in the coordinates of `F`.

For a parametric `F` pass the parameter values via `target_parameters`; they are
substituted into `F` before slicing.

# Example
```julia
@polyvar x y
F = System([x^2 + y^2 - 5])
L = rand_subspace(2; codim = 1)
result = solve(F, L)
```
"""
function solve(
        F::System, L::LinearSubspace,
        alg::TotalDegree = TotalDegree(),
        exec::AbstractExecutor = Threaded();
        target_parameters::Union{Nothing, AbstractVector{<:Number}} = nothing,
        show_progress::Bool = true,
    )::Result
    return CommonSolve.solve!(
        CommonSolve.init(
            F, L, alg, exec;
            target_parameters = target_parameters, show_progress = show_progress,
        ),
    )
end

function solve(
        F::System, L::LinearSubspace, alg::Polyhedral,
        exec::AbstractExecutor = Threaded();
        target_parameters::Union{Nothing, AbstractVector{<:Number}} = nothing,
        show_progress::Bool = true,
    )::Result
    return CommonSolve.solve!(
        CommonSolve.init(
            F, L, alg, exec;
            target_parameters = target_parameters, show_progress = show_progress,
        ),
    )
end

function solve(
        F::System, L::LinearSubspace, exec::AbstractExecutor;
        target_parameters::Union{Nothing, AbstractVector{<:Number}} = nothing,
        show_progress::Bool = true,
    )::Result
    return solve(
        F, L, TotalDegree(), exec;
        target_parameters = target_parameters, show_progress = show_progress,
    )
end
