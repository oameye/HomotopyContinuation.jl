## Slicing a system with a linear subspace, and the solve routes built on it.
#
# `V(F) ∩ L` is expressed in AMBIENT coordinates: the linear equations `A x − b`
# of `L` are appended to `F` rather than substituting intrinsic coordinates
# `F(A v + b)`. Solutions are therefore ambient points and the ordinary
# total-degree / polyhedral machinery applies unchanged.

# The full ambient space as a codim-0 subspace (`A` is `0 × n`): witness sets
# of zero-dimensional varieties slice with the whole space, so the sliced
# system is `F` itself (plus a chart row in the projective case).
full_subspace(n::Int)::LinearSubspace{ComplexF64} =
    LinearSubspace(zeros(ComplexF64, 0, n), ComplexF64[])

# Build the degree-1 polynomials `A[i,:]·vars - b[i]` describing the extrinsic
# subspace `L = {x | A x = b}`.
function linear_equations(L::LinearSubspace, vars)
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
function chart_equation(chart::AbstractVector, vars::AbstractVector)
    n = length(vars)
    return sum(chart[j] * vars[j] for j in 1:n) - 1
end

# Build the sliced system `[polys; A x − b]` in ambient coordinates, i.e. the
# zero set `V(polys) ∩ L`. For a projective (homogeneous) problem a chart
# equation `c·x − 1` is appended so the system is square.
function sliced_system(
        polys::AbstractVector, vars::AbstractVector, L::LinearSubspace,
        chart::AbstractVector, params::AbstractVector, compile::CompileMode.T,
    )::System
    lin = linear_equations(L, vars)
    eqs = vcat(collect(polys), lin)
    isempty(chart) || push!(eqs, chart_equation(chart, vars))
    return System(eqs; parameters = params, variables = vars, compile = compile)
end

"""
    slice(F::System, L::LinearSubspace; chart = ComplexF64[]) -> System

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
        F::System, L::LinearSubspace;
        chart::AbstractVector = ComplexF64[],
    )
    ambient_dim(L) == nvariables(F) || throw(
        ArgumentError(
            "The subspace lives in dimension $(ambient_dim(L)), but the system " *
                "has $(nvariables(F)) variables.",
        ),
    )
    return sliced_system(
        polynomials(F), collect(variables(F)), L, chart, collect(parameters(F)),
        F.compile_mode,
    )
end

# ── solve(F, L) ──────────────────────────────────────────────────────────────

# Every route that charts a projective system draws it here, so one guard covers all.
function random_affine_chart(
        rng::Random.MersenneTwister, F::CloneableSystem,
    )::Vector{ComplexF64}
    check_single_group(F)
    return randn(rng, ComplexF64, nvariables(F))
end

# Draw the affine chart for a projective problem. `seed` makes it reproducible.
function sliced_solve_setup(F::System, L::LinearSubspace, seed::UInt32)
    check_parameter_free(F, "`solve(F, L)`")
    chart = if is_linear(L) && is_homogeneous(F)
        random_affine_chart(Random.MersenneTwister(seed), F)
    else
        ComplexF64[]
    end
    return F, chart
end

rebuild_sliced(G::System, L::LinearSubspace, chart::Vector{ComplexF64})::System =
    isempty(chart) ? slice(G, L) : slice(G, L; chart = chart)

# The polynomial sliced system a solve route would track, with the projective
# chart row appended.
function sliced_solve_system(F::System, L::LinearSubspace, seed::UInt32)::System
    G, chart = sliced_solve_setup(F, L, seed)
    return rebuild_sliced(G, L, chart)
end

# Total degree over `[G; A x − b]`. When the sliced system is square the linear
# rows are appended by wrapping `G`'s evaluator (`SlicedSystem`), which skips CSE
# and tape construction entirely. An under- or overdetermined slice falls back to
# rebuilding the polynomial system, so the shape check and the square-up
# machinery (excess-solution checker) apply unchanged.
function init_sliced_total_degree(
        G::System, L::LinearSubspace, chart::Vector{ComplexF64},
        alg::TotalDegree, exec::E,
    )::SolveCache{E} where {E <: AbstractExecutor}
    # The square branch below reads `G.degrees` directly, so guard first: a
    # degree of -1 would size the start-solution array negatively.
    check_polynomial(G, "`TotalDegree`")
    m = size(G)[1]
    nrows = m + codim(L) + (isempty(chart) ? 0 : 1)
    nrows == nvariables(G) || return CommonSolve.init(
        rebuild_sliced(G, L, chart), alg, exec,
    )

    degs = [G.degrees; ones(Int, nrows - m)]
    subspace = convert(LinearSubspace{ComplexF64}, L)
    γ = random_gamma(algorithm_seed(alg))
    builder = SlicedStraightLineBuilder(
        degs, G, subspace, chart, γ, algorithm_tracker_options(alg), algorithm_endgame_options(alg),
    )
    return solve_cache(
        exec, builder, total_degree_start_solutions(degs), algorithm_seed(alg),
        ExcessCheckers(), algorithm_show_progress(alg), early_stop_callback(alg),
    )
end

function CommonSolve.init(
        F::System, L::LinearSubspace, alg::TotalDegree,
        exec::E = Threaded(),
    )::SolveCache{E} where {E <: AbstractExecutor}
    G, chart = sliced_solve_setup(F, L, algorithm_seed(alg))
    return init_sliced_total_degree(G, L, chart, alg, exec)
end

function CommonSolve.init(
        F::System, L::LinearSubspace, alg::Polyhedral,
        exec::E = Threaded(),
    )::PolyhedralSolveCache{E} where {E <: AbstractExecutor}
    return CommonSolve.init(sliced_solve_system(F, L, algorithm_seed(alg)), alg, exec)
end

"""
    solve(F::System, L::LinearSubspace, alg = TotalDegree(), exec = Threaded())

Solve `V(F) ∩ L` for the (affine) linear subspace `L`. The returned solutions
are ambient points, i.e. in the coordinates of `F`.

`F` must be parameter-free; for a parametric system fix the values first with
[`fix_parameters`](@ref).

# Example
```julia
@polyvar x y
F = System([x^2 + y^2 - 5])
L = rand_subspace(2; codim = 1)
result = solve(F, L)
```
"""
function solve(
        F::System, L::LinearSubspace, alg::TotalDegree = TotalDegree(),
        exec::E = Threaded(),
    )::Result where {E <: AbstractExecutor}
    cache::PathSolveCache = CommonSolve.init(F, L, alg, exec)
    return CommonSolve.solve!(cache)
end

function solve(
        F::System, L::LinearSubspace, alg::Polyhedral,
        exec::E = Threaded(),
    )::Result where {E <: AbstractExecutor}
    cache::PathSolveCache = CommonSolve.init(F, L, alg, exec)
    return CommonSolve.solve!(cache)
end

function solve(F::System, L::LinearSubspace, exec::E)::Result where {E <: AbstractExecutor}
    return solve(F, L, TotalDegree(), exec)
end
