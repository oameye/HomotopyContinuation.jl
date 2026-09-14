module SemialgebraicSetsExt

using CommonSolve: CommonSolve
using MultivariatePolynomials: MultivariatePolynomials
using SemialgebraicSets: SemialgebraicSets

using HomotopyContinuation: HomotopyContinuation
const HCN = HomotopyContinuation
const MP = MultivariatePolynomials

function HCN.System(
        V::SemialgebraicSets.AbstractAlgebraicSet;
        variables = nothing,
        variable_groups = nothing,
        compile::HCN.CompileMode.T = HCN.CompileMode.INTERPRETED,
    )::HCN.System
    eqs = SemialgebraicSets.equalities(V)
    isempty(eqs) && throw(
        ArgumentError("the set has no equalities, so there is no system to build"),
    )
    return HCN.System(
        eqs;
        variables = variables === nothing ? MP.variables(eqs) : variables,
        variable_groups = variable_groups,
        compile = compile,
    )
end

const SetAlgorithm = Union{HCN.TotalDegree, HCN.Polyhedral}

struct HCSolver{A <: SetAlgorithm, E <: HCN.AbstractExecutor} <:
    SemialgebraicSets.AbstractAlgebraicSolver
    algorithm::A
    executor::E
    real_tol::Float64
    compile::HCN.CompileMode.T
end

HCN.SemialgebraicSetsHCSolver(;
    algorithm::SetAlgorithm = HCN.TotalDegree(; show_progress = false),
    executor::HCN.AbstractExecutor = HCN.Threaded(),
    real_tol::Float64 = HCN.DEFAULT_REAL_TOL,
    compile::HCN.CompileMode.T = HCN.CompileMode.INTERPRETED,
) = HCSolver(algorithm, executor, real_tol, compile)

SemialgebraicSets.default_gröbner_basis_algorithm(::Any, ::HCSolver) =
    SemialgebraicSets.NoAlgorithm()
SemialgebraicSets.promote_for(::Type{<:Number}, ::Type{<:HCSolver}) = Float64

function Base.show(io::IO, solver::HCSolver)
    print(io, "SemialgebraicSetsHCSolver(; ")
    print(io, "algorithm = ", nameof(typeof(solver.algorithm)))
    print(io, ", executor = ", nameof(typeof(solver.executor)))
    print(io, ", real_tol = ", solver.real_tol)
    print(io, ", compile = ", solver.compile, ")")
    return
end

function _solve_set(
        V::SemialgebraicSets.AbstractAlgebraicSet, solver::HCSolver,
    )::Union{Nothing, Vector{Vector{Float64}}}
    eqs = SemialgebraicSets.equalities(V)
    isempty(eqs) && return nothing
    vars = MP.variables(eqs)
    length(eqs) >= length(vars) || return nothing
    F = HCN.System(eqs; variables = vars, compile = solver.compile)
    result = HCN.solve(F, solver.algorithm, solver.executor)
    return HCN.real_solutions(result; tol = solver.real_tol)
end

CommonSolve.solve(
    V::SemialgebraicSets.AbstractAlgebraicSet, solver::HCSolver,
)::Union{Nothing, Vector{Vector{Float64}}} = _solve_set(V, solver)

HCN.solve(
    V::SemialgebraicSets.AbstractAlgebraicSet,
    alg::SetAlgorithm = HCN.TotalDegree(),
    exec::HCN.AbstractExecutor = HCN.Threaded(),
)::HCN.Result = HCN.solve(HCN.System(V), alg, exec)

function HCN.real_solutions(
        V::SemialgebraicSets.AbstractAlgebraicSet,
        solver::HCSolver = HCN.SemialgebraicSetsHCSolver(),
    )::Vector{Vector{Float64}}
    points = _solve_set(V, solver)
    points === nothing && throw(
        ArgumentError(
            "the set has fewer equations than unknowns, so its solutions are " *
                "not finitely many",
        ),
    )
    return points
end

end # module
