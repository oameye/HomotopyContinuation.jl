## Monodromy solver.

@enumx MonodromyCode::Int8 begin
    IN_PROGRESS
    SUCCESS
    HEURISTIC_STOP
    TIMEOUT
    TERMINATED_CALLBACK
    INVALID_STARTVALUE
    INTERRUPTED
end

@enumx ReuseLoops::Int8 begin
    ALL
    RANDOM
    NONE
end

@enumx DuplicateCheck::Int8 begin
    HEURISTIC
    CERTIFIED
end

# The verdict on a solution offered to an accumulator of certified-distinct
# solutions. Declared here because it is the return code of the certification
# package's `add_solution!` and of the hook below, which core has to read.
@enumx AddSolutionCode::Int8 begin
    CERTIFIED_DISTINCT
    DUPLICATE
    NOT_CERTIFIED
end

@noinline _certification_required() = throw(
    ArgumentError(
        "`duplicate_check = DuplicateCheck.CERTIFIED` needs Krawczyk certification: run " *
            "`using HomotopyContinuationCertification`.",
    ),
)

"""
    AbstractCertifiedSolutions

Supertype of the accumulator of certified-distinct solutions that
`duplicate_check = DuplicateCheck.CERTIFIED` files monodromy endpoints into.
`HomotopyContinuationCertification` provides the implementation.
"""
abstract type AbstractCertifiedSolutions end

"""
    AbstractCertifiedCandidate

Supertype of a monodromy endpoint that has been certified but not yet filed into an
[`AbstractCertifiedSolutions`](@ref).
"""
abstract type AbstractCertifiedCandidate end

struct NoCandidate <: AbstractCertifiedCandidate end
function monodromy_certified_solutions(
        ::SystemLike, ::Union{Nothing, Vector{ComplexF64}}, ::Int, ::Bool,
    )::AbstractCertifiedSolutions
    return _certification_required()
end


function monodromy_certify_candidate(
        ::AbstractCertifiedSolutions, ::Vector{ComplexF64}, ::Int,
    )::AbstractCertifiedCandidate
    return _certification_required()
end

function monodromy_file_certified!(
        ::AbstractCertifiedSolutions, ::AbstractCertifiedCandidate, ::Int,
    )::Tuple{AddSolutionCode.T, Int, Union{Nothing, CertifiedEndpoint}}
    return _certification_required()
end

# One certification cache per task index, sized for `ntasks` before a threaded solve
# hands them out.
function monodromy_size_caches!(::AbstractCertifiedSolutions, ::Int)
    return _certification_required()
end

# Drop every stored solution, keeping the caches. Declared with the hooks above for
# the same reason: the certified route has to infer in core alone.
Base.empty!(::AbstractCertifiedSolutions) = _certification_required()

always_false(args...) = false

"""
    independent_normal(rng, p::AbstractVector)

Sample a vector where each entry is drawn independently from the complex
normal distribution by calling `randn(rng, ComplexF64)`.

    independent_normal(rng, L::LinearSubspace)

Creates a random linear subspace by calling [`rand_subspace`](@ref).

Usable as the `parameter_sampler` of [`Monodromy`](@ref), whose contract
is `sampler(rng, p)`.
"""
independent_normal(rng::Random.AbstractRNG, p::AbstractVector)::Vector{ComplexF64} =
    randn(rng, ComplexF64, length(p))
independent_normal(
    rng::Random.AbstractRNG, L::LinearSubspace,
)::LinearSubspace{ComplexF64} = convert(
    LinearSubspace{ComplexF64}, rand_subspace(rng, ambient_dim(L); dim = dim(L)),
)

"""
    weighted_normal(rng, p::AbstractVector)

Sample a vector `q` where each entry `q[i]` is drawn from the complex normal
distribution with variance `|p[i]|^2`.

    weighted_normal(rng, L::LinearSubspace)

Sample a linear subspace `A x = a` where the entries of `A` and `a` are drawn
from the complex normal distribution with variances `|B[i,j]|^2` and `|b[i]|^2`,
where `L = {B x = b}`. Unlike [`independent_normal`](@ref), this preserves the
zero structure of `L` (entries where `B`/`b` vanish stay zero), which is
essential for the structured flag subspaces used in [`Regeneration`](@ref) and
[`Decomposition`](@ref).

Usable as the `parameter_sampler` of [`Monodromy`](@ref), whose contract
is `sampler(rng, p)`.
"""
weighted_normal(rng::Random.AbstractRNG, p::AbstractVector)::Vector{ComplexF64} =
    randn(rng, ComplexF64, length(p)) .* abs.(p)
function weighted_normal(
        rng::Random.AbstractRNG, L::LinearSubspace,
    )::LinearSubspace{ComplexF64}
    E = extrinsic(L)
    B, b = E.A, E.b
    A = randn(rng, ComplexF64, size(B)...) .* abs.(B)
    a = randn(rng, ComplexF64, length(b)) .* abs.(b)
    return convert(LinearSubspace{ComplexF64}, LinearSubspace(A, a))
end

#####################
# Monodromy Options #
#####################

"""
    MonodromyOptions(; options...)

Options for [`Monodromy`](@ref). `group_actions` accepts a `Function`,
`Tuple`, `AbstractVector` or [`GroupActions`](@ref); `reuse_loops` takes a
`ReuseLoops` enum value and `duplicate_check` a `DuplicateCheck` one.
"""
struct MonodromyOptions{D, GA <: Union{Nothing, GroupActions}, CB, PS}
    check_startsolutions::Bool
    group_actions::GA
    # Callback and sampler are user closures on the cold path; the type
    # parameters keep the struct concrete.
    loop_finished_callback::CB
    parameter_sampler::PS
    equivalence_classes::Bool
    # stopping heuristics
    trace_test::Bool
    trace_test_tol::Float64
    # `typemax(Int)` is the no-target sentinel: it never equals a real solution
    # count and never compares `>=` true.
    target_solutions_count::Int
    timeout::Float64
    min_solutions::Int
    max_loops_no_progress::Int
    reuse_loops::ReuseLoops.T
    permutations::Bool
    # deduplication policy
    duplicate_check::DuplicateCheck.T
    certification_max_precision::Int
    certification_refine_solution::Bool
    # unique points options
    distance::D
    triangle_inequality::Bool
    unique_points_atol::Float64
    # `NaN` is not a missing option: the default is `uniqueness_rtol(res)`, which
    # needs the endpoint and so cannot be resolved here.
    unique_points_rtol::Float64
    single_loop_per_start_solution::Bool
end

function MonodromyOptions(;
        check_startsolutions::Bool = true,
        group_action = nothing,
        group_actions = group_action === nothing ? nothing : GroupActions(group_action),
        loop_finished_callback = always_false,
        parameter_sampler = independent_normal,
        equivalence_classes::Bool = group_actions !== nothing,
        trace_test::Bool = true,
        trace_test_tol::Float64 = 1.0e-6,
        target_solutions_count::Int = typemax(Int),
        timeout::Real = Inf,
        min_solutions::Int = 0,
        max_loops_no_progress::Int = 5,
        reuse_loops::ReuseLoops.T = ReuseLoops.ALL,
        permutations::Bool = false,
        duplicate_check::DuplicateCheck.T = DuplicateCheck.HEURISTIC,
        certification_max_precision::Int = 256,
        certification_refine_solution::Bool = true,
        distance = InfNorm(),
        triangle_inequality::Bool = satisfies_triangle_inequality(distance),
        unique_points_atol::Float64 = 1.0e-14,
        unique_points_rtol::Float64 = NaN,
        single_loop_per_start_solution::Bool = false,
    )
    if group_actions isa Function || group_actions isa Tuple ||
            group_actions isa AbstractVector
        group_actions = GroupActions(group_actions)
    end
    # Equivalence classes only make sense with group actions.
    group_actions === nothing && (equivalence_classes = false)
    return MonodromyOptions(
        check_startsolutions,
        group_actions,
        loop_finished_callback,
        parameter_sampler,
        equivalence_classes,
        trace_test,
        trace_test_tol,
        target_solutions_count,
        Float64(timeout),
        min_solutions,
        max_loops_no_progress,
        reuse_loops,
        permutations,
        duplicate_check,
        certification_max_precision,
        certification_refine_solution,
        distance,
        triangle_inequality,
        unique_points_atol,
        unique_points_rtol,
        single_loop_per_start_solution,
    )
end

# A loop returns to the base parameters, where the solutions are regular, so this
# route runs no endgame. Fixed rather than exposed: there is nothing to tune.
const _MONODROMY_ENDGAME = EndgameOptions(; endgame_start = 0.0)

"""
    Monodromy([options]; variables, parameters, dim, codim, coords, catch_interrupt, warning, options...)

Find solutions of `F(x; p)` by tracking loops in the parameter space of `F`.

Pass start data positionally: `solve(F, sols, p, Monodromy())` bases the loops at
the parameters `p`, and `solve(F, sols, L, Monodromy())` intersects with a
[`LinearSubspace`](@ref). With no start data, `solve(F, Monodromy())` computes a
start pair itself, which needs the parameters to occur linearly in `F`.

`dim` or `codim`, the expected (co)dimension of a component of `V(F)`, selects
the subspace route instead: `solve(F, Monodromy(; dim = d))` intersects a
parameter-free `F` with a subspace of the complementary dimension. Which route
runs is fixed by whether one of them was given, not by what `F` turns out to be,
so passing either for a parameterized `F` is an error rather than silently
ignored.

`variables` and `parameters` split the symbols of a polynomial `F`, exactly as
they do in [`System`](@ref); they are ignored when `F` is already a `System`.

Accepts every [`MonodromyOptions`](@ref) keyword, or a pre-built `options` object.
There is no `endgame_options`: this route runs no endgame.
"""
# `SUBSPACE` says which route `solve(F, alg)` takes when given no start data.
struct Monodromy{
        MO <: MonodromyOptions, V <: AbstractVector, P <: AbstractVector, SUBSPACE,
    } <: AbstractAlgorithm
    common::CommonOptions
    options::MO
    # Empty means "take them from the system".
    variables::V
    parameters::P
    # `-1` marks an unset dimension, as in `rand_subspace`.
    dim::Int
    codim::Int
    coords::SubspaceCoords.T
    catch_interrupt::Bool
    warning::Bool
end

# `dim`/`codim` are the subspace route's only input, so whether one was given is
# the route.
_subspace_route(::Nothing, ::Nothing)::Val{false} = Val(false)
_subspace_route(::Int, ::Nothing)::Val{true} = Val(true)
_subspace_route(::Nothing, ::Int)::Val{true} = Val(true)
_subspace_route(::Int, ::Int)::Val{true} = Val(true)

_route_dim(::Nothing)::Int = -1
_route_dim(d::Int)::Int = d

_monodromy_algorithm(
    ::Val{S}, common::CommonOptions, options::MO, variables::V, parameters::P,
    dim::Int, codim::Int, coords::SubspaceCoords.T, catch_interrupt::Bool,
    warning::Bool,
) where {S, MO <: MonodromyOptions, V <: AbstractVector, P <: AbstractVector} =
    Monodromy{MO, V, P, S}(
    common, options, variables, parameters, dim, codim, coords,
    catch_interrupt, warning,
)

# The one place monodromy's keyword surface is declared.
function Monodromy(;
        variables::AbstractVector = Expression[],
        parameters::AbstractVector = Expression[],
        dim::Union{Nothing, Int} = nothing,
        codim::Union{Nothing, Int} = nothing,
        coords::SubspaceCoords.T = SubspaceCoords.AUTO,
        catch_interrupt::Bool = true,
        warning::Bool = true,
        tracker_options::TrackerOptions = TrackerOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        show_progress::Bool = true,
        check_startsolutions::Bool = true,
        group_action = nothing,
        group_actions = group_action === nothing ? nothing : GroupActions(group_action),
        loop_finished_callback = always_false,
        parameter_sampler = independent_normal,
        equivalence_classes::Bool = group_actions !== nothing,
        trace_test::Bool = true,
        trace_test_tol::Float64 = 1.0e-6,
        target_solutions_count::Int = typemax(Int),
        timeout::Real = Inf,
        min_solutions::Int = 0,
        max_loops_no_progress::Int = 5,
        reuse_loops::ReuseLoops.T = ReuseLoops.ALL,
        permutations::Bool = false,
        duplicate_check::DuplicateCheck.T = DuplicateCheck.HEURISTIC,
        certification_max_precision::Int = 256,
        certification_refine_solution::Bool = true,
        distance = InfNorm(),
        triangle_inequality::Bool = satisfies_triangle_inequality(distance),
        unique_points_atol::Float64 = 1.0e-14,
        unique_points_rtol::Float64 = NaN,
        single_loop_per_start_solution::Bool = false,
    )
    opts = MonodromyOptions(;
        check_startsolutions,
        group_actions,
        loop_finished_callback,
        parameter_sampler,
        equivalence_classes,
        trace_test,
        trace_test_tol,
        target_solutions_count,
        timeout,
        min_solutions,
        max_loops_no_progress,
        reuse_loops,
        permutations,
        duplicate_check,
        certification_max_precision,
        certification_refine_solution,
        distance,
        triangle_inequality,
        unique_points_atol,
        unique_points_rtol,
        single_loop_per_start_solution,
    )
    return _monodromy_algorithm(
        _subspace_route(dim, codim),
        CommonOptions(tracker_options, _MONODROMY_ENDGAME, seed, show_progress),
        opts, variables, parameters, _route_dim(dim), _route_dim(codim), coords,
        catch_interrupt, warning,
    )
end

# A pre-built options object. The remaining keywords configure the algorithm
# rather than the monodromy run, so none of them reach `MonodromyOptions`.
function Monodromy(
        options::MonodromyOptions;
        variables::AbstractVector = Expression[],
        parameters::AbstractVector = Expression[],
        dim::Union{Nothing, Int} = nothing,
        codim::Union{Nothing, Int} = nothing,
        coords::SubspaceCoords.T = SubspaceCoords.AUTO,
        catch_interrupt::Bool = true,
        warning::Bool = true,
        tracker_options::TrackerOptions = TrackerOptions(),
        seed::UInt32 = rand(Random.RandomDevice(), UInt32),
        show_progress::Bool = true,
    )
    return _monodromy_algorithm(
        _subspace_route(dim, codim),
        CommonOptions(tracker_options, _MONODROMY_ENDGAME, seed, show_progress),
        options, variables, parameters, _route_dim(dim), _route_dim(codim), coords,
        catch_interrupt, warning,
    )
end

# The polynomial input forms need the variable/parameter split, which the shared
# `_as_system` cannot know. Unannotated on purpose: an annotation would assert the
# UnionAll and lose the concrete `System` parameters.
_monodromy_system(F::SystemLike, ::Monodromy) = F
_monodromy_system(F::AbstractVector{<:MP.AbstractPolynomialLike}, alg::Monodromy) =
    System(
    F;
    variables = isempty(alg.variables) ? nothing : alg.variables,
    parameters = isempty(alg.parameters) ? nothing : alg.parameters,
)
_monodromy_system(f::MP.AbstractPolynomialLike, alg::Monodromy) =
    _monodromy_system([f], alg)
