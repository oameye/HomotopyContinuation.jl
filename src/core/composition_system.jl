## Composition `G ∘ F`: the system `x ↦ G(F(x; p); p)`, evaluated by wrapping
## the two evaluators. Both stages see the same parameter vector, so a stage
## that declares no parameters ignores it.

struct ComposedSystem <: AbstractSystem
    g::SystemEvaluator
    f::SystemEvaluator
    # undoes the inner stage's equation normalization, which changes `F` as a map
    f_scale::FSVec{Float64}
    # Scratch buffers
    x0::FSVec{ComplexF64}
    p0::FSVec{ComplexF64}
    f_u::FSVec{ComplexF64}
    f_ū::FSVec{ComplexDF64}
    f_U::FSMat{ComplexF64}
    g_U::FSMat{ComplexF64}
    # Taylor coefficients of `F` along the incoming series, orders 0 to 3
    tu::TaylorVector{4, ComplexF64}
end

function ComposedSystem(
        g::SystemEvaluator, f::SystemEvaluator, f_scale::AbstractVector{Float64},
    )::ComposedSystem
    k, ng = size(g)
    m, n = size(f)
    ng == m || throw(
        ArgumentError(
            "cannot compose `G ∘ F`: `G` has $ng variables but `F` has $m equations",
        ),
    )
    length(f_scale) == m || throw(
        ArgumentError(
            "the inner scale has length $(length(f_scale)), expected $m",
        ),
    )
    np = max(nparameters(g), nparameters(f))
    return ComposedSystem(
        g, f,
        FSVec{Float64}(collect(Float64, f_scale)),
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexF64}(zeros(ComplexF64, np)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, m)),
        FSMat{ComplexF64}(zeros(ComplexF64, m, n)),
        FSMat{ComplexF64}(zeros(ComplexF64, k, m)),
        TaylorVector{4, ComplexF64}(m),
    )
end

clone_system(C::ComposedSystem)::ComposedSystem = ComposedSystem(
    clone_system_evaluator(C.g), clone_system_evaluator(C.f), C.f_scale,
)

Base.size(C::ComposedSystem)::Tuple{Int, Int} = (size(C.g)[1], size(C.f)[2])
nparameters(C::ComposedSystem)::Int = max(nparameters(C.g), nparameters(C.f))

composition_evaluator(
    g::SystemEvaluator, f::SystemEvaluator, f_scale::AbstractVector{Float64},
)::SystemEvaluator = SystemEvaluator(ComposedSystem(g, f, f_scale))

@inline function unscale_values!(
        v::FSVec{T}, scale::FSVec{Float64},
    )::Nothing where {T <: Complex}
    @inbounds for i in eachindex(v)
        v[i] *= scale[i]
    end
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, C::ComposedSystem,
        x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate!(C.f_u, C.f, x, p)
    unscale_values!(C.f_u, C.f_scale)
    evaluate!(u, C.g, C.f_u, p)
    return nothing
end

# The inner value is `G`'s argument, so an extended-precision residual needs it
# unrounded.
function evaluate!(
        u::Union{FSVec{ComplexF64}, FSVec{ComplexDF64}}, C::ComposedSystem,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate!(C.f_ū, C.f, x, p)
    unscale_values!(C.f_ū, C.f_scale)
    evaluate!(u, C.g, C.f_ū, p)
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64}, C::ComposedSystem,
        x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate_and_jacobian!(C.f_u, C.f_U, C.f, x, p)
    unscale_values!(C.f_u, C.f_scale)
    scale = C.f_scale
    @inbounds for j in axes(C.f_U, 2), i in axes(C.f_U, 1)
        C.f_U[i, j] *= scale[i]
    end
    evaluate_and_jacobian!(u, C.g_U, C.g, C.f_u, p)
    LA.mul!(U, C.g_U, C.f_U)
    return nothing
end

## Taylor: the order-K coefficient of `G(F(x(t)))` is that of `G` on the series
## of `F` truncated at order K, and a `SystemEvaluator` returns one order per
## call, so a composed order costs K + 1 runs of the inner tape.

@inline function store_taylor_order!(
        tu::TaylorVector{4, ComplexF64}, order::Int, v::FSVec{ComplexF64},
    )::Nothing
    d = tu.data
    @inbounds for i in eachindex(v)
        d[order + 1, i] = v[i]
    end
    return nothing
end

# `TaylorVector` reads rows 1 to N of its backing matrix, so a lower-order view
# shares the matrix instead of copying it.
@inline truncate_series(
    tp::TaylorVector{M, ComplexF64}, ::Val{N},
) where {M, N} = TaylorVector{N, ComplexF64}(tp.data)

@inline function truncated_parameters(
        p::FSVec{ComplexF64}, ::Val{N},
    )::FSVec{ComplexF64} where {N}
    return p
end
@inline truncated_parameters(
    tp::TaylorVector{M, ComplexF64}, v::Val{N},
) where {M, N} = truncate_series(tp, v)

@inline order0_parameters(
    p::FSVec{ComplexF64}, ::FSVec{ComplexF64},
)::FSVec{ComplexF64} = p
@inline function order0_parameters(
        tp::TaylorVector{M, ComplexF64}, buf::FSVec{ComplexF64},
    )::FSVec{ComplexF64} where {M}
    d = tp.data
    @inbounds for i in eachindex(buf)
        buf[i] = d[1, i]
    end
    return buf
end

@inline function inner_series!(
        C::ComposedSystem, ::Val{0}, tx::TaylorVector{N, ComplexF64}, p,
    )::Nothing where {N}
    x0 = C.x0
    d = tx.data
    @inbounds for i in eachindex(x0)
        x0[i] = d[1, i]
    end
    evaluate!(C.f_u, C.f, x0, order0_parameters(p, C.p0))
    unscale_values!(C.f_u, C.f_scale)
    store_taylor_order!(C.tu, 0, C.f_u)
    return nothing
end

@inline function inner_series!(
        C::ComposedSystem, ::Val{K}, tx::TaylorVector{N, ComplexF64}, p,
    )::Nothing where {K, N}
    inner_series!(C, Val(K - 1), tx, p)
    taylor!(
        C.f_u, Val(K), C.f, truncate_series(tx, Val(K + 1)),
        truncated_parameters(p, Val(K + 1)),
    )
    unscale_values!(C.f_u, C.f_scale)
    store_taylor_order!(C.tu, K, C.f_u)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, v::Val{K}, C::ComposedSystem,
        tx::TaylorVector{N, ComplexF64},
        p::Union{FSVec{ComplexF64}, TaylorVector{<:Any, ComplexF64}},
    )::Nothing where {K, N}
    inner_series!(C, v, tx, p)
    taylor!(u, v, C.g, truncate_series(C.tu, Val(K + 1)), p)
    return nothing
end

## ── User-facing composition ─────────────────────────────────────────────────

# A stage is a thunk producing a fresh evaluator, so a composition of any depth
# is one concrete type and every worker can clone the whole chain.
# Converting a stage's equations costs as much as it has terms, so they sit
# behind a second thunk.
const StageEquations = FunctionWrapper{Vector{Expression}, Tuple{}}

struct SystemCloner{S <: System}
    system::S
end

(cloner::SystemCloner)()::SystemEvaluator = clone_system_evaluator(cloner.system)

struct StageEquationsThunk{S <: System}
    system::S
end

(eqs::StageEquationsThunk)()::Vector{Expression} =
    as_expressions(collect(eqs.system.polys))

struct CompositionStage
    factory::SystemFactory
    equations::StageEquations
    variables::Vector{Expression}
    # equation scales of this stage, undone when it feeds another stage
    scales::Vector{Float64}
    degrees::Vector{Int}
    is_homogeneous::Bool
end

"""
    CompositionSystem

The composition `G(F(x; p); p)` of two systems, built by [`compose`](@ref) or
the infix operator `∘`. Accepted wherever a system is only evaluated: parameter
homotopies ([`solve`](@ref) with start solutions), [`Monodromy`](@ref) and
[`newton`](@ref). `solve(C)` works too, since the total-degree start system
needs only the folded degrees. Routes that need the composed monomials or
equations (polyhedral, witness sets) go through `System(C)`, which substitutes
the stages into each other.

`variables` (those of the innermost stage) and `parameters` come back as
[`Expression`](@ref)s whichever front-end the stages were built from.

`degrees` and `is_homogeneous` are folded stage by stage rather than read off the
composed equations: each stage is weighted by the degrees of the stage below it,
so `G ∘ F` is homogeneous when every equation of `F` is and every equation of `G`
is homogeneous in the weights `deg(fᵢ)`. As for a [`System`](@ref) the degrees
are upper bounds and homogeneity is structural, and an equation that is not
polynomial in its variables gives a degree of `-1`.
"""
struct CompositionSystem
    # innermost stage first
    stages::Vector{CompositionStage}
    evaluator::SystemEvaluator
    variables::Vector{Expression}
    parameters::Vector{Expression}
    # `-1` entries where the composed degree is not determined
    degrees::Vector{Int}
    is_homogeneous::Bool
end

"""
    SystemLike

A [`System`](@ref) or a [`CompositionSystem`](@ref): input that carries a
`SystemEvaluator` and can be cloned per worker.
"""
const SystemLike = Union{System, CompositionSystem}

composition_stages(F::System)::Vector{CompositionStage} = CompositionStage[
    CompositionStage(
        SystemFactory(SystemCloner(F)),
        StageEquations(StageEquationsThunk(F)),
        as_expression_vector(F.variables),
        equation_scales(F),
        degrees(F),
        is_homogeneous(F),
    ),
]
composition_stages(C::CompositionSystem)::Vector{CompositionStage} = copy(C.stages)

function fold_composition(stages::Vector{CompositionStage})::SystemEvaluator
    evaluator = (stages[1].factory[])()::SystemEvaluator
    for k in 2:length(stages)
        evaluator = composition_evaluator(
            (stages[k].factory[])()::SystemEvaluator, evaluator, stages[k - 1].scales,
        )
    end
    return evaluator
end

# `gⱼ ∘ F` has the degree of `gⱼ` in the weights `deg(fᵢ)`, and is homogeneous
# when `gⱼ` is homogeneous in those weights and every `fᵢ` is. Uniform `deg(fᵢ)`
# collapses that to `deg(gⱼ) · d` with `gⱼ` homogeneous, which the stage already
# records; that branch is kept because it reads no equations.
function fold_composition_degrees(
        stages::Vector{CompositionStage},
    )::Tuple{Vector{Int}, Bool}
    degs = copy(stages[1].degrees)
    homogeneous = stages[1].is_homogeneous
    for k in 2:length(stages)
        stage = stages[k]
        d = uniform_degree(degs)
        if d >= 0 && all(>=(0), stage.degrees)
            degs = stage.degrees .* d
            homogeneous &= stage.is_homogeneous
        else
            degs, homogeneous = weighted_stage_degrees(stage, degs, homogeneous)
        end
    end
    return degs, homogeneous
end

# `inner` holds one degree per variable of the stage, `-1` where it is unknown.
function weighted_stage_degrees(
        stage::CompositionStage, inner::Vector{Int}, inner_homogeneous::Bool,
    )::Tuple{Vector{Int}, Bool}
    weights = Dict{Symbol, Int}(
        Symbol(v) => inner[i] for (i, v) in enumerate(stage.variables)
    )
    degs, homogeneous = weighted_degrees(stage.equations(), weights)
    return degs, homogeneous & inner_homogeneous
end

# The degree shared by every entry, or `-1` when they differ or one is unknown.
function uniform_degree(degs::Vector{Int})::Int
    isempty(degs) && return -1
    d = @inbounds degs[1]
    d < 0 && return -1
    for e in degs
        e == d || return -1
    end
    return d
end

as_expression_vector(vars::AbstractVector)::Vector{Expression} =
    Expression[convert(Expression, v) for v in vars]

composition_variables(F::System)::Vector{Expression} =
    as_expression_vector(F.variables)
composition_variables(C::CompositionSystem)::Vector{Expression} = C.variables

composition_parameters(F::System)::Vector{Expression} =
    as_expression_vector(F.parameters)
composition_parameters(C::CompositionSystem)::Vector{Expression} = C.parameters

"""
    compose(G::SystemLike, F::SystemLike) -> CompositionSystem

Construct the composition `G(F(x; p); p)`, also written `G ∘ F`. Both stages
receive the same parameters, so their parameter lists must agree unless one of
them is parameter-free.

```julia
@polyvar x y a b
f = System([y^2 + 2x + 3, x - 1])
g = System([x + y * a, x - b]; parameters = [a, b])
C = g ∘ f
```
"""
function compose(g::SystemLike, f::SystemLike)::CompositionSystem
    ng = size(g)[2]
    m = size(f)[1]
    ng == m || throw(
        ArgumentError(
            "cannot compose `G ∘ F`: `G` has $ng variables but `F` has $m equations",
        ),
    )
    pg = composition_parameters(g)
    pf = composition_parameters(f)
    (isempty(pg) || isempty(pf) || pg == pf) || throw(
        ArgumentError("cannot compose two systems with different parameters"),
    )
    stages = CompositionStage[composition_stages(f); composition_stages(g)]
    degs, homogeneous = fold_composition_degrees(stages)
    return CompositionSystem(
        stages, fold_composition(stages),
        composition_variables(f), isempty(pg) ? pf : pg,
        degs, homogeneous,
    )
end

Base.:∘(g::SystemLike, f::SystemLike)::CompositionSystem = compose(g, f)

Base.size(C::CompositionSystem)::Tuple{Int, Int} = size(C.evaluator)
nvariables(C::CompositionSystem)::Int = size(C.evaluator)[2]
nparameters(C::CompositionSystem)::Int = nparameters(C.evaluator)
variables(C::CompositionSystem)::Vector{Expression} = C.variables
parameters(C::CompositionSystem)::Vector{Expression} = C.parameters
degrees(C::CompositionSystem)::Vector{Int} = C.degrees
is_homogeneous(C::CompositionSystem)::Bool = C.is_homogeneous

@inline function with_system_shape(f::F, C::CompositionSystem) where {F}
    m, n = size(C)
    return m == n ? f(SquareShape()) : f(OverdeterminedShape())
end

clone_system_evaluator(C::CompositionSystem)::SystemEvaluator =
    fold_composition(C.stages)

# The evaluator undoes an inner stage's normalization and leaves the outermost
# stage's in place, so the symbolic fold has to do the same.
function stage_equations(stage::CompositionStage, unscale::Bool)::Vector{Expression}
    eqs = stage.equations()
    unscale || return eqs
    scales = stage.scales
    return Expression[
        isone(scales[i]) ? e : e * scales[i] for (i, e) in enumerate(eqs)
    ]
end

"""
    System(C::CompositionSystem; compile = CompileMode.INTERPRETED) -> System

Rebuild the composed equations symbolically, substituting each stage into the
next. Routes that need the equations themselves rather than their values
(polyhedral, witness sets, certification) go through this.

This is the work a `CompositionSystem` exists to avoid: the substituted tree is
differentiated and compiled as one system, which for a deep composition of large
stages takes minutes and gigabytes. The result agrees with `C` up to a nonzero
constant factor per equation, since `System` renormalizes what it is given.
"""
function System(
        C::CompositionSystem;
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::System
    stages = C.stages
    last_stage = length(stages)
    exprs = stage_equations(stages[1], true)
    for k in 2:last_stage
        outer = stage_equations(stages[k], k < last_stage)
        exprs = subs(outer, stages[k].variables => exprs)
    end
    return System(
        exprs; variables = C.variables, parameters = C.parameters, compile = compile,
    )
end

function Base.show(io::IO, C::CompositionSystem)
    m, n = size(C)
    print(
        io, "CompositionSystem of ", length(C.stages), " stages: ",
        m, " equations, ", n, " variables, ",
        nparameters(C), " parameters",
    )
    return
end
