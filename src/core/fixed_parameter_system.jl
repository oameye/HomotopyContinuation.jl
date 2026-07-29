"""
    fix_parameters(F::System, p::AbstractVector{<:Number}) -> System
    fix_parameters(C::CompositionSystem, p::AbstractVector{<:Number}) -> FixedParameterSystem

The system with its parameters fixed at the values `p`, accepted by every route
that requires a parameter-free system.

```julia
@polyvar x y a b
F = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
solve(fix_parameters(F, [2, 4]), TotalDegree())
```

A `System` has its values substituted into the equations, which keeps the tracked
tapes one evaluator hop shorter and is what the routes reading the support or the
equations (polyhedral, witness sets) need. A composition has no equations to
substitute into and binds them instead.
"""
function fix_parameters(F::System{P, V, M}, p::AbstractVector{<:Number})::System where {P, V, M}
    values = _parameter_values(F, p)
    return System(
        _substitute(polynomials(F), collect(parameters(F)), values);
        variables = collect(variables(F)), compile = M,
    )
end

# Substitution runs through whichever front-end built the system.
_substitute(
    polys::FSVec{<:MP.AbstractPolynomialLike}, params::Vector, pc::Vector{ComplexF64},
) = [MP.polynomial(MP.subs(f, params => pc)) for f in polys]

_substitute(
    polys::FSVec{Expression}, params::Vector{Expression}, pc::Vector{ComplexF64},
)::Vector{Expression} = [subs(f, params => pc) for f in polys]

function _parameter_values(
        F, p::AbstractVector{<:Number},
    )::Vector{ComplexF64}
    np = nparameters(F)
    np == 0 && throw(
        ArgumentError(
            "Parameter values were given, but the system has no parameters.",
        ),
    )
    length(p) == np || throw(
        ArgumentError(
            "The number of parameter values ($(length(p))) does not match the " *
                "number of parameters ($np).",
        ),
    )
    return Vector{ComplexF64}(p)
end

"""
    FixedParameterSystem(F, p::AbstractVector{<:Number})

`F` with its parameters bound to `p` at the evaluator level, leaving the equations
alone. [`fix_parameters`](@ref) returns one of these for a composition, where
there are no equations to substitute into.
"""
struct FixedParameterSystem{S <: SystemLike}
    system::S
    parameters::Vector{ComplexF64}
    evaluator::SystemEvaluator
end

function FixedParameterSystem(
        F::SystemLike, p::AbstractVector{<:Number},
    )::FixedParameterSystem
    values = _parameter_values(F, p)
    return FixedParameterSystem(F, values, _bound_evaluator(F.evaluator, values))
end

fix_parameters(C::CompositionSystem, p::AbstractVector{<:Number})::FixedParameterSystem =
    FixedParameterSystem(C, p)

# Rebuilt per worker: the bound evaluator owns mutable tapes that must not be shared.
_clone_system_evaluator(F::FixedParameterSystem)::SystemEvaluator =
    _bound_evaluator(_clone_system_evaluator(F.system), F.parameters)

# Parameter values change neither the degrees in the variables nor the shape.
degrees(F::FixedParameterSystem)::Vector{Int} = degrees(F.system)
nvariables(F::FixedParameterSystem)::Int = nvariables(F.system)
nparameters(::FixedParameterSystem)::Int = 0
system_shape(F::FixedParameterSystem) = system_shape(F.system)
Base.size(F::FixedParameterSystem)::Tuple{Int, Int} = size(F.system)

"""
    CloneableSystem

Input a solve route can clone a fresh parameter-free `SystemEvaluator` from.
"""
const CloneableSystem = Union{System, CompositionSystem, FixedParameterSystem}

_bound_evaluator(
    inner::SystemEvaluator, p::AbstractVector{<:Number},
)::SystemEvaluator = SystemEvaluator(_BoundParameterSystem(inner, p))

# `F(x; p)` with `p` frozen, as a parameter-free `AbstractSystem`.
struct _BoundParameterSystem <: AbstractSystem
    system::SystemEvaluator
    p::FSVec{ComplexF64}
end

_BoundParameterSystem(system::SystemEvaluator, p::AbstractVector{<:Number}) =
    _BoundParameterSystem(system, FSVec{ComplexF64}(Vector{ComplexF64}(p)))

Base.size(F::_BoundParameterSystem)::Tuple{Int, Int} = size(F.system)
nparameters(::_BoundParameterSystem)::Int = 0

# The ignored argument below is the wrapper's own (empty) parameter vector.
function evaluate!(
        u::FSVec{ComplexF64}, F::_BoundParameterSystem,
        x::FSVec{ComplexF64}, ::FSVec{ComplexF64},
    )::Nothing
    evaluate!(u, F.system, x, F.p)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, F::_BoundParameterSystem,
        x::FSVec{ComplexDF64}, ::FSVec{ComplexF64},
    )::Nothing
    evaluate!(u, F.system, x, F.p)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexDF64}, F::_BoundParameterSystem,
        x::FSVec{ComplexDF64}, ::FSVec{ComplexF64},
    )::Nothing
    evaluate!(u, F.system, x, F.p)
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64}, F::_BoundParameterSystem,
        x::FSVec{ComplexF64}, ::FSVec{ComplexF64},
    )::Nothing
    evaluate_and_jacobian!(u, U, F.system, x, F.p)
    return nothing
end

# A frozen `p` has the series `p + 0t + …`, so both parameter forms take the
# constant call on the wrapped system.
for (K, N) in ((1, 2), (2, 3), (3, 4))
    @eval function taylor!(
            u::FSVec{ComplexF64}, v::Val{$K}, F::_BoundParameterSystem,
            tx::TaylorVector{$N, ComplexF64}, ::FSVec{ComplexF64},
        )::Nothing
        taylor!(u, v, F.system, tx, F.p)
        return nothing
    end

    @eval function taylor!(
            u::FSVec{ComplexF64}, v::Val{$K}, F::_BoundParameterSystem,
            tx::TaylorVector{$N, ComplexF64}, ::TaylorVector{$N, ComplexF64},
        )::Nothing
        taylor!(u, v, F.system, tx, F.p)
        return nothing
    end
end
