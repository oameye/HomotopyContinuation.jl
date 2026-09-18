"""
    FixedParameterHomotopy(H::AbstractHomotopy, p::AbstractVector{<:Number})

Wrap a user-defined [`AbstractHomotopy`](@ref) with its external parameters fixed
at `p`. The wrapped homotopy must report the same number of parameters through
[`nparameters`](@ref) and implement the parameter-aware forms

    evaluate!(u, H, x, t, p)
    evaluate_and_jacobian!(u, U, H, x, t, p)
    taylor!(u, Val(k), H, tx, t, p)

for the scalar types and Taylor orders it supports. These methods may use generic
`AbstractVector` / `AbstractMatrix` signatures; the tracker's fixed-size storage
is deliberately not part of the extension API. The parameter vector is bound
before `HomotopyEvaluator` erases the concrete homotopy type.
"""
struct FixedParameterHomotopy{H <: AbstractHomotopy} <: AbstractHomotopy
    homotopy::H
    parameters::Vector{ComplexF64}

    # An inner constructor is deliberate: otherwise Julia generates the more
    # specific `(H, ::Vector{ComplexF64})` field constructor, which would bypass
    # the public arity validation for already-complex parameter vectors.
    function FixedParameterHomotopy(
            H::T, p::AbstractVector{<:Number},
        ) where {T <: AbstractHomotopy}
        np = nparameters(H)
        np == 0 && throw(
            ArgumentError("Parameter values were given, but the homotopy has no parameters."),
        )
        length(p) == np || throw(
            ArgumentError(
                "The number of parameter values ($(length(p))) does not match the " *
                    "number of homotopy parameters ($np).",
            ),
        )
        return new{T}(H, Vector{ComplexF64}(p))
    end
end

fix_parameters(
    H::AbstractHomotopy, p::AbstractVector{<:Number},
)::FixedParameterHomotopy = FixedParameterHomotopy(H, p)

Base.size(H::FixedParameterHomotopy)::Tuple{Int, Int} = size(H.homotopy)
nvariables(H::FixedParameterHomotopy)::Int = nvariables(H.homotopy)
nparameters(::FixedParameterHomotopy)::Int = 0
variables(H::FixedParameterHomotopy)::Vector{Expression} = variables(H.homotopy)
parameters(::FixedParameterHomotopy)::Vector{Expression} = Expression[]
variable_groups(H::FixedParameterHomotopy)::Vector{Vector{Int}} =
    variable_groups(H.homotopy)

_clone_homotopy(H::FixedParameterHomotopy)::FixedParameterHomotopy =
    FixedParameterHomotopy(_clone_homotopy(H.homotopy), copy(H.parameters))

function evaluate!(
        u::AbstractVector, H::FixedParameterHomotopy,
        x::AbstractVector, t::ComplexF64,
    )::Nothing
    evaluate!(u, H.homotopy, x, t, H.parameters)
    return nothing
end

function evaluate_and_jacobian!(
        u::AbstractVector, U::AbstractMatrix, H::FixedParameterHomotopy,
        x::AbstractVector, t::ComplexF64,
    )::Nothing
    evaluate_and_jacobian!(u, U, H.homotopy, x, t, H.parameters)
    return nothing
end

function taylor!(
        u::AbstractVector, v::Val, H::FixedParameterHomotopy,
        tx::AbstractVector, t::ComplexF64,
    )::Nothing
    taylor!(u, v, H.homotopy, tx, t, H.parameters)
    return nothing
end

function set_solution!(
        x::AbstractVector, H::FixedParameterHomotopy,
        y::AbstractVector, t::ComplexF64,
    )::Nothing
    set_solution!(x, H.homotopy, y, t)
    return nothing
end

function get_solution!(
        out::AbstractVector, H::FixedParameterHomotopy,
        x::AbstractVector, t::ComplexF64,
    )::Nothing
    get_solution!(out, H.homotopy, x, t)
    return nothing
end
