"""
    FixedParameterHomotopy(H::AbstractHomotopy, p::AbstractVector{<:Number})

Wrap a user-defined [`AbstractHomotopy`](@ref) with its external parameters fixed
at `p`. The wrapped homotopy must implement the parameter-aware forms

    evaluate!(u, H, x, t, p)
    evaluate_and_jacobian!(u, U, H, x, t, p)
    taylor!(u, Val(k), H, tx, t, p)

for the scalar types/Taylor orders it supports. The tracker still sees the normal
parameter-free `(x, t)` homotopy interface; the parameter vector is bound before
`HomotopyEvaluator` erases the concrete homotopy type.
"""
struct FixedParameterHomotopy{H <: AbstractHomotopy} <: AbstractHomotopy
    homotopy::H
    parameters::Vector{ComplexF64}
end

function FixedParameterHomotopy(
        H::AbstractHomotopy, p::AbstractVector{<:Number},
    )::FixedParameterHomotopy
    return FixedParameterHomotopy(H, Vector{ComplexF64}(p))
end

fix_parameters(
    H::AbstractHomotopy, p::AbstractVector{<:Number},
)::FixedParameterHomotopy = FixedParameterHomotopy(H, p)

Base.size(H::FixedParameterHomotopy)::Tuple{Int, Int} = size(H.homotopy)
nvariables(H::FixedParameterHomotopy)::Int = last(size(H))
nparameters(::FixedParameterHomotopy)::Int = 0
variables(H::FixedParameterHomotopy) = variables(H.homotopy)
parameters(::FixedParameterHomotopy) = Expression[]
variable_groups(H::FixedParameterHomotopy) = variable_groups(H.homotopy)

_clone_homotopy(H::FixedParameterHomotopy)::FixedParameterHomotopy =
    FixedParameterHomotopy(_clone_homotopy(H.homotopy), copy(H.parameters))

function evaluate!(
        u::FSVec{ComplexF64}, H::FixedParameterHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    evaluate!(u, H.homotopy, x, t, H.parameters)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, H::FixedParameterHomotopy,
        x::FSVec{ComplexDF64}, t::ComplexF64,
    )::Nothing
    evaluate!(u, H.homotopy, x, t, H.parameters)
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64}, H::FixedParameterHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    evaluate_and_jacobian!(u, U, H.homotopy, x, t, H.parameters)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, H::FixedParameterHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    taylor!(u, Val(1), H.homotopy, x, t, H.parameters)
    return nothing
end

for (K, N) in ((2, 3), (3, 4))
    @eval function taylor!(
            u::FSVec{ComplexF64}, ::Val{$K}, H::FixedParameterHomotopy,
            tx::TaylorVector{$N, ComplexF64}, t::ComplexF64,
        )::Nothing
        taylor!(u, Val($K), H.homotopy, tx, t, H.parameters)
        return nothing
    end
end

function set_solution!(
        x::FSVec{ComplexF64}, H::FixedParameterHomotopy,
        y::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    set_solution!(x, H.homotopy, y, t)
    return nothing
end

function get_solution!(
        out::FSVec{ComplexF64}, H::FixedParameterHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    get_solution!(out, H.homotopy, x, t)
    return nothing
end
