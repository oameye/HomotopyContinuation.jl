# internal only
include("homotopies/toric_homotopy.jl")

# public, these should be linked on the top
include("homotopies/mixed_homotopy.jl")
include("homotopies/affine_chart_homotopy.jl")
include("homotopies/parameter_homotopy.jl")
include("homotopies/coefficient_homotopy.jl")
include("homotopies/subspace_homotopies.jl")
include("homotopies/straight_line_homotopy.jl")
include("homotopies/fixed_parameter_homotopy.jl")

"""
    fixed(H::Homotopy; compile = Val($(COMPILE_DEFAULT[])))

Constructs either a [`CompiledHomotopy`](@ref) (if `compile = Val(:all)`), an
[`InterpretedHomotopy`](@ref) (if `compile = Val(:none)`) or a
[`MixedHomotopy`](@ref) (`compile = Val(:mixed)`).
"""
fixed(H::Homotopy; compile::Val = COMPILE_DEFAULT[], kwargs...) =
    fixed(H, compile; kwargs...)
fixed(H::AbstractHomotopy; compile::Val = COMPILE_DEFAULT[], kwargs...) = H

fixed(H::Homotopy, ::Val{:all}; kwargs...) = CompiledHomotopy(H; kwargs...)
fixed(H::Homotopy, ::Val{:none}; kwargs...) = InterpretedHomotopy(H; kwargs...)
fixed(H::Homotopy, ::Val{:mixed}; kwargs...) = MixedHomotopy(H; kwargs...)
fixed(H::AbstractHomotopy, ::Val; kwargs...) = H

function set_solution!(x::AbstractVector, H::AbstractHomotopy, y::AbstractVector, t)
    x .= y
end
get_solution(H::AbstractHomotopy, x::AbstractVector, t) = copy(x)

start_parameters!(H::AbstractHomotopy, p) = H
target_parameters!(H::AbstractHomotopy, p) = H

ModelKit.taylor!(u, v::Val, H::AbstractHomotopy, tx, t, incremental::Bool) =
    taylor!(u, v, H, tx, t)
