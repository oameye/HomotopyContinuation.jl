export fixed

include("systems/mixed_system.jl")
include("systems/affine_chart_system.jl")
include("systems/composition_system.jl")
include("systems/fixed_parameter_system.jl")
include("systems/randomized_system.jl")
include("systems/sliced_system.jl")
include("systems/start_pair_system.jl")

"""
    fixed(F::System; compile = Val($(COMPILE_DEFAULT[])))

Constructs either a [`CompiledSystem`](@ref) (if `compile = Val(:all)` / `Val(true)`), an
[`InterpretedSystem`](@ref) (if `compile = Val(:none)` / `Val(false)`) or a
[`MixedSystem`](@ref) (`compile = Val(:mixed)`).
"""
fixed(F::System; compile::Val = COMPILE_DEFAULT[], kwargs...) =
    fixed(F, compile; kwargs...)
fixed(F::AbstractSystem; compile::Val = COMPILE_DEFAULT[], kwargs...) = F

# Val-dispatched methods — each returns a concrete type
fixed(F::System, ::Val{true}; kwargs...) = CompiledSystem(F; kwargs...)
fixed(F::System, ::Val{:all}; kwargs...) = CompiledSystem(F; kwargs...)
fixed(F::System, ::Val{false}; kwargs...) = InterpretedSystem(F; kwargs...)
fixed(F::System, ::Val{:none}; kwargs...) = InterpretedSystem(F; kwargs...)
fixed(F::System, ::Val{:mixed}; kwargs...) = MixedSystem(F; kwargs...)
fixed(F::AbstractSystem, ::Val; kwargs...) = F

set_solution!(x, ::AbstractSystem, y) = (x .= y; x)
