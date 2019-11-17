export AbstractInput,
       StartTargetInput,
       TargetSystemInput,
       HomotopyInput,
       ParameterSystemInput

const Inputs = Union{<:AbstractSystem,<:MPPolys,<:Composition}
const MPPolyInputs = Union{<:MPPolys,<:Composition}

const input_supported_keywords = [
    :parameters,
    :generic_parameters,
    :start_parameters,
    :target_parameters,
    :target_gamma,
    :start_gamma,
    :p₁,
    :p₀,
    :γ₁,
    :γ₀,
    :variable_ordering,
    # deprecated
    :startparameters,
    :targetparameters,
    :targetgamma,
    :startgamma,
]



"""
    AbstractInput

An abstract type to model different input types.
"""
abstract type AbstractInput end

"""
    StartTargetInput(start::MPPolyInputs, target::MPPolyInputs, startsolutions)

Construct a `StartTargetInput` out of two systems of polynomials.
"""
struct StartTargetInput{P1<:Inputs,P2<:Inputs} <: AbstractInput
    start::P1
    target::P2

    function StartTargetInput{P1,P2}(start::P1, target::P2) where {P1<:Inputs,P2<:Inputs}
        if length(start) ≠ length(target)
            error("Cannot construct `StartTargetInput` since the lengths of `start` and `target` don't match.")
        end
        new(start, target)
    end
end
function StartTargetInput(start::P1, target::P2) where {P1<:Inputs,P2<:Inputs}
    StartTargetInput{P1,P2}(start, target)
end


"""
    HomotopyInput(H::Homotopy, startsolutions)

Construct a `HomotopyInput` for a homotopy `H` with given `startsolutions`.
"""
struct HomotopyInput{Hom<:AbstractHomotopy} <: AbstractInput
    H::Hom
end


"""
    TargetSystemInput(system::Inputs)

Construct a `TargetSystemInput`. This indicates that the system `system`
is the target system and a start system should be assembled.
"""
struct TargetSystemInput{S<:Inputs} <: AbstractInput
    system::S
end

"""
    ParameterSystemInput(F, parameters, p₁, p₀, startsolutions, γ₁=nothing, γ₂=nothing)

Construct a `ParameterSystemInput`.
"""
struct ParameterSystemInput{S<:Inputs} <: AbstractInput
    system::S
    parameters::Union{Nothing,Vector{<:MP.AbstractVariable}}
    p₁::AbstractVector
    p₀::AbstractVector
    γ₁::Union{Nothing,ComplexF64}
    γ₀::Union{Nothing,ComplexF64}
end

"""
    input_startsolutions(F::MPPolyInputs)
    input_startsolutions(F::AbstractSystem)
    input_startsolutions(G::MPPolyInputs, F::MPPolyInputs, startsolutions)
    input_startsolutions(F::MPPolyInputs, parameters, startsolutions; kwargs...)
    input_startsolutions(H::AbstractHomotopy, startsolutions)

Construct an `AbstractInput` and pass through startsolutions if provided.
Returns a named tuple `(input=..., startsolutions=...)`.
"""
function input_startsolutions(
    F::MPPolyInputs;
    variable_ordering = nothing,
    parameters = nothing,
    kwargs...,
)
    # if parameters !== nothing this is actually the
    # input constructor for a parameter homotopy, but no startsolutions
    # are provided
    if parameters !== nothing
        startsolutions = [randn(ComplexF64, nvariables(F, parameters))]
        return input_startsolutions(
            F,
            startsolutions;
            variable_ordering = variable_ordering,
            parameters = parameters,
            kwargs...,
        )
    end

    if variable_ordering !== nothing && nvariables(F) != length(variable_ordering)
        throw(ArgumentError("Number of assigned variables is too small."))
    end

    remove_zeros!(F)
    if has_constant_polynomial(F)
        throw(ArgumentError("System contains a non-zero constant polynomial."))
    end

    (input = TargetSystemInput(F), startsolutions = nothing)
end

function input_startsolutions(F::AbstractSystem; variable_ordering = nothing)
    (input = TargetSystemInput(F), startsolutions = nothing)
end
function input_startsolutions(
    F::Vector{<:ModelKit.Expression};
    variable_ordering::Vector{ModelKit.Variable} = error("`variable_ordering = ...` needs to be passed as a keyword argument."),
)
    input_startsolutions(ModelKit.System(F, variable_ordering))
end

function input_startsolutions(F::ModelKit.System; variable_ordering = nothing)
    (input = TargetSystemInput(ModelKitSystem(F)), startsolutions = nothing)
end

function input_startsolutions(
    G::MPPolyInputs,
    F::MPPolyInputs,
    startsolutions = nothing;
    variable_ordering = nothing,
)
    if length(G) ≠ length(F)
        throw(ArgumentError("Start and target system don't have the same length"))
    end
    if variable_ordering !== nothing && (
    	nvariables(F) != length(variable_ordering) || nvariables(G) != length(variable_ordering)
    )
        throw(ArgumentError("Number of assigned variables is too small."))
    end

    check_zero_dimensional(F)
    if startsolutions === nothing
        startsolutions = [randn(ComplexF64, nvariables(F))]
    elseif isa(startsolutions, AbstractVector{<:Number})
        startsolutions = [startsolutions]
    end
    (input = StartTargetInput(G, F), startsolutions = startsolutions)
end


# need
input_startsolutions(F::MPPolyInputs, starts; kwargs...) =
    parameter_homotopy(F, starts; kwargs...)
input_startsolutions(F::AbstractSystem, starts; variable_ordering = nothing, kwargs...) =
    parameter_homotopy(F, starts; kwargs...)

function parameter_homotopy(
    F::Inputs,
    startsolutions;
    variable_ordering = nothing,
    parameters = (
    	isa(F, AbstractSystem) ? nothing :
        error(ArgumentError("You need to pass `parameters=...` as a keyword argument."))
    ),
    generic_parameters = nothing,
    start_parameters = generic_parameters,
    p₁ = start_parameters,
    target_parameters = generic_parameters,
    p₀ = target_parameters,
    start_gamma = nothing,
    γ₁ = start_gamma,
    target_gamma = nothing,
    γ₀ = target_gamma,
    # deprecated in 0.7
    startparameters = nothing,
    targetparameters = nothing,
    startgamma = nothing,
    targetgamma = nothing,
)

    # deprecation handling
    @deprecatekwarg startparameters start_parameters
    @deprecatekwarg targetparameters target_parameters
    @deprecatekwarg startgamma start_gamma
    @deprecatekwarg targetgamma target_gamma

    if γ₁ === nothing
        γ₁ = start_gamma
    end
    if γ₀ === nothing
        γ₀ = target_gamma
    end
    if p₁ === nothing
        p₁ = start_parameters
    end
    if p₀ === nothing
        p₀ = target_parameters
    end


    if p₁ === nothing
        error("You need to pass `generic_parameters=`, `start_parameters=` or `p₁=` as a keyword argument")
    elseif p₀ === nothing
        error("`target_parameters=` or `p₀=` need to be passed as a keyword argument.")
    end

    if length(p₁) != length(
        p₀,
    ) || (parameters !== nothing && length(parameters) != length(p₀))
        error("Number of parameters doesn't match!")
    end
    if startsolutions === nothing && parameters !== nothing
        startsolutions = [randn(ComplexF64, nvariables(F, parameters))]
    elseif isa(startsolutions, AbstractVector{<:Number})
        startsolutions = [startsolutions]
    end

    if variable_ordering !== nothing &&
       nvariables(F, parameters) != length(variable_ordering)
        throw(ArgumentError("Number of assigned variables is too small."))
    end

    (
     input = ParameterSystemInput(F, parameters, p₁, p₀, γ₁, γ₀),
     startsolutions = startsolutions,
    )
end

function input_startsolutions(
    H::AbstractHomotopy,
    startsolutions;
    variable_ordering = nothing,
)
    if isa(startsolutions, AbstractVector{<:Number})
        startsolutions = [startsolutions]
    end
    (input = HomotopyInput(H), startsolutions = startsolutions)
end
