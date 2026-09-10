from pathlib import Path

p = Path("src/core/fixed_parameter_homotopy.jl")
s = p.read_text()
old = '''struct FixedParameterHomotopy{H <: AbstractHomotopy} <: AbstractHomotopy
    homotopy::H
    parameters::Vector{ComplexF64}
end

function FixedParameterHomotopy(
        H::AbstractHomotopy, p::AbstractVector{<:Number},
    )::FixedParameterHomotopy
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
    return FixedParameterHomotopy(H, Vector{ComplexF64}(p))
end
'''
new = '''struct FixedParameterHomotopy{H <: AbstractHomotopy} <: AbstractHomotopy
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
'''
if old not in s:
    raise SystemExit("missing outer constructor block")
p.write_text(s.replace(old, new, 1))
