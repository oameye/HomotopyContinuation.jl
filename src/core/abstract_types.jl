## Abstract types and interface contracts for systems and homotopies.

abstract type AbstractSystem end
abstract type AbstractHomotopy end

# ── AbstractSystem interface ──

# Base.size(F::AbstractSystem) -> (nequations, nvariables) — must be implemented by subtypes.
function Base.size(::AbstractSystem)::Tuple{Int, Int}
    error("size(::AbstractSystem) must be implemented by subtypes")
end

# Base.size(H::AbstractHomotopy) -> (nequations, nvariables) — must be implemented by subtypes.
function Base.size(::AbstractHomotopy)::Tuple{Int, Int}
    error("size(::AbstractHomotopy) must be implemented by subtypes")
end

"""
    nparameters(F::AbstractSystem) -> Int

Number of parameters. Default: 0.
"""
nparameters(::AbstractSystem)::Int = 0

"""
    evaluate!(u, F::AbstractSystem, x, p) -> nothing

Evaluate F at (x, p), writing result into u.
"""
function evaluate! end

"""
    evaluate_and_jacobian!(u, U, F::AbstractSystem, x, p) -> nothing

Evaluate F and its Jacobian at (x, p), writing into u and U.
"""
function evaluate_and_jacobian! end

"""
    taylor!(u, ::Val{K}, F::AbstractSystem, tx, p) -> nothing

Compute order-K Taylor coefficient of F.
"""
function taylor! end

# ── AbstractHomotopy interface ──

# Required: evaluate!, evaluate_and_jacobian!, taylor! (defined above).
# Higher-order homotopy Taylor methods use the same five-argument contract as
# systems: taylor!(u, ::Val{K}, H, tx, t).
# Required: Base.size(H) -> (nequations, nvariables)

# Optional defaults:
"""
    set_solution!(x, y, t) -> nothing

Map y to the internal representation x at time t. Default: copy.
"""
set_solution!(x::AbstractVector, y::AbstractVector, ::ComplexF64)::Nothing =
    (copyto!(x, y); nothing)

"""
    get_solution!(out, x, t) -> nothing

Extract solution from internal representation. Default: copy.
"""
get_solution!(out::AbstractVector, x::AbstractVector, ::ComplexF64)::Nothing =
    (copyto!(out, x); nothing)

"""
    start_parameters!(H::AbstractHomotopy, p) -> H
"""
start_parameters!(H::AbstractHomotopy, ::AbstractVector) = H

"""
    target_parameters!(H::AbstractHomotopy, ::AbstractVector) -> H
"""
target_parameters!(H::AbstractHomotopy, ::AbstractVector) = H
