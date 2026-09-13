## Abstract types and interface contracts for systems and homotopies.

abstract type AbstractSystem end

"""
    AbstractHomotopy

Interface for a user-defined homotopy `H(x, t)`. A subtype must implement
`size(H) == (nequations, nvariables)` and the mutating evaluation methods

    evaluate!(u, H, x, t)
    evaluate_and_jacobian!(u, U, H, x, t)
    taylor!(u, Val(k), H, tx, t)

for Taylor orders `k = 1, 2, 3`. Method signatures should accept ordinary
`AbstractVector` / `AbstractMatrix` arguments rather than HomotopyContinuation's
private fixed-size storage types. During residual refinement, `x` may carry
extended-precision scalar values, so implementations should not unnecessarily
restrict its element type.

For a parameterized custom homotopy, define `parameters(H)` (or `nparameters(H)`
directly) and the corresponding methods with a final parameter vector `p`, then
bind values with [`fix_parameters`](@ref) before tracking.

[`set_solution!`](@ref) and [`get_solution!`](@ref) are optional coordinate
transforms and default to copying their input. Parallel solves rebuild custom
homotopies with `deepcopy`; if that is not appropriate, use the
`solve(build_homotopy, starts, alg, exec)` form to construct independent worker
instances explicitly.
"""
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

# Rebuild `F` with its own mutable state; read-only data may be shared.
function _clone_system end

# `deepcopy` gives independent buffers, and the `SystemEvaluator` hook rebuilds
# tapes rather than copying a pointer into the original ones. A method that
# shares the read-only data is faster.
_clone_system(F::AbstractSystem) = deepcopy(F)

# ── AbstractHomotopy interface ──

Base.size(H::AbstractHomotopy, i::Integer)::Int = size(H)[i]
Base.length(H::AbstractHomotopy)::Int = size(H, 1)

"""Number of variables tracked by `H`, inferred from `size(H)`."""
nvariables(H::AbstractHomotopy)::Int = size(H, 2)

"""Symbolic external parameters of `H`. Parameter-free homotopies return none."""
parameters(::AbstractHomotopy)::Vector{Expression} = Expression[]

"""Number of external parameters of `H`, inferred from `parameters(H)`."""
nparameters(H::AbstractHomotopy)::Int = length(parameters(H))

"""Variable groups of `H`. The default is no grouping metadata."""
variable_groups(::AbstractHomotopy)::Vector{Vector{Int}} = Vector{Int}[]

# Required: evaluate!, evaluate_and_jacobian!, taylor! (defined above).
# Higher-order homotopy Taylor methods use the same five-argument contract as
# systems: taylor!(u, ::Val{K}, H, tx, t).
# Required: Base.size(H) -> (nequations, nvariables)

# Optional coordinate transforms. Keep the legacy three-argument copy helpers
# and make the actual AbstractHomotopy interface delegate to them.
set_solution!(x::AbstractVector, y::AbstractVector, ::ComplexF64)::Nothing =
    (copyto!(x, y); nothing)
get_solution!(out::AbstractVector, x::AbstractVector, ::ComplexF64)::Nothing =
    (copyto!(out, x); nothing)

"""
    set_solution!(x, H::AbstractHomotopy, y, t) -> nothing

Map an external representative `y` to `H`'s internal coordinates at `t`.
The default is an identity copy.
"""
set_solution!(
    x::AbstractVector, ::AbstractHomotopy, y::AbstractVector, t::ComplexF64,
)::Nothing = set_solution!(x, y, t)

"""
    get_solution!(out, H::AbstractHomotopy, x, t) -> nothing

Map `H`'s internal coordinates `x` back to the reported representative at `t`.
The default is an identity copy.
"""
get_solution!(
    out::AbstractVector, ::AbstractHomotopy, x::AbstractVector, t::ComplexF64,
)::Nothing = get_solution!(out, x, t)

# Rebuild `H` with its own mutable state, cloning the evaluators it holds.
function _clone_homotopy end

_clone_homotopy(H::AbstractHomotopy) = deepcopy(H)

"""
    start_parameters!(H::AbstractHomotopy, p) -> H
"""
start_parameters!(H::AbstractHomotopy, ::AbstractVector) = H

"""
    target_parameters!(H::AbstractHomotopy, ::AbstractVector) -> H
"""
target_parameters!(H::AbstractHomotopy, ::AbstractVector) = H
