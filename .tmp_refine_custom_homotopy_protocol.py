from pathlib import Path

# Public extension surface: expose the abstract type, metadata hooks and the
# mutating evaluation protocol, but keep the tracker's storage types private.
p = Path("src/HomotopyContinuationNext.jl")
s = p.read_text()
old = "export ParameterHomotopy, Homotopy, expressions, equation_scales\n"
new = """export AbstractHomotopy\nexport ParameterHomotopy, Homotopy, expressions, equation_scales\nexport nvariables, nparameters, variables, parameters, variable_groups\nexport evaluate!, evaluate_and_jacobian!, taylor!, set_solution!, get_solution!\n"""
if old not in s:
    raise SystemExit("missing public homotopy export anchor")
p.write_text(s.replace(old, new, 1))

Path("src/core/abstract_types.jl").write_text(r'''## Abstract types and interface contracts for systems and homotopies.

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
''')

Path("src/core/fixed_parameter_homotopy.jl").write_text(r'''"""
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

fix_parameters(
    H::AbstractHomotopy, p::AbstractVector{<:Number},
)::FixedParameterHomotopy = FixedParameterHomotopy(H, p)

Base.size(H::FixedParameterHomotopy)::Tuple{Int, Int} = size(H.homotopy)
nvariables(H::FixedParameterHomotopy)::Int = nvariables(H.homotopy)
nparameters(::FixedParameterHomotopy)::Int = 0
variables(H::FixedParameterHomotopy) = variables(H.homotopy)
parameters(::FixedParameterHomotopy)::Vector{Expression} = Expression[]
variable_groups(H::FixedParameterHomotopy) = variable_groups(H.homotopy)

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
''')

# The public parallel escape hatch is the builder form; don't tell downstream
# users to extend a private cloning hook.
p = Path("src/solving/homotopy_solve.jl")
s = p.read_text()
old = """Anything but `Serial()` rebuilds `H` per task, since it owns the buffers it
evaluates through. Any homotopy can be rebuilt; a `_clone_homotopy` method for
your own type makes it cheaper by sharing its read-only data.
"""
new = """Anything but `Serial()` rebuilds `H` per task, since it owns the buffers it
evaluates through. Custom homotopies use `deepcopy` by default. If a type cannot
be deep-copied safely or cheaply, use `solve(build_homotopy, starts, alg, exec)`
to construct one independent homotopy per worker.
"""
if old not in s:
    raise SystemExit("missing custom homotopy cloning doc anchor")
p.write_text(s.replace(old, new, 1))

Path("test/fixed_parameter_homotopy_test.jl").write_text(r'''using Test
using HomotopyContinuationNext
import HomotopyContinuationNext:
    evaluate!, evaluate_and_jacobian!, taylor!, parameters, variables, variable_groups

@var ξ α

# This intentionally uses only the public custom-homotopy protocol and ordinary
# AbstractVector/AbstractMatrix signatures. No tracker storage type is named here.
# H(x,t;p) = x - [p + (1-p)t], so the root moves exactly from 1 at t=1 to p at t=0.
struct _ParametricLineHomotopy <: AbstractHomotopy end
Base.size(::_ParametricLineHomotopy) = (1, 1)
variables(::_ParametricLineHomotopy) = [ξ]
parameters(::_ParametricLineHomotopy) = [α]
variable_groups(::_ParametricLineHomotopy) = [[1]]

function evaluate!(
        u::AbstractVector, ::_ParametricLineHomotopy,
        x::AbstractVector, t::ComplexF64, p::AbstractVector,
    )
    u[1] = ComplexF64(x[1]) - (p[1] + (1 - p[1]) * t)
    return nothing
end

function evaluate_and_jacobian!(
        u::AbstractVector, U::AbstractMatrix, H::_ParametricLineHomotopy,
        x::AbstractVector, t::ComplexF64, p::AbstractVector,
    )
    evaluate!(u, H, x, t, p)
    U[1, 1] = 1
    return nothing
end

function taylor!(
        u::AbstractVector, ::Val{K}, ::_ParametricLineHomotopy,
        tx::AbstractVector, ::ComplexF64, p::AbstractVector,
    ) where {K}
    if K == 1
        u[1] = p[1] - 1
    else
        # TaylorVector elements use zero-based coefficient indexing, but the
        # protocol only requires an AbstractVector here.
        u[1] = tx[1][K]
    end
    return nothing
end

struct _ParameterFreeHomotopy <: AbstractHomotopy end
Base.size(::_ParameterFreeHomotopy) = (1, 1)

@testset "public custom homotopy protocol" begin
    H = _ParametricLineHomotopy()
    @test nvariables(H) == 1
    @test nparameters(H) == 1
    @test parameters(H) == [α]

    # Optional coordinate transforms really are optional: the AbstractHomotopy
    # defaults have the same arity used by HomotopyEvaluator.
    x = zeros(ComplexF64, 1)
    set_solution!(x, H, ComplexF64[2], 0.5 + 0im)
    @test x == ComplexF64[2]
    y = zeros(ComplexF64, 1)
    get_solution!(y, H, x, 0.5 + 0im)
    @test y == x

    # Parameter arity is rejected at the binding boundary, not later in a path.
    @test_throws ArgumentError fix_parameters(H, ComplexF64[])
    @test_throws ArgumentError fix_parameters(H, [1, 2])
    H0 = _ParameterFreeHomotopy()
    @test nparameters(H0) == 0
    @test_throws ArgumentError fix_parameters(H0, [1])
end

@testset "FixedParameterHomotopy" begin
    H = _ParametricLineHomotopy()
    F = fix_parameters(H, [3.0])
    @test F isa FixedParameterHomotopy
    @test size(F) == (1, 1)
    @test F.parameters isa Vector{ComplexF64}
    @test F.parameters == ComplexF64[3]
    @test nvariables(F) == 1
    @test nparameters(F) == 0
    @test variables(F) == [ξ]
    @test isempty(parameters(F))
    @test variable_groups(F) == [[1]]

    # The wrapper is itself usable through the public mutating protocol with
    # ordinary Julia arrays; its inner callback receives the fixed values.
    u = zeros(ComplexF64, 1)
    x = ComplexF64[2]
    evaluate!(u, F, x, 0.5 + 0im)
    @test u[1] == 0

    U = zeros(ComplexF64, 1, 1)
    evaluate_and_jacobian!(u, U, F, x, 0.5 + 0im)
    @test u[1] == 0
    @test U[1, 1] == 1
    taylor!(u, Val(1), F, x, 0.5 + 0im)
    @test u[1] == 2

    # Most importantly, generic downstream methods survive the monomorphic
    # evaluator firewall and the threaded cloning path without naming internals.
    r = solve(F, [[1.0]], Continuation(; show_progress = false), Threaded())
    @test nsolutions(r) == 1
    @test abs(only(solutions(r))[1] - 3) < 1.0e-10
end
''')
