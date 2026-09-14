using Test
using HomotopyContinuation
import HomotopyContinuation:
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
