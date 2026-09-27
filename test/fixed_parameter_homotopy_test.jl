using Test
using HomotopyContinuation
import HomotopyContinuation:
    evaluate!, evaluate_and_jacobian!, taylor!, parameters, variables, variable_groups

@var ξ α

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
    @test variables(H) == [ξ]
    @test parameters(H) == [α]
    @test variable_groups(H) == [[1]]

    x = zeros(ComplexF64, 1)
    set_solution!(x, H, ComplexF64[2], 0.5 + 0im)
    @test x == ComplexF64[2]
    y = zeros(ComplexF64, 1)
    get_solution!(y, H, x, 0.5 + 0im)
    @test y == x

    @test_throws ArgumentError fix_parameters(H, ComplexF64[])
    @test_throws ArgumentError fix_parameters(H, [1, 2])
    H0 = _ParameterFreeHomotopy()
    @test nparameters(H0) == 0
    @test_throws ArgumentError fix_parameters(H0, [1])
end

@testset "FixedParameterHomotopy public behavior" begin
    H = _ParametricLineHomotopy()
    F = fix_parameters(H, [3.0])
    @test F isa FixedParameterHomotopy
    @test size(F) == (1, 1)
    @test nvariables(F) == 1
    @test nparameters(F) == 0
    @test variables(F) == [ξ]
    @test isempty(parameters(F))
    @test variable_groups(F) == [[1]]

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

    serial = solve(F, [[1.0]], Continuation(; show_progress = false), Serial())
    threaded = solve(F, [[1.0]], Continuation(; show_progress = false), Threaded())
    @test nsolutions(serial) == nsolutions(threaded) == 1
    @test abs(only(solutions(serial))[1] - 3) < 1.0e-10
    @test abs(only(solutions(threaded))[1] - 3) < 1.0e-10
end
