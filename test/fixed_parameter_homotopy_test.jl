using Test
using HomotopyContinuationNext
import HomotopyContinuationNext as HCN
using HomotopyContinuationNext: AbstractHomotopy, FixedParameterHomotopy, HomotopyEvaluator,
    FSVec, FSMat, TaylorVector, ComplexDF64, evaluate!, evaluate_and_jacobian!, taylor!,
    fix_parameters, solve, solutions, nsolutions, Continuation, Serial, _clone_homotopy

@var ξ

# H(x,t;p) = x - [p + (1-p)t].  Its root moves exactly from x=1 at t=1
# to x=p at t=0, making all parameter-binding and Taylor expectations explicit.
struct _ParametricLineHomotopy <: AbstractHomotopy end
Base.size(::_ParametricLineHomotopy) = (1, 1)
HCN.variables(::_ParametricLineHomotopy) = [ξ]
HCN.variable_groups(::_ParametricLineHomotopy) = [[1]]
HCN.nparameters(::_ParametricLineHomotopy) = 1

function HCN.evaluate!(
        u::FSVec{ComplexF64}, ::_ParametricLineHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64, p::Vector{ComplexF64},
    )
    u[1] = x[1] - (p[1] + (1 - p[1]) * t)
    return nothing
end

function HCN.evaluate!(
        u::FSVec{ComplexF64}, ::_ParametricLineHomotopy,
        x::FSVec{ComplexDF64}, t::ComplexF64, p::Vector{ComplexF64},
    )
    u[1] = ComplexF64(x[1]) - (p[1] + (1 - p[1]) * t)
    return nothing
end

function HCN.evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64}, H::_ParametricLineHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64, p::Vector{ComplexF64},
    )
    evaluate!(u, H, x, t, p)
    U[1, 1] = 1
    return nothing
end

function HCN.taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, ::_ParametricLineHomotopy,
        ::FSVec{ComplexF64}, ::ComplexF64, p::Vector{ComplexF64},
    )
    u[1] = p[1] - 1
    return nothing
end

for (K, N) in ((2, 3), (3, 4))
    @eval function HCN.taylor!(
            u::FSVec{ComplexF64}, ::Val{$K}, ::_ParametricLineHomotopy,
            tx::TaylorVector{$N, ComplexF64}, ::ComplexF64, ::Vector{ComplexF64},
        )
        u[1] = tx[1][$K]
        return nothing
    end
end

HCN.set_solution!(
    x::FSVec{ComplexF64}, ::_ParametricLineHomotopy,
    y::FSVec{ComplexF64}, ::ComplexF64,
) = (copyto!(x, y); nothing)

HCN.get_solution!(
    out::FSVec{ComplexF64}, ::_ParametricLineHomotopy,
    x::FSVec{ComplexF64}, ::ComplexF64,
) = (copyto!(out, x); nothing)

@testset "FixedParameterHomotopy" begin
    H = _ParametricLineHomotopy()
    F = fix_parameters(H, [3.0])
    @test F isa FixedParameterHomotopy
    @test size(F) == (1, 1)
    @test F.parameters isa Vector{ComplexF64}
    @test F.parameters == ComplexF64[3]
    @test HCN.nvariables(F) == 1
    @test HCN.nparameters(F) == 0
    @test HCN.variables(F) == [ξ]
    @test isempty(HCN.parameters(F))
    @test HCN.variable_groups(F) == [[1]]

    # The concrete wrapper feeds a parameter-free interface to HomotopyEvaluator.
    E = HomotopyEvaluator(F)
    u = FSVec{ComplexF64}(zeros(ComplexF64, 1))
    x = FSVec{ComplexF64}(ComplexF64[2])
    evaluate!(u, E, x, 0.5 + 0im)
    @test u[1] == 0

    U = FSMat{ComplexF64}(zeros(ComplexF64, 1, 1))
    evaluate_and_jacobian!(u, U, E, x, 0.5 + 0im)
    @test u[1] == 0
    @test U[1, 1] == 1

    # Extended-precision residual input uses the same bound parameter values.
    xdf = FSVec{ComplexDF64}(ComplexDF64.(ComplexF64[2]))
    evaluate!(u, E, xdf, 0.5 + 0im)
    @test u[1] == 0

    taylor!(u, Val(1), E, x, 0.5 + 0im)
    @test u[1] == 2
    for K in 2:3
        tx = TaylorVector{K + 1, ComplexF64}(1)
        tx.data .= 0
        tx.data[K + 1, 1] = 0.25K
        taylor!(u, Val(K), E, tx, 0.5 + 0im)
        @test u[1] == 0.25K
    end

    clone = _clone_homotopy(F)
    @test clone isa FixedParameterHomotopy
    @test clone !== F
    @test clone.homotopy !== F.homotopy || isbitstype(typeof(H))
    @test clone.parameters == F.parameters
    @test clone.parameters !== F.parameters

    # Most importantly, the wrapper is accepted by the ordinary monomorphic
    # homotopy-tracking route and follows the expected root from 1 to p.
    r = solve(F, [[1.0]], Continuation(; show_progress = false), Serial())
    @test nsolutions(r) == 1
    @test abs(only(solutions(r))[1] - 3) < 1.0e-10
end
