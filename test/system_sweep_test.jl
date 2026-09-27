using Test, Random
using HomotopyContinuation
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: MultivariatePolynomials as MP

include("test_systems.jl")
include("minors_polys.jl")

const MODES = (CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL)

_at(f, vars, x, params, p) = isempty(params) ? f(vars => x) : f(vars => x, params => p)

mp_eval(polys, vars, x, params, p) =
    ComplexF64[_at(f, vars, x, params, p) for f in polys]

function mp_jacobian(polys, vars, x, params, p)
    J = zeros(ComplexF64, length(polys), length(vars))
    for j in eachindex(vars), i in eachindex(polys)
        J[i, j] = _at(MP.differentiate(polys[i], vars[j]), vars, x, params, p)
    end
    return J
end

_public_evaluate(F, x, p) = isempty(p) ? evaluate(F, x) : evaluate(F, x, p)
_public_jacobian(F, x, p) = isempty(p) ? jacobian(F, x) : jacobian(F, x, p)

const TEST_SYSTEMS = [(name, build()...) for (name, build) in TEST_SYSTEM_COLLECTION]

@testset "System public sweep: $name" for (name, polys, vars, params) in TEST_SYSTEMS
    rng = MersenneTwister(0x00051ee7 + length(name))
    m, n, r = length(polys), length(vars), length(params)
    xvals = randn(rng, ComplexF64, n)
    pvals = randn(rng, ComplexF64, r)
    truth_u = mp_eval(polys, vars, xvals, params, pvals)
    truth_J = mp_jacobian(polys, vars, xvals, params, pvals)

    @testset "$mode" for mode in MODES
        F = System(polys; variables = vars, parameters = params, compile = mode)
        @test size(F) == (m, n)
        @test nvariables(F) == n
        @test nparameters(F) == r
        @test collect(variables(F)) == collect(vars)
        @test collect(parameters(F)) == collect(params)
        @test _public_evaluate(F, xvals, pvals) ≈ truth_u rtol = 1.0e-10
        @test _public_jacobian(F, xvals, pvals) ≈ truth_J rtol = 1.0e-10
    end
end
