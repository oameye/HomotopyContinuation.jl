using Test, Random
using HomotopyContinuation
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: MultivariatePolynomials as MP
using MultivariatePolynomials: differentiate as mp_diff
using HomotopyContinuation: TaylorVector, ComplexDF64
using FixedSizeArrays: FixedSizeVectorDefault

const MODES = (CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL)

@static if VERSION < v"1.11"
    buffer(v::AbstractVector{T}) where {T} = Vector{T}(v)
else
    buffer(v::AbstractVector{T}) where {T} = FixedSizeVectorDefault{T}(v)
end

function eval_mp(polys, vars, x, params = Any[], p = ComplexF64[])
    return ComplexF64[
        isempty(params) ? f(vars => x) : f(vars => x, params => p) for f in polys
    ]
end

function eval_mp_jacobian(polys, vars, x, params = Any[], p = ComplexF64[])
    J = zeros(ComplexF64, length(polys), length(vars))
    for j in eachindex(vars), i in eachindex(polys)
        df = mp_diff(polys[i], vars[j])
        J[i, j] = isempty(params) ? df(vars => x) : df(vars => x, params => p)
    end
    return J
end

function test_eval_and_jac(polys, vars; params = nothing, npoints = 3)
    params = isnothing(params) ? eltype(vars)[] : params
    rng = MersenneTwister(0x00c011ec + length(polys) + length(vars))
    for mode in MODES
        F = System(polys; variables = vars, parameters = params, compile = mode)
        scales = equation_scales(F)
        for _ in 1:npoints
            x = randn(rng, ComplexF64, length(vars))
            p = randn(rng, ComplexF64, length(params))
            truth_u = eval_mp(polys, vars, x, params, p)
            truth_J = eval_mp_jacobian(polys, vars, x, params, p)
            if isempty(params)
                @test evaluate(F, x) .* scales ≈ truth_u rtol = 1.0e-10
                @test jacobian(F, x) .* scales ≈ truth_J rtol = 1.0e-10
            else
                @test evaluate(F, x, p) .* scales ≈ truth_u rtol = 1.0e-10
                @test jacobian(F, x, p) .* scales ≈ truth_J rtol = 1.0e-10
            end
        end
    end
    return nothing
end

@testset "polynomial input public behavior" begin
    @testset "evaluation and Jacobian edge cases" begin
        @polyvar x y z
        test_eval_and_jac([x^3 - 1], [x])
        test_eval_and_jac([x + y, x * y, x - y + 3], [x, y])
        test_eval_and_jac([x^7 - x^3 + x - 1], [x])
        test_eval_and_jac([x^2 + y^2, x^2 - y^2], [x, y])
        test_eval_and_jac(
            [2x + 3y - z + 1, x - y + 2z - 3, -x + y + z],
            [x, y, z],
        )
        test_eval_and_jac([x^2 * y + y^2 * z + z^2 * x - 1], [x, y, z])
        test_eval_and_jac([-x * y, -x^2 - y^2 + 1, -2x + 3y - 1], [x, y])
        test_eval_and_jac(
            [(1.0 + 2.0im) * x + (3.0 - 1.0im) * y, x * y - (2.0im) * x],
            [x, y],
        )
        test_eval_and_jac(
            [x^2 + (1.0im) * x * y + y^2, 2x - (3.0 + 1.0im) * y + 1],
            [x, y],
        )
        test_eval_and_jac(
            [x * y * z + x + y, x * y * z + y + z, x * y * z + z + x],
            [x, y, z],
        )
        test_eval_and_jac([2x * y, 2x * y], [x, y])
        test_eval_and_jac([2x + 2y, 2x * y, 2x^2 + 2y^2], [x, y])
        # Exponents from 4 up, zero and constant outputs, bare variables as outputs,
        # and a complex coefficient on a product.
        test_eval_and_jac([x^5 - y^7 + 3x^4 * y^4], [x, y])
        test_eval_and_jac([-x + y, x - y], [x, y])
        test_eval_and_jac([x^2 - 1, 0 * x + 2], [x, y])
        # The zero equation has no coefficients, and normalization reduces over them.
        @test_broken evaluate(System([x^2 - 1, 0 * x]; variables = [x, y]), [1, 2]) ==
            ComplexF64[0, 0]
        test_eval_and_jac([x * y, x * y, x^2], [x, y])
        test_eval_and_jac([x + 0 * y, y + 0 * x], [x, y])
        test_eval_and_jac([(2.0 + 3.0im) * x^2 * y - (0.5 - 0.25im), x - y], [x, y])
        @polyvar p
        test_eval_and_jac([x + 0 * y, p + 0 * x], [x, y]; params = [p])
    end

    @testset "parameters participate in values but not variable Jacobian columns" begin
        @polyvar x y z a b
        vars = [x, y, z]
        params = [a, b]
        polys = [a * x^2 + b * y * z, x * y - a * z^2 + b, x + y + z - a * b]
        test_eval_and_jac(polys, vars; params = params, npoints = 5)

        F = System(polys; variables = vars, parameters = params)
        @test nvariables(F) == 3
        @test nparameters(F) == 2
        @test variables(F) == vars
        @test parameters(F) == params
        @test size(jacobian(F, ComplexF64[1, 2, 3], ComplexF64[4, 5])) == (3, 3)
    end

    @testset "variable declarations are validated" begin
        @polyvar x y
        @polyvar u1 u2 u3 u4
        big = [u1^2 + u2 * u3 - 1, u1 + u2^2 - u3, u1 * u2 * u3 - u4, u4^2 - u1 - 1]
        @test_throws ArgumentError System([x^2 + y - 1, x + y^2 - 1]; variables = [x])
        @test_throws ArgumentError System(big; variables = [u1, u2, u3])
        @test_throws ArgumentError System(big; variable_groups = [[u1, u2], [u3]])
    end

    @testset "real benchmark systems agree with polynomial ground truth" begin
        @polyvar x0 x1 x2 x3
        katsura = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        test_eval_and_jac(katsura, [x0, x1, x2, x3]; npoints = 5)

        @polyvar c[1:5]
        cyclic5 = [
            sum(c),
            sum(c[i] * c[mod1(i + 1, 5)] for i in 1:5),
            sum(c[i] * c[mod1(i + 1, 5)] * c[mod1(i + 2, 5)] for i in 1:5),
            sum(
                c[i] * c[mod1(i + 1, 5)] * c[mod1(i + 2, 5)] * c[mod1(i + 3, 5)] for
                    i in 1:5
            ),
            prod(c) - 1,
        ]
        test_eval_and_jac(cyclic5, collect(c); npoints = 5)

        @polyvar u v
        test_eval_and_jac(
            [u^3 - 3u * v^2 + v^3, -u^3 + u * v^2 - v],
            [u, v];
            npoints = 5,
        )
    end

    # The mutating entry points of a `System` are those of the parameter homotopy
    # through it; a parameter-free one has H(x, t) = F(x) for every t.
    @testset "Taylor coefficients agree with polynomial substitution" begin
        @polyvar x y λ
        cases = (
            [x^3 + x * y - 2, y^2 - x],
            [x^5 - y^7 + 3x^4 * y^4, (2.0 + 3.0im) * x^2 * y - (0.5 - 0.25im)],
            [2x * y, 2x * y],
        )
        rng = MersenneTwister(0x07a1)
        for polys in cases, mode in MODES
            F = System(polys; variables = [x, y], compile = mode)
            H = ParameterHomotopy(F, ComplexF64[], ComplexF64[])
            scales = equation_scales(F)
            X = 0.5 .* randn(rng, ComplexF64, 4, 2)
            u = buffer(zeros(ComplexF64, 2))
            for K in 2:3
                tx = TaylorVector{K + 1, ComplexF64}(2)
                tx[1] = Tuple(X[1:(K + 1), 1])
                tx[2] = Tuple(X[1:(K + 1), 2])
                taylor!(u, Val(K), H, tx, complex(0.0))
                series = [sum(X[k + 1, i] * λ^k for k in 0:K) for i in 1:2]
                truth = [MP.coefficient(f([x, y] => series), λ^K) for f in polys]
                @test u .* scales ≈ truth rtol = 1.0e-10
            end
        end
    end

    @testset "double-double evaluation agrees with the Float64 value" begin
        @polyvar x y
        rng = MersenneTwister(0xdf)
        for polys in (
                [x^3 + x * y - 2, y^2 - x],
                [(1.0 + 2.0im) * x^2 + (3.0 - 1.0im) * y, x * y - (2.0im)],
            )
            F = System(polys; variables = [x, y])
            H = ParameterHomotopy(F, ComplexF64[], ComplexF64[])
            u = buffer(zeros(ComplexF64, 2))
            for _ in 1:3
                xv = randn(rng, ComplexF64, 2)
                evaluate!(u, H, buffer(ComplexDF64.(xv)), complex(0.0))
                @test u .* equation_scales(F) ≈ eval_mp(polys, [x, y], xv) rtol = 1.0e-14
            end
        end
    end
end
