using Test
using HomotopyContinuation
using HomotopyContinuation: TaylorVector
using DynamicPolynomials: @polyvar, subs, differentiate
using MultivariatePolynomials: coefficient

@static if VERSION < v"1.11"
    buffer(A::AbstractArray) = collect(A)
else
    using FixedSizeArrays: FixedSizeArrayDefault
    buffer(A::AbstractArray) = FixedSizeArrayDefault(A)
end

function _parameter_solution_distance(a, b)
    isempty(a) && return Inf
    return maximum(minimum(maximum(abs.(x .- y)) for y in b) for x in a)
end

@testset "ParameterHomotopy public behavior" begin
    @testset "nonlinear parameter dependence reaches the target fiber" begin
        @polyvar x[1:2] p[1:2]
        F = System(
            [
                x[1]^2 * p[1]^2 + x[2] * p[2]^3 + p[1] * p[2] * x[1] * x[2] - 1,
                x[1] + x[2] - p[1],
            ];
            variables = x, parameters = p,
        )
        pstart = ComplexF64[1.0 + 0.2im, -0.7 + 1.1im]
        ptarget = ComplexF64[0.3 - 0.9im, 1.4 + 0.5im]
        seed = UInt32(0x5eed)

        start = solve(
            fix_parameters(F, pstart),
            TotalDegree(; seed, show_progress = false),
            Serial(),
        )
        target = solve(
            fix_parameters(F, ptarget),
            TotalDegree(; seed, show_progress = false),
            Serial(),
        )
        tracked = solve(
            F,
            start,
            pstart,
            ptarget,
            Continuation(; seed, show_progress = false),
            Serial(),
        )

        @test nsolutions(start) == nsolutions(target) == nsolutions(tracked) == 2
        @test _parameter_solution_distance(solutions(tracked), solutions(target)) < 1.0e-8
        for s in solutions(tracked)
            @test maximum(abs, evaluate(F, s, ptarget)) < 1.0e-8
        end
    end

    @testset "exact parameter tangent keeps a simple path short" begin
        @polyvar y q
        F = System([y^2 - q]; variables = [y], parameters = [q])
        result = solve(
            F,
            [[1.0 + 0.0im]],
            [1.0 + 0im],
            [9.0 + 0im],
            Continuation(; seed = UInt32(1), show_progress = false),
            Serial(),
        )
        @test nsolutions(result) == 1
        path = only(path_results(result))
        @test solution(path)[1] ≈ 3.0 atol = 1.0e-8
        @test accepted_steps(path) < 20
    end

    @testset "the same start fiber accepts vectors, Result, and ResultIterator" begin
        @polyvar w z c
        F = System([w^2 - c, z^2 - c]; variables = [w, z], parameters = [c])
        base_system = fix_parameters(F, [1.0])
        base = solve(base_system, TotalDegree(; show_progress = false), Serial())

        key(R) = sort(
            [
                sort([(round(real(v); digits = 8), round(imag(v); digits = 8)) for v in s])
                    for s in solutions(R)
            ]
        )
        reference = key(
            solve(
                F,
                solutions(base),
                [1.0],
                [4.0],
                Continuation(; show_progress = false),
                Serial(),
            )
        )
        @test length(reference) == 4
        @test key(
            solve(F, base, [1.0], [4.0], Continuation(; show_progress = false), Serial()),
        ) == reference
        @test key(
            solve(
                F,
                result_iterator(base_system),
                [1.0],
                [4.0],
                Continuation(; show_progress = false),
                Serial(),
            ),
        ) == reference

        @test length(
            solve(F, base, [1.0], [[4.0], [9.0]], Sweep(; show_progress = false), Serial()),
        ) == 2
        @test key(Result(result_iterator(F, base, [1.0], [4.0]))) == reference

        @test_throws ArgumentError solve(
            F,
            "not start solutions",
            [1.0],
            [4.0],
            Continuation(; show_progress = false),
            Serial(),
        )
    end

    @testset "public ParameterHomotopy validates and solves" begin
        @polyvar y q
        F = System([y^2 - q]; variables = [y], parameters = [q])
        H = ParameterHomotopy(F, [1.0], [9.0])
        result = solve(H, [[1.0]], Continuation(; show_progress = false), Serial())
        @test nsolutions(result) == 1
        @test solution(only(path_results(result)))[1] ≈ 3.0 atol = 1.0e-8

        @test_throws ArgumentError ParameterHomotopy(F, Float64[], [9.0])
        @test_throws ArgumentError ParameterHomotopy(F, [1.0], Float64[])
    end

    @testset "evaluate!, Jacobian and taylor! follow the parameter line" begin
        @polyvar x[1:2] p[1:2] s
        polys = [
            x[1]^2 * p[1]^2 + x[2] * p[2]^3 + p[1] * p[2] * x[1] * x[2] - 1,
            x[1] + x[2] - p[1],
        ]
        F = System(polys; variables = x, parameters = p)
        pstart = ComplexF64[1.0 + 0.2im, -0.7 + 1.1im]
        ptarget = ComplexF64[0.3 - 0.9im, 1.4 + 0.5im]
        H = ParameterHomotopy(F, pstart, ptarget)
        # H(x, t) = F(x, t * pstart + (1 - t) * ptarget).
        p_of(t) = t .* pstart .+ (1 .- t) .* ptarget
        xv = ComplexF64[0.4 + 0.3im, -1.2 + 0.1im]
        u = buffer(zeros(ComplexF64, 2))
        U = buffer(zeros(ComplexF64, 2, 2))

        for t in (complex(0.37), 0.2 + 0.6im)
            evaluate!(u, H, buffer(xv), t)
            @test u ≈ [ComplexF64(f(x => xv, p => p_of(t))) for f in polys] atol = 1.0e-13

            evaluate_and_jacobian!(u, U, H, buffer(xv), t)
            @test u ≈ [ComplexF64(f(x => xv, p => p_of(t))) for f in polys] atol = 1.0e-13
            @test U ≈ [
                ComplexF64(differentiate(f, xj)(x => xv, p => p_of(t)))
                    for f in polys, xj in x
            ] atol = 1.0e-13
        end

        # The order-K coefficient in λ of H(x₀ + x₁λ + … + x_Kλ^K, t₀ + λ),
        # computed symbolically.
        t0 = complex(0.37)
        X = ComplexF64[
            0.4 + 0.3im -1.2 + 0.1im
            0.7 - 0.2im 0.3 + 0.5im
            -0.1 + 0.6im 0.9 - 0.4im
            0.5 + 0.5im -0.3 - 0.8im
        ]
        taylor!(u, Val(1), H, buffer(xv), t0)
        @test u ≈ [
            ComplexF64(
                differentiate(subs(f, x => xv, p => p_of(t0) .+ s .* (pstart .- ptarget)), s)(s => 0),
            ) for f in polys
        ] atol = 1.0e-12
        for K in 2:3
            tx = TaylorVector{K + 1, ComplexF64}(2)
            for i in 1:2
                tx[i] = Tuple(X[1:(K + 1), i])
            end
            taylor!(u, Val(K), H, tx, t0)
            path = [sum(X[k + 1, i] * s^k for k in 0:K) for i in 1:2]
            params = p_of(t0) .+ s .* (pstart .- ptarget)
            expected = [ComplexF64(coefficient(f(x => path, p => params), s^K)) for f in polys]
            @test u ≈ expected atol = 1.0e-11
        end
    end
end
