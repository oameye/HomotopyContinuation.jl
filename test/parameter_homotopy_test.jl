using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar

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
end
