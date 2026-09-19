using Test
using HomotopyContinuation

function same_solution_set(a, b; atol = 1.0e-8)
    length(a) == length(b) || return false
    return all(sa -> any(sb -> maximum(abs.(sa .- sb)) < atol, b), a)
end

@testset "public solve workflows" begin
    @testset "basic solve and result inspection" begin
        @var x y
        F = System([x^2 - 1, y^2 - 4])
        result = solve(F, TotalDegree(; seed = UInt32(0x44), show_progress = false), Serial())

        @test nsolutions(result) == 4
        @test nfailed(result) == 0
        @test length(real_solutions(result)) == 4
        for sol in solutions(result)
            @test abs(sol[1]^2 - 1) < 1.0e-8
            @test abs(sol[2]^2 - 4) < 1.0e-8
        end
    end

    @testset "explicit total-degree and polyhedral routes agree" begin
        @var x y
        F = System([x^2 - 1, y^2 - 4])
        td = solve(F, TotalDegree(; seed = UInt32(0x45), show_progress = false), Serial())
        ph = solve(F, Polyhedral(; seed = UInt32(0x45), show_progress = false), Serial())

        @test nsolutions(td) == 4
        @test nsolutions(ph) == 4
        @test all(s -> any(t -> isapprox(s, t; atol = 1.0e-8), solutions(ph)), solutions(td))
    end

    @testset "reuse a parameterized System across target parameters" begin
        @var x y a b
        F = System([x^2 - a, y - b]; variables = [x, y], parameters = [a, b])
        p₀ = ComplexF64[1, 2]
        starts = solutions(
            solve(
                fix_parameters(F, p₀),
                TotalDegree(; seed = UInt32(0x46), show_progress = false),
                Serial(),
            ),
        )
        targets = [[4.0, 3.0], [9.0, -1.0]]
        results = solve(F, starts, p₀, targets, Sweep(; show_progress = false), Serial())

        @test length(results) == length(targets)
        for ((result, p), target) in zip(results, targets)
            @test p == target
            @test nsolutions(result) == 2
            for sol in solutions(result)
                @test abs(sol[1]^2 - target[1]) < 1.0e-7
                @test abs(sol[2] - target[2]) < 1.0e-7
            end
        end
    end
end

@testset "public solver routes" begin
    @polyvar x y
    equations = [x^2 + y - 1, x * y - 0.5]

    interpreted = System(equations; variables = [x, y], compile = CompileMode.INTERPRETED)
    compiled = System(equations; variables = [x, y], compile = CompileMode.COMPILED)
    algorithm = TotalDegree(; seed = UInt32(0x1234), show_progress = false)

    serial = solve(interpreted, algorithm, Serial())
    generated = solve(compiled, algorithm, Serial())
    threaded = solve(interpreted, algorithm, Threaded())

    @test nfailed(serial) == 0
    @test nsolutions(serial) == nsolutions(generated) == nsolutions(threaded)
    @test same_solution_set(solutions(serial), solutions(generated))
    @test same_solution_set(solutions(serial), solutions(threaded))
end

@testset "public overdetermined route" begin
    @polyvar x y
    F = System([x^2 - 1, y^2 - 1, x * y - 1]; variables = [x, y])
    result = solve(
        F, TotalDegree(; seed = UInt32(0x42), show_progress = false), Serial(),
    )

    @test nfailed(result) == 0
    @test nsolutions(result) == 2
    @test same_solution_set(
        solutions(result),
        [ComplexF64[1, 1], ComplexF64[-1, -1]],
    )
end

@testset "public subspace and witness routes" begin
    @polyvar x y
    F = System([x^2 + y^2 - 5]; variables = [x, y])
    L = LinearSubspace(reshape([1.0, 0.0], 1, 2), [1.0])

    sliced = solve(
        F, L, TotalDegree(; seed = UInt32(0x1234), show_progress = false), Serial(),
    )
    @test nsolutions(sliced) == 2
    @test all(abs(s[1] - 1) < 1.0e-8 for s in solutions(sliced))

    W = solve(F, Witness(; seed = UInt32(0x1234), show_progress = false), Serial())
    @test dim(W) == 1
    @test degree(W) == 2
    @test length(solutions(W)) == 2
    @test trace_test(W) < 1.0e-7
end

@testset "public monodromy and completeness routes" begin
    @polyvar u[1:2] p[1:2]
    F = System(
        [u[1]^2 + u[2]^2 - p[1], u[1] + u[2] - p[2]];
        variables = u, parameters = p,
    )

    result = solve(
        F,
        Monodromy(;
            target_solutions_count = 2,
            seed = UInt32(21),
            show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(result) == 2
    @test is_success(result)
    @test verify_solution_completeness(
        F, result, Monodromy(; show_progress = false),
    ) == Completeness.COMPLETE
end

@testset "public lazy result route" begin
    @polyvar x y
    F = System([x^2 + y^2 - 5]; variables = [x, y])
    L = LinearSubspace(reshape([1.0, 0.0], 1, 2), [1.0])

    first_result = first(
        result_iterator(
            F, L, TotalDegree(; seed = UInt32(0x1234), show_progress = false),
        )
    )
    @test is_success(first_result)

    result = Result(
        result_iterator(
            F, L, TotalDegree(; seed = UInt32(0x1234), show_progress = false),
        )
    )
    @test nsolutions(result) == 2
end

@testset "public solution and parameter I/O" begin
    expected_solutions = [ComplexF64[1 + 2im, -3], ComplexF64[0.5, 4 - im]]
    expected_parameters = ComplexF64[2, -3.2 + 2im]

    mktempdir() do dir
        solutions_path = joinpath(dir, "solutions.txt")
        parameters_path = joinpath(dir, "parameters.txt")

        write_solutions(solutions_path, expected_solutions)
        write_parameters(parameters_path, expected_parameters)

        @test read_solutions(solutions_path) == expected_solutions
        @test read_parameters(parameters_path) == expected_parameters
    end
end
