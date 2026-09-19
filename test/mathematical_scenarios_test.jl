using Test
using HomotopyContinuation

function _same_solution_set(a, b; atol = 1.0e-8)
    length(a) == length(b) || return false
    return all(sa -> any(sb -> maximum(abs.(sa .- sb)) < atol, b), a)
end

@testset "mathematical scenarios across numerical backends" begin
    @testset "Katsura-3 is backend independent" begin
        @polyvar x0 x1 x2 x3
        equations = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        systems = [
            System(equations; compile = mode) for mode in
                (CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL)
        ]

        probe = ComplexF64[0.3, 0.5, -0.2, 0.1]
        values = evaluate.(systems, Ref(probe))
        jacobians = jacobian.(systems, Ref(probe))
        @test all(v -> isapprox(v, first(values); atol = 1.0e-13), values)
        @test all(J -> isapprox(J, first(jacobians); atol = 1.0e-13), jacobians)

        algorithm = TotalDegree(; seed = UInt32(0x5101), show_progress = false)
        results = [solve(F, algorithm, Serial()) for F in systems]
        @test all(r -> nfailed(r) == 0, results)
        @test all(r -> nsolutions(r) == nsolutions(first(results)), results)
        @test all(r -> _same_solution_set(solutions(r), solutions(first(results))), results)
        for (F, result) in zip(systems, results), sol in solutions(result)
            @test maximum(abs.(evaluate(F, sol))) < 1.0e-8
        end
    end

    @testset "two-parameter quadratic family preserves its algebraic invariants" begin
        @polyvar x y a b
        F = System(
            [x^2 + y^2 - a, x * y - b];
            variables = [x, y], parameters = [a, b],
        )
        p0 = ComplexF64[5, 2]
        starts = solutions(
            solve(
                fix_parameters(F, p0),
                TotalDegree(; seed = UInt32(0x5102), show_progress = false),
                Serial(),
            )
        )
        @test length(starts) == 4

        targets = [ComplexF64[10, 3], ComplexF64[13, 6]]
        tracked = solve(F, starts, p0, targets, Sweep(; show_progress = false), Serial())
        @test length(tracked) == length(targets)
        for ((result, p), target) in zip(tracked, targets)
            @test p == target
            @test nfailed(result) == 0
            @test nsolutions(result) == 4
            for sol in solutions(result)
                @test abs(sol[1]^2 + sol[2]^2 - target[1]) < 1.0e-8
                @test abs(sol[1] * sol[2] - target[2]) < 1.0e-8
            end
        end
    end

    @testset "ill-conditioned root is recovered with extended precision" begin
        @polyvar x y
        eps = 1.0e-8
        F = System([x + y - 2, x + (1 + eps) * y - (2 + eps)])
        result = newton(F, ComplexF64[1.2, 0.8]; extended_precision = true)

        @test is_success(result)
        @test result.residual < 1.0e-12
        @test isapprox(result.x, ComplexF64[1, 1]; atol = 1.0e-7)
    end

    @testset "transcendental Newton problem agrees across compile modes" begin
        @var x y
        equations = [exp(x) + sin(y) - 1, x + cos(y) - 1]
        systems = [
            System(equations; variables = [x, y], compile = mode) for mode in
                (CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL)
        ]
        x0 = ComplexF64[0.1, -0.1]

        for F in systems
            result = newton(F, x0)
            @test is_success(result)
            @test maximum(abs.(evaluate(F, result.x))) < 1.0e-10
            @test maximum(abs.(result.x)) < 1.0e-8
        end
    end
end
