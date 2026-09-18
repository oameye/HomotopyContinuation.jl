@testset "Public API acceptance: solve workflows" begin
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
