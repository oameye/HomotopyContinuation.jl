@testset "Solver" begin
    @testset "constructor" begin
        @polyvar x y z
        f = [x^2 - 2, x + y - 1]
        solver, starts = solver_startsolutions(f; system = FPSystem)
        @test isa(solver, Solver)
        result = solve!(solver, starts)
        @test all(is_success, result)
    end

    @testset "path jumping" begin
        solver, starts = solver_startsolutions(
            equations(katsura(5));
            system = FPSystem,
            seed = 124232,
            max_corrector_iters = 5,
            accuracy = 1e-3,
        )
        result_jumping = solve!(solver, starts; path_jumping_check = false)
        @test nsolutions(result_jumping) < 32

        result = solve!(solver, starts; path_jumping_check = true)
        @test nsolutions(result) == 32
        @test all(is_nonsingular, result)
        @test all(is_success, result)

        # check that path_jumping_check is on by default
        result2 = solve!(solver, starts)
        @test nsolutions(result2) == 32
    end
end
