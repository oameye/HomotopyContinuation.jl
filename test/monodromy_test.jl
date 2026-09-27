using Test, Random
using HomotopyContinuation
using DynamicPolynomials: @polyvar

@testset "Monodromy" begin
    @testset "find_start_pair" begin
        Random.seed!(0xf00d)
        @polyvar y[1:2] p[1:2]
        F = System(
            [y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]];
            variables = y,
            parameters = p,
        )
        pair = find_start_pair(F)
        @test pair.found
        @test abs(pair.x[1]^2 + pair.x[2]^2 - pair.p[1]) < 1.0e-10
        @test abs(pair.x[1] + pair.x[2] - pair.p[2]) < 1.0e-10

        @polyvar q[1:2]
        G = System(
            [y[1]^2 * q[1]^2 + y[2] * q[2]^3 - 1, y[1] + y[2] - q[1]];
            variables = y,
            parameters = q,
        )
        pair2 = find_start_pair(G)
        @test pair2.found
        @test abs(pair2.x[1]^2 * pair2.p[1]^2 + pair2.x[2] * pair2.p[2]^3 - 1) < 1.0e-8
        @test abs(pair2.x[1] + pair2.x[2] - pair2.p[1]) < 1.0e-8

        @polyvar u
        pair3 = find_start_pair(System([u^2 - 4]; variables = [u]))
        @test pair3.found
        @test isempty(pair3.p)
        @test abs(pair3.x[1]^2 - 4) < 1.0e-8
    end

    @testset "known two-sheeted family and permutations" begin
        @polyvar y[1:2] p[1:2]
        F = System(
            [y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]];
            variables = y,
            parameters = p,
        )

        result = solve(
            F,
            Monodromy(; permutations = true, seed = UInt32(4242), show_progress = false),
            Serial(),
        )
        @test nsolutions(result) == 2
        @test is_success(result) || is_heuristic_stop(result)
        for x in solutions(result)
            p0 = parameters(result)
            @test abs(x[1]^2 + x[2]^2 - p0[1]) < 1.0e-8
            @test abs(x[1] + x[2] - p0[2]) < 1.0e-8
        end

        perm = permutations(result)
        @test size(perm, 1) == 2
        @test all(j -> sort(perm[:, j]) == [1, 2], axes(perm, 2))
        @test any(j -> perm[:, j] == [2, 1], axes(perm, 2))

        counted = solve(
            F,
            Monodromy(; target_solutions_count = 2, seed = UInt32(7), show_progress = false),
            Serial(),
        )
        @test nsolutions(counted) == 2
        @test is_success(counted)
    end

    @testset "public start-solution routes" begin
        @polyvar y[1:2] p[1:2]
        F = System(
            [y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]];
            variables = y,
            parameters = p,
        )
        p0 = ComplexF64[3, 1]
        base = solve(
            System([y[1]^2 + y[2]^2 - 3, y[1] + y[2] - 1]; variables = y),
            TotalDegree(; seed = UInt32(5), show_progress = false),
        )
        @test nsolutions(base) == 2

        from_result = solve(
            F,
            base,
            p0,
            Monodromy(; target_solutions_count = 2, seed = UInt32(5), show_progress = false),
            Serial(),
        )
        @test nsolutions(from_result) == 2

        from_one = solve(
            F,
            [solutions(base)[1]],
            p0,
            Monodromy(;
                target_solutions_count = 2,
                max_loops_no_progress = 50,
                seed = UInt32(5),
                show_progress = false,
            ),
            Serial(),
        )
        @test nsolutions(from_one) == 2
    end

    @testset "group actions change orbit counting, not the solved fiber" begin
        @polyvar u q
        F = System([u^2 - q]; variables = [u], parameters = [q])

        quotient = solve(
            F,
            [[2.0 + 0im]],
            [4.0 + 0im],
            Monodromy(;
                group_action = s -> ([-s[1]],),
                seed = UInt32(100),
                show_progress = false,
            ),
            Serial(),
        )
        ordinary = solve(
            F,
            [[2.0 + 0im]],
            [4.0 + 0im],
            Monodromy(; seed = UInt32(100), show_progress = false),
            Serial(),
        )

        @test nsolutions(quotient) == 1
        @test nsolutions(ordinary) == 2
        @test all(x -> abs(x[1]^2 - 4) < 1.0e-8, solutions(ordinary))
    end

    @testset "subspace monodromy and trace test" begin
        @polyvar z[1:3]
        Q = System(
            [z[1]^2 + 2z[2]^2 + 3z[3]^2 + z[1] * z[2] - 1];
            variables = z,
        )
        result = solve(
            Q,
            Monodromy(; dim = 2, seed = UInt32(99), show_progress = false),
            Serial(),
        )
        @test nsolutions(result) == 2
        @test is_success(result)
        @test !isnan(trace(result))
        @test trace(result) < 1.0e-10
        for x in solutions(result)
            @test abs(x[1]^2 + 2x[2]^2 + 3x[3]^2 + x[1] * x[2] - 1) < 1.0e-8
        end

        @test_throws ArgumentError solve(
            Q,
            Monodromy(; dim = 1, seed = UInt32(99), show_progress = false),
            Serial(),
        )
    end

    @testset "Serial and Threaded recover the same four-sheeted fiber" begin
        @polyvar y[1:2] q[1:2]
        F = System(
            [y[1]^2 + y[2]^2 - q[1]^2, y[1] * y[2] - q[2]^3];
            variables = y,
            parameters = q,
        )
        algorithm = Monodromy(;
            seed = UInt32(123),
            show_progress = false,
            target_solutions_count = 4,
            max_loops_no_progress = 50,
        )
        serial = solve(F, algorithm, Serial())
        threaded = solve(F, algorithm, Threaded())

        @test nsolutions(serial) == nsolutions(threaded) == 4
        for x in solutions(serial)
            @test any(y -> maximum(abs.(x .- y)) < 1.0e-8, solutions(threaded))
        end
    end

    @testset "solution completeness" begin
        @polyvar y[1:2] p[1:2]
        F = System(
            [y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]];
            variables = y,
            parameters = p,
        )
        result = solve(
            F,
            Monodromy(; target_solutions_count = 2, seed = UInt32(21), show_progress = false),
            Serial(),
        )
        @test verify_solution_completeness(F, result, Monodromy(; show_progress = false)) ==
            Completeness.COMPLETE

        incomplete = verify_solution_completeness(
            F,
            [solutions(result)[1]],
            Vector(parameters(result)),
            Monodromy(; show_progress = false),
        )
        @test incomplete != Completeness.COMPLETE
    end

    @testset "trace discriminates a complete slice" begin
        @polyvar z[1:3]
        Q = System(
            [z[1]^2 + 2z[2]^2 + 3z[3]^2 + z[1] * z[2] - 1];
            variables = z,
        )
        result = solve(
            Q,
            Monodromy(; dim = 2, seed = UInt32(202), show_progress = false),
            Serial(),
        )
        @test is_success(result)
        @test trace(result) < 1.0e-6
    end

    @testset "loop reuse policies and heuristic stopping" begin
        @polyvar y[1:2] p[1:2]
        F = System(
            [y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]];
            variables = y,
            parameters = p,
        )
        for reuse in (ReuseLoops.ALL, ReuseLoops.RANDOM, ReuseLoops.NONE)
            result = solve(
                F,
                Monodromy(;
                    reuse_loops = reuse,
                    target_solutions_count = 2,
                    seed = UInt32(303),
                    show_progress = false,
                ),
                Serial(),
            )
            @test nsolutions(result) == 2
        end

        stopped = solve(
            F,
            Monodromy(;
                max_loops_no_progress = 2,
                seed = UInt32(300),
                show_progress = false,
            ),
            Serial(),
        )
        @test nsolutions(stopped) == 2
        @test is_heuristic_stop(stopped)
    end

    @testset "duplicate-check public modes" begin
        @polyvar y[1:2] p[1:2]
        F = System(
            [y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]];
            variables = y,
            parameters = p,
        )

        heuristic = solve(
            F,
            Monodromy(; target_solutions_count = 2, seed = UInt32(404), show_progress = false),
            Serial(),
        )
        @test nsolutions(heuristic) == 2
        @test ncertified_distinct(heuristic) == 0
        @test ndiscarded_uncertified(heuristic) == 0

        @test_throws TypeError MonodromyOptions(; duplicate_check = :certified)
        @test_throws ArgumentError solve(
            F,
            Monodromy(;
                duplicate_check = DuplicateCheck.CERTIFIED,
                show_progress = false,
            ),
            Serial(),
        )
    end
end
