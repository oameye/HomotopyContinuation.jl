using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar
using CommonSolve: CommonSolve

@testset "Solve" begin
    @testset "TotalDegree: basic solver semantics" begin
        @polyvar x y

        linear = solve(System([x - 2, y - 3]), TotalDegree(; show_progress = false))
        @test nsolutions(linear) == 1
        @test solutions(linear)[1] ≈ [2.0, 3.0] atol = 1.0e-8

        complete = solve(System([x^2 - 1, y^2 - 4]), TotalDegree(; show_progress = false))
        @test nsolutions(complete) == 4
        @test length(real_solutions(complete)) == 4
        for sol in real_solutions(complete)
            @test abs(sol[1]^2 - 1) < 1.0e-6
            @test abs(sol[2]^2 - 4) < 1.0e-6
        end

        complex_only =
            solve(System([x^2 + 1, y - 1]), TotalDegree(; show_progress = false))
        @test nreal(complex_only) == 0
        @test nsolutions(complex_only) == 2
    end

    @testset "TotalDegree: nonlinear residuals and reproducibility" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 0.5])

        result = solve(F, TotalDegree(; seed = UInt32(42), show_progress = false))
        for sol in solutions(result)
            @test abs(sol[1]^2 + sol[2] - 1) < 1.0e-6
            @test abs(sol[1] * sol[2] - 0.5) < 1.0e-6
        end

        repeated = solve(F, TotalDegree(; seed = UInt32(42), show_progress = false))
        @test nsolutions(result) == nsolutions(repeated)
        for a in solutions(result)
            @test any(b -> isapprox(a, b; atol = 1.0e-10), solutions(repeated))
        end
    end

    @testset "TotalDegree: Katsura-3" begin
        @polyvar x0 x1 x2 x3
        F = System(
            [
                x0 + 2x1 + 2x2 + 2x3 - 1,
                x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
                2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
                x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
            ],
        )
        result = solve(F, TotalDegree(; show_progress = false))
        @test nsolutions(result) >= 2
        for sol in solutions(result)
            residual = maximum(
                abs.(
                    [
                        sol[1] + 2sol[2] + 2sol[3] + 2sol[4] - 1,
                        sol[1]^2 + 2sol[2]^2 + 2sol[3]^2 + 2sol[4]^2 - sol[1],
                        2sol[1] * sol[2] + 2sol[2] * sol[3] + 2sol[3] * sol[4] - sol[2],
                        sol[2]^2 + 2sol[1] * sol[3] + 2sol[2] * sol[4] - sol[3],
                    ],
                ),
            )
            @test residual < 1.0e-6
        end
    end

    @testset "CommonSolve interface" begin
        @polyvar x y a

        td_cache = CommonSolve.init(System([x^2 - 1, y - 2]), TotalDegree())
        @test nsolutions(CommonSolve.solve!(td_cache)) == 2

        poly_cache = CommonSolve.init(System([x^2 - 1, y^2 - 4]), Polyhedral())
        @test nsolutions(CommonSolve.solve!(poly_cache)) == 4

        F = System([x^2 - a, y - 1]; parameters = [a])
        starts = solutions(solve(System([x^2 - 1, y - 1]), TotalDegree(; show_progress = false)))
        parameter_cache = CommonSolve.init(F, starts, [1.0], [4.0])
        @test nsolutions(CommonSolve.solve!(parameter_cache)) == 2
    end

    @testset "Result display" begin
        @polyvar x y
        result = solve(System([x - 1, y - 2]), TotalDegree(; show_progress = false))
        buf = IOBuffer()
        show(buf, result)
        text = String(take!(buf))
        @test contains(text, "tracked paths")
        @test contains(text, "solutions")
    end

    @testset "Tracker options are honored through the public solver" begin
        @polyvar x y
        result = solve(
            System([x^2 - 1, y^2 - 1]),
            TotalDegree(;
                tracker_options = TrackerOptions(; max_steps = 100),
                show_progress = false,
            ),
        )
        @test nsolutions(result) == 4
    end

    @testset "Polyhedral: sparse and dense path geometry" begin
        @polyvar x y
        sparse = System([x^2 + y - 1, x * y - 2])
        result = solve(sparse, Polyhedral(; show_progress = false))
        for sol in solutions(result)
            @test abs(sol[1]^2 + sol[2] - 1) < 1.0e-6
            @test abs(sol[1] * sol[2] - 2) < 1.0e-6
        end

        td = solve(sparse, TotalDegree(; show_progress = false))
        @test nsolutions(td) == nsolutions(result)

        dense = System(
            [
                x^2 + x * y + y^2 + x + y + 1,
                x^2 + 2 * x * y - y^2 + x - y + 2,
            ],
        )
        @test mixed_volume(dense) == 4
        @test paths_to_track(dense, TotalDegree()) == 4
    end

    @testset "Polyhedral: torus support" begin
        @polyvar x₁ x₂ s
        F = System(
            [
                x₁^3 * s^15 + x₁ * x₂ * s + x₂^3 + s^12,
                x₁^2 * s^9 + x₁ * x₂^2 + x₂ * s^3,
                x₁^2 * x₂ * s^5 + x₁ * s^8 + x₂^2,
            ],
        )
        all_alg = Polyhedral(; seed = UInt32(1), show_progress = false)
        torus_alg =
            Polyhedral(; only_torus = true, seed = UInt32(1), show_progress = false)

        @test paths_to_track(F, all_alg) == 92
        @test paths_to_track(F, torus_alg) == 54
        @test mixed_volume(F) == 54

        all_result = solve(F, all_alg, Serial())
        torus_result = solve(F, torus_alg, Serial())
        @test nsolutions(torus_result) <= nsolutions(all_result)
        @test all(sol -> all(!iszero, sol), solutions(torus_result))
        for sol in solutions(torus_result)
            @test any(
                candidate -> maximum(abs.(sol .- candidate)) < 1.0e-8,
                solutions(all_result),
            )
        end
    end

    @testset "Polyhedral and TotalDegree agree on complete root sets" begin
        @polyvar x y
        for F in (
                System([x^2 - 1, y^2 - 4]),
                System([x^2 + y^2 - 1, x * y - 0.25]),
            )
            td = solve(F, TotalDegree(; seed = UInt32(1), show_progress = false))
            ph = solve(F, Polyhedral(; seed = UInt32(1), show_progress = false))
            @test nsolutions(td) == nsolutions(ph)
            for sol in solutions(td)
                @test any(other -> isapprox(sol, other; atol = 1.0e-8), solutions(ph))
            end
        end
    end

    projective_normalize(sols) = [s ./ s[argmax(abs.(s))] for s in sols]
    function same_point_set(a, b)
        normalized_b = projective_normalize(b)
        return length(a) == length(b) && all(
            u -> any(v -> maximum(abs.(u .- v)) < 1.0e-7, normalized_b),
            projective_normalize(a),
        )
    end

    @testset "projective solve: chart-independent roots" begin
        @polyvar x y z
        F = System(
            [
                2.3 * x^2 + 1.2 * y^2 + 3x * z - 2y * z + 3 * z^2,
                2.3 * x^2 + 1.2 * y^2 + 5x * z + 2y * z - 5 * z^2,
            ],
        )
        run(alg, exec = Serial()) = solve(F, alg, exec)

        reference = run(TotalDegree(; seed = UInt32(0x1234)))
        for result in (reference, run(Polyhedral(; seed = UInt32(0x1234))))
            @test ntracked(result) == 4
            @test count(is_success, path_results(result)) == 4
            @test nsolutions(result) == 4
            for v in projective_normalize(solutions(result))
                @test abs(
                    2.3v[1]^2 + 1.2v[2]^2 + 3v[1] * v[3] - 2v[2] * v[3] +
                        3v[3]^2,
                ) < 1.0e-8
                @test abs(
                    2.3v[1]^2 + 1.2v[2]^2 + 5v[1] * v[3] + 2v[2] * v[3] -
                        5v[3]^2,
                ) < 1.0e-8
            end
        end

        @test count(
            s -> abs(s[3]) < 1.0e-8 * maximum(abs.(s)),
            solutions(reference),
        ) == 2

        for (alg, exec) in (
                (TotalDegree(; seed = UInt32(99)), Serial()),
                (Polyhedral(; seed = UInt32(99)), Serial()),
                (TotalDegree(; seed = UInt32(0x1234)), Threaded()),
            )
            @test same_point_set(solutions(run(alg, exec)), solutions(reference))
        end
    end

    @testset "projective solve: overdetermined after charting" begin
        @polyvar x y z
        F = System(
            [
                (x^2 + y^2 + x * y - 3 * z^2) * (x + 3z),
                (x^2 + y^2 + x * y - 3 * z^2) * (y - x + 2z),
                2x + 5y - 3z,
            ],
        )
        for alg in (
                TotalDegree(; seed = UInt32(0x1234), show_progress = false),
                Polyhedral(; seed = UInt32(0x1234), show_progress = false),
            )
            result = solve(F, alg, Serial())
            @test ntracked(result) == 9
            @test count(is_success, path_results(result)) == 2
            @test nsolutions(result) == 2
            @test nexcess_solutions(result) == 3
        end

        G = System(
            [
                (x^2 + y^2 + x * y - 3 * z^2) * (x + 3z),
                2x + 5y - 3z,
                (x^2 + y^2 + x * y - 3 * z^2) * (y^2 - x * z + 2 * z^2),
            ],
        )
        result = solve(
            G,
            TotalDegree(; seed = UInt32(0x1234), show_progress = false),
            Serial(),
        )
        @test ntracked(result) == 12
        @test count(is_success, path_results(result)) == 2
    end

    @testset "projective solve: composition and fixed parameters" begin
        @polyvar u v w a
        run(F) = solve(F, TotalDegree(; seed = UInt32(5), show_progress = false), Serial())
        L = System([2u - v + w, u + 3w, v - w]; variables = [u, v, w])

        composition =
            System([u * v - w^2, u^2 + v * w]; variables = [u, v, w]) ∘ L
        composition_result = run(composition)
        @test ntracked(composition_result) == 4
        @test nsolutions(composition_result) == 4
        @test same_point_set(
            solutions(composition_result),
            solutions(run(System(composition))),
        )

        parametric = System(
            [u * v - a * w^2, u^2 + v * w];
            variables = [u, v, w],
            parameters = [a],
        ) ∘ L
        fixed = fix_parameters(parametric, [2.0])
        @test fixed isa FixedParameterSystem
        fixed_result = run(fixed)
        @test nsolutions(fixed_result) == 4
        @test same_point_set(
            solutions(fixed_result),
            solutions(run(fix_parameters(System(parametric), [2.0]))),
        )
    end

    @testset "parameter continuation" begin
        @polyvar x y a b
        F = System([x^2 + a * y - 1, x * y - b]; parameters = [a, b])
        starts = solutions(
            solve(
                System([x^2 + y - 1, x * y - 0.5]),
                TotalDegree(; show_progress = false),
            ),
        )
        result = solve(
            F,
            starts,
            [1.0, 0.5],
            [2.0, 1.0],
            Continuation(; show_progress = false),
        )
        @test nsolutions(result) >= 2
        for sol in solutions(result)
            @test abs(sol[1]^2 + 2.0 * sol[2] - 1) < 1.0e-6
            @test abs(sol[1] * sol[2] - 1.0) < 1.0e-6
        end
    end

    @testset "parameter continuation preserves all four real branches" begin
        @polyvar x y a
        F = System([x^2 - a, y^2 - a]; parameters = [a])
        starts = solutions(
            solve(System([x^2 - 1, y^2 - 1]), TotalDegree(; show_progress = false)),
        )
        result = solve(
            F,
            starts,
            [1.0],
            [4.0],
            Continuation(; show_progress = false),
        )
        @test nsolutions(result) == 4
        @test length(real_solutions(result)) == 4
        for sol in real_solutions(result)
            @test abs(sol[1]^2 - 4) < 1.0e-6
            @test abs(sol[2]^2 - 4) < 1.0e-6
        end
    end

    @testset "start-system routes reject parametric systems without values" begin
        @polyvar x y a
        F = System([x^2 + y^2 - a, x * y - 1]; variables = [x, y], parameters = [a])
        for alg in (
                TotalDegree(; show_progress = false),
                Polyhedral(; show_progress = false),
            )
            @test_throws ArgumentError solve(F, alg, Serial())
            @test_throws ArgumentError CommonSolve.init(F, alg, Serial())
        end
        @test_throws ArgumentError solve(
            F,
            rand_subspace(2; codim = 1),
            TotalDegree(; show_progress = false),
            Serial(),
        )

        nonparametric = System([x^2 + y^2 - 5, x * y - 1]; variables = [x, y])
        @test nsolutions(
            solve(nonparametric, TotalDegree(; show_progress = false), Serial()),
        ) == 4
    end

    @testset "executor validation and parity" begin
        @test_throws ArgumentError Threaded(Threads.nthreads() + 1)
        @test_throws ArgumentError Threaded(0)
        @test_throws ArgumentError Threaded(-1)

        @polyvar x y
        F = System([x^2 + y - 1, x * y - 0.5])
        for alg in (
                TotalDegree(; seed = UInt32(99), show_progress = false),
                Polyhedral(; seed = UInt32(99), show_progress = false),
            )
            serial = solve(F, alg, Serial())
            threaded = solve(F, alg, Threaded())

            @test nsolutions(serial) == nsolutions(threaded)
            @test nreal(serial) == nreal(threaded)
            @test ntracked(serial) == ntracked(threaded)

            serial_solutions = sort(solutions(serial); by = s -> (real(s[1]), imag(s[1])))
            threaded_solutions =
                sort(solutions(threaded); by = s -> (real(s[1]), imag(s[1])))
            for (a, b) in zip(serial_solutions, threaded_solutions)
                @test a ≈ b atol = 1.0e-6
            end
        end
    end

    @testset "parameter continuation: Serial and Threaded parity" begin
        @polyvar x y a
        F = System([x^2 - a, y^2 - a]; parameters = [a])
        starts = solutions(
            solve(System([x^2 - 1, y^2 - 1]), TotalDegree(; show_progress = false)),
        )

        serial = solve(
            F,
            starts,
            [1.0],
            [4.0],
            Continuation(; show_progress = false),
            Serial(),
        )
        threaded = solve(
            F,
            starts,
            [1.0],
            [4.0],
            Continuation(; show_progress = false),
            Threaded(),
        )
        @test nsolutions(serial) == nsolutions(threaded) == 4
        for sol in real_solutions(serial)
            @test any(other -> isapprox(sol, other; atol = 1.0e-8), real_solutions(threaded))
        end
    end

    @testset "Serial and Threaded preserve cluster structure" begin
        @polyvar x0 x1 x2 x3
        F = System(
            [
                x0 + 2x1 + 2x2 + 2x3 - 1,
                x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
                2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
                x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
            ],
        )
        serial = solve(F, TotalDegree(; seed = UInt32(7), show_progress = false), Serial())
        threaded =
            solve(F, TotalDegree(; seed = UInt32(7), show_progress = false), Threaded())

        @test nsolutions(serial) == nsolutions(threaded)
        @test ntracked(serial) == ntracked(threaded)
        @test sort(length.(clusters(serial))) == sort(length.(clusters(threaded)))
    end

    @testset "paths_to_track agrees with executed path count" begin
        @polyvar x y z
        @test paths_to_track(System([x^2 + y^2 - 4, x * y - 1])) == 4
        @test paths_to_track(
            System([2y + 3 * y^2 - x * y^3, x + 4 * x^2 - 2 * x^3 * y]),
        ) == 16

        for F in (
                System([x^2 + y^2 - 4, x * y - 1]),
                System([x^2 + y^2 - z^2, x * y - z^2]),
                System([(x^2 - 4) * (x * y - 2), x * y - 2, x^2 - 4]),
                System([x^2 - y, x + y - 1]) ∘ System([x + y, x - y]),
            )
            alg = TotalDegree(; seed = UInt32(11), show_progress = false)
            @test paths_to_track(F, alg) == ntracked(solve(F, alg, Serial()))
        end
    end

    @testset "early_stop_callback preserves executed-path accounting" begin
        @polyvar x y
        F = System([x^3 + y^2 - 3, x * y^2 - 2])

        for exec in (Serial(), Threaded(1))
            full = solve(F, TotalDegree(; seed = UInt32(5), show_progress = false), exec)
            @test ntracked(full) == 9

            hits = Threads.Atomic{Int}(0)
            stopped = solve(
                F,
                TotalDegree(;
                    seed = UInt32(5),
                    show_progress = false,
                    early_stop_callback = function (pr)
                        Threads.atomic_add!(hits, 1)
                        return true
                    end,
                ),
                exec,
            )
            @test hits[] >= 1
            @test 0 < ntracked(stopped) < 9
            @test nfailed(stopped) == 0
            @test length(path_results(stopped)) == ntracked(stopped)

            never = solve(
                F,
                TotalDegree(;
                    seed = UInt32(5),
                    show_progress = false,
                    early_stop_callback = _ -> false,
                ),
                exec,
            )
            @test ntracked(never) == 9
            @test nsolutions(never) == nsolutions(full)
        end

        if Threads.nthreads() > 1
            stopped = solve(
                F,
                TotalDegree(;
                    seed = UInt32(5),
                    show_progress = false,
                    early_stop_callback = _ -> true,
                ),
                Threaded(min(4, Threads.nthreads())),
            )
            prs = path_results(stopped)
            @test issorted(path_number.(prs))
            @test allunique(path_number.(prs))
            for pr in prs
                is_success(pr) || continue
                i = path_number(pr)
                @test multiplicity(stopped, i) == multiplicity(pr)
                @test any(c -> solution(c) == solution(pr), cluster_of(stopped, i))
            end
            @test_throws BoundsError cluster_of(stopped, 10)
        end
    end
end
