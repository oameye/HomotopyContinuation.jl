using Test
using Random: MersenneTwister
using HomotopyContinuationNext
using HomotopyContinuationNext: TotalDegree, Polyhedral, Serial, Threaded,
    solutions, nsolutions, nexcess_solutions, is_success, is_homogeneous,
    variables, variable_groups, multi_degrees, degrees, fix_parameters,
    paths_to_track, rand_subspace, LinearSubspace, CompileMode,
    evaluate!, FSVec,
    _group_dims, _bezout_assignments, _multi_bezout_count, _multi_start_coefficients,
    _multi_start_system, _multi_start_solutions
using DynamicPolynomials: @polyvar

same_points(a, b) =
    length(a) == length(b) && all(u -> any(v -> maximum(abs.(u .- v)) < 1.0e-7, b), a)

# Two multi-projective solutions agree when every group is proportional. By minors,
# not by normalizing: no stable divisor when coordinates share the largest magnitude.
function group_proportional(u, v, groups)
    for group in groups
        scale = maximum(abs, view(u, group)) * maximum(abs, view(v, group))
        for i in group, j in group
            abs(u[i] * v[j] - u[j] * v[i]) < 1.0e-7 * scale || return false
        end
    end
    return true
end

same_group_points(a, b, groups) =
    length(a) == length(b) && all(u -> any(v -> group_proportional(u, v, groups), b), a)

run_td(F, seed, exec = Serial()) =
    solve(F, TotalDegree(; seed = UInt32(seed), show_progress = false), exec)

@testset "Variable groups" begin

    @testset "construction" begin
        @polyvar x y v w
        F = System([x * y - 2v * w, x^2 - 4 * v^2]; variable_groups = [[x, v], [y, w]])
        # The groups fix the variable order when `variables` is not given.
        @test collect(variables(F)) == [x, v, y, w]
        @test variable_groups(F) == [[1, 2], [3, 4]]
        @test multi_degrees(F) == [1 2; 1 0]
        @test degrees(F) == [2, 2]
        # Homogeneous in each group separately.
        @test is_homogeneous(F)

        G = System([x * y - 2, x^2 - 4]; variable_groups = [[x], [y]])
        @test multi_degrees(G) == [1 2; 1 0]
        @test !is_homogeneous(G)

        # Homogeneous in all variables at once is not enough.
        H = System([x^2 + y * v, x * y - v^2]; variable_groups = [[x, y], [v]])
        @test is_homogeneous(System([x^2 + y * v, x * y - v^2]))
        @test !is_homogeneous(H)

        # An ungrouped system has the single row of total degrees.
        @test multi_degrees(System([x^2 + y - 1, x * y - 2])) == [2 2]
        @test variable_groups(System([x^2 + y - 1, x * y - 2])) == Vector{Int}[]

        @test_throws ArgumentError System(
            [x * y - 2v * w, x^2 - 4 * v^2]; variable_groups = [[x, v], [y, w, x]],
        )
        @test_throws ArgumentError System(
            [x * y - 2v * w, x^2 - 4 * v^2];
            variables = [x, v, y, w], variable_groups = [[x, v], [y]],
        )
    end

    @testset "multi-homogeneous Bezout number" begin
        # Two groups of one coordinate each: the two ways of assigning the
        # equations, minus the assignment through the zero degree.
        @test _multi_bezout_count([1 2; 1 0], [1, 1]) == 2
        # Every degree 1, three groups: one path per permutation.
        @test _multi_bezout_count(ones(Int, 3, 3), [1, 1, 1]) == 6
        # A single group covering everything is the plain Bezout number.
        @test _multi_bezout_count(reshape([2, 3, 4], 1, 3), [3]) == 24
        # An equation of degree zero in every group cannot be assigned.
        @test _multi_bezout_count([1 0; 1 0], [1, 1]) == 0
    end

    @testset "paths_to_track" begin
        @polyvar x y v w
        @test paths_to_track(System([x * y - 2, x^2 - 4])) == 4
        @test paths_to_track(
            System([x * y - 2, x^2 - 4]; variable_groups = [[x], [y]]),
        ) == 2
        for F in (
                System([x * y - 2, x^2 - 4]),
                System([x * y - 2, x^2 - 4]; variable_groups = [[x], [y]]),
                System([x * y - 2v * w, x^2 - 4 * v^2]; variable_groups = [[x, v], [y, w]]),
                System(
                    [(x^2 - 4) * (x * y - 2), x * y - 2, x^2 - 4];
                    variable_groups = [[x], [y]],
                ),
            )
            alg = TotalDegree(; seed = UInt32(5), show_progress = false)
            @test paths_to_track(F, alg) ==
                solve(F, alg, Serial()).tracked_paths
        end
    end

    @testset "start solutions solve the start system" begin
        @polyvar x y v w
        for (F, n) in (
                (System([x * y - 2, x^2 - 4]; variable_groups = [[x], [y]]), 2),
                (
                    System(
                        [x * y - 2v * w, x^2 - 4 * v^2];
                        variable_groups = [[x, v], [y, w]],
                    ), 4,
                ),
            )
            groups = variable_groups(F)
            homogeneous = is_homogeneous(F)
            k = _group_dims(groups, homogeneous)
            D = multi_degrees(F)
            C = _multi_start_coefficients(MersenneTwister(0x1234), k, sum(k))
            G = _multi_start_system(D, k, groups, C, homogeneous, n)
            starts = _multi_start_solutions(
                D, k, groups, C, homogeneous, n, _bezout_assignments(D, k),
            )
            @test length(starts) == _multi_bezout_count(D, k)
            @test size(G) == (n, n)
            u = FSVec{ComplexF64}(zeros(ComplexF64, n))
            p = FSVec{ComplexF64}(ComplexF64[])
            for s in starts
                evaluate!(u, G.evaluator, FSVec{ComplexF64}(s), p)
                @test maximum(abs, u) < 1.0e-12
            end
            # Distinct start points, or paths would collide.
            @test length(unique(s -> round.(s; digits = 8), starts)) == length(starts)
        end
    end

    @testset "affine groups" begin
        @polyvar x y
        F = System([x * y - 2, x^2 - 4]; variable_groups = [[x], [y]])
        r = run_td(F, 5)
        # Half the Bezout number 2 * 2, and both solutions are found.
        @test r.tracked_paths == 2
        @test nsolutions(r) == 2
        plain = run_td(System([x * y - 2, x^2 - 4]), 5)
        @test plain.tracked_paths == 4
        @test same_points(solutions(r), solutions(plain))

        # Overdetermined: the excess equation is folded in and its solutions
        # filtered afterwards.
        ov = System(
            [(x^2 - 4) * (x * y - 2), x * y - 2, x^2 - 4]; variable_groups = [[x], [y]],
        )
        rov = run_td(ov, 5)
        @test rov.tracked_paths == 5
        @test nsolutions(rov) == 2
        @test nexcess_solutions(rov) == 2
        @test same_points(solutions(rov), solutions(r))
    end

    @testset "multi-projective groups" begin
        @polyvar x y v w
        groups = [[1, 2], [3, 4]]
        F = System([x * y - 2v * w, x^2 - 4 * v^2]; variable_groups = [[x, v], [y, w]])
        r = run_td(F, 5)
        @test r.tracked_paths == 2
        @test count(is_success, r.path_results) == 2
        @test nsolutions(r) == 2
        for s in solutions(r)
            @test abs(s[1] * s[3] - 2s[2] * s[4]) < 1.0e-8
            @test abs(s[1]^2 - 4 * s[2]^2) < 1.0e-8
        end

        # Independent of the charts drawn, the executor, and the compile mode.
        @test same_group_points(solutions(run_td(F, 99)), solutions(r), groups)
        @test same_group_points(solutions(run_td(F, 5, Threaded())), solutions(r), groups)
        for mode in (CompileMode.COMPILED, CompileMode.COMPILED_ALL)
            Fc = System(
                [x * y - 2v * w, x^2 - 4 * v^2];
                variable_groups = [[x, v], [y, w]], compile = mode,
            )
            @test same_group_points(solutions(run_td(Fc, 5)), solutions(r), groups)
        end

        # Overdetermined: the chart rows go in before the fold, so the folded
        # degrees stay [3 2; 1 1] and the count is 3 * 1 + 1 * 2.
        ov = System(
            [(x^2 - 4 * v^2) * (x * y - v * w), x * y - v * w, x^2 - v^2];
            variable_groups = [[x, v], [y, w]],
        )
        rov = run_td(ov, 5)
        @test rov.tracked_paths == 5
        @test nsolutions(rov) == 2
        @test nexcess_solutions(rov) == 2
        for s in solutions(rov)
            @test abs(s[1] * s[3] - s[2] * s[4]) < 1.0e-8
            @test abs(s[1]^2 - s[2]^2) < 1.0e-8
        end
    end

    @testset "one group is the plain route" begin
        @polyvar x y v
        polys = [x^2 + y^2 - v^2, x * y - v^2]
        grouped = System(polys; variable_groups = [[x, y, v]])
        plain = System(polys)
        rg = run_td(grouped, 4)
        rp = run_td(plain, 4)
        @test rg.tracked_paths == rp.tracked_paths == 4
        @test nsolutions(rg) == nsolutions(rp) == 4
        @test same_group_points(solutions(rg), solutions(rp), [[1, 2, 3]])
    end

    @testset "parameters" begin
        @polyvar x y v w a
        F = System(
            [x * y - a * v * w, x^2 - 4 * v^2];
            parameters = [a], variable_groups = [[x, v], [y, w]],
        )
        G = fix_parameters(F, [2.0])
        @test variable_groups(G) == [[1, 2], [3, 4]]
        @test is_homogeneous(G)
        r = run_td(G, 5)
        @test r.tracked_paths == 2
        @test nsolutions(r) == 2
    end

    @testset "too few equations for the groups" begin
        @polyvar x y v w
        F = System([x * y - 2v * w]; variable_groups = [[x, v], [y, w]])
        @test_throws ArgumentError run_td(F, 1)
        @test_throws "one affine chart per variable group" run_td(F, 1)
    end

    @testset "routes that chart the variables as a whole reject groups" begin
        @polyvar x y v w
        F = System([x * y - 2v * w, x^2 - 4 * v^2]; variable_groups = [[x, v], [y, w]])
        # A linear slice makes these routes draw the single chart groups cannot live on.
        L = LinearSubspace(
            randn(MersenneTwister(0x07), ComplexF64, 2, 4), zeros(ComplexF64, 2),
        )
        @test_throws ArgumentError solve(F, Polyhedral(; seed = UInt32(1), show_progress = false), Serial())
        @test_throws "variable groups" solve(F, L, TotalDegree(; seed = UInt32(1), show_progress = false), Serial())
        @test_throws "variable groups" solve(F, L, Polyhedral(; seed = UInt32(1), show_progress = false), Serial())
        @test_throws "variable groups" solve(F, Witness())

        # An affine slice draws no chart, so the shape is rejected, not the groups.
        @test_throws "positive-dimensional" solve(
            F,
            rand_subspace(4; codim = 1),
            TotalDegree(; seed = UInt32(1), show_progress = false),
            Serial(),
        )

        # A single group is the ordinary projective problem, so it is accepted.
        P = System([x^2 + y^2 - v^2, x * y - v^2]; variable_groups = [[x, y, v]])
        @test nsolutions(
            solve(P, Polyhedral(; seed = UInt32(1), show_progress = false), Serial()),
        ) == 4
    end
end
