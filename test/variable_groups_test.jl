using Test
using Random: MersenneTwister
using HomotopyContinuation
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

run_td(F, seed_value, exec = Serial()) =
    solve(F, TotalDegree(; seed = UInt32(seed_value), show_progress = false), exec)

@testset "Variable groups public behavior" begin
    @testset "construction and validation" begin
        @polyvar x y v w
        F = System([x * y - 2v * w, x^2 - 4 * v^2]; variable_groups = [[x, v], [y, w]])
        @test collect(variables(F)) == [x, v, y, w]
        @test variable_groups(F) == [[1, 2], [3, 4]]
        @test degrees(F) == [2, 2]

        plain = System([x^2 + y - 1, x * y - 2])
        @test variable_groups(plain) == Vector{Int}[]

        @test_throws ArgumentError System(
            [x * y - 2v * w, x^2 - 4 * v^2]; variable_groups = [[x, v], [y, w, x]],
        )
        @test_throws ArgumentError System(
            [x * y - 2v * w, x^2 - 4 * v^2];
            variables = [x, v, y, w], variable_groups = [[x, v], [y]],
        )
    end

    @testset "grouping reduces the continuation path count" begin
        @polyvar x y v w
        plain = System([x * y - 2, x^2 - 4])
        grouped = System([x * y - 2, x^2 - 4]; variable_groups = [[x], [y]])
        projective = System(
            [x * y - 2v * w, x^2 - 4 * v^2]; variable_groups = [[x, v], [y, w]],
        )
        overdetermined = System(
            [(x^2 - 4) * (x * y - 2), x * y - 2, x^2 - 4];
            variable_groups = [[x], [y]],
        )

        @test paths_to_track(plain) == 4
        @test paths_to_track(grouped) == 2
        for F in (plain, grouped, projective, overdetermined)
            alg = TotalDegree(; seed = UInt32(5), show_progress = false)
            @test paths_to_track(F, alg) == ntracked(solve(F, alg, Serial()))
        end
    end

    @testset "affine groups preserve the physical solutions" begin
        @polyvar x y
        F = System([x * y - 2, x^2 - 4]; variable_groups = [[x], [y]])
        grouped = run_td(F, 5)
        plain = run_td(System([x * y - 2, x^2 - 4]), 5)
        @test ntracked(grouped) == 2
        @test ntracked(plain) == 4
        @test nsolutions(grouped) == nsolutions(plain) == 2
        @test same_points(solutions(grouped), solutions(plain))

        overdetermined = System(
            [(x^2 - 4) * (x * y - 2), x * y - 2, x^2 - 4]; variable_groups = [[x], [y]],
        )
        result = run_td(overdetermined, 5)
        @test ntracked(result) == 5
        @test nsolutions(result) == 2
        @test nexcess_solutions(result) == 2
        @test same_points(solutions(result), solutions(grouped))
    end

    @testset "multi-projective groups are chart, executor, and compile-mode independent" begin
        @polyvar x y v w
        groups = [[1, 2], [3, 4]]
        F = System([x * y - 2v * w, x^2 - 4 * v^2]; variable_groups = [[x, v], [y, w]])
        reference = run_td(F, 5)
        @test ntracked(reference) == 2
        @test nsolutions(reference) == 2
        for s in solutions(reference)
            @test abs(s[1] * s[3] - 2s[2] * s[4]) < 1.0e-8
            @test abs(s[1]^2 - 4s[2]^2) < 1.0e-8
        end

        @test same_group_points(solutions(run_td(F, 99)), solutions(reference), groups)
        @test same_group_points(solutions(run_td(F, 5, Threaded())), solutions(reference), groups)
        for mode in (CompileMode.COMPILED, CompileMode.COMPILED_ALL)
            compiled = System(
                [x * y - 2v * w, x^2 - 4 * v^2];
                variable_groups = [[x, v], [y, w]], compile = mode,
            )
            @test same_group_points(solutions(run_td(compiled, 5)), solutions(reference), groups)
        end

        overdetermined = System(
            [(x^2 - 4v^2) * (x * y - v * w), x * y - v * w, x^2 - v^2];
            variable_groups = [[x, v], [y, w]],
        )
        result = run_td(overdetermined, 5)
        @test ntracked(result) == 5
        @test nsolutions(result) == 2
        @test nexcess_solutions(result) == 2
        for s in solutions(result)
            @test abs(s[1] * s[3] - s[2] * s[4]) < 1.0e-8
            @test abs(s[1]^2 - s[2]^2) < 1.0e-8
        end
    end

    @testset "one group is the ordinary projective route" begin
        @polyvar x y v
        polys = [x^2 + y^2 - v^2, x * y - v^2]
        grouped = System(polys; variable_groups = [[x, y, v]])
        plain = System(polys)
        rg = run_td(grouped, 4)
        rp = run_td(plain, 4)
        @test ntracked(rg) == ntracked(rp) == 4
        @test nsolutions(rg) == nsolutions(rp) == 4
        @test same_group_points(solutions(rg), solutions(rp), [[1, 2, 3]])
    end

    @testset "fixed parameters preserve groups" begin
        @polyvar x y v w a
        F = System(
            [x * y - a * v * w, x^2 - 4v^2];
            parameters = [a], variable_groups = [[x, v], [y, w]],
        )
        G = fix_parameters(F, [2.0])
        @test variable_groups(G) == [[1, 2], [3, 4]]
        result = run_td(G, 5)
        @test ntracked(result) == 2
        @test nsolutions(result) == 2
    end

    @testset "unsupported shapes and algorithms reject groups" begin
        @polyvar x y v w
        underdetermined = System([x * y - 2v * w]; variable_groups = [[x, v], [y, w]])
        @test_throws ArgumentError run_td(underdetermined, 1)
        @test_throws "one affine chart per variable group" run_td(underdetermined, 1)

        F = System([x * y - 2v * w, x^2 - 4v^2]; variable_groups = [[x, v], [y, w]])
        L = LinearSubspace(
            randn(MersenneTwister(0x07), ComplexF64, 2, 4), zeros(ComplexF64, 2),
        )
        @test_throws ArgumentError solve(
            F, Polyhedral(; seed = UInt32(1), show_progress = false), Serial(),
        )
        @test_throws "variable groups" solve(
            F, L, TotalDegree(; seed = UInt32(1), show_progress = false), Serial(),
        )
        @test_throws "variable groups" solve(
            F, L, Polyhedral(; seed = UInt32(1), show_progress = false), Serial(),
        )
        @test_throws "variable groups" solve(F, Witness())
        @test_throws "positive-dimensional" solve(
            F,
            rand_subspace(4; codim = 1),
            TotalDegree(; seed = UInt32(1), show_progress = false),
            Serial(),
        )

        projective = System([x^2 + y^2 - v^2, x * y - v^2]; variable_groups = [[x, y, v]])
        @test nsolutions(
            solve(projective, Polyhedral(; seed = UInt32(1), show_progress = false), Serial()),
        ) == 4
    end
end
