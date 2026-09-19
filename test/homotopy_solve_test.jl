using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar
using CommonSolve: init, solve!
using LinearAlgebra: norm
using Random: MersenneTwister

_normalize_projective(x) = x ./ x[argmax(abs.(x))]

function max_distance(tracked, reference)
    isempty(tracked) && return Inf
    return maximum(minimum(norm(a - b) for b in reference) for a in tracked)
end

function public_path_parity(a, b; atol = 1.0e-10)
    pa = sort(path_results(a); by = path_number)
    pb = sort(path_results(b); by = path_number)
    length(pa) == length(pb) || return false
    for (u, v) in zip(pa, pb)
        path_number(u) == path_number(v) || return false
        isapprox(solution(u), solution(v); atol) || return false
        isapprox(start_solution(u), start_solution(v); atol) || return false
        steps(u) == steps(v) || return false
    end
    return true
end

@testset "Homotopy solve" begin
    @testset "start-target fixed systems" begin
        @polyvar x y a b
        family = System(
            [x^2 - a, x * y - a + b];
            variables = [x, y], parameters = [a, b],
        )
        start = fix_parameters(family, [1, 0])
        target = fix_parameters(family, [2, 4])

        result = solve(
            start,
            target,
            [[1.0, 1.0]],
            Continuation(; show_progress = false),
        )
        @test nsolutions(result) == 1
        @test ntracked(result) == 1
        @test max_distance(solutions(result), [[sqrt(2), -sqrt(2)]]) < 1.0e-8

        h = System([x^2 - a, y - b]; variables = [x, y], parameters = [a, b])
        composed_target = fix_parameters(h ∘ System([x, y]; variables = [x, y]), [4, 3])
        composed = solve(
            fix_parameters(h, [1, 2]),
            composed_target,
            [[1.0, 2.0]],
            Continuation(; show_progress = false),
        )
        @test nsolutions(composed) == 1
        @test max_distance(solutions(composed), [[2.0, 3.0]]) < 1.0e-8
    end

    @testset "start-target agrees with direct solving" begin
        @polyvar x y
        G = System([x^2 + y^2 - 5, x * y - 2]; variables = [x, y])
        F = System([x^2 + 3y^2 - 7, x * y + x - 3]; variables = [x, y])
        starts = solve(G, TotalDegree(; show_progress = false))
        reference = solutions(solve(F, TotalDegree(; show_progress = false)))

        result = solve(G, F, starts, Continuation(; show_progress = false))
        @test nsolutions(result) == 4
        @test max_distance(solutions(result), reference) < 1.0e-8
        @test nsolutions(
            solve(G, F, solutions(starts), Continuation(; show_progress = false)),
        ) == 4
        @test nsolutions(
            solve(
                G,
                F,
                result_iterator(G, TotalDegree()),
                Continuation(; show_progress = false),
            ),
        ) == 4
    end

    @testset "seeded Serial and Threaded paths agree" begin
        @polyvar x y
        G = System([x^2 + y^2 - 5, x * y - 2]; variables = [x, y])
        F = System([x^2 + 3y^2 - 7, x * y + x - 3]; variables = [x, y])
        starts = solutions(solve(G, TotalDegree(; show_progress = false)))
        alg = Continuation(; seed = UInt32(0x1234), show_progress = false)

        serial = solve(G, F, starts, alg, Serial())
        threaded = solve(G, F, starts, alg, Threaded())
        @test public_path_parity(serial, threaded)
        @test seed(serial) == seed(threaded) == UInt32(0x1234)

        other = solve(
            G,
            F,
            starts,
            Continuation(; seed = UInt32(0x99), show_progress = false),
            Serial(),
        )
        @test nsolutions(other) == nsolutions(serial)
    end

    @testset "projective start representatives" begin
        @polyvar x y z
        monomials = [x^2, x * y, x * z, y^2, y * z, z^2]
        rng = MersenneTwister(3)
        coefficients = [randn(rng, 6) for _ in 1:4]
        G = System([sum(coefficients[1] .* monomials), sum(coefficients[2] .* monomials)])
        F = System([sum(coefficients[3] .* monomials), sum(coefficients[4] .* monomials)])

        starts = solutions(solve(G, TotalDegree(; show_progress = false)))
        scaled = [(3.7 - 1.2im) .* s for s in starts]
        reference = _normalize_projective.(solutions(solve(F, TotalDegree(; show_progress = false))))

        result = solve(G, F, scaled, Continuation(; show_progress = false))
        @test nsolutions(result) == 4
        @test max_distance(_normalize_projective.(solutions(result)), reference) < 1.0e-8
    end

    @testset "overdetermined start-target continuation" begin
        @polyvar x y
        G = System([x^2 - 1, y^2 - 1, x - y]; variables = [x, y])
        F = System([x^2 - 4, y^2 - 4, x - y]; variables = [x, y])
        result = solve(
            G,
            F,
            [[1.0, 1.0], [-1.0, -1.0]],
            Continuation(; show_progress = false),
        )
        @test nsolutions(result) == 2
        @test max_distance(solutions(result), [[2.0, 2.0], [-2.0, -2.0]]) < 1.0e-8
    end

    @testset "start-target input validation" begin
        @polyvar x y a
        parametric = System([x^2 - a, y - 1]; variables = [x, y], parameters = [a])
        square = System([x^2 - 1, y - 1]; variables = [x, y])

        @test_throws ArgumentError solve(
            parametric,
            square,
            [[1.0, 1.0]],
            Continuation(; show_progress = false),
        )
        @test_throws ArgumentError solve(
            square,
            System([x^2 - 4]; variables = [x]),
            [[1.0, 1.0]],
            Continuation(; show_progress = false),
        )
        @test_throws ArgumentError solve(
            System([x^2 - y^2]; variables = [x, y]),
            System([x^2 - y^2 - 1]; variables = [x, y]),
            [[1.0, 1.0]],
            Continuation(; show_progress = false),
        )
        @test_throws ArgumentError solve(
            System([x * y - 1]; variables = [x, y]),
            System([x * y - 4]; variables = [x, y]),
            [[1.0, 1.0]],
            Continuation(; show_progress = false),
        )
        @test_throws ArgumentError solve(
            square,
            square,
            [[1.0]],
            Continuation(; show_progress = false),
        )
    end

    @testset "explicit ParameterHomotopy" begin
        @polyvar x y a b
        family = System(
            [x^2 - a, x * y - a + b];
            variables = [x, y], parameters = [a, b],
        )
        H = ParameterHomotopy(family, [1, 0], [2, 4])
        result = solve(H, [[1.0, 1.0]], Continuation(; show_progress = false))
        @test nsolutions(result) == 1
        @test max_distance(solutions(result), [[sqrt(2), -sqrt(2)]]) < 1.0e-8

        typed = solve(
            family,
            [[1.0, 1.0]],
            [1, 0],
            [2, 4],
            Continuation(; show_progress = false),
            Serial(),
        )
        @test max_distance(solutions(result), solutions(typed)) < 1.0e-8

        serial = solve(
            H,
            [[1.0, 1.0]],
            Continuation(; seed = UInt32(4), show_progress = false),
            Serial(),
        )
        threaded = solve(
            H,
            [[1.0, 1.0]],
            Continuation(; seed = UInt32(4), show_progress = false),
            Threaded(),
        )
        @test public_path_parity(serial, threaded)

        again = solve(
            H,
            [[1.0, 1.0]],
            Continuation(; seed = UInt32(4), show_progress = false),
            Serial(),
        )
        @test public_path_parity(serial, again)
    end

    @testset "explicit homotopy input validation" begin
        @polyvar x y a
        family = System([x^2 - a, y - 1]; variables = [x, y], parameters = [a])
        H = ParameterHomotopy(family, [1], [4])
        @test_throws ArgumentError solve(
            H,
            [[1.0]],
            Continuation(; show_progress = false),
        )

        underdetermined = System([x * y - a]; variables = [x, y], parameters = [a])
        @test_throws ArgumentError solve(
            ParameterHomotopy(underdetermined, [1], [4]),
            [[1.0, 1.0]],
            Continuation(; show_progress = false),
        )
    end

    @testset "builder constructs independent homotopies" begin
        @polyvar x y a b
        starts = [ComplexF64[sx, sy] for sx in (-1, 1) for sy in (-2, 2)]
        expected = [ComplexF64[sx, sy] for sx in (-2, 2) for sy in (-3, 3)]

        function build_homotopy()
            family = System(
                [x^2 - a, y^2 - b];
                variables = [x, y], parameters = [a, b],
            )
            return ParameterHomotopy(family, [1, 4], [4, 9])
        end

        serial = solve(
            build_homotopy,
            starts,
            Continuation(; seed = UInt32(9), show_progress = false),
            Serial(),
        )
        threaded = solve(
            build_homotopy,
            starts,
            Continuation(; seed = UInt32(9), show_progress = false),
            Threaded(),
        )
        @test nsolutions(serial) == nsolutions(threaded) == 4
        @test max_distance(solutions(serial), expected) < 1.0e-8
        @test max_distance(solutions(threaded), expected) < 1.0e-8
        @test public_path_parity(serial, threaded)

        @test_throws ArgumentError solve(
            () -> System([x^2 - 1, y^2 - 4]; variables = [x, y]),
            starts,
            Continuation(; show_progress = false),
            Serial(),
        )
    end

    @testset "CommonSolve cache is reusable" begin
        @polyvar x y
        G = System([x^2 + y^2 - 5, x * y - 2]; variables = [x, y])
        F = System([x^2 + 3y^2 - 7, x * y + x - 3]; variables = [x, y])
        starts = solutions(solve(G, TotalDegree(; show_progress = false)))
        cache = init(
            G,
            F,
            starts,
            Continuation(; seed = UInt32(11), show_progress = false),
            Serial(),
        )
        first_result = solve!(cache)
        second_result = solve!(cache)
        @test public_path_parity(first_result, second_result)
        @test seed(first_result) == seed(second_result) == UInt32(11)
    end
end
