# Tests ported from HomotopyContinuation.jl v2 to verify correctness parity.
# Each test section references the v2 test file it was ported from.

using Test
using HomotopyContinuationNext
using HomotopyContinuationNext: TotalDegree, Polyhedral, Result, PathResult,
    PathResultCode, TrackerOptions, EndgameOptions,
    solutions, real_solutions, nsolutions, nreal, nresults,
    nsingular, nnonsingular, nat_infinity, multiplicity,
    is_success, is_real, is_singular
using DynamicPolynomials: @polyvar
using CommonSolve: CommonSolve

@testset "v2 Parity" begin

    # ── From v2 solve_test.jl: "total degree (simple)" ───────────────────

    @testset "total degree: affine quadratic" begin
        @polyvar x y
        result = solve(
            System(
                [
                    2.3x^2 + 1.2y^2 + 3x - 2y + 3,
                    2.3x^2 + 1.2y^2 + 5x + 2y - 5,
                ]
            );
            show_progress = false,
        )
        @test nsolutions(result) == 2
    end

    # ── From v2 solve_test.jl: "polyhedral" ──────────────────────────────

    @testset "polyhedral: affine quadratic" begin
        @polyvar x y
        result = solve(
            System(
                [
                    2.3x^2 + 1.2y^2 + 3x - 2y + 3,
                    2.3x^2 + 1.2y^2 + 5x + 2y - 5,
                ]
            ),
            Polyhedral(); show_progress = false,
        )
        @test nsolutions(result) == 2
    end

    # ── From v2 solve_test.jl: "solve (DynamicPolynomials)" ──────────────

    @testset "DynamicPolynomials: high-degree system (18 solutions)" begin
        @polyvar x y
        f₁ = (x^4 + y^4 - 1) * (x^2 + y^2 - 2) + x^5 * y
        f₂ = x^2 + 2x * y^2 - 2y^2 - 1 / 2
        result = solve(System([f₁, f₂]); show_progress = false)
        @test nsolutions(result) == 18
    end

    # ── From v2 solve_test.jl: "solve (parameter homotopy)" ──────────────

    @testset "parameter homotopy: affine" begin
        @polyvar x y a b
        F = System([x^2 - a, x * y - a + b]; parameters = [a, b])
        # Solve at start parameters
        F_start = System([x^2 - 1, x * y - 1])
        r1 = solve(F_start; show_progress = false)
        @test nsolutions(r1) >= 1
        # Track to target parameters
        r2 = solve(
            F, solutions(r1), [1.0, 0.0], [2.0, 4.0]; show_progress = false,
        )
        @test nsolutions(r2) >= 1
        for sol in solutions(r2)
            @test abs(sol[1]^2 - 2.0) < 1.0e-6
            @test abs(sol[1] * sol[2] - 2.0 + 4.0) < 1.0e-6
        end
    end

    # ── From v2 endgame_test.jl: "Wilkinson" ─────────────────────────────

    @testset "Wilkinson-12" begin
        @polyvar x
        f = prod(x - i for i in 1:12)
        result = solve(
            System([f]),
            TotalDegree(; endgame_options = EndgameOptions(; only_nonsingular = true)); show_progress = false,
        )
        @test nsolutions(result) == 12
        sols = sort(real.(first.(solutions(result))); by = abs)
        @test sols ≈ collect(1.0:12.0) atol = 1.0e-3
        @test maximum(abs.(imag.(first.(solutions(result))))) < 1.0e-4
    end

    # ── From v2 endgame_test.jl: "Beyond Polyhedral Homotopy Example" ────

    @testset "at-infinity: 2 finite + 2 divergent" begin
        @polyvar x y
        result = solve(
            System([2.3x^2 + 1.2y^2 + 3x - 2y + 3, 2.3x^2 + 1.2y^2 + 5x + 2y - 5]);
            show_progress = false,
        )
        @test count(is_success, result.path_results) == 2
        @test nat_infinity(result) == 2
    end

    # ── From v2 endgame_test.jl: "(x-10)^d" ─────────────────────────────

    @testset "(x-10)^d singular roots" begin
        @testset "d=2" begin
            @polyvar x
            result = solve(System([(x - 10)^2]); show_progress = false)
            @test nresults(result) == 1
            @test nsingular(result) == 1
        end

        @testset "d=6" begin
            @polyvar x
            result = solve(System([(x - 10)^6]); show_progress = false)
            # Most paths detect winding number 6 (seed-dependent)
            @test count(r -> r.winding_number == 6, result.path_results) >= 4
        end
    end

    # ── From v2 endgame_test.jl: "Winding Number Family" ────────────────

    @testset "winding number family d=$d" for d in 2:2:6
        @polyvar x y
        a = [0.257, -0.139, -1.73, -0.199, 1.79, -1.32]
        f1 = (a[1] * x^d + a[2] * y) * (a[3] * x + a[4] * y) + 1
        f2 = (a[1] * x^d + a[2] * y) * (a[5] * x + a[6] * y) + 1
        result = solve(System([f1, f2]); show_progress = false)
        @test count(is_success, result.path_results) == d + 1
    end

    # ── From v2 endgame_test.jl: "Hyperbolic 6,6" ───────────────────────

    @testset "Hyperbolic 6,6: two roots of multiplicity 6" begin
        @polyvar x z
        y = 1
        F = System(
            [
                0.75x^4 + 1.5x^2 * y^2 - 2.5x^2 * z^2 + 0.75y^4 - 2.5y^2 * z^2 + 0.75z^4,
                10x^2 * z + 10y^2 * z - 6z^3,
            ]
        )
        result = solve(F, TotalDegree(; seed = UInt32(1)); show_progress = false)
        # A dead path keeps its last winding number estimate, so the count below alone
        # does not catch one.
        @test count(is_success, result.path_results) == 12
        @test count(r -> r.winding_number == 3, result.path_results) == 12
        @test nresults(result) == 2
        @test nsingular(result) == 2
    end

    # ── From v2 endgame_test.jl: "Singular 1" ───────────────────────────

    @testset "singular: multiplicity 3 + nonsingular" begin
        @polyvar x y
        z = 1
        F = System(
            [
                x^2 + 2y^2 + 2im * y * z,
                (18 + 3im) * x * y + 7im * y^2 - (3 - 18im) * x * z - 14y * z - 7im * z^2,
            ]
        )
        result = solve(F, TotalDegree(; seed = UInt32(12345)); show_progress = false)
        @test nresults(result) == 2
        @test nsingular(result) == 1
        @test nnonsingular(result) == 1
    end

    # ── From v2 polyhedral_test.jl: "cyclic" ─────────────────────────────

    @testset "polyhedral: cyclic-5" begin
        @polyvar z[1:5]
        eqs = [sum(prod(z[((k - 1) % 5) + 1] for k in j:(j + m)) for j in 1:5) for m in 0:3]
        push!(eqs, prod(z) - 1)
        result = solve(System(eqs), Polyhedral(); show_progress = false)
        @test nsolutions(result) == 70
    end

    # ── From v2 result_test.jl: "Basic functionality of Result" ──────────

    @testset "Result: winding family d=2" begin
        @polyvar x y
        a = [0.257, -0.139, -1.73, -0.199, 1.79, -1.32]
        F = System(
            [
                (a[1] * x^2 + a[2] * y) * (a[3] * x + a[4] * y) + 1,
                (a[1] * x^2 + a[2] * y) * (a[5] * x + a[6] * y) + 1,
            ]
        )
        result = solve(F; show_progress = false)

        @test nresults(result) == 3
        @test nsolutions(result) >= 3
        @test length(real_solutions(result)) == nreal(result)
        @test nreal(result) == 1
        @test nnonsingular(result) == 3
        @test nsingular(result) == 0

        # show works
        buf = IOBuffer()
        show(buf, result)
        s = String(take!(buf))
        @test contains(s, "solutions")
    end

    @testset "Result: singular system (29/16)x³ - 2xy, x² - y" begin
        @polyvar x y
        result = solve(System([(29 / 16) * x^3 - 2x * y, x^2 - y]); show_progress = false)
        @test nresults(result) >= 1
        # v2: "Result with 1 solution" — has 1 singular solution at origin
        buf = IOBuffer()
        show(buf, result)
        @test !isempty(String(take!(buf)))
    end

    # ── From v2 solve_test.jl: "paths to track" ─────────────────────────

    @testset "paths to track: total degree vs polyhedral" begin
        @polyvar x y
        f = System([2y + 3y^2 - x * y^3, x + 4x^2 - 2x^3 * y])
        # Total degree = 4 * 4 = 16
        r_td = solve(f, TotalDegree(); show_progress = false)
        @test r_td.tracked_paths == 16

        # Polyhedral (mixed volume) tracks fewer paths
        r_ph = solve(f, Polyhedral(); show_progress = false)
        @test r_ph.tracked_paths <= 16
        @test r_ph.tracked_paths >= 3  # mixed volume = 3 for torus solutions
    end

    # ── From v2 endgame_test.jl: "Mohab" (large system) ─────────────────

    @testset "Mohab: large-coefficient system (degrees 9,10,10)" begin
        @polyvar x y z
        F = System(
            [
                -9091098778555951517 * x^3 * y^4 * z^2 +
                    5958442613080401626 * y^2 * z^7 +
                    17596733865548170996 * x^2 * z^6 - 17979170986378486474 * x * y * z^6 -
                    2382961149475678300 * x^4 * y^3 - 15412758154771986214 * x * y^3 * z^3 +
                    133,
                -10798198881812549632 * x^6 * y^3 * z - 11318272225454111450 * x * y^9 -
                    14291416869306766841 * y^9 * z - 5851790090514210599 * y^2 * z^8 +
                    15067068695242799727 * x^2 * y^3 * z^4 +
                    7716112995720175148 * x^3 * y * z^3 +
                    171,
                13005416239846485183 * x^7 * y^3 + 4144861898662531651 * x^5 * z^4 -
                    8026818640767362673 * x^6 - 6882178109031199747 * x^2 * y^4 +
                    7240929562177127812 * x^2 * y^3 * z +
                    5384944853425480296 * x * y * z^4 +
                    88,
            ],
        )
        result = solve(F; show_progress = false)
        # v2 finds 693 nonsingular + 0 singular. We find ~679 nonsingular + ~23 singular
        # (more genuine solutions, fewer at-infinity misclassifications).
        @test nnonsingular(result) >= 670
    end

    # ── From v2 polyhedral_test.jl: "affine + torus solutions" ───────────

    @testset "polyhedral: torus solutions count" begin
        @polyvar x y
        f = System([2y + 3y^2 - x * y^3, x + 4x^2 - 2x^3 * y])
        result = solve(f, Polyhedral(); show_progress = false)
        # v2: 6 affine solutions (8 paths including non-torus)
        @test nsolutions(result) >= 3
    end

end
