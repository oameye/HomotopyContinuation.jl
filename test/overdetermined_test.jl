using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: System, RandomizedSystem, SystemEvaluator,
    evaluate!, evaluate_and_jacobian!, taylor!, nparameters,
    TaylorVector, TruncatedTaylorSeries, ComplexDF64,
    solve, TotalDegree, Polyhedral, Serial, Threaded,
    nsolutions, solutions, real_solutions, nresults, results, nsingular,
    nexcess_solutions, nat_infinity, is_excess_solution, is_success,
    ExcessSolutionChecker, check_excess_solution
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

include("minors_polys.jl")

@testset "Overdetermined" begin

    # ── RandomizedSystem: G_i(x) = F_perm[i](x) + Σ_j A[i,j]·F_perm[n+j](x) ──

    @testset "RandomizedSystem evaluation" begin
        @polyvar x y
        F = System([x^2 + y^2 - 1, x - y, x * y - 0.25])
        @test size(F.evaluator) == (3, 2)
        A = FSMat{ComplexF64}(reshape(ComplexF64[0.5 + 0.25im, -1.0 + 2.0im], 2, 1))
        perm = [1, 2, 3]
        R = RandomizedSystem(F.evaluator, A, perm)
        @test size(R) == (2, 2)
        @test nparameters(R) == 0

        Reval = SystemEvaluator(R)
        p = FSVec{ComplexF64}(ComplexF64[])
        for _ in 1:5
            xv = randn(ComplexF64, 2)
            x_fs = FSVec{ComplexF64}(xv)

            # Reference: evaluate the full m-vector, fold rows manually
            u_full = FSVec{ComplexF64}(zeros(ComplexF64, 3))
            U_full = FSMat{ComplexF64}(zeros(ComplexF64, 3, 2))
            evaluate_and_jacobian!(u_full, U_full, F.evaluator, x_fs, p)
            u_ref = [u_full[i] + A[i, 1] * u_full[3] for i in 1:2]
            U_ref = [U_full[i, j] + A[i, 1] * U_full[3, j] for i in 1:2, j in 1:2]

            u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            evaluate!(u, Reval, x_fs, p)
            @test u ≈ u_ref atol = 1.0e-13

            # DF64 variant
            x_df = FSVec{ComplexDF64}(ComplexDF64.(xv))
            fill!(u, zero(ComplexF64))
            evaluate!(u, Reval, x_df, p)
            @test u ≈ u_ref atol = 1.0e-13

            U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
            fill!(u, zero(ComplexF64))
            evaluate_and_jacobian!(u, U, Reval, x_fs, p)
            @test u ≈ u_ref atol = 1.0e-13
            @test U ≈ U_ref atol = 1.0e-13

            # Taylor orders 1-3 vs inner taylor + fold
            for K in 1:3
                N = K + 1
                tx = TaylorVector{N, ComplexF64}(2)
                for i in 1:2
                    tx[i] = TruncatedTaylorSeries(ntuple(_ -> randn(ComplexF64), Val(N)))
                end
                ut_full = FSVec{ComplexF64}(zeros(ComplexF64, 3))
                taylor!(ut_full, Val(K), F.evaluator, tx, p)
                ut_ref = [ut_full[i] + A[i, 1] * ut_full[3] for i in 1:2]
                ut = FSVec{ComplexF64}(zeros(ComplexF64, 2))
                taylor!(ut, Val(K), Reval, tx, p)
                @test ut ≈ ut_ref atol = 1.0e-12

                # TaylorVector-parameter variant (Cauchy product path)
                tp = TaylorVector{N, ComplexF64}(0)
                fill!(ut, zero(ComplexF64))
                taylor!(ut, Val(K), Reval, tx, tp)
                @test ut ≈ ut_ref atol = 1.0e-12
            end
        end
    end

    @testset "RandomizedSystem permutation" begin
        @polyvar x y
        F = System([x - y, x^2 + y^2 - 1, x * y - 0.25])  # degrees 1, 2, 2
        A = FSMat{ComplexF64}(reshape(ComplexF64[0.3 - 0.7im, 1.1 + 0.2im], 2, 1))
        perm = [2, 3, 1]  # identity block on the two degree-2 equations
        R = RandomizedSystem(F.evaluator, A, perm)
        Reval = SystemEvaluator(R)
        p = FSVec{ComplexF64}(ComplexF64[])

        xv = randn(ComplexF64, 2)
        x_fs = FSVec{ComplexF64}(xv)
        u_full = FSVec{ComplexF64}(zeros(ComplexF64, 3))
        evaluate!(u_full, F.evaluator, x_fs, p)
        u_ref = [u_full[perm[i]] + A[i, 1] * u_full[perm[3]] for i in 1:2]
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u, Reval, x_fs, p)
        @test u ≈ u_ref atol = 1.0e-13

        # Non-permutation vectors are rejected (the fold indexes @inbounds)
        @test_throws ArgumentError RandomizedSystem(F.evaluator, A, [1, 1, 2])
        @test_throws ArgumentError RandomizedSystem(F.evaluator, A, [0, 1, 2])
    end

    # The fold G_i = F_i + Σ_j A[i,j]·F_j cancels at an excess solution (G = 0
    # while F stays O(1)). Rounding the inner F-residual to Float64 before the
    # fold floors |G| at 1e-16 there; folding in DF64 reaches the 1e-32 noise
    # floor. This is what the DF64-output evaluator path is for.
    @testset "RandomizedSystem DF64 extended precision" begin
        @polyvar x y
        F = System([x^2 + y^2 - 1, x - y, x * y - 0.5])
        A = FSMat{ComplexF64}(reshape(ComplexF64[0.5 + 0.25im, -1.0 + 2.0im], 2, 1))
        R = RandomizedSystem(F.evaluator, A, [1, 2, 3])
        Reval = SystemEvaluator(R)
        p = FSVec{ComplexF64}(ComplexF64[])

        # Locate an excess solution of G, then refine it with Newton in BigFloat.
        r = solve(F, TotalDegree(; seed = UInt32(7)), Serial(); show_progress = false)
        excess = filter(is_excess_solution, r.path_results)
        @test !isempty(excess)
        setprecision(BigFloat, 512) do
            a1, a2 = (big(real(a)) + im * big(imag(a)) for a in (A[1, 1], A[2, 1]))
            G(v) = [
                (v[1]^2 + v[2]^2 - 1) + a1 * (v[1] * v[2] - big"0.5"),
                (v[1] - v[2]) + a2 * (v[1] * v[2] - big"0.5"),
            ]
            J(v) = [
                2v[1] + a1 * v[2]  2v[2] + a1 * v[1]
                1 + a2 * v[2]      -1 + a2 * v[1]
            ]
            vb = [big(real(z)) + im * big(imag(z)) for z in excess[1].solution]
            for _ in 1:50
                vb .-= J(vb) \ G(vb)
            end
            @test maximum(abs.(G(vb))) < big"1e-100"
            @test abs(vb[1] * vb[2] - big"0.5") > 0.1  # genuinely excess: F_3 is O(1)

            todf(z) = ComplexDF64(
                HC.DoubleF64(Float64(real(z)), Float64(real(z) - big(Float64(real(z))))),
                HC.DoubleF64(Float64(imag(z)), Float64(imag(z) - big(Float64(imag(z))))),
            )
            x_df = FSVec{ComplexDF64}(todf.(vb))

            # DF64-input, F64-output: fold must happen in DF64 (old floor: 5.6e-17)
            u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            evaluate!(u, Reval, x_df, p)
            @test maximum(abs.(u)) < 1.0e-28

            # DF64-output variant keeps the extended-precision residual
            ū = FSVec{ComplexDF64}(zeros(ComplexDF64, 2))
            evaluate!(ū, Reval, x_df, p)
            @test maximum(Float64.(abs.(ū))) < 1.0e-28
        end
    end

    # ── End-to-end solve ─────────────────────────────────────────────────

    # x² = 1, y² = 1, xy = 1 has exactly the two solutions (1,1) and (-1,-1).
    # The squared-up 2×2 system generically has Bezout-many (4) solutions; the
    # extra endpoints must be flagged as excess, not returned as solutions.

    @testset "solve overdetermined: total degree ($(nameof(typeof(exec))))" for exec in (Serial(), Threaded())
        @polyvar x y
        F = System([x^2 - 1, y^2 - 1, x * y - 1])
        result = solve(F, TotalDegree(; seed = UInt32(0x42)), exec; show_progress = false)
        @test result.tracked_paths == 4
        @test nsolutions(result) == 2
        for s in solutions(result)
            @test abs(s[1]^2 - 1) < 1.0e-8
            @test abs(s[2]^2 - 1) < 1.0e-8
            @test abs(s[1] * s[2] - 1) < 1.0e-8
        end
        rsols = sort(real_solutions(result); by = first)
        @test length(rsols) == 2
        @test rsols[1] ≈ [-1.0, -1.0] atol = 1.0e-8
        @test rsols[2] ≈ [1.0, 1.0] atol = 1.0e-8
        # The non-genuine paths are excess solutions (or diverged); none of them
        # may be reported as a success.
        @test count(is_success, result.path_results) == 2
        @test nexcess_solutions(result) + nat_infinity(result) +
            count(r -> !is_success(r) && !is_excess_solution(r) && !HC.is_at_infinity(r), result.path_results) == 2
    end

    @testset "solve overdetermined: polyhedral ($(nameof(typeof(exec))))" for exec in (Serial(), Threaded())
        @polyvar x y
        F = System([x^2 - 1, y^2 - 1, x * y - 1])
        result = solve(F, Polyhedral(; seed = UInt32(0x42)), exec; show_progress = false)
        @test nsolutions(result) == 2
        for s in solutions(result)
            @test abs(s[1]^2 - 1) < 1.0e-8
            @test abs(s[2]^2 - 1) < 1.0e-8
            @test abs(s[1] * s[2] - 1) < 1.0e-8
        end
        @test count(is_success, result.path_results) == 2
    end

    @testset "solve overdetermined: seed reproducibility" begin
        @polyvar x y
        F = System([x^2 - 1, y^2 - 1, x * y - 1])
        r1 = solve(F, TotalDegree(; seed = UInt32(7)), Serial(); show_progress = false)
        r2 = solve(F, TotalDegree(; seed = UInt32(7)), Serial(); show_progress = false)
        @test sort(map(first, solutions(r1)); by = real) ≈
            sort(map(first, solutions(r2)); by = real)
        @test nexcess_solutions(r1) == nexcess_solutions(r2)
    end

    @testset "solve overdetermined: singular solution" begin
        # (1,1) is a singular solution: the Jacobian of [(x-1)², y-1, (x-1)y]
        # drops rank at x=1.
        @polyvar x y
        F = System([(x - 1)^2, y - 1, (x - 1) * y])
        result = solve(F, TotalDegree(; seed = UInt32(3)), Serial(); show_progress = false)
        @test nresults(result) >= 1
        found = any(results(result)) do r
            abs(r.solution[1] - 1) < 1.0e-4 && abs(r.solution[2] - 1) < 1.0e-4
        end
        @test found
    end

    @testset "underdetermined systems throw" begin
        @polyvar x y
        F = System([x * y - 1])
        @test_throws ArgumentError solve(F, TotalDegree(; seed = UInt32(1)), Serial(); show_progress = false)
        @test_throws ArgumentError solve(F, Polyhedral(; seed = UInt32(1)), Serial(); show_progress = false)
    end

    # 10 equations of degree 6 in 3 variables. The squared-up system tracks
    # 6³ = 216 paths of which exactly 80 end at solutions of the original
    # system and 136 are excess solutions of the randomization.
    @testset "3 by 5 minors" begin
        F = System(minors_polys())
        @test size(F.evaluator) == (10, 3)
        result = solve(F, TotalDegree(; seed = UInt32(0x1234)), Threaded(); show_progress = false)
        @test result.tracked_paths == 216
        @test count(is_success, result.path_results) == 80
        @test nexcess_solutions(result) == 136
    end

    # Underdetermined input throws for total degree and polyhedral, both
    # affine and projective.
    @testset "underdetermined throws" begin
        @polyvar x y z
        affine_under = System([2.3 * x^2 + 1.2 * y^2 + 3 * x - 2 * y + 3])
        @test_throws ArgumentError solve(affine_under, TotalDegree(; seed = UInt32(2)), Serial(); show_progress = false)
        @test_throws ArgumentError solve(affine_under, Polyhedral(; seed = UInt32(2)), Serial(); show_progress = false)

        proj_under = System([2.3 * x^2 + 1.2 * y^2 + 3 * x * z])
        @test_throws ArgumentError solve(proj_under, TotalDegree(; seed = UInt32(2)), Serial(); show_progress = false)
        @test_throws ArgumentError solve(proj_under, Polyhedral(; seed = UInt32(2)), Serial(); show_progress = false)
    end

    # Parameter homotopy: underdetermined systems throw, affine and
    # projective variants.
    @testset "parameter homotopy underdetermined throws" begin
        @polyvar x y z a b
        F = System([x^2 - a]; variables = [x, y], parameters = [a, b])
        @test_throws ArgumentError solve(
            F, [[1.0, 1.0]], [1, 0], [2, 4], Serial(); show_progress = false,
        )

        F_proj = System([x * y + (b - a) * z^2]; variables = [x, y, z], parameters = [a, b])
        @test_throws ArgumentError solve(
            F_proj, [[1.0, 1.0, 1.0]], [1, 0], [2, 4], Serial(); show_progress = false,
        )
    end

    # Evaluation consistency for a randomized parametric system
    # (3 equations, 2 variables, 2 parameters).
    @testset "RandomizedSystem with parameters" begin
        @polyvar x y a b
        g = System(
            [
                x^2 + 3 * x * y + y^2 * b,
                (x + 3y - 2)^2 / 2,
                3 * a * b * (2x - y + y^3 - 4x^2),
            ];
            parameters = [a, b],
        )
        A = FSMat{ComplexF64}(reshape(ComplexF64[0.8 - 0.3im, -0.4 + 1.6im], 2, 1))
        perm = [1, 2, 3]
        R = RandomizedSystem(g.evaluator, A, perm)
        @test size(R) == (2, 2)
        @test nparameters(R) == 2
        Reval = SystemEvaluator(R)

        for _ in 1:5
            xv = randn(ComplexF64, 2)
            pv = randn(ComplexF64, 2)
            x_fs = FSVec{ComplexF64}(xv)
            p_fs = FSVec{ComplexF64}(pv)

            u_full = FSVec{ComplexF64}(zeros(ComplexF64, 3))
            U_full = FSMat{ComplexF64}(zeros(ComplexF64, 3, 2))
            evaluate_and_jacobian!(u_full, U_full, g.evaluator, x_fs, p_fs)
            u_ref = [u_full[i] + A[i, 1] * u_full[3] for i in 1:2]
            U_ref = [U_full[i, j] + A[i, 1] * U_full[3, j] for i in 1:2, j in 1:2]

            u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
            evaluate!(u, Reval, x_fs, p_fs)
            @test u ≈ u_ref atol = 1.0e-13

            x_df = FSVec{ComplexDF64}(ComplexDF64.(xv))
            fill!(u, zero(ComplexF64))
            evaluate!(u, Reval, x_df, p_fs)
            @test u ≈ u_ref atol = 1.0e-13

            fill!(u, zero(ComplexF64))
            evaluate_and_jacobian!(u, U, Reval, x_fs, p_fs)
            @test u ≈ u_ref atol = 1.0e-13
            @test U ≈ U_ref atol = 1.0e-13

            for K in 1:3
                N = K + 1
                tx = TaylorVector{N, ComplexF64}(2)
                for i in 1:2
                    tx[i] = TruncatedTaylorSeries(ntuple(_ -> randn(ComplexF64), Val(N)))
                end
                ut_full = FSVec{ComplexF64}(zeros(ComplexF64, 3))
                taylor!(ut_full, Val(K), g.evaluator, tx, p_fs)
                ut_ref = [ut_full[i] + A[i, 1] * ut_full[3] for i in 1:2]
                ut = FSVec{ComplexF64}(zeros(ComplexF64, 2))
                taylor!(ut, Val(K), Reval, tx, p_fs)
                @test ut ≈ ut_ref atol = 1.0e-12

                # TaylorVector-parameter variant (Cauchy product path)
                tp = TaylorVector{N, ComplexF64}(2)
                for i in 1:2
                    tp[i] = TruncatedTaylorSeries(
                        ntuple(k -> k == 1 ? pv[i] : randn(ComplexF64), Val(N)),
                    )
                end
                fill!(ut_full, zero(ComplexF64))
                taylor!(ut_full, Val(K), g.evaluator, tx, tp)
                ut_ref = [ut_full[i] + A[i, 1] * ut_full[3] for i in 1:2]
                fill!(ut, zero(ComplexF64))
                taylor!(ut, Val(K), Reval, tx, tp)
                @test ut ≈ ut_ref atol = 1.0e-12
            end
        end
    end

    @testset "square systems unaffected" begin
        @polyvar x y
        F = System([x^2 - 1, y^2 - 4])
        result = solve(F, TotalDegree(; seed = UInt32(11)), Serial(); show_progress = false)
        @test nsolutions(result) == 4
        @test nexcess_solutions(result) == 0
    end
end
