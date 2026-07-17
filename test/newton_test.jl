using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: System, newton, NewtonResult, NewtonCache, is_success
using DynamicPolynomials: @polyvar

@testset "Standalone newton(F, x₀)" begin

    # ── Square system: circle ∩ diagonal, roots at ±(1/√2, 1/√2) ──────────
    @polyvar x y
    F = System([x^2 + y^2 - 1, x - y])
    root = 1 / sqrt(2)

    @testset "converges from a nearby guess" begin
        r = newton(F, [root + 0.05, root - 0.05])
        @test r isa NewtonResult
        @test is_success(r)
        @test r.residual < 1.0e-12
        @test abs(r.x[1] - root) < 1.0e-8
        @test abs(r.x[2] - root) < 1.0e-8
        @test r.iters >= 1
    end

    @testset "converges to the other real root" begin
        r = newton(F, [-root - 0.05, -root + 0.05])
        @test is_success(r)
        @test abs(r.x[1] + root) < 1.0e-8
        @test abs(r.x[2] + root) < 1.0e-8
    end

    @testset "extended precision reaches near machine-eps residual" begin
        r = newton(F, [root + 0.05, root - 0.05]; extended_precision = true)
        @test is_success(r)
        @test r.residual < 1.0e-14
    end

    @testset "extended_precision defaults to false" begin
        # The default call must reproduce the explicit `extended_precision = false`
        # call bit-for-bit, not the extended-precision one.
        x0 = [root + 0.05, root - 0.05]
        @test newton(F, x0).x == newton(F, x0; extended_precision = false).x
    end

    @testset "reports max_iters when it cannot converge in the budget" begin
        r = newton(F, [root + 0.05, root - 0.05]; max_iters = 1)
        @test !is_success(r)
        @test r.return_code == HC.NewtonReturnCode.NEWTON_MAX_ITERS
        @test r.iters == 1
    end

    # ── Overdetermined but consistent: adds x·y − 1/2, satisfied at the root ─
    @testset "overdetermined consistent system (least squares)" begin
        G = System([x^2 + y^2 - 1, x - y, x * y - 0.5])
        r = newton(G, [root + 0.02, root - 0.02])
        @test is_success(r)
        @test r.residual < 1.0e-10
        @test abs(r.x[1] - root) < 1.0e-8
    end

    # ── Underdetermined (m < n): column-pivoted QR least-squares step ─────
    @testset "underdetermined system converges to a point on the variety" begin
        # One equation, two unknowns: the unit circle.
        C = System([x^2 + y^2 - 1])
        r = newton(C, [1.1, 0.2])
        @test is_success(r)
        @test abs(r.x[1]^2 + r.x[2]^2 - 1) < 1.0e-10
    end

    @testset "underdetermined with preallocated cache" begin
        C = System([x^2 + y^2 - 1])
        cache = NewtonCache(C)
        r1 = newton(C, [1.1, 0.2]; cache = cache)
        r2 = newton(C, [0.1, -1.3]; cache = cache)
        @test is_success(r1)
        @test is_success(r2)
        @test abs(r2.x[1]^2 + r2.x[2]^2 - 1) < 1.0e-10
    end

    @testset "underdetermined extended precision" begin
        C = System([x^2 + y^2 - 1])
        r = newton(C, [1.1, 0.2]; extended_precision = true)
        @test is_success(r)
        @test r.residual < 1.0e-14
    end

    # ── Preallocated cache: repeated calls reuse the workspace ─────────────
    @testset "preallocated NewtonCache" begin
        cache = NewtonCache(F)
        r1 = newton(F, [root + 0.05, root - 0.05]; cache = cache)
        r2 = newton(F, [-root - 0.05, -root + 0.05]; cache = cache)
        @test is_success(r1)
        @test is_success(r2)
    end

    # ── First-update rejection guard ──────────────────────────────────────
    @testset "rejects a wild first update" begin
        r = newton(F, [root + 0.05, root - 0.05]; max_abs_norm_first_update = 1.0e-6)
        @test !is_success(r)
        @test r.return_code == HC.NewtonReturnCode.NEWTON_REJECTED
    end

end
