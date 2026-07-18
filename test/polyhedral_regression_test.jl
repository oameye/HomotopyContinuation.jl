# Regression tests for polyhedral homotopy fixes:
# 1. Taylor cross-term correctness (ToricHomotopy and CoefficientHomotopy)
# 2. Two-stage toric reparameterization (max_weight >= 10)
# 3. Toric+coefficient step counter aggregation

using Test

using HomotopyContinuationNext
using HomotopyContinuationNext:
    ToricHomotopy, CoefficientHomotopy, HomotopyEvaluator,
    SystemEvaluator, System, FSVec, FSMat, TaylorVector, PathResultCode,
    update_weights!, evaluate!, evaluate_and_jacobian!, taylor!,
    _update_toric_coeffs!, _update_toric_dt_coeffs!,
    _update_toric_d2t_coeffs!, _update_toric_d3t_coeffs!,
    _copy_prefix!, _update_p!, PolyhedralSolveCache, EndgameCode

using DynamicPolynomials: @polyvar
using CommonSolve: init as cs_init, solve! as cs_solve!

@testset "Polyhedral regression" begin

    # ── 1. Taylor cross-term correctness ─────────────────────────────────

    @testset "CoefficientHomotopy Taylor order 2 includes cross term" begin
        # Build F(x; p) = p₁x² + p₂x + p₃ with p(t) = t·start + (1-t)·target.
        # Analytically:
        #   [H]₂ = [B(x)]₂ · p₀ + [B(x)]₁ · p₁
        # where [B(x)]₁ · p₁ is the cross term added by the fix.
        # We verify this by comparing taylor!(Val(2), H, ...) against the
        # system-only result (which lacks the cross term).
        @polyvar px pc1 pc2 pc3
        param_sys = System(
            [pc1 * px^2 + pc2 * px + pc3];
            variables = [px], parameters = [pc1, pc2, pc3],
        )

        start = ComplexF64[2.0, 3.0, 1.0]
        target = ComplexF64[1.0, -1.0, 2.0]
        H = CoefficientHomotopy(param_sys.evaluator, start, target)

        n = 1
        x0 = FSVec{ComplexF64}(ComplexF64[0.7 + 0.3im])
        t0 = complex(0.6)

        # Compute x₁ via implicit differentiation: J·x₁ = -H_t
        u_jac = FSVec{ComplexF64}(zeros(ComplexF64, n))
        U_jac = FSMat{ComplexF64}(zeros(ComplexF64, n, n))
        evaluate_and_jacobian!(u_jac, U_jac, H, x0, t0)
        u1 = FSVec{ComplexF64}(zeros(ComplexF64, n))
        taylor!(u1, Val(1), H, x0, t0)
        x1_val = -(U_jac[1, 1] \ u1[1])

        # Build TaylorVector: [x₀, x₁, 0]
        tx = TaylorVector{3, ComplexF64}(n)
        tx.data[1, 1] = x0[1]
        tx.data[2, 1] = x1_val
        tx.data[3, 1] = zero(ComplexF64)

        # Full homotopy Taylor order 2 (with cross term fix)
        u2_full = FSVec{ComplexF64}(zeros(ComplexF64, n))
        taylor!(u2_full, Val(2), H, tx, t0)

        # System-only Taylor order 2 (no cross term — what old code computed)
        _update_p!(H, t0)
        u2_system_only = FSVec{ComplexF64}(zeros(ComplexF64, n))
        taylor!(u2_system_only, Val(2), param_sys.evaluator, tx, H.p_t)

        # The cross term [B(x)]₁ · dp should make these differ
        dp = start - target
        @test any(!iszero, dp)
        @test !iszero(x1_val)
        cross_term = u2_full[1] - u2_system_only[1]
        @test abs(cross_term) > 1.0e-10

        # Verify the cross term equals taylor!(Val(1), system, tx[:2], dt_coeffs)
        tx1 = TaylorVector{2, ComplexF64}(n)
        tx1.data[1, 1] = tx.data[1, 1]
        tx1.data[2, 1] = tx.data[2, 1]
        u_cross = FSVec{ComplexF64}(zeros(ComplexF64, n))
        taylor!(u_cross, Val(1), param_sys.evaluator, tx1, H.dp)
        @test cross_term ≈ u_cross[1] atol = 1.0e-12
    end

    @testset "CoefficientHomotopy Taylor via Cauchy product matches multi-call" begin
        # Build F(x; p) = p₁x² + p₂x + p₃
        @polyvar crx crp1 crp2 crp3
        param_sys = System(
            [crp1 * crx^2 + crp2 * crx + crp3];
            variables = [crx], parameters = [crp1, crp2, crp3],
        )

        start = ComplexF64[2.0, 3.0, 1.0]
        target = ComplexF64[1.0, -1.0, 2.0]
        H = CoefficientHomotopy(param_sys.evaluator, start, target)

        x0 = FSVec{ComplexF64}(ComplexF64[0.7 + 0.3im])
        t0 = complex(0.6)

        # Compute x₁ for the TaylorVector
        u_jac = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        U_jac = FSMat{ComplexF64}(zeros(ComplexF64, 1, 1))
        evaluate_and_jacobian!(u_jac, U_jac, H, x0, t0)
        u1 = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        taylor!(u1, Val(1), H, x0, t0)
        x1 = -(U_jac[1, 1] \ u1[1])

        # Build tx for order 2
        tx2 = TaylorVector{3, ComplexF64}(1)
        tx2.data[1, 1] = x0[1]; tx2.data[2, 1] = x1; tx2.data[3, 1] = zero(ComplexF64)

        # Reference: compute via multi-call (the old approach)
        _update_p!(H, t0)
        u_sys = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        taylor!(u_sys, Val(2), param_sys.evaluator, tx2, H.p_t)  # [B(x)]₂ · p₀
        tx1 = TaylorVector{2, ComplexF64}(1)
        tx1.data[1, 1] = x0[1]; tx1.data[2, 1] = x1
        u_cross = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        taylor!(u_cross, Val(1), param_sys.evaluator, tx1, H.dp)  # [B(x)]₁ · p₁
        ref_result = u_sys[1] + u_cross[1]

        # Actual: compute via homotopy (uses Cauchy product internally)
        u_hom = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        taylor!(u_hom, Val(2), H, tx2, t0)

        @test u_hom[1] ≈ ref_result atol = 1.0e-12
    end

    @testset "ToricHomotopy higher-order parameter derivatives" begin
        @polyvar tx tc1 tc2 tc3
        param_sys = System(
            [tc1 * tx^2 + tc2 * tx + tc3];
            variables = [tx], parameters = [tc1, tc2, tc3],
        )

        start_coeffs = [ComplexF64[2.0, 3.0, 1.0]]
        H = ToricHomotopy(param_sys.evaluator, start_coeffs)

        # Manually set weights: w = [3, 1, 0]
        H.weights[1] = 3.0
        H.weights[2] = 1.0
        H.weights[3] = 0.0
        H.t_cache[] = complex(NaN)
        H.dt_cache[] = complex(NaN)
        H.d2t_cache[] = complex(NaN)
        H.d3t_cache[] = complex(NaN)

        t = complex(0.5)
        _update_toric_coeffs!(H, t)
        _update_toric_dt_coeffs!(H, t)
        _update_toric_d2t_coeffs!(H, t)
        _update_toric_d3t_coeffs!(H, t)

        # p₀ = c * t^w
        @test H.coeffs[1] ≈ 2.0 * 0.5^3.0
        @test H.coeffs[2] ≈ 3.0 * 0.5^1.0
        @test H.coeffs[3] ≈ 1.0  # w=0 → t^0 = 1

        # p₁ = w * c * t^{w-1}
        @test H.dt_coeffs[1] ≈ 3.0 * 2.0 * 0.5^2.0
        @test H.dt_coeffs[2] ≈ 1.0 * 3.0 * 0.5^0.0
        @test H.dt_coeffs[3] ≈ 0.0

        # p₂ = w(w-1)/2 * c * t^{w-2}
        @test H.d2t_coeffs[1] ≈ 3.0 * 2.0 / 2.0 * 2.0 * 0.5^1.0
        @test H.d2t_coeffs[2] ≈ 0.0  # w=1 → w(w-1) = 0
        @test H.d2t_coeffs[3] ≈ 0.0

        # p₃ = w(w-1)(w-2)/6 * c * t^{w-3}
        @test H.d3t_coeffs[1] ≈ 3.0 * 2.0 * 1.0 / 6.0 * 2.0 * 0.5^0.0
        @test H.d3t_coeffs[2] ≈ 0.0
        @test H.d3t_coeffs[3] ≈ 0.0
    end

    @testset "ToricHomotopy Taylor via Cauchy product matches multi-call" begin
        @polyvar ttx ttc1 ttc2 ttc3
        param_sys = System(
            [ttc1 * ttx^2 + ttc2 * ttx + ttc3];
            variables = [ttx], parameters = [ttc1, ttc2, ttc3],
        )

        start_coeffs = [ComplexF64[2.0, 3.0, 1.0]]
        H = ToricHomotopy(param_sys.evaluator, start_coeffs)
        H.weights[1] = 3.0; H.weights[2] = 1.0; H.weights[3] = 0.0
        H.t_cache[] = complex(NaN); H.dt_cache[] = complex(NaN)
        H.d2t_cache[] = complex(NaN); H.d3t_cache[] = complex(NaN)

        t = complex(0.5)
        x0 = FSVec{ComplexF64}(ComplexF64[0.7 + 0.3im])

        # Compute x₁
        u_jac = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        U_jac = FSMat{ComplexF64}(zeros(ComplexF64, 1, 1))
        evaluate_and_jacobian!(u_jac, U_jac, H, x0, t)
        u1 = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        taylor!(u1, Val(1), H, x0, t)
        x1 = -(U_jac[1, 1] \ u1[1])

        # Build tx for order 2
        tx2 = TaylorVector{3, ComplexF64}(1)
        tx2.data[1, 1] = x0[1]; tx2.data[2, 1] = x1; tx2.data[3, 1] = zero(ComplexF64)

        # Reference: multi-call approach (old code logic)
        _update_toric_coeffs!(H, t)
        _update_toric_dt_coeffs!(H, t)
        _update_toric_d2t_coeffs!(H, t)
        u_p0 = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        taylor!(u_p0, Val(2), param_sys.evaluator, tx2, H.coeffs)  # [B]₂·p₀
        tx1 = TaylorVector{2, ComplexF64}(1)
        tx1.data[1, 1] = x0[1]; tx1.data[2, 1] = x1
        u_p1 = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        taylor!(u_p1, Val(1), param_sys.evaluator, tx1, H.dt_coeffs)  # [B]₁·p₁
        u_p2 = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        evaluate!(u_p2, param_sys.evaluator, x0, H.d2t_coeffs)  # [B]₀·p₂
        ref_result = u_p0[1] + u_p1[1] + u_p2[1]

        # Actual: Cauchy product via homotopy
        u_hom = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        taylor!(u_hom, Val(2), H, tx2, t)

        @test u_hom[1] ≈ ref_result atol = 1.0e-12
    end

    # ── 2. Two-stage toric reparameterization ────────────────────────────

    @testset "Polyhedral: cyclic-4 high-weight cells mostly succeed" begin
        # cyclic-4 has mixed cells with max_weight up to 30+, triggering two-stage.
        # Most paths should succeed; allow up to 2 failures from the hardest cells.
        @polyvar c1 c2 c3 c4
        F = System(
            [
                c1 + c2 + c3 + c4,
                c1 * c2 + c2 * c3 + c3 * c4 + c4 * c1,
                c1 * c2 * c3 + c2 * c3 * c4 + c3 * c4 * c1 + c4 * c1 * c2,
                c1 * c2 * c3 * c4 - 1,
            ]
        )
        r = solve(F, Polyhedral(; seed = UInt32(42)))
        @test r.tracked_paths == 16  # mixed volume of cyclic-4
        n_success = count(
            p -> p.return_code == PathResultCode.PATH_SUCCESS, r.path_results,
        )
        @test n_success >= 14  # at least 14/16 paths succeed
        @test nresults(r) > 0
    end

    # ── 3. Toric+coefficient step counter aggregation ────────────────────

    @testset "Polyhedral: PathResult includes toric phase steps" begin
        @polyvar sx sy
        F = System([sx^2 + sy - 1, sx * sy - 2])
        r = solve(F, Polyhedral(; seed = UInt32(123)))

        for p in r.path_results
            if p.return_code == PathResultCode.PATH_SUCCESS
                # Both phases take ≥ 1 accepted step, so combined must be ≥ 2
                @test p.accepted_steps >= 2
            end
        end

        # Total steps should be positive
        total_steps = sum(p.accepted_steps + p.rejected_steps for p in r.path_results)
        @test total_steps > 0

        # Second solve with same System + seed should also find solutions
        r2 = solve(F, Polyhedral(; seed = UInt32(123)))
        @test nsolutions(r2) == nsolutions(r)
        # Step counts should also include toric phase
        for p in r2.path_results
            if p.return_code == PathResultCode.PATH_SUCCESS
                @test p.accepted_steps >= 2
            end
        end
    end

    @testset "Polyhedral: toric steps included in count" begin
        # Directly verify that reported steps exceed what a single phase would report.
        # The toric phase of x²+y-1, xy-2 takes ~5-10 accepted steps.
        # The coefficient phase takes ~10-15 accepted steps.
        # The combined PathResult should reflect both.
        @polyvar qx qy
        F = System([qx^2 + qy - 1, qx * qy - 2])
        r = solve(F, Polyhedral(; seed = UInt32(456)))

        # Every successful path should have more than ~5 steps (even a single phase
        # needs several). With two phases combined, expect at least 10.
        for p in r.path_results
            if p.return_code == PathResultCode.PATH_SUCCESS
                @test p.accepted_steps >= 10
            end
        end
    end
end
