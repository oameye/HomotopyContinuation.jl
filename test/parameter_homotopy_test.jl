using Test, Random
using HomotopyContinuationNext
using HomotopyContinuationNext: HomotopyEvaluator, ParameterHomotopy, parameters!,
    evaluate!, evaluate_and_jacobian!, taylor!, FSVec, FSMat, TaylorVector,
    start_parameters!, target_parameters!
using HomotopyContinuationNext: Tracker, TrackerCode, track!
using DynamicPolynomials: @polyvar, subs, differentiate

@testset "ParameterHomotopy" begin
    Random.seed!(0x5eed)
    @polyvar x[1:2] p[1:2]
    # Nonlinear in the parameters on purpose (prototype 6 system class).
    polys = [
        x[1]^2 * p[1]^2 + x[2] * p[2]^3 + p[1] * p[2] * x[1] * x[2] - 1,
        x[1] + x[2] - p[1],
    ]
    F = System(polys; variables = x, parameters = p)
    pstart = [1.0 + 0.2im, -0.7 + 1.1im]
    ptarget = [0.3 - 0.9im, 1.4 + 0.5im]
    H = ParameterHomotopy(F.evaluator, pstart, ptarget)

    p_of(t) = t .* pstart .+ (1 .- t) .* ptarget
    xv = [0.4 + 0.3im, -1.2 + 0.1im]
    xf = FSVec{ComplexF64}(ComplexF64.(xv))
    u = FSVec{ComplexF64}(zeros(ComplexF64, 2))

    @testset "evaluate! matches F(x; p(t))" begin
        for t in (complex(0.37), 0.2 + 0.6im)
            evaluate!(u, H, xf, ComplexF64(t))
            expected = [ComplexF64(f(x => xv, p => p_of(t))) for f in polys]
            @test u ≈ expected atol = 1.0e-13
        end
    end

    @testset "evaluate_and_jacobian!" begin
        t = 0.2 + 0.6im
        U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        evaluate_and_jacobian!(u, U, H, xf, ComplexF64(t))
        J = [
            ComplexF64(differentiate(f, xj)(x => xv, p => p_of(t)))
                for f in polys, xj in x
        ]
        @test U ≈ J atol = 1.0e-13
    end

    @testset "taylor! Val(1) is the exact tangent" begin
        # d/dt F(x; p(t)) via the symbolic series in s: F(x; p(t) + s*(pstart - ptarget))
        t = complex(0.37)
        @polyvar s
        dp = pstart .- ptarget
        series = [subs(f, x => xv, p => p_of(t) .+ s .* dp) for f in polys]
        expected = [ComplexF64(differentiate(g, s)(s => 0.0)) for g in series]
        taylor!(u, Val(1), H, xf, ComplexF64(t))
        @test u ≈ expected atol = 1.0e-12
    end

    @testset "taylor! Val(2)/Val(3) match the symbolic series" begin
        # constant path: order-k coefficient of s ↦ F(x; p(t0) + s*(pstart - ptarget))
        t0 = complex(0.37)
        @polyvar s
        dp = pstart .- ptarget
        series = [subs(f, x => xv, p => p_of(t0) .+ s .* dp) for f in polys]
        dseries = series
        for k in (1, 2, 3)
            dseries = differentiate.(dseries, s)
            expected = [ComplexF64(g(s => 0.0)) / factorial(k) for g in dseries]
            if k == 1
                taylor!(u, Val(1), H, xf, ComplexF64(t0))
            else
                tv = TaylorVector{k + 1, ComplexF64}(2)
                tv.data[1, :] .= ComplexF64.(xv)
                taylor!(u, Val(k), H, tv, ComplexF64(t0))
            end
            @test u ≈ expected atol = 1.0e-11
        end
    end

    @testset "parameters! retarget invalidates caches" begin
        t = complex(0.5)
        evaluate!(u, H, xf, ComplexF64(t))
        before = copy(Vector(u))
        p2 = [2.0 + 0.0im, -1.0 + 0.5im]
        q2 = [0.1 + 0.1im, 0.9 - 0.2im]
        parameters!(H, p2, q2)
        evaluate!(u, H, xf, ComplexF64(t))
        expected = [ComplexF64(f(x => xv, p => (t .* p2 .+ (1 - t) .* q2))) for f in polys]
        @test u ≈ expected atol = 1.0e-13
        @test !(u ≈ before)
    end

    @testset "tracks a nonlinear-in-p path in few steps" begin
        @polyvar y q
        G = System([y^2 - q]; variables = [y], parameters = [q])
        Hg = ParameterHomotopy(G.evaluator, [1.0 + 0im], [9.0 + 0im])
        tracker = Tracker(HomotopyEvaluator(Hg))
        code = track!(tracker, [1.0 + 0.0im])
        @test code == TrackerCode.TRACKER_SUCCESS
        @test tracker.state.x[1] ≈ 3.0 atol = 1.0e-10
        # CoefficientHomotopy needed 204 accepted steps here (invalid tangent shortcut).
        @test tracker.state.accepted_steps < 20
    end

    @testset "solve parameter homotopy step-count regression" begin
        @polyvar y q
        G = System([y^2 - q]; variables = [y], parameters = [q])
        res = solve(
            G, [[1.0 + 0.0im]], Serial();
            start_parameters = [1.0 + 0im], target_parameters = [9.0 + 0im],
            seed = UInt32(1), show_progress = false,
        )
        @test nsolutions(res) == 1
        r = first(path_results(res))
        @test solution(r)[1] ≈ 3.0 atol = 1.0e-8
        # With CoefficientHomotopy's invalid Val(1) shortcut this took 204 accepted
        # steps; the exact tangent needs a handful.
        @test accepted_steps(r) < 20
    end
end
