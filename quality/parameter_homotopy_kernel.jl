using Test, Random
using HomotopyContinuation
using HomotopyContinuation: parameters!, FSVec, FSMat, TaylorVector
using DynamicPolynomials: @polyvar, subs, differentiate

@testset "ParameterHomotopy numerical kernel" begin
    Random.seed!(0x5eed)
    @polyvar x[1:2] p[1:2]
    polys = [
        x[1]^2 * p[1]^2 + x[2] * p[2]^3 + p[1] * p[2] * x[1] * x[2] - 1,
        x[1] + x[2] - p[1],
    ]
    F = System(polys; variables = x, parameters = p)
    pstart = [1.0 + 0.2im, -0.7 + 1.1im]
    ptarget = [0.3 - 0.9im, 1.4 + 0.5im]
    H = ParameterHomotopy(F, pstart, ptarget)

    p_of(t) = t .* pstart .+ (1 .- t) .* ptarget
    xv = [0.4 + 0.3im, -1.2 + 0.1im]
    xf = FSVec{ComplexF64}(ComplexF64.(xv))
    u = FSVec{ComplexF64}(zeros(ComplexF64, 2))

    @testset "value and Jacobian follow parameter interpolation" begin
        for t in (complex(0.37), 0.2 + 0.6im)
            evaluate!(u, H, xf, ComplexF64(t))
            expected = [ComplexF64(f(x => xv, p => p_of(t))) for f in polys]
            @test u ≈ expected atol = 1.0e-13
        end

        t = 0.2 + 0.6im
        U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        evaluate_and_jacobian!(u, U, H, xf, ComplexF64(t))
        expected_J = [
            ComplexF64(differentiate(f, xj)(x => xv, p => p_of(t)))
                for f in polys, xj in x
        ]
        @test U ≈ expected_J atol = 1.0e-13
    end

    @testset "Taylor coefficients match symbolic parameter series" begin
        t0 = complex(0.37)
        @polyvar s
        dp = pstart .- ptarget
        series = [subs(f, x => xv, p => p_of(t0) .+ s .* dp) for f in polys]
        dseries = series
        for k in 1:3
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

    @testset "retargeting invalidates the interpolation cache" begin
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
end
