using Test
using HomotopyContinuation
using HomotopyContinuation:
    ComplexDF64, DoubleF64, FSMat, FSVec, RandomizedSystem, SystemEvaluator, evaluate!

@testset "RandomizedSystem extended-precision cancellation" begin
    @polyvar x y
    F = System([x^2 + y^2 - 1, x - y, x * y - 0.5])
    A = FSMat{ComplexF64}(
        reshape(ComplexF64[0.5 + 0.25im, -1.0 + 2.0im], 2, 1),
    )
    randomized = SystemEvaluator(RandomizedSystem(F.evaluator, A, [1, 2, 3]))
    p = FSVec{ComplexF64}(ComplexF64[])

    # A public overdetermined solve supplies a point in the excess branch. Refine
    # that branch against this fixed randomized square system in high precision.
    result = solve(F, TotalDegree(; seed = UInt32(7), show_progress = false), Serial())
    excess = filter(is_excess_solution, path_results(result))
    @test !isempty(excess)

    setprecision(BigFloat, 512) do
        a1, a2 = (
            big(real(a)) + im * big(imag(a)) for a in (A[1, 1], A[2, 1])
        )
        G(v) = [
            (v[1]^2 + v[2]^2 - 1) + a1 * (v[1] * v[2] - big"0.5"),
            (v[1] - v[2]) + a2 * (v[1] * v[2] - big"0.5"),
        ]
        J(v) = [
            2v[1] + a1 * v[2]  2v[2] + a1 * v[1]
            1 + a2 * v[2]      -1 + a2 * v[1]
        ]

        vb = [
            big(real(z)) + im * big(imag(z)) for z in solution(first(excess))
        ]
        for _ in 1:50
            vb .-= J(vb) \ G(vb)
        end

        @test maximum(abs.(G(vb))) < big"1e-100"
        # The point is genuinely excess: the discarded original equation is O(1).
        @test abs(vb[1] * vb[2] - big"0.5") > 0.1

        function to_df64(z)
            zr = real(z)
            zi = imag(z)
            hr = Float64(zr)
            hi = Float64(zi)
            return ComplexDF64(
                DoubleF64(hr, Float64(zr - big(hr))),
                DoubleF64(hi, Float64(zi - big(hi))),
            )
        end

        x_df = FSVec{ComplexDF64}(to_df64.(vb))

        # If the inner residual were rounded to Float64 before the randomized
        # fold, cancellation would stop near 1e-16. Both output paths must retain
        # the extended-precision fold and reach the double-double noise floor.
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u, randomized, x_df, p)
        @test maximum(abs.(u)) < 1.0e-28

        u_df = FSVec{ComplexDF64}(zeros(ComplexDF64, 2))
        evaluate!(u_df, randomized, x_df, p)
        @test maximum(Float64.(abs.(u_df))) < 1.0e-28
    end
end
