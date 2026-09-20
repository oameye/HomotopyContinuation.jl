using Test
import HomotopyContinuation as HC

@testset "endgame numerical kernel" begin
    @testset "winding-number estimator" begin
        for (slopes, expected) in (
                ((1.02, 0.98), 1),
                ((0.51, 0.49), 2),
                ((0.34, 0.33), 3),
            )
            val = HC.Valuation(2)
            val.val_tẋ[1] = slopes[1]
            val.val_tẋ[2] = slopes[2]

            winding, error = HC.estimate_winding_number(val, 2, 6)
            @test winding == expected
            @test error < 0.05
        end
    end

    @testset "valuation derivative agrees with finite difference" begin
        x_of_t(t) = ComplexF64(1 + t + t^2)
        ẋ_of_t(t) = ComplexF64(1 + 2t)
        t = 0.5
        h = 1.0e-7

        ν, dν_dt = HC._val_dval(x_of_t(t), ẋ_of_t(t), ComplexF64(2.0), t)
        ν_fd = (
            HC._val(x_of_t(t + h), ẋ_of_t(t + h), t + h) -
                HC._val(x_of_t(t - h), ẋ_of_t(t - h), t - h)
        ) / (2h)

        @test ν ≈ HC._val(x_of_t(t), ẋ_of_t(t), t) atol = 1.0e-12
        @test dν_dt ≈ ν_fd rtol = 1.0e-6 atol = 1.0e-8
    end

    @testset "cubic Hermite extrapolation is exact for a linear path" begin
        ty0 = HC.TaylorVector{2, ComplexF64}(1)
        ty1 = HC.TaylorVector{2, ComplexF64}(1)
        ty0.data[1, 1] = 3.5 + 0im
        ty0.data[2, 1] = 3.0 + 0im
        ty1.data[1, 1] = 2.75 + 0im
        ty1.data[2, 1] = 3.0 + 0im

        x_hat = HC.FSVec{ComplexF64}(zeros(ComplexF64, 1))
        HC.cubic_hermite!(x_hat, ty0, 0.5, ty1, 0.25, 0.0)

        @test abs(x_hat[1] - 2.0) < 1.0e-10
    end
end
