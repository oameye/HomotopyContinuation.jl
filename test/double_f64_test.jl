using Test
using HomotopyContinuationNext: DoubleF64, ComplexDF64

@testset "DoubleF64" begin
    @testset "construction and conversion" begin
        x = DoubleF64(1.0)
        @test x.hi == 1.0
        @test x.lo == 0.0

        y = DoubleF64(big"3.141592653589793238462643383279502884197")
        @test abs(BigFloat(y) - big(π)) < 1e-30

        @test DoubleF64(1) == DoubleF64(1.0)
        @test Float64(DoubleF64(3.14)) == 3.14
    end

    @testset "arithmetic vs BigFloat" begin
        a = DoubleF64(big"1.23456789012345678901234567890")
        b = DoubleF64(big"9.87654321098765432109876543210")

        @test abs(BigFloat(a + b) - (big(a) + big(b))) < 1e-29
        @test abs(BigFloat(a - b) - (big(a) - big(b))) < 1e-29
        @test abs(BigFloat(a * b) - (big(a) * big(b))) < 1e-28
        @test abs(BigFloat(a / b) - (big(a) / big(b))) < 1e-28
    end

    @testset "integer power" begin
        x = DoubleF64(2.0)
        @test abs(BigFloat(x^10) - big(2.0)^10) < 1e-25
        @test DoubleF64(3.0)^0 == DoubleF64(1.0)
        @test DoubleF64(5.0)^1 == DoubleF64(5.0)
    end

    @testset "sqrt" begin
        x = DoubleF64(2.0)
        @test abs(BigFloat(sqrt(x)) - sqrt(big(2.0))) < 1e-30
    end

    @testset "comparison" begin
        @test DoubleF64(1.0) < DoubleF64(2.0)
        @test DoubleF64(2.0) == DoubleF64(2.0)
        @test DoubleF64(3.0) <= DoubleF64(3.0)
    end

    @testset "special values" begin
        @test iszero(zero(DoubleF64))
        @test isone(one(DoubleF64))
        @test isnan(DoubleF64(NaN, NaN))
        @test isinf(DoubleF64(Inf))
        @test isfinite(DoubleF64(1.0))
    end

    @testset "isbits" begin
        @test isbits(DoubleF64(1.0))
        @test isbits(ComplexDF64(DoubleF64(1.0), DoubleF64(2.0)))
    end

    @testset "ComplexDF64" begin
        z = Complex(DoubleF64(1.0), DoubleF64(2.0))
        @test z isa ComplexDF64
        w = Complex(DoubleF64(3.0), DoubleF64(4.0))
        @test abs(BigFloat(real(z + w)) - 4.0) < 1e-30
        @test abs(BigFloat(imag(z + w)) - 6.0) < 1e-30
    end

    @testset "promotion" begin
        @test DoubleF64(1.0) + 2.0 isa DoubleF64
        @test 3 + DoubleF64(1.0) isa DoubleF64
        @test promote_type(DoubleF64, Float64) == DoubleF64
        @test promote_type(DoubleF64, Int) == DoubleF64
    end

    @testset "abs and abs2" begin
        z = Complex(DoubleF64(3.0), DoubleF64(4.0))
        @test abs(BigFloat(abs2(z)) - 25.0) < 1e-28
    end
end
