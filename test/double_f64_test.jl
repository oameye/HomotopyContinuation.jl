using Test
using HomotopyContinuationNext: DoubleF64, ComplexDF64, wide_add, wide_sub, wide_mul, wide_div

@testset "DoubleF64" begin
    @testset "construction and conversion" begin
        @test DoubleF64(1.0).hi == 1.0
        @test DoubleF64(1.0).lo == 0.0
        @test DoubleF64(Float32(2.0)) == DoubleF64(2.0)
        @test DoubleF64(Float16(2.0)) == DoubleF64(2.0)
        @test DoubleF64(2) == DoubleF64(2.0)
        @test DoubleF64(π) == DoubleF64(big(π))

        # BigFloat roundtrip preserves ~30 digits
        y = DoubleF64(big"3.141592653589793238462643383279502884197")
        @test abs(BigFloat(y) - big(π)) < 1.0e-30

        @test Float64(DoubleF64(3.14)) == 3.14
        @test convert(Integer, DoubleF64(2.0)) isa Int64
    end

    @testset "arithmetic precision vs BigFloat" begin
        a = DoubleF64(big"1.23456789012345678901234567890")
        b = DoubleF64(big"9.87654321098765432109876543210")

        @test abs(BigFloat(a + b) - (big(a) + big(b))) < 1.0e-29
        @test abs(BigFloat(a - b) - (big(a) - big(b))) < 1.0e-29
        @test abs(BigFloat(a * b) - (big(a) * big(b))) < 1.0e-28
        @test abs(BigFloat(a / b) - (big(a) / big(b))) < 1.0e-28
    end

    @testset "randomized arithmetic accuracy" begin
        for _ in 1:3
            x = DoubleF64(rand()) * 20 - 10
            y = DoubleF64(rand()) * 20 - 10
            @test x * y ≈ big(x) * big(y) atol = 1.0e-29
            @test x + y ≈ big(x) + big(y) atol = 1.0e-30
            @test x - y ≈ big(x) - big(y) atol = 1.0e-30
            @test x / y ≈ big(x) / big(y) atol = 1.0e-26
            u = rand()
            @test x / u ≈ big(x) / big(u) atol = 1.0e-26
            @test x^5 ≈ BigFloat(x)^5 atol = 1.0e-25
        end
    end

    @testset "integer power edge cases" begin
        @test abs(BigFloat(DoubleF64(2.0)^10) - big(2.0)^10) < 1.0e-25
        @test DoubleF64(3.0)^0 == DoubleF64(1.0)
        @test DoubleF64(5.0)^1 == DoubleF64(5.0)
    end

    @testset "sqrt" begin
        @test abs(BigFloat(sqrt(DoubleF64(2.0))) - sqrt(big(2.0))) < 1.0e-30
    end

    @testset "comparison and promotion" begin
        x = DoubleF64(2.321)
        @test x < 2 * x
        @test x ≤ x
        @test convert(Float64, x) ≤ x
        @test DoubleF64(2.0) == 2.0
        @test 2.0 == DoubleF64(2.0)
        @test promote_type(DoubleF64, Float64) == DoubleF64
        @test promote_type(DoubleF64, Int) == DoubleF64
        @test DoubleF64(1.0) + 2.0 isa DoubleF64
        @test 3 + DoubleF64(1.0) isa DoubleF64
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
        @test im * DoubleF64(rand()) isa ComplexDF64
        w = Complex(DoubleF64(3.0), DoubleF64(4.0))
        @test abs(BigFloat(real(z + w)) - 4.0) < 1.0e-30
        @test abs(BigFloat(abs2(z)) - 5.0) < 1.0e-28
    end

    @testset "wide_* correctness" begin
        a, b = 1.5, 2.5
        @test abs(BigFloat(wide_add(a, b)) - (big(a) + big(b))) < 1.0e-30
        @test abs(BigFloat(wide_sub(a, b)) - (big(a) - big(b))) < 1.0e-30
        @test abs(BigFloat(wide_mul(a, b)) - (big(a) * big(b))) < 1.0e-30
        @test abs(BigFloat(wide_div(a, b)) - (big(a) / big(b))) < 1.0e-30
    end

    @testset "floor, ceil, trunc, isinteger" begin
        @test floor(DoubleF64(3.2)) == 3.0
        @test floor(Int, DoubleF64(3.2)) == 3
        @test ceil(DoubleF64(3.2)) == 4.0
        @test ceil(Int, DoubleF64(3.2)) == 4
        @test trunc(DoubleF64(3.2)) == 3.0
        @test trunc(Int, DoubleF64(3.2)) == 3
        @test isinteger(DoubleF64(3.2)) == false
        @test isinteger(DoubleF64(3.0)) == true
    end

    @testset "decompose" begin
        for _ in 1:3
            x = DoubleF64(rand()) * 20 - 10
            n, p, d = Base.decompose(x)
            @test x ≈ BigFloat(n) * BigFloat(2)^p / BigFloat(d)
        end
    end
end
