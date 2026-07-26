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
        @test Int64(DoubleF64(2.0)) === Int64(2)
        @test UInt8(DoubleF64(2.0)) === UInt8(2)
        @test UInt64(DoubleF64(2.0)) === UInt64(2)
        @test convert(Integer, DoubleF64(2.0)) isa Int64
        @test convert(UInt16, DoubleF64(2.0)) === UInt16(2)
        @test convert(DoubleF64, UInt32(7)) == DoubleF64(7.0)
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
        @test promote_type(DoubleF64, UInt8) == DoubleF64
        @test promote_type(DoubleF64, UInt64) == DoubleF64
        @test promote_type(DoubleF64, BigInt) == BigFloat
        @test DoubleF64(1.0) + 2.0 isa DoubleF64
        @test 3 + DoubleF64(1.0) isa DoubleF64
        @test DoubleF64(1.0) + UInt8(2) isa DoubleF64
        @test UInt16(3) + DoubleF64(1.0) isa DoubleF64

        a, b = promote(DoubleF64(1.0), UInt8(2))
        @test a isa DoubleF64
        @test b isa DoubleF64
        @test b == DoubleF64(2.0)
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

    @testset "transcendental functions" begin
        args = (0.0, 1.0e-8, 0.05, 0.3, 1.0, 2.7, -3.3, 10.0, -50.25, 100.5)
        @testset "$name" for (f, name) in (
                (exp, "exp"), (sin, "sin"), (cos, "cos"),
                (sinh, "sinh"), (cosh, "cosh"),
            )
            for x in args
                got = BigFloat(f(DoubleF64(x)))
                want = f(BigFloat(x))
                err = iszero(want) ? abs(got) : abs((got - want) / want)
                @test err < 1.0e-29
            end
        end

        @test sincos(DoubleF64(0.7)) == (sin(DoubleF64(0.7)), cos(DoubleF64(0.7)))
        @test exp(DoubleF64(0.0)) == DoubleF64(1.0)
        @test iszero(sinh(DoubleF64(0.0)))
        @test cosh(DoubleF64(0.0)) == DoubleF64(1.0)
        @test iszero(exp(DoubleF64(-1000.0)))
        @test isinf(exp(DoubleF64(1000.0)))
        @test isnan(exp(DoubleF64(NaN)))

        # Identities that exercise the reduction branches.
        for x in (0.4, 3.9, -7.1)
            d = DoubleF64(x)
            @test abs(BigFloat(sin(d)^2 + cos(d)^2 - 1)) < 1.0e-30
            @test abs(BigFloat(cosh(d)^2 - sinh(d)^2 - 1)) < 1.0e-28
        end

        @testset "large arguments keep at least Float64 accuracy" begin
            for x in (0x1p53, 1.0e16, 1.0e20, 1.0e32, -1.0e32, 1.0e300)
                d = DoubleF64(x)
                for (f, want) in ((sin, sin(BigFloat(x))), (cos, cos(BigFloat(x))))
                    @test abs(BigFloat(f(d)) - want) < 1.0e-15
                end
                @test abs(BigFloat(sin(d)^2 + cos(d)^2 - 1)) < 1.0e-15
            end
        end

        @testset "sinh and cosh overflow to infinities" begin
            for x in (41.0, -41.0, 300.0, 710.0)
                d = DoubleF64(x)
                for (f, want) in ((sinh, sinh(BigFloat(x))), (cosh, cosh(BigFloat(x))))
                    @test abs((BigFloat(f(d)) - want) / want) < 1.0e-29
                end
            end
            for x in (1000.0, Inf)
                @test sinh(DoubleF64(x)).hi == Inf
                @test sinh(DoubleF64(-x)).hi == -Inf
                @test cosh(DoubleF64(x)).hi == Inf
                @test cosh(DoubleF64(-x)).hi == Inf
            end
            @test isnan(sinh(DoubleF64(NaN)))
            @test isnan(cosh(DoubleF64(NaN)))
        end
    end

    @testset "ComplexDF64 transcendental functions" begin
        z = ComplexDF64(DoubleF64(0.7), DoubleF64(-1.3))
        zb = big(0.7) - big(1.3) * im
        for f in (sin, cos, sqrt, exp)
            got = f(z)
            g = Complex(BigFloat(real(got)), BigFloat(imag(got)))
            @test abs(g - f(zb)) / abs(f(zb)) < 1.0e-29
        end
    end
end
