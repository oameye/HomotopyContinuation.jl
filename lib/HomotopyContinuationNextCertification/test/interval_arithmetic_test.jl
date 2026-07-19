using Test
using HomotopyContinuationNextCertification:
    Interval, IComplex, IComplexF64, interval, mid, diam, rad, mig, mag, hull,
    isinterior, sqr, inf_norm_bound

@testset "Interval arithmetic" begin
    @testset "Interval construction and queries" begin
        a = Interval(1.0, 2.0)
        @test a.lo == 1.0
        @test a.hi == 2.0
        @test mid(a) == 1.5
        @test mag(a) == 2.0
        @test mig(a) == 1.0
        @test mig(Interval(-1.0, 2.0)) == 0.0  # contains zero
        @test diam(a) ≥ 1.0
        @test rad(a) ≥ 0.5

        @test Interval(2) == Interval(2.0, 2.0)
        @test eltype(Interval(1.0)) === Float64
        @test convert(Interval{Float64}, Interval(1, 1)) == Interval(1.0, 1.0)

        @test interval(1.0, 2.0) == Interval(1.0, 2.0)
        @test_throws ArgumentError interval(2.0, 1.0)
        @test_throws ArgumentError interval(Inf, 1.0)
    end

    @testset "Interval arithmetic (containment)" begin
        a = Interval(1.0, 2.0)
        b = Interval(3.0, 4.0)
        for x in (1.0, 1.5, 2.0), y in (3.0, 3.5, 4.0)
            @test (x + y) ∈ (a + b)
            @test (x - y) ∈ (a - b)
            @test (x * y) ∈ (a * b)
            @test (x / y) ∈ (a / b)
        end
        # multiplication sign cases
        for u in (Interval(-2.0, 3.0), Interval(-3.0, -1.0), Interval(1.0, 2.0)),
                v in (Interval(-2.0, 3.0), Interval(-3.0, -1.0), Interval(1.0, 2.0))
            for x in (u.lo, mid(u), u.hi), y in (v.lo, mid(v), v.hi)
                @test (x * y) ∈ (u * v)
            end
        end
    end

    @testset "sqr, pow, inv, hull" begin
        @test 4.0 ∈ sqr(Interval(-2.0, 3.0))
        @test 9.0 ∈ sqr(Interval(-2.0, 3.0))
        @test 0.0 ∈ sqr(Interval(-2.0, 3.0))
        @test (Interval(2.0, 3.0)^2) == sqr(Interval(2.0, 3.0))
        @test 8.0 ∈ (Interval(2.0, 2.0)^3)
        @test 0.5 ∈ inv(Interval(2.0, 2.0))
        @test 1.0 ∈ hull(Interval(0.0, 0.5), Interval(0.9, 1.0))
    end

    @testset "IComplex arithmetic" begin
        z = IComplexF64(1.0, 1.0)   # 1 + i
        w = IComplexF64(2.0, -1.0)  # 2 - i
        @test (1.0 + 1.0im) ∈ z
        @test (3.0 + 1.0im) ∈ (z * w)   # (1+i)(2-i) = 3+i
        @test (3.0 + 0.0im) ∈ (z + w)
        @test (0.4 + 0.2im) ∈ inv(w)    # 1/(2-i) = (2+i)/5
        @test ((1.0 + 1.0im) / (2.0 - 1.0im)) ∈ (z / w)
        @test mid(z * w) ≈ 3.0 + 1.0im
        @test (4.0 + 2.0im) ∈ muladd(z, w, z)  # z*w + z = 3+i + 1+i = 4+2i
        @test conj(z) == IComplexF64(1.0, -1.0)
    end

    @testset "isinterior and inf_norm_bound" begin
        big = IComplex(Interval(0.0, 2.0), Interval(0.0, 2.0))
        small = IComplex(Interval(0.9, 1.1), Interval(0.9, 1.1))
        @test isinterior(small, big)
        @test !isinterior(big, small)

        M = [
            IComplexF64(0.1, 0.0) IComplexF64(0.0, 0.1);
            IComplexF64(0.05, 0.0) IComplexF64(0.0, 0.0)
        ]
        # row sums: 0.2 and 0.05, times √2 upper bound
        @test inf_norm_bound(M) ≥ 0.2
        @test inf_norm_bound(M) < 0.29
    end
end
