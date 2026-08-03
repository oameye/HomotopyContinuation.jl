using Test
using Random: MersenneTwister
using HomotopyContinuationNextCertification:
    Interval, IComplex, IComplexF64, interval, mid, diam, rad, mig, mag, hull,
    isinterior, sqr, inf_norm_bound

# `f(t)` lies inside the enclosure `itv` for every sampled `t` in the box.
function samples_enclosed(rng, f, itv, lo, hi; n::Int = 40)
    for _ in 1:n
        t = lo + rand(rng) * (hi - lo)
        f(t) ∈ itv || return false
    end
    return true
end

function complex_samples_enclosed(rng, f, itv, z::IComplexF64; n::Int = 40)
    for _ in 1:n
        t = complex(
            real(z).lo + rand(rng) * diam(real(z)),
            imag(z).lo + rand(rng) * diam(imag(z)),
        )
        v = f(t)
        (real(v) ∈ real(itv) && imag(v) ∈ imag(itv)) || return false
    end
    return true
end

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

    @testset "real sqrt, sinh, cosh" begin
        @test 2.0 ∈ sqrt(Interval(4.0, 4.0))
        @test sqrt(Interval(1.0, 9.0)) ⊆ Interval(0.99, 3.01)
        @test isempty(sqrt(Interval(-4.0, -1.0)))
        @test sqrt(Interval(-1.0, 4.0)) ⊆ Interval(0.0, 2.01)

        @test sinh(1.0) ∈ sinh(Interval(1.0, 1.0))
        @test cosh(1.0) ∈ cosh(Interval(1.0, 1.0))
        @test cosh(Interval(-1.0, 2.0)).lo == 1.0
        @test cosh(2.0) ∈ cosh(Interval(-1.0, 2.0))
    end

    @testset "real sin and cos" begin
        @test sin(1.0) ∈ sin(Interval(1.0, 1.0))
        @test cos(1.0) ∈ cos(Interval(1.0, 1.0))

        @test sin(Interval(0.0, 7.0)) == Interval(-1.0, 1.0)
        @test cos(Interval(0.0, 7.0)) == Interval(-1.0, 1.0)
        @test sin(Interval(1.5, 1.7)).hi == 1.0
        @test cos(Interval(3.0, 3.3)).lo == -1.0
        @test sin(Interval(0.0, 1.0)).hi < 1.0
        @test cos(Interval(0.0, 1.0)).lo > 0.0

        for a in (Interval(-1.0e6, -1.0e6 + 0.1), Interval(1.0e8, 1.0e8 + 1.0))
            @test sin(a) ⊆ Interval(-1.0, 1.0)
            @test cos(a) ⊆ Interval(-1.0, 1.0)
        end
    end

    @testset "complex sqrt, sin and cos" begin
        @test 2.0 ∈ real(sqrt(IComplexF64(4.0, 0.0)))
        @test sqrt(2.0im) ≈ mid(sqrt(IComplexF64(0.0, 2.0)))
        @test sin(1.0 + 0.5im) ≈ mid(sin(IComplexF64(1.0, 0.5)))
        @test cos(1.0 + 0.5im) ≈ mid(cos(IComplexF64(1.0, 0.5)))

        @test isempty(real(sqrt(IComplex(Interval(-2.0, -1.0), Interval(-0.1, 0.1)))))
        @test !isempty(real(sqrt(IComplex(Interval(-2.0, -1.0), Interval(0.1, 0.2)))))

        # `2uv = Im z` keeps a box around a positive real number tight.
        w = 1.0e-8
        q = sqrt(IComplex(Interval(4.0 - w, 4.0 + w), Interval(-w, w)))
        @test rad(real(q)) < 1.0e-8
        @test rad(imag(q)) < 1.0e-8
    end

    @testset "real exp, log and atan" begin
        for x in (-30.0, -1.0, 0.0, 0.5, 3.0, 20.0)
            @test exp(x) ∈ exp(Interval(x, x))
            @test atan(x) ∈ atan(Interval(x, x))
        end
        @test exp(Interval(-800.0, -700.0)).lo == 0.0
        @test exp(Interval(1.0, 2.0)) ⊆ Interval(2.71, 7.39)
        @test atan(Interval(-1.0e300, 1.0e300)) ⊆ Interval(-1.5708, 1.5708)

        for x in (1.0e-30, 0.5, 1.0, 7.0, 1.0e30)
            @test log(x) ∈ log(Interval(x, x))
        end
        @test log(Interval(1.0, exp(2.0))) ⊆ Interval(-1.0e-15, 2.01)
        # No finite enclosure once the box reaches zero.
        @test isempty(log(Interval(0.0, 1.0)))
        @test isempty(log(Interval(-1.0, 1.0)))
        @test isempty(log(Interval(-2.0, -1.0)))
    end

    @testset "complex exp, log, hyperbolics and inverse trig" begin
        z = IComplexF64(0.7, -1.3)
        zm = 0.7 - 1.3im
        for f in (exp, sinh, cosh, tan, tanh, log, asin, acos)
            @test f(zm) ≈ mid(f(z))
        end
        @test (zm^1.5) ≈ mid(z^IComplex(Interval(1.5), Interval(0.0)))

        # `log` and a non-integer power are cut along the negative real axis.
        for f in (log, w -> w^IComplex(Interval(1.5), Interval(0.0)))
            @test isempty(real(f(IComplex(Interval(-2.0, -1.0), Interval(-0.1, 0.1)))))
            @test isempty(real(f(IComplex(Interval(-1.0, 1.0), Interval(-1.0, 1.0)))))
            @test !isempty(real(f(IComplex(Interval(-2.0, -1.0), Interval(0.1, 0.2)))))
        end
        # `asin` and `acos` are cut along |Re| > 1 of the real axis.
        for f in (asin, acos)
            @test isempty(real(f(IComplex(Interval(1.5, 2.0), Interval(-0.1, 0.1)))))
            @test isempty(real(f(IComplex(Interval(-2.0, -1.5), Interval(-0.1, 0.1)))))
            @test !isempty(real(f(IComplex(Interval(-0.5, 0.5), Interval(-0.1, 0.1)))))
        end
        # A pole inside the box leaves the real denominator straddling zero.
        @test isempty(real(tan(IComplex(Interval(1.5, 1.7), Interval(-0.1, 0.1)))))
        @test isempty(real(tanh(IComplex(Interval(-0.1, 0.1), Interval(1.5, 1.7)))))
        @test !isempty(real(tan(IComplex(Interval(0.1, 0.2), Interval(-0.1, 0.1)))))

        # The argument stays tight next to either axis.
        w = 1.0e-8
        for b in (
                IComplex(Interval(3.0 - w, 3.0 + w), Interval(-w, w)),
                IComplex(Interval(-w, w), Interval(3.0 - w, 3.0 + w)),
                IComplex(Interval(-3.0 - w, -3.0 + w), Interval(-w - 2.0, w - 2.0)),
            )
            @test rad(imag(log(b))) < 1.0e-7
        end
    end

    @testset "enclosure soundness on random boxes" begin
        rng = MersenneTwister(0x5eed)
        real_ok = true
        for _ in 1:400
            c = (rand(rng) - 0.5) * 200
            w = exp(rand(rng) * log(20) - 6)
            a = Interval(c - w, c + w)
            real_ok &= samples_enclosed(rng, sin, sin(a), a.lo, a.hi)
            real_ok &= samples_enclosed(rng, cos, cos(a), a.lo, a.hi)
            real_ok &= samples_enclosed(rng, atan, atan(a), a.lo, a.hi)
            b = Interval(abs(c) - min(w, abs(c)), abs(c) + w)
            real_ok &= samples_enclosed(rng, sqrt, sqrt(b), b.lo, b.hi)
            e = Interval(c / 40 - w, c / 40 + w)
            real_ok &= samples_enclosed(rng, exp, exp(e), e.lo, e.hi)
            b.lo > 0.0 && (real_ok &= samples_enclosed(rng, log, log(b), b.lo, b.hi))
        end
        @test real_ok

        complex_ok = true
        # Each function that can decline a box gets counted, so a sweep that only
        # ever rejects cannot pass as a sweep that only ever encloses.
        cut_boxes = Dict(f => 0 for f in (sqrt, log, asin, acos, tan, tanh))
        pow = w -> w^IComplex(Interval(1.5), Interval(0.0))
        for _ in 1:400
            cx, cy = (rand(rng) - 0.5) * 20, (rand(rng) - 0.5) * 20
            wx, wy = rand(rng) / 2, rand(rng) / 2
            z = IComplex(Interval(cx - wx, cx + wx), Interval(cy - wy, cy + wy))
            complex_ok &= complex_samples_enclosed(rng, sin, sin(z), z)
            complex_ok &= complex_samples_enclosed(rng, cos, cos(z), z)
            complex_ok &= complex_samples_enclosed(rng, sinh, sinh(z), z)
            complex_ok &= complex_samples_enclosed(rng, cosh, cosh(z), z)
            complex_ok &= complex_samples_enclosed(rng, exp, exp(z), z)
            for f in (sqrt, log, asin, acos, tan, tanh)
                v = f(z)
                if isempty(real(v)) || isempty(imag(v))
                    cut_boxes[f] += 1
                else
                    complex_ok &= complex_samples_enclosed(rng, f, v, z)
                end
            end
            p = pow(z)
            isempty(real(p)) || (complex_ok &= complex_samples_enclosed(rng, w -> w^1.5, p, z))
        end
        @test complex_ok
        @test all(>(0), values(cut_boxes))
    end
end
