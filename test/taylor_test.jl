using Test
using HomotopyContinuationNext:
    TruncatedTaylorSeries, TaylorVector, vectors,
    taylor_op_add, taylor_op_mul, taylor_op_div,
    taylor_op_neg, taylor_op_inv, taylor_op_sqrt
using FixedSizeArrays: FixedSizeArray
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

@testset "TruncatedTaylorSeries" begin
    @testset "construction" begin
        t = TruncatedTaylorSeries((1.0 + 0im, 2.0 + 0im, 3.0 + 0im))
        @test t[0] == 1.0 + 0im
        @test length(t) == 3
    end

    @testset "from scalar (zero-padded)" begin
        t = TruncatedTaylorSeries{3, ComplexF64}(2.0 + 0im)
        @test t[0] == 2.0 + 0im
        @test t[1] == 0.0 + 0im
        @test t[2] == 0.0 + 0im
    end
end

@testset "taylor_op_* correctness" begin
    a = TruncatedTaylorSeries((2.0 + 0im, 1.0 + 0im, 0.5 + 0im))
    b = TruncatedTaylorSeries((3.0 + 0im, -1.0 + 0im, 0.0 + 0im))

    @testset "add" begin
        r = taylor_op_add(a, b)
        @test r[0] ≈ 5.0 + 0im
        @test r[1] ≈ 0.0 + 0im
        @test r[2] ≈ 0.5 + 0im
    end

    @testset "neg" begin
        r = taylor_op_neg(a)
        @test r[0] ≈ -a[0]
        @test r[1] ≈ -a[1]
    end

    @testset "mul (Cauchy product)" begin
        r = taylor_op_mul(a, b)
        @test r[0] ≈ a[0] * b[0]
        @test r[1] ≈ a[0] * b[1] + a[1] * b[0]
        @test r[2] ≈ a[0] * b[2] + a[1] * b[1] + a[2] * b[0]
    end

    @testset "inv: inv(b) * b ≈ (1, 0, 0)" begin
        r = taylor_op_inv(b)
        product = taylor_op_mul(r, b)
        @test product[0] ≈ 1.0 + 0im atol = 1.0e-12
        @test product[1] ≈ 0.0 + 0im atol = 1.0e-12
        @test product[2] ≈ 0.0 + 0im atol = 1.0e-12
    end

    @testset "div: (a/b) * b ≈ a" begin
        r = taylor_op_div(a, b)
        product = taylor_op_mul(r, b)
        for k in 0:2
            @test product[k] ≈ a[k] atol = 1.0e-12
        end
    end

    @testset "sqrt: sqrt(a)^2 ≈ a" begin
        r = taylor_op_sqrt(a)
        product = taylor_op_mul(r, r)
        for k in 0:2
            @test product[k] ≈ a[k] atol = 1.0e-12
        end
    end
end

@testset "TaylorVector" begin
    @testset "getindex / setindex!" begin
        tv = TaylorVector{2, ComplexF64}(3)
        tv[1] = TruncatedTaylorSeries((1.0 + 0im, 2.0 + 0im))
        t = tv[1]
        @test t[0] == 1.0 + 0im
        @test t[1] == 2.0 + 0im
    end

    @testset "vectors" begin
        tv = TaylorVector{2, ComplexF64}(3)
        tv[1] = TruncatedTaylorSeries((1.0 + 0im, 10.0 + 0im))
        tv[2] = TruncatedTaylorSeries((2.0 + 0im, 20.0 + 0im))
        tv[3] = TruncatedTaylorSeries((3.0 + 0im, 30.0 + 0im))

        vs = vectors(tv)
        @test length(vs) == 2
        @test vs[1][1] == 1.0 + 0im
        @test vs[2][3] == 30.0 + 0im
    end
end
