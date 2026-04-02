# Compare primitive operations between HomotopyContinuationNext and HomotopyContinuation v2.
# Covers: DoubleF64, norms, LU solve, condition estimation.

using Test
using LinearAlgebra: LinearAlgebra, I, norm, ldiv!, diagm

using HomotopyContinuationNext
using HomotopyContinuation

const Next = HomotopyContinuationNext
const HC = HomotopyContinuation

using FixedSizeArrays: FixedSizeArray
const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}

@testset "Compare v2: DoubleF64" begin
    @testset "arithmetic matches" begin
        for _ in 1:20
            a_f64 = randn()
            b_f64 = randn()
            a_next = Next.DoubleF64(a_f64)
            b_next = Next.DoubleF64(b_f64)
            a_hc = HC.DoubleF64(a_f64)
            b_hc = HC.DoubleF64(b_f64)

            @test Float64(a_next + b_next) == Float64(a_hc + b_hc)
            @test Float64(a_next - b_next) == Float64(a_hc - b_hc)
            @test Float64(a_next * b_next) == Float64(a_hc * b_hc)
            if b_f64 != 0.0
                @test Float64(a_next / b_next) == Float64(a_hc / b_hc)
            end
        end
    end

    @testset "sqrt matches" begin
        for _ in 1:10
            x = abs(randn()) + 0.01
            @test Float64(sqrt(Next.DoubleF64(x))) == Float64(sqrt(HC.DoubleF64(x)))
        end
    end

    @testset "integer power matches" begin
        for p in [0, 1, 2, 3, 5, 10]
            x = 1.5
            @test Float64(Next.DoubleF64(x)^p) == Float64(HC.DoubleF64(x)^p)
        end
    end
end

@testset "Compare v2: Norms" begin
    @testset "inf_norm matches" begin
        for n in [3, 8, 32]
            x_data = randn(ComplexF64, n)
            x_fs = FSVec{ComplexF64}(x_data)
            x_vec = Vector{ComplexF64}(x_data)

            hc_inf = HC.InfNorm()
            @test Next.inf_norm(x_fs) ≈ hc_inf(x_vec)
        end
    end

    @testset "inf_norm overflow (exp2(700))" begin
        x_data = [2.0im, 3.0 - 1im, 5.0 + 2.0im]
        huge_data = exp2(700) .* x_data

        x_fs = FSVec{ComplexF64}(huge_data)
        x_vec = Vector{ComplexF64}(huge_data)

        hc_inf = HC.InfNorm()
        @test Next.inf_norm(x_fs) ≈ hc_inf(x_vec)
    end

    @testset "weighted_norm matches" begin
        for n in [3, 8, 32]
            x_data = randn(ComplexF64, n)
            x_fs = FSVec{ComplexF64}(x_data)
            x_vec = Vector{ComplexF64}(x_data)

            w_next = Next.WeightedNorm(n)
            Next.init!(w_next, x_fs)

            w_hc = HC.WeightedNorm(HC.InfNorm(), n)
            HC.init!(w_hc, x_vec)

            @test Next.weighted_norm(x_fs, w_next) ≈ w_hc(x_vec)
        end
    end

    @testset "weighted_norm overflow (exp2(700))" begin
        x_data = [2.0im, 3.0 - 1im, 5.0 + 2.0im]
        huge_data = exp2(700) .* x_data

        x_fs = FSVec{ComplexF64}(huge_data)
        x_vec = Vector{ComplexF64}(huge_data)

        w_next = Next.WeightedNorm(FSVec{Float64}([4.0, 2.0, 2.0]))
        w_hc = HC.WeightedNorm(HC.InfNorm(), 3)
        w_hc .= [4.0, 2.0, 2.0]

        @test Next.weighted_norm(x_fs, w_next) ≈ w_hc(x_vec)
    end
end

@testset "Compare v2: LU solve" begin
    @testset "same solution for well-conditioned" begin
        for n in [3, 8, 16]
            A_data = randn(ComplexF64, n, n) + 5.0I
            b_data = randn(ComplexF64, n)

            WS_next = Next.MatrixWorkspace(n, n)
            copyto!(WS_next.A, A_data)
            Next.updated!(WS_next)
            x_next = FSVec{ComplexF64}(zeros(ComplexF64, n))
            ldiv!(x_next, WS_next, FSVec{ComplexF64}(copy(b_data)))

            WS_hc = HC.MatrixWorkspace(n, n)
            copyto!(WS_hc.A, A_data)
            HC.updated!(WS_hc)
            x_hc = zeros(ComplexF64, n)
            ldiv!(x_hc, WS_hc, copy(b_data))

            @test Vector(x_next) ≈ x_hc rtol = 1.0e-12
        end
    end

    @testset "same solution for ill-conditioned" begin
        # Moderately ill-conditioned via diagonal scaling
        n = 6
        d = exp10.(range(-3; stop = 3, length = n))
        A_data = ComplexF64.(diagm(d) * randn(n, n) + 0.1I)
        b_data = randn(ComplexF64, n)

        WS_next = Next.MatrixWorkspace(n, n)
        copyto!(WS_next.A, A_data)
        Next.updated!(WS_next)
        x_next = FSVec{ComplexF64}(zeros(ComplexF64, n))
        ldiv!(x_next, WS_next, FSVec{ComplexF64}(copy(b_data)))

        WS_hc = HC.MatrixWorkspace(n, n)
        copyto!(WS_hc.A, A_data)
        HC.updated!(WS_hc)
        x_hc = zeros(ComplexF64, n)
        ldiv!(x_hc, WS_hc, copy(b_data))

        # Both should get similar answers (not exact — LU pivot order may differ)
        @test norm(Vector(x_next) - x_hc) / norm(x_hc) < 1.0e-8
    end
end

@testset "Compare v2: Condition estimation" begin
    for _ in 1:5
        n = 6
        A_data = ComplexF64.(randn(n, n) + 3.0I)

        WS_next = Next.MatrixWorkspace(n, n)
        copyto!(WS_next.A, A_data)
        Next.updated!(WS_next)

        WS_hc = HC.MatrixWorkspace(n, n)
        copyto!(WS_hc.A, A_data)
        HC.updated!(WS_hc)

        cond_next = LinearAlgebra.cond(WS_next)
        cond_hc = LinearAlgebra.cond(WS_hc)

        # Both are estimates — should be within 10x of each other
        @test 0.1 ≤ cond_next / cond_hc ≤ 10.0
    end
end
