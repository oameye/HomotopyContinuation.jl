using Test, Random
using LinearAlgebra: norm
import HomotopyContinuation as HC
using DynamicPolynomials: @polyvar, differentiate

@polyvar z[1:3]
const QUADRIC = HC.System(
    [z[1]^2 + 2z[2]^2 + 3z[3]^2 + z[1] * z[2] - 1];
    variables = z,
)

function taylor_oracle(H, x::Vector{ComplexF64}, t0::ComplexF64, K::Int; r = 0.1, n = 64)
    m = size(H)[1]
    u = HC.FSVec{ComplexF64}(zeros(ComplexF64, m))
    xf = HC.FSVec{ComplexF64}(x)
    vals = Matrix{ComplexF64}(undef, m, n)
    for k in 0:(n - 1)
        HC.evaluate!(u, H, xf, t0 + r * cis(2π * k / n))
        vals[:, k + 1] .= Vector(u)
    end
    return [
        sum(vals[i, k + 1] * cis(-2π * k * K / n) for k in 0:(n - 1)) / (n * r^K)
            for i in 1:m
    ]
end

function taylor_constant_path(H, ::Val{K}, x::Vector{ComplexF64}, t0::ComplexF64) where {K}
    m, n = size(H)
    u = HC.FSVec{ComplexF64}(zeros(ComplexF64, m))
    tv = HC.TaylorVector{K + 1, ComplexF64}(n)
    tv.data[1, :] .= x
    HC.taylor!(u, Val(K), H, tv, t0)
    return Vector(u)
end

@testset "subspace-homotopy numerical kernel" begin
    @testset "intrinsic first Taylor coefficient agrees with finite differences" begin
        Random.seed!(12)
        V = HC.rand_subspace(3; dim = 1)
        W = HC.rand_subspace(3; dim = 1)
        H = HC.IntrinsicSubspaceHomotopy(QUADRIC.evaluator, V, W)
        m, n = size(H)
        derivative = HC.FSVec{ComplexF64}(zeros(ComplexF64, m))
        x = HC.FSVec{ComplexF64}(randn(ComplexF64, n))
        t = complex(0.6)

        HC.taylor!(derivative, Val(1), H, x, t)
        h = 1.0e-6
        plus = HC.FSVec{ComplexF64}(zeros(ComplexF64, m))
        minus = HC.FSVec{ComplexF64}(zeros(ComplexF64, m))
        HC.evaluate!(plus, H, x, t + h)
        HC.evaluate!(minus, H, x, t - h)
        finite_difference = (Vector(plus) .- Vector(minus)) ./ (2h)

        @test Vector(derivative) ≈ finite_difference atol = 1.0e-5
    end

    @testset "extrinsic value, Jacobian, and Taylor coefficients" begin
        Random.seed!(41)
        @polyvar x[1:4]
        f1 = sum(randn(ComplexF64) * x[i] * x[j] for i in 1:4 for j in i:4) +
            sum(randn(ComplexF64) .* x) + randn(ComplexF64)
        f2 = sum(randn(ComplexF64) * x[i] * x[j] for i in 1:4 for j in i:4) +
            sum(randn(ComplexF64) .* x) + randn(ComplexF64)
        polys = [f1, f2]
        F = HC.System(polys; variables = x)
        A = HC.rand_subspace(4; codim = 2)
        B = HC.rand_subspace(4; codim = 2)
        H = HC.ExtrinsicSubspaceHomotopy(F, A, B; gamma = one(ComplexF64))

        Q, Q_cos, Θ = H.path.Q, H.path.Q_cos, H.path.Θ
        γ_at(t) = Q_cos .* transpose(cos.(t .* Θ)) .+ Q .* transpose(sin.(t .* Θ))

        xv = randn(ComplexF64, 4)
        xf = HC.FSVec{ComplexF64}(xv)
        value = HC.FSVec{ComplexF64}(zeros(ComplexF64, 4))
        for t in (complex(0.3), 0.8 - 0.4im)
            HC.evaluate!(value, H, xf, ComplexF64(t))
            expected = [
                [ComplexF64(f(x => xv)) for f in polys]
                transpose(γ_at(t)) * xv .- (t .* H.a0 .+ (1 .- t) .* H.b0)
            ]
            @test Vector(value) ≈ expected rtol = 1.0e-12
        end

        t = 0.7 + 0.2im
        jacobian = HC.FSMat{ComplexF64}(zeros(ComplexF64, 4, 4))
        HC.evaluate_and_jacobian!(value, jacobian, H, xf, ComplexF64(t))
        JF = [ComplexF64(differentiate(f, xj)(x => xv)) for f in polys, xj in x]
        @test Matrix(jacobian) ≈ [JF; transpose(γ_at(t))] rtol = 1.0e-12

        t0 = complex(0.42)
        first = HC.FSVec{ComplexF64}(zeros(ComplexF64, 4))
        HC.taylor!(first, Val(1), H, xf, t0)
        @test Vector(first) ≈ taylor_oracle(H, xv, t0, 1) rtol = 1.0e-9
        @test taylor_constant_path(H, Val(2), xv, t0) ≈
            taylor_oracle(H, xv, t0, 2) rtol = 1.0e-8
        @test taylor_constant_path(H, Val(3), xv, t0) ≈
            taylor_oracle(H, xv, t0, 3) rtol = 1.0e-7
    end

    @testset "intrinsic geodesic embedding and higher Taylor coefficients" begin
        Random.seed!(42)
        V = HC.rand_subspace(3; dim = 1)
        W = HC.rand_subspace(3; dim = 1)
        H = HC.IntrinsicSubspaceHomotopy(QUADRIC.evaluator, V, W; gamma = one(ComplexF64))

        Q, Q_cos, Θ = H.path.Q, H.path.Q_cos, H.path.Θ
        γ_at(t) = Q_cos .* transpose(cos.(t .* Θ)) .+ Q .* transpose(sin.(t .* Θ))
        b_target = copy(HC.intrinsic(H.target).b)
        a_minus_b = copy(H.a_minus_b)
        ambient(v, t) = γ_at(t) * v .+ b_target .+ t .* a_minus_b

        v = randn(ComplexF64, 1)
        value = HC.FSVec{ComplexF64}(zeros(ComplexF64, 1))
        for t in (complex(0.25), 0.6 + 0.3im)
            HC.evaluate!(value, H, HC.FSVec{ComplexF64}(v), ComplexF64(t))
            xa = ambient(v, t)
            expected = xa[1]^2 + 2xa[2]^2 + 3xa[3]^2 + xa[1] * xa[2] - 1
            @test value[1] ≈ expected rtol = 1.0e-12
        end

        for (t, L) in ((1.0, V), (0.0, W))
            xa = ambient(v, complex(t))
            @test norm(HC.extrinsic(L).A * xa - HC.extrinsic(L).b) < 1.0e-10
        end

        t0 = complex(0.37)
        v2 = randn(ComplexF64, 1)
        @test taylor_constant_path(H, Val(2), v2, t0) ≈
            taylor_oracle(H, v2, t0, 2) rtol = 1.0e-8
        @test taylor_constant_path(H, Val(3), v2, t0) ≈
            taylor_oracle(H, v2, t0, 3) rtol = 1.0e-7
    end
end
