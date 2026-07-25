using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: AbstractSystem, AbstractHomotopy,
    evaluate!, evaluate_and_jacobian!, taylor!,
    nparameters, set_solution!, get_solution!,
    start_parameters!, target_parameters!,
    SystemEvaluator, HomotopyEvaluator,
    System,
    StraightLineHomotopy, CoefficientHomotopy,
    TaylorVector, TruncatedTaylorSeries, DoubleF64, ComplexDF64,
    execute!, execute_taylor!
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: differentiate as mp_diff
using FixedSizeArrays: FixedSizeArray

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

# ═══════════════════════════════════════════════════════════════════════════════
# Test AbstractSystem for wrapping via HomotopyEvaluator
# ═══════════════════════════════════════════════════════════════════════════════

struct QuadraticSystem <: AbstractSystem end
Base.size(::QuadraticSystem) = (2, 2)

function HC.evaluate!(u::AbstractVector, ::QuadraticSystem, x::AbstractVector, ::AbstractVector)
    u[1] = x[1]^2 + x[2] - 1
    u[2] = x[1] * x[2] - 2
    return nothing
end

function HC.evaluate_and_jacobian!(
        u::AbstractVector, U::AbstractMatrix,
        ::QuadraticSystem, x::AbstractVector, ::AbstractVector,
    )
    u[1] = x[1]^2 + x[2] - 1
    u[2] = x[1] * x[2] - 2
    U[1, 1] = 2x[1]
    U[1, 2] = one(eltype(x))
    U[2, 1] = x[2]
    U[2, 2] = x[1]
    return nothing
end

function HC.taylor!(
        u::AbstractVector, ::Val{1}, ::QuadraticSystem,
        tx::TaylorVector, ::AbstractVector,
    )
    x1, x2 = tx[1], tx[2]
    u[1] = 2 * x1[0] * x1[1] + x2[1]
    u[2] = x1[0] * x2[1] + x1[1] * x2[0]
    return nothing
end

function HC.taylor!(
        u::AbstractVector, ::Val{K}, ::QuadraticSystem,
        tx::TaylorVector, ::AbstractVector,
    ) where {K}
    fill!(u, zero(eltype(u)))
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# Test AbstractHomotopy for wrapping via HomotopyEvaluator
# ═══════════════════════════════════════════════════════════════════════════════

struct ManualSLH <: AbstractHomotopy
    γ::ComplexF64
end
Base.size(::ManualSLH) = (2, 2)

# H(x,t) = γt(x₁-1, x₂-1) + (1-t)(x₁²-1, x₂²-1)
function HC.evaluate!(u::AbstractVector, H::ManualSLH, x::AbstractVector, t::ComplexF64)
    γt = H.γ * t
    t1 = one(ComplexF64) - t
    u[1] = γt * (x[1] - 1) + t1 * (x[1]^2 - 1)
    u[2] = γt * (x[2] - 1) + t1 * (x[2]^2 - 1)
    return nothing
end

function HC.evaluate_and_jacobian!(
        u::AbstractVector, U::AbstractMatrix,
        H::ManualSLH, x::AbstractVector, t::ComplexF64,
    )
    HC.evaluate!(u, H, x, t)
    γt = H.γ * t
    t1 = one(ComplexF64) - t
    U[1, 1] = γt + t1 * 2x[1]
    U[1, 2] = zero(ComplexF64)
    U[2, 1] = zero(ComplexF64)
    U[2, 2] = γt + t1 * 2x[2]
    return nothing
end

function HC.taylor!(u::AbstractVector, ::Val{1}, H::ManualSLH, x::AbstractVector, t::ComplexF64)
    u[1] = H.γ * (x[1] - 1) - (x[1]^2 - 1)
    u[2] = H.γ * (x[2] - 1) - (x[2]^2 - 1)
    return nothing
end

function HC.taylor!(
        u::AbstractVector, ::Val{K}, ::ManualSLH,
        tx::TaylorVector, t::ComplexF64,
    ) where {K}
    fill!(u, zero(eltype(u)))
    return nothing
end

function HC.set_solution!(
        x::AbstractVector, ::ManualSLH, y::AbstractVector, ::ComplexF64,
    )
    copyto!(x, y)
    return nothing
end

function HC.get_solution!(
        out::AbstractVector, ::ManualSLH, x::AbstractVector, ::ComplexF64,
    )
    copyto!(out, x)
    return nothing
end

# ═══════════════════════════════════════════════════════════════════════════════
# Helper: MP ground truth
# ═══════════════════════════════════════════════════════════════════════════════

function eval_mp(F, vars, x_vals)
    return ComplexF64[p(vars => x_vals) for p in F]
end

function eval_mp_jac(F, vars, x_vals)
    m, n = length(F), length(vars)
    J = zeros(ComplexF64, m, n)
    for j in 1:n, i in 1:m
        dp = mp_diff(F[i], vars[j])
        J[i, j] = dp(vars => x_vals)
    end
    return J
end

# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

@testset "Core Types" begin

    # ── SystemEvaluator from AbstractSystem ──────────────────────────────

    @testset "SystemEvaluator from AbstractSystem" begin
        F = QuadraticSystem()
        seval = SystemEvaluator(F)

        @test size(seval) == (2, 2)
        @test nparameters(seval) == 0

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        x = FSVec{ComplexF64}(ComplexF64[2.0, 3.0])
        p = FSVec{ComplexF64}(ComplexF64[])

        evaluate!(u, seval, x, p)
        @test u[1] ≈ 6.0 + 0im  # 4+3-1
        @test u[2] ≈ 4.0 + 0im  # 6-2

        U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        evaluate_and_jacobian!(u, U, seval, x, p)
        @test U[1, 1] ≈ 4.0 + 0im
        @test U[1, 2] ≈ 1.0 + 0im
        @test U[2, 1] ≈ 3.0 + 0im
        @test U[2, 2] ≈ 2.0 + 0im

        tv = TaylorVector{2, ComplexF64}(2)
        tv[1] = TruncatedTaylorSeries((ComplexF64(2.0), ComplexF64(1.0)))
        tv[2] = TruncatedTaylorSeries((ComplexF64(3.0), ComplexF64(0.0)))
        taylor!(u, Val(1), seval, tv, p)
        @test u[1] ≈ 4.0 + 0im  # 2*2*1 + 0
        @test u[2] ≈ 3.0 + 0im  # 2*0 + 1*3
    end

    # ── FW transparency: SystemEvaluator must match raw Interpreter ──────

    @testset "SystemEvaluator matches raw Interpreter" begin
        @polyvar x y
        F = [x^3 * y - x * y^2 + x^2 - 3, x^2 * y + y^3 - x + 2]
        sys = System(F)
        seval = sys.evaluator
        I_eval = sys._interp_f64
        I_jac = sys._interp_jac
        I_t1 = sys._interp_t1
        I_t2 = sys._interp_t2
        I_t3 = sys._interp_t3

        for _ in 1:10
            xvals = randn(ComplexF64, 2)
            xv_fs = FSVec{ComplexF64}(xvals)
            p_fs = FSVec{ComplexF64}(ComplexF64[])

            # eval: FW vs raw
            u_fw = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            u_raw = zeros(ComplexF64, 2)
            evaluate!(u_fw, seval, xv_fs, p_fs)
            execute!(u_raw, I_eval, xvals)
            @test u_fw ≈ u_raw atol = 1.0e-14

            # jacobian: FW vs raw
            U_fw = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
            U_raw = zeros(ComplexF64, 2, 2)
            evaluate_and_jacobian!(u_fw, U_fw, seval, xv_fs, p_fs)
            execute!(u_raw, U_raw, I_jac, xvals)
            @test u_fw ≈ u_raw atol = 1.0e-14
            @test U_fw ≈ U_raw atol = 1.0e-14

            # Taylor orders 1-3: FW vs raw
            for (K, I_t) in ((1, I_t1), (2, I_t2), (3, I_t3))
                N = K + 1
                coeffs = ntuple(_ -> randn(ComplexF64), Val(N))
                tv_fw = TaylorVector{N, ComplexF64}(2)
                tv_raw = TaylorVector{N, ComplexF64}(2)
                for i in 1:2
                    tts = TruncatedTaylorSeries(ntuple(k -> randn(ComplexF64), Val(N)))
                    tv_fw[i] = tts
                    tv_raw[i] = tts
                end
                u_tfw = FSVec{ComplexF64}(zeros(ComplexF64, 2))
                u_traw = zeros(ComplexF64, 2)
                taylor!(u_tfw, Val(K), seval, tv_fw, p_fs)
                execute_taylor!(u_traw, Val(K), I_t, tv_raw, ComplexF64[])
                @test u_tfw ≈ u_traw atol = 1.0e-13
            end
        end
    end

    # ── HomotopyEvaluator from user AbstractHomotopy ─────────────────────

    @testset "HomotopyEvaluator from AbstractHomotopy" begin
        H = ManualSLH(ComplexF64(1.0))
        heval = HomotopyEvaluator(H)

        @test size(heval) == (2, 2)

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        xv = FSVec{ComplexF64}(ComplexF64[2.0, 3.0])

        # At t=0: H = F = (x^2-1, y^2-1) = (3, 8)
        evaluate!(u, heval, xv, ComplexF64(0.0))
        @test u[1] ≈ 3.0 + 0im
        @test u[2] ≈ 8.0 + 0im

        # At t=1: H = γ*G = (x-1, y-1) = (1, 2)
        evaluate!(u, heval, xv, ComplexF64(1.0))
        @test u[1] ≈ 1.0 + 0im
        @test u[2] ≈ 2.0 + 0im

        # Jacobian at random t
        U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        t = ComplexF64(0.3)
        evaluate_and_jacobian!(u, U, heval, xv, t)
        @test U[1, 1] ≈ 0.3 + 0.7 * 4.0 + 0im  # γt + (1-t)*2x
        @test U[2, 2] ≈ 0.3 + 0.7 * 6.0 + 0im
        @test U[1, 2] ≈ 0.0 + 0im
        @test U[2, 1] ≈ 0.0 + 0im

        # Taylor order 1: ∂H/∂t
        taylor!(u, Val(1), heval, xv, t)
        @test u[1] ≈ (2.0 - 1) - (4.0 - 1) + 0im  # γ*(x-1) - (x²-1) = 1 - 3 = -2
        @test u[2] ≈ (3.0 - 1) - (9.0 - 1) + 0im  # = 2 - 8 = -6

        # set/get roundtrip
        y = FSVec{ComplexF64}(ComplexF64[1.0 + 2im, 3.0 - 1im])
        x = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        out = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        set_solution!(x, heval, y, ComplexF64(0.5))
        get_solution!(out, heval, x, ComplexF64(0.5))
        @test out ≈ y
    end

    # ── HomotopyEvaluator matches direct calls on AbstractHomotopy ───────

    @testset "HomotopyEvaluator is transparent wrapper" begin
        γ = cis(1.23)
        H = ManualSLH(γ)
        heval = HomotopyEvaluator(H)

        for _ in 1:5
            xvals = randn(ComplexF64, 2)
            xv = FSVec{ComplexF64}(xvals)
            t = ComplexF64(rand())

            # Direct call on AbstractHomotopy
            u_direct = zeros(ComplexF64, 2)
            HC.evaluate!(u_direct, H, xv, t)

            # Through HomotopyEvaluator
            u_wrapped = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            evaluate!(u_wrapped, heval, xv, t)

            @test u_wrapped ≈ u_direct atol = 1.0e-14

            # Jacobian
            U_direct = zeros(ComplexF64, 2, 2)
            HC.evaluate_and_jacobian!(u_direct, U_direct, H, xv, t)
            U_wrapped = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
            evaluate_and_jacobian!(u_wrapped, U_wrapped, heval, xv, t)

            @test U_wrapped ≈ U_direct atol = 1.0e-14
        end
    end

    # ── System: polynomial input ─────────────────────────────────────────

    @testset "System: metadata" begin
        @polyvar x y
        F = [x^2 + y - 1, x * y - 2]
        sys = System(F)

        @test sys isa System
        @test sys.evaluator isa SystemEvaluator
        @test size(sys) == (2, 2)
        @test HC.nparameters(sys) == 0
        @test HC.degrees(sys) == [2, 2]
        @test HC.nvariables(sys) == 2
        @test collect(HC.polynomials(sys)) == F
        @test collect(HC.variables(sys)) == [x, y]
        @test isempty(HC.parameters(sys))
        supp, coeffs = HC.support_coefficients(sys)
        @test supp == sys.support
        @test coeffs == sys.coefficients
        @test collect(sys.polys) == F
        @test collect(sys.variables) == [x, y]
        @test isempty(sys.parameters)
        @test HC.is_homogeneous(sys) == false
        @test sys.is_homogeneous == false
    end

    @testset "System: homogeneous metadata" begin
        @polyvar x y a
        F = [x^2 + a * y^2, x * y]
        sys = System(F; parameters = [a])

        @test collect(sys.variables) == [x, y]
        @test collect(sys.parameters) == [a]
        @test collect(HC.parameters(sys)) == [a]
        @test collect(HC.variables(sys)) == [x, y]
        @test HC.is_homogeneous(sys) == true
        @test sys.is_homogeneous == true
        @test_throws ArgumentError HC.support_coefficients(sys)
        # `a * y^2` has total degree 3, but degree 2 in `[x, y]`.
        @test HC.degrees(sys) == [2, 2]
    end

    @testset "System: degrees ignore the parameters" begin
        @polyvar x y a b
        # A higher power, a product of parameters, and a parameter-only term.
        sys = System(
            [a^3 * x^2 + y, a * b * x * y - a^5, b^2 - x];
            variables = [x, y], parameters = [a, b],
        )
        @test HC.degrees(sys) == [2, 2, 1]
    end

    @testset "System: eval+jac vs MP ground truth" begin
        @polyvar x y
        F = [x^3 * y - x * y^2 + x^2 - 3, x^2 * y + y^3 - x + 2]
        sys = System(F)
        seval = sys.evaluator
        vars = [x, y]

        for _ in 1:10
            xvals = randn(ComplexF64, 2)
            xv = FSVec{ComplexF64}(xvals)
            p = FSVec{ComplexF64}(ComplexF64[])
            u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))

            evaluate!(u, seval, xv, p)
            @test u ≈ eval_mp(F, vars, xvals) atol = 1.0e-12

            evaluate_and_jacobian!(u, U, seval, xv, p)
            @test u ≈ eval_mp(F, vars, xvals) atol = 1.0e-12
            @test U ≈ eval_mp_jac(F, vars, xvals) atol = 1.0e-12
        end
    end

    @testset "System: with parameters" begin
        @polyvar x y a b
        F = [x^2 + a * y, x * y - b]
        sys = System(F; parameters = [a, b])
        seval = sys.evaluator

        @test nparameters(seval) == 2
        @test HC.nparameters(sys) == 2

        xv = FSVec{ComplexF64}(ComplexF64[2.0, 3.0])
        p = FSVec{ComplexF64}(ComplexF64[1.0, 2.0])
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u, seval, xv, p)
        @test u[1] ≈ 7.0 + 0im   # 4 + 1*3
        @test u[2] ≈ 4.0 + 0im   # 6 - 2
    end

    @testset "System: DF64 matches F64" begin
        @polyvar x y
        F = [x^3 - y^2 + 1, x * y^2 - x^2]
        sys = System(F)
        seval = sys.evaluator

        for _ in 1:5
            xvals = randn(ComplexF64, 2)

            u_f64 = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            xv_f64 = FSVec{ComplexF64}(xvals)
            p = FSVec{ComplexF64}(ComplexF64[])
            evaluate!(u_f64, seval, xv_f64, p)

            u_df64 = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            xv_df64 = FSVec{ComplexDF64}(ComplexDF64.(xvals))
            evaluate!(u_df64, seval, xv_df64, p)

            @test u_df64 ≈ u_f64 atol = 1.0e-12
        end
    end

    @testset "System: Taylor vs finite differences" begin
        @polyvar x y
        F = [x^2 + y - 1, x * y - 2]
        sys = System(F)
        seval = sys.evaluator
        p = FSVec{ComplexF64}(ComplexF64[])

        # Pick a base point and direction
        x0 = ComplexF64[1.5, 2.5]
        dx = ComplexF64[0.3 + 0.1im, -0.2 + 0.4im]

        # Taylor order 1: should match directional derivative
        tv1 = TaylorVector{2, ComplexF64}(2)
        tv1[1] = TruncatedTaylorSeries((x0[1], dx[1]))
        tv1[2] = TruncatedTaylorSeries((x0[2], dx[2]))
        u_t1 = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        taylor!(u_t1, Val(1), seval, tv1, p)

        # FD order 1: (F(x0+ε*dx) - F(x0-ε*dx)) / (2ε) — central difference
        ε = 1.0e-5
        u0 = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        up = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        um = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u0, seval, FSVec{ComplexF64}(x0), p)
        evaluate!(up, seval, FSVec{ComplexF64}(x0 .+ ε .* dx), p)
        evaluate!(um, seval, FSVec{ComplexF64}(x0 .- ε .* dx), p)
        fd1 = (up .- um) ./ (2ε)

        @test u_t1[1] ≈ fd1[1] atol = 1.0e-5
        @test u_t1[2] ≈ fd1[2] atol = 1.0e-5

        # Taylor order 2: second Taylor coefficient = f''/(2!)
        # FD: (F(x0+ε*dx) - 2F(x0) + F(x0-ε*dx)) / ε² / 2
        tv2 = TaylorVector{3, ComplexF64}(2)
        tv2[1] = TruncatedTaylorSeries((x0[1], dx[1], zero(ComplexF64)))
        tv2[2] = TruncatedTaylorSeries((x0[2], dx[2], zero(ComplexF64)))
        u_t2 = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        taylor!(u_t2, Val(2), seval, tv2, p)

        fd2 = (up .- 2 .* u0 .+ um) ./ ε^2 ./ 2

        @test u_t2[1] ≈ fd2[1] atol = 1.0e-4
        @test u_t2[2] ≈ fd2[2] atol = 1.0e-4
    end

    @testset "System: katsura-3 eval+jac vs MP" begin
        @polyvar x0 x1 x2 x3
        F = [
            x0 + 2x1 + 2x2 + 2x3 - 1,
            x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
            2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
            x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
        ]
        vars = [x0, x1, x2, x3]
        sys = System(F)
        seval = sys.evaluator

        for _ in 1:5
            xvals = randn(ComplexF64, 4)
            xv = FSVec{ComplexF64}(xvals)
            p = FSVec{ComplexF64}(ComplexF64[])
            u = FSVec{ComplexF64}(zeros(ComplexF64, 4))
            U = FSMat{ComplexF64}(zeros(ComplexF64, 4, 4))

            evaluate_and_jacobian!(u, U, seval, xv, p)
            @test u ≈ eval_mp(F, vars, xvals) atol = 1.0e-12
            @test U ≈ eval_mp_jac(F, vars, xvals) atol = 1.0e-12
        end
    end

    # ── StraightLineHomotopy ─────────────────────────────────────────────

    @testset "StraightLineHomotopy: boundary conditions" begin
        @polyvar x y
        eval_G = System([x - 1, y - 1])
        eval_F = System([x^2 - 1, y^2 - 1])

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = ComplexF64(1.0))
        @test size(H) == (2, 2)
        heval = HomotopyEvaluator(H)

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        xv = FSVec{ComplexF64}(ComplexF64[2.0, 3.0])

        # t=1: H = γ*G = (x-1, y-1) = (1, 2)
        evaluate!(u, heval, xv, ComplexF64(1.0))
        @test u[1] ≈ 1.0 + 0im
        @test u[2] ≈ 2.0 + 0im

        # t=0: H = F = (x^2-1, y^2-1) = (3, 8)
        evaluate!(u, heval, xv, ComplexF64(0.0))
        @test u[1] ≈ 3.0 + 0im
        @test u[2] ≈ 8.0 + 0im

        # t=0.5: H = 0.5*(1+3, 2+8) = (2, 5)
        evaluate!(u, heval, xv, ComplexF64(0.5))
        @test u[1] ≈ 2.0 + 0im
        @test u[2] ≈ 5.0 + 0im
    end

    # Mid-path H(x,t) ≈ 0 means γt·G(x) and (1-t)·F(x) cancel while each term
    # is O(1). The DF64 evaluate! must combine unrounded DF64 residuals; a
    # Float64 round of G and F before combining floors |H| at 1e-16.
    @testset "StraightLineHomotopy: DF64 extended precision" begin
        @polyvar x
        eval_F = System([x^2 - 2])
        start_eval = HC._total_degree_startevaluator([2])
        H = StraightLineHomotopy(start_eval, eval_F.evaluator; γ = ComplexF64(1.0))

        # H(x,t) = t(x²-1) + (1-t)(x²-2) = x² - 2 + t, root x = √1.5 at t = 0.5
        setprecision(BigFloat, 512) do
            xb = sqrt(big"1.5")
            x_hi = Float64(xb)
            x_lo = Float64(xb - x_hi)
            x_df = FSVec{ComplexDF64}([ComplexDF64(DoubleF64(x_hi, x_lo))])
            u = FSVec{ComplexF64}(zeros(ComplexF64, 1))
            evaluate!(u, H, x_df, ComplexF64(0.5))
            @test abs(u[1]) < 1.0e-28
        end
    end

    @testset "StraightLineHomotopy: jacobian via finite differences" begin
        @polyvar x y
        eval_G = System([x^2 - 1, x * y + y^2])
        eval_F = System([x^3 + y - 2, x * y^2 - 1])

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator)
        heval = HomotopyEvaluator(H)

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        u_ε = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))

        for _ in 1:5
            xv = FSVec{ComplexF64}(randn(ComplexF64, 2))
            t = ComplexF64(rand())
            ε = 1.0e-7

            evaluate_and_jacobian!(u, U, heval, xv, t)

            for j in 1:2
                x_ε = FSVec{ComplexF64}(copy(Vector(xv)))
                x_ε[j] += ε
                evaluate!(u_ε, heval, x_ε, t)
                fd_col = (u_ε .- u) ./ ε
                @test U[1, j] ≈ fd_col[1] atol = 1.0e-5
                @test U[2, j] ≈ fd_col[2] atol = 1.0e-5
            end
        end
    end

    @testset "StraightLineHomotopy: SLH matches ManualSLH" begin
        @polyvar x y
        eval_G = System([x - 1, y - 1])
        eval_F = System([x^2 - 1, y^2 - 1])
        γ = cis(0.7)

        H_slh = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = γ)
        heval_slh = HomotopyEvaluator(H_slh)

        H_manual = ManualSLH(γ)
        heval_manual = HomotopyEvaluator(H_manual)

        for _ in 1:5
            xv = FSVec{ComplexF64}(randn(ComplexF64, 2))
            t = ComplexF64(rand())

            u_slh = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            u_man = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            evaluate!(u_slh, heval_slh, xv, t)
            evaluate!(u_man, heval_manual, xv, t)
            @test u_slh ≈ u_man atol = 1.0e-12

            U_slh = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
            U_man = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
            evaluate_and_jacobian!(u_slh, U_slh, heval_slh, xv, t)
            evaluate_and_jacobian!(u_man, U_man, heval_manual, xv, t)
            @test U_slh ≈ U_man atol = 1.0e-12

            # Taylor order 1
            u_t_slh = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            u_t_man = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            taylor!(u_t_slh, Val(1), heval_slh, xv, t)
            taylor!(u_t_man, Val(1), heval_manual, xv, t)
            @test u_t_slh ≈ u_t_man atol = 1.0e-12
        end
    end

    # Higher-order homotopy Taylor coefficients are taken along z ↦ H(x(z), t + z),
    # which is the quantity used by the tracker's implicit differentiation.
    @testset "StraightLineHomotopy: Taylor order 2 via finite differences (coupled x,t)" begin
        @polyvar x y
        eval_G = System([x^2 + x * y - 1, x * y + y^2 - 2])
        eval_F = System([x^3 + x * y + y - 2, x * y^2 + x^2 - 1])

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = ComplexF64(1.0))
        heval = HomotopyEvaluator(H)

        x0 = ComplexF64[1.5, -0.7]
        dx = ComplexF64[0.3, 0.2]
        t = ComplexF64(0.4)
        ε = 1.0e-4

        function eval_h_xt2(z)
            u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            xv = FSVec{ComplexF64}(x0 .+ z .* dx)
            evaluate!(u, heval, xv, t + z)
            return Vector(u)
        end

        h0 = eval_h_xt2(0.0)
        hp = eval_h_xt2(ε)
        hm = eval_h_xt2(-ε)

        fd2 = (hp .- 2 .* h0 .+ hm) ./ ε^2 ./ 2

        tv2 = TaylorVector{3, ComplexF64}(2)
        tv2[1] = TruncatedTaylorSeries((x0[1], dx[1], zero(ComplexF64)))
        tv2[2] = TruncatedTaylorSeries((x0[2], dx[2], zero(ComplexF64)))
        u_t2 = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        taylor!(u_t2, Val(2), heval, tv2, t)
        @test u_t2[1] ≈ fd2[1] atol = 1.0e-3
        @test u_t2[2] ≈ fd2[2] atol = 1.0e-3
    end

    @testset "StraightLineHomotopy: Taylor order 3 via finite differences (coupled x,t)" begin
        @polyvar x y
        eval_G = System([x^3 + x * y - y^2, x * y^2 + y^3])
        eval_F = System([x^3 + y^3 - 1, x^2 * y - x * y^2 + x])

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator; γ = ComplexF64(1.0))
        heval = HomotopyEvaluator(H)

        x0 = ComplexF64[1.0, -0.5]
        dx = ComplexF64[0.2, 0.3]
        t = ComplexF64(0.6)
        ε = 1.0e-3

        function eval_h_xt3(z)
            u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
            xv = FSVec{ComplexF64}(x0 .+ z .* dx)
            evaluate!(u, heval, xv, t + z)
            return Vector(u)
        end

        hp1 = eval_h_xt3(ε)
        hp2 = eval_h_xt3(2ε)
        hm1 = eval_h_xt3(-ε)
        hm2 = eval_h_xt3(-2ε)

        fd3 = (hp2 .- 2 .* hp1 .+ 2 .* hm1 .- hm2) ./ (2 * ε^3) ./ 6

        tv3 = TaylorVector{4, ComplexF64}(2)
        tv3[1] = TruncatedTaylorSeries((x0[1], dx[1], zero(ComplexF64), zero(ComplexF64)))
        tv3[2] = TruncatedTaylorSeries((x0[2], dx[2], zero(ComplexF64), zero(ComplexF64)))
        u_t3 = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        taylor!(u_t3, Val(3), heval, tv3, t)
        @test u_t3[1] ≈ fd3[1] atol = 1.0e-2
        @test u_t3[2] ≈ fd3[2] atol = 1.0e-2
    end

    @testset "StraightLineHomotopy: random γ has unit magnitude" begin
        @polyvar x y
        eval_G = System([x - 1, y - 1])
        eval_F = System([x^2 - 1, y^2 - 1])
        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator)
        @test abs(H.γ) ≈ 1.0 atol = 1.0e-14
    end

    # ── Zero-allocation checks ───────────────────────────────────────────

    @testset "zero allocations: SystemEvaluator eval + jac" begin
        @polyvar x y
        sys = System([x^2 + y - 1, x * y - 2])
        seval = sys.evaluator

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        xv = FSVec{ComplexF64}(ComplexF64[2.0, 3.0])
        p = FSVec{ComplexF64}(ComplexF64[])
        U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))

        evaluate!(u, seval, xv, p)
        evaluate_and_jacobian!(u, U, seval, xv, p)

        @test (@allocated evaluate!(u, seval, xv, p)) == 0
        @test (@allocated evaluate_and_jacobian!(u, U, seval, xv, p)) == 0
    end

    @testset "zero allocations: HomotopyEvaluator eval + jac" begin
        @polyvar x y
        eval_G = System([x - 1, y - 1])
        eval_F = System([x^2 - 1, y^2 - 1])

        H = StraightLineHomotopy(eval_G.evaluator, eval_F.evaluator)
        heval = HomotopyEvaluator(H)

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        xv = FSVec{ComplexF64}(ComplexF64[2.0, 3.0])
        U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        t = ComplexF64(0.5)

        evaluate!(u, heval, xv, t)
        evaluate_and_jacobian!(u, U, heval, xv, t)

        @test (@allocated evaluate!(u, heval, xv, t)) == 0
        @test (@allocated evaluate_and_jacobian!(u, U, heval, xv, t)) == 0
    end

    # ═══════════════════════════════════════════════════════════════════════════
    # CoefficientHomotopy
    # ═══════════════════════════════════════════════════════════════════════════

    @testset "CoefficientHomotopy: interpolation" begin
        # Build a parametric system: f(x; a,b) = [a*x^2 + b*x - 1]
        @polyvar x a b
        F = System([a * x^2 + b * x - 1]; parameters = [a, b])

        start_coeffs = ComplexF64[2.0, 3.0]   # a=2, b=3 at t=1
        target_coeffs = ComplexF64[1.0, -1.0]  # a=1, b=-1 at t=0

        H = CoefficientHomotopy(F.evaluator, start_coeffs, target_coeffs)
        m, n = size(H)
        @test m == 1
        @test n == 1

        u = FSVec{ComplexF64}(zeros(ComplexF64, m))
        xv = FSVec{ComplexF64}(ComplexF64[0.5])

        # At t=1: coeffs = start → F(x; 2, 3) = 2*0.25 + 3*0.5 - 1 = 1.0
        evaluate!(u, H, xv, ComplexF64(1.0))
        @test abs(u[1] - 1.0) < 1.0e-12

        # At t=0: coeffs = target → F(x; 1, -1) = 1*0.25 + (-1)*0.5 - 1 = -1.25
        evaluate!(u, H, xv, ComplexF64(0.0))
        @test abs(u[1] - (-1.25)) < 1.0e-12

        # At t=0.5: coeffs = (1.5, 1.0) → 1.5*0.25 + 1.0*0.5 - 1 = -0.125
        evaluate!(u, H, xv, ComplexF64(0.5))
        @test abs(u[1] - (-0.125)) < 1.0e-12
    end

    @testset "CoefficientHomotopy: jacobian" begin
        @polyvar x y a b
        F = System([a * x + b * y, x * y - a]; parameters = [a, b])

        H = CoefficientHomotopy(F.evaluator, ComplexF64[2.0, 1.0], ComplexF64[1.0, 1.0])
        m, n = size(H)
        u = FSVec{ComplexF64}(zeros(ComplexF64, m))
        U = FSMat{ComplexF64}(zeros(ComplexF64, m, n))
        xv = FSVec{ComplexF64}(ComplexF64[1.0, 2.0])

        evaluate_and_jacobian!(u, U, H, xv, ComplexF64(0.0))

        # At t=0: a=1, b=1. F = [x+y, xy-1]. J = [1 1; y x] = [1 1; 2 1]
        @test abs(U[1, 1] - 1.0) < 1.0e-12
        @test abs(U[1, 2] - 1.0) < 1.0e-12
        @test abs(U[2, 1] - 2.0) < 1.0e-12
        @test abs(U[2, 2] - 1.0) < 1.0e-12
    end

    @testset "CoefficientHomotopy: taylor order 1" begin
        @polyvar x a
        F = System([a * x^2 - 1]; parameters = [a])

        start = ComplexF64[3.0]
        target = ComplexF64[1.0]
        H = CoefficientHomotopy(F.evaluator, start, target)

        u = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        xv = FSVec{ComplexF64}(ComplexF64[0.5])

        # Taylor order 1: dH/dt = F(x; start - target) = F(x; 2) = 2*0.25 - 1 = -0.5
        taylor!(u, Val(1), H, xv, ComplexF64(0.5))
        @test abs(u[1] - (-0.5)) < 1.0e-12
    end

    @testset "CoefficientHomotopy: zero allocations" begin
        @polyvar x a
        F = System([a * x^2 - 1]; parameters = [a])
        H = CoefficientHomotopy(F.evaluator, ComplexF64[2.0], ComplexF64[1.0])
        heval = HomotopyEvaluator(H)

        m, n = size(heval)
        u = FSVec{ComplexF64}(zeros(ComplexF64, m))
        U = FSMat{ComplexF64}(zeros(ComplexF64, m, n))
        xv = FSVec{ComplexF64}(ComplexF64[0.5])
        t = ComplexF64(0.5)

        evaluate!(u, heval, xv, t)
        evaluate_and_jacobian!(u, U, heval, xv, t)

        @test (@allocated evaluate!(u, heval, xv, t)) == 0
        @test (@allocated evaluate_and_jacobian!(u, U, heval, xv, t)) == 0
    end
end
