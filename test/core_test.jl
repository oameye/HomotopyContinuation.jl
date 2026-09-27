using Test, Random
using HomotopyContinuation
using HomotopyContinuation: TaylorVector, DoubleF64, ComplexDF64
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: MultivariatePolynomials as MP
using FixedSizeArrays: FixedSizeVectorDefault, FixedSizeMatrixDefault
using Preferences: load_preference

mp_value(polys, vars, x, params, p) =
    ComplexF64[isempty(params) ? f(vars => x) : f(vars => x, params => p) for f in polys]

function mp_jacobian(polys, vars, x, params, p)
    J = zeros(ComplexF64, length(polys), length(vars))
    for j in eachindex(vars), i in eachindex(polys)
        df = MP.differentiate(polys[i], vars[j])
        J[i, j] = isempty(params) ? df(vars => x) : df(vars => x, params => p)
    end
    return J
end

@testset "System public behavior" begin
    @testset "metadata" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2]; variables = [x, y])

        @test size(F) == (2, 2)
        @test nvariables(F) == 2
        @test nparameters(F) == 0
        @test collect(variables(F)) == [x, y]
        @test isempty(parameters(F))
        @test degrees(F) == [2, 2]

        @polyvar a b
        P = System(
            [a^3 * x^2 + y, a * b * x * y - a^5, b^2 - x];
            variables = [x, y], parameters = [a, b],
        )
        @test size(P) == (3, 2)
        @test nvariables(P) == 2
        @test nparameters(P) == 2
        @test collect(variables(P)) == [x, y]
        @test collect(parameters(P)) == [a, b]
        @test degrees(P) == [2, 2, 1]

        # `a * y^2` has total degree 3 but degree 2 in the variables.
        Q = System([x^2 + a * y^2, x * y]; variables = [x, y], parameters = [a])
        @test degrees(Q) == [2, 2]
    end

    @testset "evaluation and Jacobian" begin
        @polyvar x y
        F = System([x^2 + y - 1, x * y - 2]; variables = [x, y])
        point = [2.0, 3.0]

        @test evaluate(F, point) ≈ ComplexF64[6, 4]
        @test F(point) ≈ ComplexF64[6, 4]
        @test jacobian(F, point) ≈ ComplexF64[4 1; 3 2]

        @polyvar a b
        P = System(
            [x^2 + a * y, x * y - b];
            variables = [x, y], parameters = [a, b],
        )
        params = [1.0, 2.0]
        @test evaluate(P, point, params) ≈ ComplexF64[7, 4]
        @test P(point, params) ≈ ComplexF64[7, 4]
        @test jacobian(P, point, params) ≈ ComplexF64[4 1; 3 2]
    end

    @testset "public evaluation validates dimensions" begin
        @polyvar x y a b
        F = System(
            [x^2 + a * y, x * y - b];
            variables = [x, y], parameters = [a, b],
        )

        @test_throws ArgumentError evaluate(F, [1.0], [2.0, 3.0])
        @test_throws ArgumentError evaluate(F, [1.0, 2.0], [3.0])
        @test_throws ArgumentError jacobian(F, [1.0], [2.0, 3.0])
        @test_throws ArgumentError jacobian(F, [1.0, 2.0], [3.0])
    end
end

# `evaluate!`, `evaluate_and_jacobian!` and `taylor!` of a `System` are reached
# through the homotopy `H(x, t) = F(x; t·p₁ + (1 - t)·p₀)`. Its buffers are the
# fixed-size arrays the package tracks with.
@static if VERSION < v"1.11"
    buffer(v::AbstractVector{T}) where {T} = Vector{T}(v)
    buffer(M::AbstractMatrix{T}) where {T} = Matrix{T}(M)
else
    buffer(v::AbstractVector{T}) where {T} = FixedSizeVectorDefault{T}(v)
    buffer(M::AbstractMatrix{T}) where {T} = FixedSizeMatrixDefault{T}(M)
end

# Taylor series whose order-k coefficients are the rows of `X`.
function taylor_vector(X::AbstractMatrix{ComplexF64})
    tv = TaylorVector{size(X, 1), ComplexF64}(size(X, 2))
    for i in axes(X, 2)
        tv[i] = Tuple(X[:, i])
    end
    return tv
end

_at(f, vars, x, params, p) = isempty(params) ? f(vars => x) : f(vars => x, params => p)

# Order-K coefficient of λ ↦ F(x(λ); p(t + λ)), with x(λ) = Σₖ X[k+1, :] λᵏ and
# p(t + λ) = p(t) + λ·(p₁ - p₀), by exact polynomial substitution.
function series_coefficient(polys, vars, X, params, pt, dp, K)
    @polyvar λ
    xs = [sum(X[k + 1, i] * λ^k for k in 0:(size(X, 1) - 1)) for i in eachindex(vars)]
    ps = [pt[i] + dp[i] * λ for i in eachindex(params)]
    return ComplexF64[MP.coefficient(_at(f, vars, xs, params, ps), λ^K) for f in polys]
end

const MODES = (CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL)

# `@stable` heap-allocates the closures the evaluators call through, so the
# zero-allocation contract describes the shipped configuration only.
const INSTRUMENTED =
    load_preference(HomotopyContinuation, "dispatch_doctor_mode", "disable") != "disable"

@testset "System mutating evaluation through ParameterHomotopy" begin
    @polyvar x y a b
    vars, params = [x, y], [a, b]
    polys = [x^3 * y - a * x * y^2 + x^2 - 3b, x^2 * y + b * y^3 - a^2 * x + 2]
    p1 = ComplexF64[0.8 - 0.3im, 1.4 + 0.2im]
    p0 = ComplexF64[-0.5 + 0.9im, 0.6 - 1.1im]
    t = 0.37 + 0.21im
    pt = t .* p1 .+ (1 - t) .* p0
    dp = p1 .- p0

    @testset "$mode" for mode in MODES
        F = System(polys; variables = vars, parameters = params, compile = mode)
        H = ParameterHomotopy(F, p1, p0)
        @test size(H) == (2, 2)
        rng = MersenneTwister(0x0000c07e)
        for _ in 1:5
            xvals = randn(rng, ComplexF64, 2)
            X = randn(rng, ComplexF64, 4, 2)
            X[1, :] = xvals
            truth_u = mp_value(polys, vars, xvals, params, pt)
            truth_J = mp_jacobian(polys, vars, xvals, params, pt)

            u = buffer(zeros(ComplexF64, 2))
            U = buffer(zeros(ComplexF64, 2, 2))
            evaluate!(u, H, buffer(xvals), t)
            @test u ≈ truth_u rtol = 1.0e-12
            fill!(u, 0)
            evaluate_and_jacobian!(u, U, H, buffer(xvals), t)
            @test u ≈ truth_u rtol = 1.0e-12
            @test U ≈ truth_J rtol = 1.0e-12

            # Order 1 holds x fixed: ∂H/∂t.
            taylor!(u, Val(1), H, buffer(xvals), t)
            @test u ≈ series_coefficient(polys, vars, X[1:1, :], params, pt, dp, 1) rtol = 1.0e-12

            # Higher orders move x and t together, as the tracker's predictor does.
            for K in 2:3
                fill!(u, 0)
                taylor!(u, Val(K), H, taylor_vector(X[1:(K + 1), :]), t)
                @test u ≈ series_coefficient(polys, vars, X[1:(K + 1), :], params, pt, dp, K) rtol = 1.0e-11
            end
        end
    end

    @testset "the homotopy interpolates the parameters linearly" begin
        F = System(polys; variables = vars, parameters = params)
        H = ParameterHomotopy(F, p1, p0)
        xvals = ComplexF64[0.4 - 0.2im, -1.1 + 0.3im]
        u = buffer(zeros(ComplexF64, 2))
        evaluate!(u, H, buffer(xvals), complex(1.0))
        @test u ≈ evaluate(F, xvals, p1) rtol = 1.0e-14
        evaluate!(u, H, buffer(xvals), complex(0.0))
        @test u ≈ evaluate(F, xvals, p0) rtol = 1.0e-14
    end

    @testset "a parameter-free system has Taylor coefficients in x alone" begin
        F = System([x^2 + y - 1, x * y - 2])
        H = ParameterHomotopy(F, ComplexF64[], ComplexF64[])
        x0 = ComplexF64[1.5, 2.5]
        u = buffer(zeros(ComplexF64, 2))
        taylor!(u, Val(1), H, buffer(x0), complex(0.5))
        @test u == zeros(ComplexF64, 2)
        # (x₀ + x₁λ + x₂λ²)² + (y₀ + y₁λ + y₂λ²) - 1 and the product xy, at order 2.
        X = ComplexF64[1.5 2.5; 0.3 + 0.1im -0.2 + 0.4im; 0.7 -0.1im]
        taylor!(u, Val(2), H, taylor_vector(X), complex(0.5))
        @test u[1] ≈ X[2, 1]^2 + 2 * X[1, 1] * X[3, 1] + X[3, 2]
        @test u[2] ≈ X[1, 1] * X[3, 2] + X[2, 1] * X[2, 2] + X[3, 1] * X[1, 2]
    end
end

@testset "extended-precision residual" begin
    # H(x, t) = x² - (t + 2(1 - t)) has the root x = √1.5 at t = 1/2. The residual
    # there is a cancellation of O(1) terms, so only an unrounded double-double
    # evaluation reaches below the Float64 floor of about 1e-16.
    @polyvar x a
    F = System([x^2 - a]; parameters = [a])
    H = ParameterHomotopy(F, [1.0], [2.0])
    u = buffer(zeros(ComplexF64, 1))
    setprecision(BigFloat, 256) do
        root = sqrt(big"1.5")
        hi = Float64(root)
        lo = Float64(root - hi)
        evaluate!(u, H, buffer([ComplexDF64(DoubleF64(hi, lo))]), complex(0.5))
        @test abs(u[1]) < 1.0e-28
        evaluate!(u, H, buffer([complex(hi)]), complex(0.5))
        @test abs(u[1]) > 1.0e-17
    end

    @testset "$mode agrees with the double-precision value" for mode in MODES
        @polyvar y
        G = System([x^3 - y^2 + 1, x * y^2 - x^2]; compile = mode)
        HG = ParameterHomotopy(G, ComplexF64[], ComplexF64[])
        rng = MersenneTwister(0x0000df64)
        v = buffer(zeros(ComplexF64, 2))
        for _ in 1:5
            xvals = randn(rng, ComplexF64, 2)
            evaluate!(v, HG, buffer(ComplexDF64.(xvals)), complex(0.0))
            @test v ≈ evaluate(G, xvals) rtol = 1.0e-13
        end
    end
end

# Measured inside a function, so that the global-scope `Val` construction and
# argument boxing of a testset are not charged to the call.
taylor_allocations(u, v::Val, H, x, t) = @allocated taylor!(u, v, H, x, t)
evaluate_allocations(u, H, x, t) = @allocated evaluate!(u, H, x, t)
jacobian_allocations(u, U, H, x, t) = @allocated evaluate_and_jacobian!(u, U, H, x, t)

@testset "zero allocations on the mutating entry points" begin
    @polyvar x y a
    F = System([x^2 + a * y - 1, x * y - a^2]; parameters = [a])
    H = ParameterHomotopy(F, [2.0], [1.0])
    u = buffer(zeros(ComplexF64, 2))
    U = buffer(zeros(ComplexF64, 2, 2))
    xv = buffer(ComplexF64[2.0, 3.0])
    xd = buffer(ComplexDF64.(ComplexF64[2.0, 3.0]))
    t = complex(0.5)
    tx2 = taylor_vector(randn(MersenneTwister(1), ComplexF64, 3, 2))
    tx3 = taylor_vector(randn(MersenneTwister(2), ComplexF64, 4, 2))

    evaluate_allocations(u, H, xv, t)
    evaluate_allocations(u, H, xd, t)
    jacobian_allocations(u, U, H, xv, t)
    taylor_allocations(u, Val(1), H, xv, t)
    taylor_allocations(u, Val(2), H, tx2, t)
    taylor_allocations(u, Val(3), H, tx3, t)

    INSTRUMENTED || @test evaluate_allocations(u, H, xv, t) == 0
    INSTRUMENTED || @test evaluate_allocations(u, H, xd, t) == 0
    INSTRUMENTED || @test jacobian_allocations(u, U, H, xv, t) == 0
    INSTRUMENTED || @test taylor_allocations(u, Val(1), H, xv, t) == 0
    INSTRUMENTED || @test taylor_allocations(u, Val(2), H, tx2, t) == 0
    INSTRUMENTED || @test taylor_allocations(u, Val(3), H, tx3, t) == 0
end
