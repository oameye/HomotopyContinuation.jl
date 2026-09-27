using Test
using LinearAlgebra: LinearAlgebra, Diagonal, cond, ldiv!, norm
using Random: Random
using Preferences: load_preference
using HomotopyContinuation: HomotopyContinuation, MatrixWorkspace, updated!, factorize!,
    iterative_refinement!

@static if VERSION < v"1.11"
    const Vec{T} = Vector{T}
else
    using FixedSizeArrays: FixedSizeVectorDefault
    const Vec{T} = FixedSizeVectorDefault{T}
end

const INSTRUMENTED =
    load_preference(HomotopyContinuation, "dispatch_doctor_mode", "disable") != "disable"

cvec(v::AbstractVector) = Vec{ComplexF64}(collect(ComplexF64, v))
cvec(n::Integer) = Vec{ComplexF64}(zeros(ComplexF64, n))

function workspace(A::AbstractMatrix)
    WS = MatrixWorkspace(size(A)...)
    copyto!(WS, A)
    updated!(WS)
    return WS
end

function solve(WS::MatrixWorkspace, b::AbstractVector)
    x = cvec(size(WS, 2))
    ldiv!(x, WS, cvec(b))
    return collect(x)
end

function bigfloat_solution(A, b)
    return setprecision(BigFloat, 256) do
        ComplexF64.(Complex{BigFloat}.(A) \ Complex{BigFloat}.(b))
    end
end

relerr(x, y) = norm(x - y, Inf) / norm(y, Inf)

@testset "construction" begin
    @test size(MatrixWorkspace(4, 4)) == (4, 4)
    @test size(MatrixWorkspace(6, 4)) == (6, 4)
    @test_throws ArgumentError MatrixWorkspace(2, 5)
end

@testset "1x1 system" begin
    WS = MatrixWorkspace(1, 1)
    WS[1, 1] = 3.0 + 1.0im
    updated!(WS)
    @test only(solve(WS, [2.0 + 0.5im])) ≈ (2.0 + 0.5im) / (3.0 + 1.0im)
    @test cond(WS) == 1.0
    WS[1, 1] = 0.0
    updated!(WS)
    @test cond(WS) == Inf
end

@testset "square LU solve against dense LinearAlgebra" begin
    Random.seed!(0x1a2b)
    for n in (2, 3, 7, 13)
        WS = MatrixWorkspace(n, n)
        # The same workspace is refilled, so every solve must refactorize.
        for _ in 1:3
            A = randn(ComplexF64, n, n)
            b = randn(ComplexF64, n)
            copyto!(WS, A)
            updated!(WS)
            @test solve(WS, b) ≈ A \ b rtol = 1.0e-10
        end
    end
end

@testset "repeated solves after changing single entries" begin
    Random.seed!(0x3c4d)
    n = 5
    A = randn(ComplexF64, n, n)
    WS = workspace(A)
    bs = [randn(ComplexF64, n) for _ in 1:3]
    for b in bs
        @test solve(WS, b) ≈ A \ b rtol = 1.0e-10
    end
    for (i, j) in ((1, 1), (3, 5), (5, 2))
        A[i, j] += 2.0 - 1.0im
        WS[i, j] = A[i, j]
        updated!(WS)
        for b in bs
            @test solve(WS, b) ≈ A \ b rtol = 1.0e-10
        end
    end

    # An explicit `factorize!` after `updated!` serves every right-hand side that
    # follows.
    A[2, 4] -= 1.5im
    WS[2, 4] = A[2, 4]
    updated!(WS)
    factorize!(WS)
    factorize!(WS)
    for b in bs
        @test solve(WS, b) ≈ A \ b rtol = 1.0e-10
    end
end

@testset "overdetermined least squares against dense QR" begin
    Random.seed!(0x5e6f)
    for (m, n) in ((6, 4), (9, 2), (13, 7))
        WS = MatrixWorkspace(m, n)
        for _ in 1:2
            A = randn(ComplexF64, m, n)
            copyto!(WS, A)
            updated!(WS)
            # A consistent system is solved exactly, an inconsistent one in the
            # least-squares sense.
            x_true = randn(ComplexF64, n)
            @test solve(WS, A * x_true) ≈ x_true rtol = 1.0e-10
            b = randn(ComplexF64, m)
            x = solve(WS, b)
            @test x ≈ A \ b rtol = 1.0e-10
            @test norm(A' * (A * x - b)) < 1.0e-10 * norm(A)^2 * norm(b)
        end
    end
end

@testset "condition estimate against dense cond(A, Inf)" begin
    Random.seed!(0x7a8b)
    scales = exp10.(range(-6, 6; length = 6))
    for _ in 1:5
        for A in (
                randn(ComplexF64, 6, 6),
                randn(ComplexF64, 6, 6) * Diagonal(scales),
                Diagonal(reverse(scales)) * randn(ComplexF64, 6, 6),
            )
            exact = cond(A, Inf)
            estimate = cond(workspace(A))
            # A 1-norm estimator of the inverse: a lower bound, rarely far below.
            @test exact / 10 <= estimate <= exact * (1 + 1.0e-8)
        end
    end
    n = 10
    hilbert = ComplexF64[1 / (i + j - 1) for i in 1:n, j in 1:n]
    @test cond(workspace(hilbert)) ≈ cond(hilbert, Inf) rtol = 0.1
end

@testset "solves and condition are invariant under power-of-two scaling" begin
    Random.seed!(0x9c0d)
    A = randn(ComplexF64, 5, 5)
    b = randn(ComplexF64, 5)
    x = solve(workspace(A), b)
    κ = cond(workspace(A))
    for σ in (exp2(-40.0), exp2(20.0), exp2(60.0))
        @test solve(workspace(σ .* A), σ .* b) == x
        @test solve(workspace(σ .* A), b) == x ./ σ
        @test cond(workspace(σ .* A)) ≈ κ rtol = 4 * eps()
    end
end

@testset "iterative refinement recovers accuracy on ill-conditioned systems" begin
    Random.seed!(0xe1f2)
    for n in (8, 10)
        A = ComplexF64[1 / (i + j - 1) for i in 1:n, j in 1:n] .* cis(0.3)
        b = A * randn(ComplexF64, n)
        x_ref = bigfloat_solution(A, b)
        WS = workspace(A)
        x = cvec(n)
        ldiv!(x, WS, cvec(b))
        before = relerr(collect(x), x_ref)
        # cond(A) ≈ 1e10 and 1e13: the Float64 solve alone loses many digits.
        @test before > 1.0e-10
        result = iterative_refinement!(x, WS, cvec(b); tol = 1.0e-20, max_iters = 10)
        @test relerr(collect(x), x_ref) <= 4 * eps()
        @test result.accuracy < 1.0e-15
    end

    @testset "badly scaled system" begin
        A = ComplexF64[
            1.0e8 1.0 0.0 0.0
            1.0 1.0e-8 1.0 0.0
            0.0 1.0 1.0e6 1.0
            1.0 0.0 1.0 1.0e-6
        ]
        b = A * ComplexF64[1.0e-3, -2.0, 5.0e1, -3.0e-2]
        x_ref = bigfloat_solution(A, b)
        WS = workspace(A)
        x = cvec(4)
        ldiv!(x, WS, cvec(b))
        before = relerr(collect(x), x_ref)
        result = iterative_refinement!(x, WS, cvec(b))
        @test relerr(collect(x), x_ref) <= min(before, 4 * eps())
        @test !result.diverged
    end

    @testset "a well-conditioned solve converges in one round" begin
        A = randn(ComplexF64, 6, 6) + 5.0 * LinearAlgebra.I
        b = randn(ComplexF64, 6)
        WS = workspace(A)
        x = cvec(6)
        ldiv!(x, WS, cvec(b))
        result = iterative_refinement!(x, WS, cvec(b); max_iters = 1)
        @test result.accuracy < sqrt(eps())
        @test !result.diverged
        @test relerr(collect(x), bigfloat_solution(A, b)) <= 4 * eps()
    end
end

function refactorize_and_solve!(x, WS, b)
    updated!(WS)
    ldiv!(x, WS, b)
    return x
end

@testset "ldiv! does not allocate" begin
    Random.seed!(0x1357)
    for (m, n) in ((3, 3), (13, 13), (8, 5))
        WS = workspace(randn(ComplexF64, m, n))
        b = cvec(randn(ComplexF64, m))
        x = cvec(n)
        refactorize_and_solve!(x, WS, b)
        ldiv!(x, WS, b)
        # DispatchDoctor instrumentation allocates on its own, and before Julia
        # 1.11 the buffers are ordinary Vectors and the counter is process wide.
        if !INSTRUMENTED && VERSION >= v"1.11"
            @test (@allocated refactorize_and_solve!(x, WS, b)) == 0
            @test (@allocated ldiv!(x, WS, b)) == 0
        end
    end
end
