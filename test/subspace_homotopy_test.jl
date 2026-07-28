using Test, Random
using LinearAlgebra
using HomotopyContinuationNext
using HomotopyContinuationNext: IntrinsicSubspaceHomotopy, ExtrinsicSubspaceHomotopy,
    set_subspaces!, target_parameters!, intrinsic_coordinates!, ambient_coordinates!,
    rand_subspace, extrinsic, intrinsic, LinearSubspace,
    HomotopyEvaluator, Tracker, TrackerCode, track!, evaluate!, evaluate_and_jacobian!,
    taylor!, FSVec, FSMat, TaylorVector
using DynamicPolynomials: @polyvar, differentiate

# Quadric surface in C^3, witness points on a random line.
@polyvar z[1:3]
quadric = System([z[1]^2 + 2z[2]^2 + 3z[3]^2 + z[1] * z[2] - 1]; variables = z)

# Exact witness points of X ∩ L for a dim-1 subspace L: substitute the line
# parametrization x = A*u + b into the quadric and solve the univariate quadratic.
function witness_points(L)
    A, b = intrinsic(L).A, intrinsic(L).b
    f(u) = begin
        x = A .* u .+ b
        x[1]^2 + 2x[2]^2 + 3x[3]^2 + x[1] * x[2] - 1
    end
    # f(u) = α u^2 + β u + γ0 via three evaluations
    γ0 = f(0.0 + 0im)
    α = (f(1.0 + 0im) + f(-1.0 + 0im)) / 2 - γ0
    β = (f(1.0 + 0im) - f(-1.0 + 0im)) / 2
    disc = sqrt(β^2 - 4α * γ0)
    us = [(-β + disc) / (2α), (-β - disc) / (2α)]
    return [vec(A .* u .+ b) for u in us]
end

@testset "IntrinsicSubspaceHomotopy tracks witness points" begin
    Random.seed!(11)
    V = rand_subspace(3; dim = 1)
    W = rand_subspace(3; dim = 1)
    H = IntrinsicSubspaceHomotopy(quadric.evaluator, V, W)
    tracker = Tracker(HomotopyEvaluator(H))

    wV = witness_points(V)
    wW = witness_points(W)
    n_u = size(H)[2]
    u = zeros(ComplexF64, n_u)
    x = zeros(ComplexF64, 3)
    for x_start in wV
        intrinsic_coordinates!(u, H, ComplexF64.(x_start), complex(1.0))
        code = track!(tracker, u)
        @test code == TrackerCode.TRACKER_SUCCESS
        ambient_coordinates!(x, H, Vector(tracker.state.x), complex(0.0))
        # endpoint lies on the quadric and on W
        @test abs(x[1]^2 + 2x[2]^2 + 3x[3]^2 + x[1] * x[2] - 1) < 1.0e-10
        @test norm(extrinsic(W).A * x - extrinsic(W).b) < 1.0e-10
        @test minimum(norm(x - w) for w in wW) < 1.0e-8
    end
end

@testset "taylor! against finite differences" begin
    Random.seed!(12)
    V = rand_subspace(3; dim = 1)
    W = rand_subspace(3; dim = 1)
    H = IntrinsicSubspaceHomotopy(quadric.evaluator, V, W)
    m, n = size(H)
    u1 = FSVec{ComplexF64}(zeros(ComplexF64, m))
    xu = FSVec{ComplexF64}(randn(ComplexF64, n))
    t = complex(0.6)
    taylor!(u1, Val(1), H, xu, t)
    # central finite difference of evaluate! in t at fixed intrinsic u
    h = 1.0e-6
    up = FSVec{ComplexF64}(zeros(ComplexF64, m))
    um = FSVec{ComplexF64}(zeros(ComplexF64, m))
    evaluate!(up, H, xu, t + h)
    evaluate!(um, H, xu, t - h)
    fd = (Vector(up) .- Vector(um)) ./ (2h)
    @test Vector(u1) ≈ fd atol = 1.0e-5
end

@testset "set_subspaces! retarget" begin
    Random.seed!(13)
    V = rand_subspace(3; dim = 1)
    W = rand_subspace(3; dim = 1)
    W2 = rand_subspace(3; dim = 1)
    H = IntrinsicSubspaceHomotopy(quadric.evaluator, V, W; gamma = nothing)
    m, n = size(H)
    u = FSVec{ComplexF64}(zeros(ComplexF64, m))
    xu = FSVec{ComplexF64}(randn(ComplexF64, n))
    set_subspaces!(H, V, W2)
    # First query at exactly t = 0 must reflect the NEW target (NaN invalidation)
    evaluate!(u, H, xu, complex(0.0))
    H2 = IntrinsicSubspaceHomotopy(quadric.evaluator, V, W2; gamma = nothing)
    u2 = FSVec{ComplexF64}(zeros(ComplexF64, m))
    evaluate!(u2, H2, xu, complex(0.0))
    @test Vector(u) ≈ Vector(u2) atol = 1.0e-13
end

# γ belongs to the caller's start subspace, so retargeting leaves the start alone.
# The retarget test above uses `gamma = nothing`, where re-rotating is a no-op.
@testset "target_parameters! leaves the start subspace fixed" begin
    Random.seed!(29)
    V = rand_subspace(3; dim = 1)
    W = rand_subspace(3; dim = 1)
    targets = [rand_subspace(3; dim = 1) for _ in 1:3]
    g = cis(2π * 0.37)

    @testset "$name" for (name, Homotopy) in (
            ("intrinsic", IntrinsicSubspaceHomotopy),
            ("extrinsic", ExtrinsicSubspaceHomotopy),
        )
        H = Homotopy(quadric.evaluator, V, W; gamma = g)
        start0 = extrinsic(H.start)
        for q in targets
            target_parameters!(H, q)
            @test extrinsic(H.start).A == start0.A
            @test extrinsic(H.start).b == start0.b
        end

        # Retargeting is equivalent to building the homotopy for that target.
        H = Homotopy(quadric.evaluator, V, W; gamma = g)
        for q in targets
            target_parameters!(H, q)
        end
        direct = Homotopy(quadric.evaluator, V, targets[end]; gamma = g)
        m, n = size(H)
        xu = FSVec{ComplexF64}(randn(ComplexF64, n))
        for t in (complex(0.0), complex(0.4, 0.2), complex(1.0))
            u = FSVec{ComplexF64}(zeros(ComplexF64, m))
            ud = FSVec{ComplexF64}(zeros(ComplexF64, m))
            evaluate!(u, H, xu, t)
            evaluate!(ud, direct, xu, t)
            @test Vector(u) ≈ Vector(ud) atol = 1.0e-12
        end
    end
end

# K-th Taylor coefficient of t ↦ H(x, t) at t0 for a constant path x, via a
# Cauchy integral on a circle of radius r (trapezoid rule = FFT, exact up to
# aliasing since H is analytic in t).
function taylor_oracle(H, x::Vector{ComplexF64}, t0::ComplexF64, K::Int; r = 0.1, n = 64)
    m = size(H)[1]
    u = FSVec{ComplexF64}(zeros(ComplexF64, m))
    xf = FSVec{ComplexF64}(x)
    vals = Matrix{ComplexF64}(undef, m, n)
    for k in 0:(n - 1)
        evaluate!(u, H, xf, t0 + r * cis(2π * k / n))
        vals[:, k + 1] .= Vector(u)
    end
    return [
        sum(vals[i, k + 1] * cis(-2π * k * K / n) for k in 0:(n - 1)) / (n * r^K)
            for i in 1:m
    ]
end

# taylor! with a constant path: coefficient rows 2:K+1 are zero.
function taylor_constant_path(H, ::Val{K}, x::Vector{ComplexF64}, t0::ComplexF64) where {K}
    m, n = size(H)
    u = FSVec{ComplexF64}(zeros(ComplexF64, m))
    tv = TaylorVector{K + 1, ComplexF64}(n)
    tv.data[1, :] .= x
    taylor!(u, Val(K), H, tv, t0)
    return Vector(u)
end

@testset "ExtrinsicSubspaceHomotopy against explicit formula" begin
    Random.seed!(41)
    @polyvar x4[1:4]
    f1 = sum(randn(ComplexF64) * x4[i] * x4[j] for i in 1:4 for j in i:4) +
        sum(randn(ComplexF64) .* x4) + randn(ComplexF64)
    f2 = sum(randn(ComplexF64) * x4[i] * x4[j] for i in 1:4 for j in i:4) +
        sum(randn(ComplexF64) .* x4) + randn(ComplexF64)
    F4 = System([f1, f2]; variables = x4)
    A = rand_subspace(4; codim = 2)
    B = rand_subspace(4; codim = 2)
    H = ExtrinsicSubspaceHomotopy(F4, A, B; gamma = nothing)
    @test size(H) == (4, 4)

    Q, Q_cos, Θ = H.path.Q, H.path.Q_cos, H.path.Θ
    γ_at(t) = Q_cos .* transpose(cos.(t .* Θ)) .+ Q .* transpose(sin.(t .* Θ))

    xv = randn(ComplexF64, 4)
    xf = FSVec{ComplexF64}(xv)
    u = FSVec{ComplexF64}(zeros(ComplexF64, 4))
    for t in (complex(0.3), 0.8 - 0.4im)
        evaluate!(u, H, xf, ComplexF64(t))
        expected = [
            [ComplexF64(f(x4 => xv)) for f in F4.polys]
            transpose(γ_at(t)) * xv .- (t .* H.a0 .+ (1 .- t) .* H.b0)
        ]
        @test Vector(u) ≈ expected rtol = 1.0e-12
    end

    # jacobian: [J_F(x); transpose(γ(t))]
    t = 0.7 + 0.2im
    U = FSMat{ComplexF64}(zeros(ComplexF64, 4, 4))
    evaluate_and_jacobian!(u, U, H, xf, ComplexF64(t))
    JF = [ComplexF64(differentiate(f, xj)(x4 => xv)) for f in F4.polys, xj in x4]
    @test Matrix(U) ≈ [JF; transpose(γ_at(t))] rtol = 1.0e-12

    # Taylor orders 1:3 against the Cauchy-integral oracle
    t0 = complex(0.42)
    tay1 = FSVec{ComplexF64}(zeros(ComplexF64, 4))
    taylor!(tay1, Val(1), H, xf, t0)
    @test Vector(tay1) ≈ taylor_oracle(H, xv, t0, 1) rtol = 1.0e-9
    @test taylor_constant_path(H, Val(2), xv, t0) ≈ taylor_oracle(H, xv, t0, 2) rtol = 1.0e-8
    @test taylor_constant_path(H, Val(3), xv, t0) ≈ taylor_oracle(H, xv, t0, 3) rtol = 1.0e-7
end

@testset "IntrinsicSubspaceHomotopy against explicit formula" begin
    Random.seed!(42)
    V = rand_subspace(3; dim = 1)
    W = rand_subspace(3; dim = 1)
    H = IntrinsicSubspaceHomotopy(quadric.evaluator, V, W; gamma = nothing)

    Q, Q_cos, Θ = H.path.Q, H.path.Q_cos, H.path.Θ
    γ_at(t) = Q_cos .* transpose(cos.(t .* Θ)) .+ Q .* transpose(sin.(t .* Θ))
    # H.offset is a t-cache mutated by evaluate!; capture the t-independent data
    b_target = copy(intrinsic(H.target).b)
    a_minus_b = copy(H.a_minus_b)
    ambient(v, t) = γ_at(t) * v .+ b_target .+ t .* a_minus_b

    v0 = randn(ComplexF64, 1)
    u = FSVec{ComplexF64}(zeros(ComplexF64, 1))
    for t in (complex(0.25), 0.6 + 0.3im)
        evaluate!(u, H, FSVec{ComplexF64}(v0), ComplexF64(t))
        xa = ambient(v0, t)
        expected = xa[1]^2 + 2xa[2]^2 + 3xa[3]^2 + xa[1] * xa[2] - 1
        @test u[1] ≈ expected rtol = 1.0e-12
    end

    # endpoints of the geodesic: ambient point lies on V at t=1 and on W at t=0
    for (t, L) in ((1.0, V), (0.0, W))
        xa = ambient(v0, complex(t))
        @test norm(extrinsic(L).A * xa - extrinsic(L).b) < 1.0e-10
    end

    # Taylor orders 2:3 against the Cauchy-integral oracle (order 1 is covered
    # by the finite-difference testset above)
    t0 = complex(0.37)
    v1 = randn(ComplexF64, 1)
    @test taylor_constant_path(H, Val(2), v1, t0) ≈ taylor_oracle(H, v1, t0, 2) rtol = 1.0e-8
    @test taylor_constant_path(H, Val(3), v1, t0) ≈ taylor_oracle(H, v1, t0, 3) rtol = 1.0e-7
end

using HomotopyContinuationNext: EndgameTracker, EndgameOptions, EndgameCode, PathResult,
    is_success, solution

@testset "subspace homotopies between perpendicular spaces" begin
    @polyvar x y z
    p = (x * y - x^2) + 1 - z
    q = x^4 + x^2 - y - 1
    f = [
        p * q * (x - 3) * (x - 5),
        p * q * (y - 3) * (y - 5),
        p * (z - 3) * (z - 5),
    ]
    F = System(f; variables = [x, y, z])

    L1 = LinearSubspace(reshape([1; 0; 0.0], 1, 3), [1.0])   # {x = 1}
    L2 = LinearSubspace(reshape([0; 1; 0.0], 1, 3), [1.0])   # {y = 1}

    # (1,1,3) is a regular witness point of F ∩ L1: q(1,1) = 0 kills f₁ and f₂,
    # z = 3 kills f₃, and the sliced Jacobian there is nonsingular. (The point
    # (1,1,1) on {p = q = 0} is NOT usable: both factors vanish, so ∇f₁ = ∇f₂ = 0
    # and every homotopy would start at a singular point.)
    start = ComplexF64[1.0, 1.0, 3.0]
    @test all(fi -> abs(ComplexF64(fi((x, y, z) => (1.0, 1.0, 3.0)))) < 1.0e-14, f)

    H1 = ExtrinsicSubspaceHomotopy(F, L1, L2)
    H2 = IntrinsicSubspaceHomotopy(F, L1, L2)

    T1 = EndgameTracker(Tracker(HomotopyEvaluator(H1)))
    code1 = track!(T1, start)
    @test code1 == EndgameCode.SUCCESS
    res1 = PathResult(T1)
    @test is_success(res1)

    u_start = zeros(ComplexF64, size(H2)[2])
    intrinsic_coordinates!(u_start, H2, start, complex(1.0))
    T2 = EndgameTracker(Tracker(HomotopyEvaluator(H2)))
    code2 = track!(T2, u_start)
    @test code2 == EndgameCode.SUCCESS
    res2 = PathResult(T2)
    @test is_success(res2)

    # both endpoints lie on the variety and on L2
    for (H, res) in ((H1, res1), (H2, res2))
        xe = zeros(ComplexF64, 3)
        if H === H2
            ambient_coordinates!(xe, H, solution(res), complex(0.0))
        else
            xe .= solution(res)
        end
        residual = maximum(
            fi -> abs(ComplexF64(fi((x, y, z) => (xe[1], xe[2], xe[3])))), f,
        )
        @test residual < 1.0e-6
        @test norm(extrinsic(L2).A * xe - extrinsic(L2).b) < 1.0e-8
    end
end

using HomotopyContinuationNext: AffineChartHomotopy, AffineChartSystem, on_affine_chart,
    on_chart!, linear_subspace_homotopy, SystemEvaluator

@testset "affine chart" begin
    Random.seed!(14)
    @polyvar w[1:3]
    # homogeneous quadric in P^2; its witness slice is dim 2 > codim 1, so the
    # dispatch takes the extrinsic branch and chart-wraps the homotopy.
    Fh = System([w[1]^2 + w[2]^2 - w[3]^2]; variables = w)
    V = rand_subspace(3; dim = 2, affine = false)
    W = rand_subspace(3; dim = 2, affine = false)
    H = linear_subspace_homotopy(Fh, V, W)
    @test H isa AffineChartHomotopy

    x = randn(ComplexF64, 3)
    on_chart!(x, H)
    # after normalization the chart row is satisfied: v'x == 1
    @test isapprox(sum(H.chart .* x), 1.0 + 0im; atol = 1.0e-12)

    # intrinsic + homogeneous: chart row goes on the system instead
    Vp = rand_subspace(3; dim = 1, affine = false)
    Wp = rand_subspace(3; dim = 1, affine = false)
    Hp = linear_subspace_homotopy(Fh, Vp, Wp)
    @test Hp isa IntrinsicSubspaceHomotopy
    @test size(Hp) == (2, 1)   # system row + chart row, one intrinsic coordinate
end

@testset "linear_subspace_homotopy dispatch" begin
    Random.seed!(15)
    V = rand_subspace(3; dim = 1)
    W = rand_subspace(3; dim = 1)
    H = linear_subspace_homotopy(quadric, V, W)          # affine system
    @test H isa IntrinsicSubspaceHomotopy                # dim 1 <= codim 2
    He = linear_subspace_homotopy(quadric, V, W; intrinsic = false)
    @test He isa ExtrinsicSubspaceHomotopy
end
