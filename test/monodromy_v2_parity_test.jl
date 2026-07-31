using Test, Random
using LinearAlgebra
using HomotopyContinuationNext
using HomotopyContinuationNext: find_start_pair, permutations,
    is_heuristic_stop, verify_solution_completeness, parameters, trace,
    SymmetricGroup, multiplicities, InfNorm
using DynamicPolynomials: @polyvar, subs, differentiate, monomials, coefficient
using HomotopyContinuationNext: FSVec, FSMat, evaluate!, evaluate_and_jacobian!,
    nparameters

# The ED-discriminant system of a toric variety: 21 solutions for the twisted
# cubic exponent matrix used below.
function toric_ed_system()
    A = [3 2 1 0; 0 1 2 3]
    d, n = size(A)
    @polyvar tv[1:d] yv[1:n] uv[1:n]
    φ = [prod(tv[i]^A[i, j] for i in 1:d) for j in 1:n]
    Dφ = [differentiate(φ[j], tv[i]) for j in 1:n, i in 1:d]
    F = System(
        [φ .+ yv .- uv; transpose(Dφ) * yv];
        variables = [tv; yv], parameters = uv,
    )
    return F, uv
end

function rand_poly(vars, d::Int; homogeneous::Bool = false)
    mons = monomials(vars, homogeneous ? (d:d) : 0:d)
    return sum(randn(ComplexF64) * m for m in mons)
end

@testset "toric ED: Monodromy options" begin
    F, uv = toric_ed_system()

    # out of the box: heuristic stop after finding all 21 solutions
    r0 = solve(F, Monodromy(; seed = UInt32(0x008b8683), show_progress = false), Serial())
    @test is_heuristic_stop(r0)
    @test nsolutions(r0) == 21

    r = solve(
        F,
        Monodromy(;
            target_solutions_count = 21, max_loops_no_progress = 50,
            seed = UInt32(0x008b8684), show_progress = false,
        ),
        Serial(),
    )
    @test is_success(r)
    @test nsolutions(r) == 21
    @test isempty(multiplicities(solutions(r)))
    @test isempty(sprint(show, r)) == false

    # seed reproducibility: identical loop counts
    r2 = solve(
        F,
        Monodromy(;
            target_solutions_count = 21, max_loops_no_progress = 50, seed = r.seed,
            show_progress = false,
        ),
        Serial(),
    )
    @test r2.statistics.tracked_loops[] == r.statistics.tracked_loops[]

    # threading
    rt = solve(
        F,
        Monodromy(;
            target_solutions_count = 21, max_loops_no_progress = 50,
            seed = UInt32(0x008b8685), show_progress = false,
        ),
        Threaded(),
    )
    @test is_success(rt)
    @test nsolutions(rt) == 21

    # timeout stops the run before all solutions are found
    r_timeout = solve(
        F,
        Monodromy(;
            target_solutions_count = 21, timeout = 1.0e-12, seed = UInt32(0x008b8686),
            show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(r_timeout) < 21
end

@testset "toric ED: start pairs, distances, dedup options" begin
    F, uv = toric_ed_system()
    Random.seed!(0x008b8687)
    x₀, p₀ = find_start_pair(F)

    # input of length > 1 (duplicated start solutions)
    r = solve(
        F,
        [x₀ for _ in 1:30],
        p₀,
        Monodromy(;
            target_solutions_count = 21, max_loops_no_progress = 50,
            seed = UInt32(0x008b8688), show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(r) == 21

    # raw polynomial input with explicit parameters
    r = solve(
        System(collect(F.polys); parameters = collect(uv)),
        [x₀],
        p₀,
        Monodromy(;
            target_solutions_count = 21, max_loops_no_progress = 50,
            seed = UInt32(0x008b8689), show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(r) == 21

    # degenerate distance: everything is identified
    r = solve(
        F,
        [x₀],
        p₀,
        Monodromy(;
            distance = (x, y) -> 0.0, seed = UInt32(0x008b868a), show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(r) == 1

    # distance without the triangle inequality (squared Euclidean)
    r = solve(
        F,
        [x₀],
        p₀,
        Monodromy(;
            distance = (x, y) -> norm(x - y, 2)^2, target_solutions_count = 21,
            max_loops_no_progress = 50, seed = UInt32(0x008b868b), show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(r) == 21

    # explicit triangle_inequality choices
    for ti in (false, true)
        r = solve(
            F,
            [x₀],
            p₀,
            Monodromy(;
                target_solutions_count = 21, max_loops_no_progress = 50,
                seed = UInt32(0x008b868c), show_progress = false,
                triangle_inequality = ti,
            ),
            Serial(),
        )
        @test nsolutions(r) == 21
    end

    # heuristic stop without a target count
    r = solve(
        F,
        [x₀],
        p₀,
        Monodromy(; seed = UInt32(0x008b868d), show_progress = false),
        Serial(),
    )
    @test is_heuristic_stop(r)
end

@testset "toric ED: group action and reuse_loops" begin
    F, uv = toric_ed_system()
    Random.seed!(0x008b868e)
    x₀, p₀ = find_start_pair(F)

    roots_of_unity(s) = begin
        t = cis(π * 2 / 3)
        t² = t * t
        (vcat(t * s[1], t * s[2], s[3:end]), vcat(t² * s[1], t² * s[2], s[3:end]))
    end

    # group action without equivalence classes still finds all 21
    r = solve(
        F,
        [x₀],
        p₀,
        Monodromy(;
            target_solutions_count = 21, max_loops_no_progress = 100,
            equivalence_classes = false, group_action = roots_of_unity,
            seed = UInt32(0x008b868f), show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(r) == 21

    # equivalence classes: 21 solutions collapse into 7 orbits
    r = solve(
        F,
        [x₀],
        p₀,
        Monodromy(;
            equivalence_classes = true, target_solutions_count = 7,
            max_loops_no_progress = 50, group_actions = roots_of_unity,
            seed = UInt32(0x008b8690), show_progress = false,
        ),
        Serial(),
    )
    @test nresults(r) == 7

    # equivalence classes are on by default when a group action is given
    r = solve(
        F,
        [x₀],
        p₀,
        Monodromy(;
            group_action = roots_of_unity, max_loops_no_progress = 50,
            seed = UInt32(0x008b8691), show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(r) == 7

    for rl in (:all, :random, :none)
        r = solve(
            F,
            [x₀],
            p₀,
            Monodromy(;
                group_action = roots_of_unity, target_solutions_count = 7, reuse_loops = rl,
                max_loops_no_progress = 200, seed = UInt32(0x008b8692),
                show_progress = false,
            ),
            Serial(),
        )
        @test nresults(r) == 7
    end
end

# Gaussian mixture model of 3 univariate Gaussians, matched to the first 9
# moments. 225 solutions, with an S3 relabeling symmetry.
function moments3_system()
    @polyvar a[1:3] x[1:3] s[1:3] m[1:9]

    f0 = a[1] + a[2] + a[3]
    f1 = a[1] * x[1] + a[2] * x[2] + a[3] * x[3]
    f2 = a[1] * (x[1]^2 + s[1]) + a[2] * (x[2]^2 + s[2]) + a[3] * (x[3]^2 + s[3])
    f3 =
        a[1] * (x[1]^3 + 3 * s[1] * x[1]) +
        a[2] * (x[2]^3 + 3 * s[2] * x[2]) +
        a[3] * (x[3]^3 + 3 * s[3] * x[3])
    f4 =
        a[1] * (x[1]^4 + 6 * s[1] * x[1]^2 + 3 * s[1]^2) +
        a[2] * (x[2]^4 + 6 * s[2] * x[2]^2 + 3 * s[2]^2) +
        a[3] * (x[3]^4 + 6 * s[3] * x[3]^2 + 3 * s[3]^2)
    f5 =
        a[1] * (x[1]^5 + 10 * s[1] * x[1]^3 + 15 * x[1] * s[1]^2) +
        a[2] * (x[2]^5 + 10 * s[2] * x[2]^3 + 15 * x[2] * s[2]^2) +
        a[3] * (x[3]^5 + 10 * s[3] * x[3]^3 + 15 * x[3] * s[3]^2)
    f6 =
        a[1] * (x[1]^6 + 15 * s[1] * x[1]^4 + 45 * x[1]^2 * s[1]^2 + 15 * s[1]^3) +
        a[2] * (x[2]^6 + 15 * s[2] * x[2]^4 + 45 * x[2]^2 * s[2]^2 + 15 * s[2]^3) +
        a[3] * (x[3]^6 + 15 * s[3] * x[3]^4 + 45 * x[3]^2 * s[3]^2 + 15 * s[3]^3)
    f7 =
        a[1] * (x[1]^7 + 21 * s[1] * x[1]^5 + 105 * x[1]^3 * s[1]^2 + 105 * x[1] * s[1]^3) +
        a[2] * (x[2]^7 + 21 * s[2] * x[2]^5 + 105 * x[2]^3 * s[2]^2 + 105 * x[2] * s[2]^3) +
        a[3] * (x[3]^7 + 21 * s[3] * x[3]^5 + 105 * x[3]^3 * s[3]^2 + 105 * x[3] * s[3]^3)
    f8 =
        a[1] * (
        x[1]^8 + 28 * s[1] * x[1]^6 + 210 * x[1]^4 * s[1]^2 +
            420 * x[1]^2 * s[1]^3 + 105 * s[1]^4
    ) +
        a[2] * (
        x[2]^8 + 28 * s[2] * x[2]^6 + 210 * x[2]^4 * s[2]^2 +
            420 * x[2]^2 * s[2]^3 + 105 * s[2]^4
    ) +
        a[3] * (
        x[3]^8 + 28 * s[3] * x[3]^6 + 210 * x[3]^4 * s[3]^2 +
            420 * x[3]^2 * s[3]^3 + 105 * s[3]^4
    )
    return System(
        [f0, f1, f2, f3, f4, f5, f6, f7, f8] .- m;
        variables = [a; x; s], parameters = m,
    )
end

@testset "method of moments" begin
    f = moments3_system()
    S₃ = SymmetricGroup(3)
    relabeling(v) = map(p -> [v[1:3][p]..., v[4:6][p]..., v[7:9][p]...], S₃)

    R = solve(
        f,
        Monodromy(;
            group_action = relabeling, show_progress = false, max_loops_no_progress = 1,
            seed = UInt32(0x6d31),
        ),
        Serial(),
    )
    @test nsolutions(R) ≤ 225

    for threading in (true, false)
        R = solve(
            f,
            Monodromy(;
                group_action = relabeling, show_progress = false,
                max_loops_no_progress = 20, target_solutions_count = 225,
                seed = UInt32(0x6d32),
            ),
            Threaded(),
        )
        @test nsolutions(R) == 225
    end
end

@testset "projective + group actions" begin
    @polyvar a[1:2] x[1:2] s[1:2] z m[1:6]

    f0 = a[1] + a[2]
    f1 = a[1] * x[1] + a[2] * x[2]
    f2 = a[1] * (x[1]^2 + s[1] * z) + a[2] * (x[2]^2 + s[2] * z)
    f3 = a[1] * (x[1]^3 + 3 * s[1] * x[1] * z) + a[2] * (x[2]^3 + 3 * s[2] * x[2] * z)
    f4 =
        a[1] * (x[1]^4 + 6 * s[1] * x[1]^2 * z + 3 * s[1]^2 * z^2) +
        a[2] * (x[2]^4 + 6 * s[2] * x[2]^2 * z + 3 * s[2]^2 * z^2)
    f5 =
        a[1] * (x[1]^5 + 10 * s[1] * x[1]^3 * z + 15 * x[1] * s[1]^2 * z^2) +
        a[2] * (x[2]^5 + 10 * s[2] * x[2]^3 * z + 15 * x[2] * s[2]^2 * z^2)

    M2 = System(
        [f0, f1, f2, f3, f4, f5] .- m .* z .^ (1:6);
        variables = [a; x; s; z], parameters = m,
    )

    relabeling = let S₂ = SymmetricGroup(2)
        v -> map(p -> [v[1:2][p]..., v[3:4][p]..., v[5:6][p]..., v[7]], S₂)
    end

    R = solve(
        M2,
        Monodromy(;
            group_action = relabeling, show_progress = false, max_loops_no_progress = 10,
            seed = UInt32(0x9a01),
        ),
        Serial(),
    )
    @test nsolutions(R) == 9
end

@testset "permutations (circle pair)" begin
    @polyvar w[1:2] a b c
    c₁ = (w[1] - 2)^2 + w[2]^2 - 1
    c₂ = (w[1] + 2)^2 + w[2]^2 - 1

    F = System([c₁ * c₂, a * w[1] + b * w[2] - c]; variables = w, parameters = [a, b, c])
    S = solve(
        F,
        [[1.0, 0.0]],
        [1, 1, 1],
        Monodromy(;
            permutations = true, max_loops_no_progress = 20, seed = UInt32(0xbe01),
            show_progress = false,
        ),
        Serial(),
    )
    A = permutations(S)
    B = permutations(S; reduced = false)

    @test size(A) == (2, 2)
    @test A == [1 2; 2 1] || A == [2 1; 1 2]
    @test size(B, 1) == 2
    @test size(B, 2) > 2

    # all 4 solutions of the cut system as start solutions
    F₁ = System(
        [subs(f, a => 1, b => 1, c => 1) for f in [c₁ * c₂, a * w[1] + b * w[2] - c]];
        variables = w,
    )
    Random.seed!(0xbe02)
    start_res = solve(F₁, TotalDegree(; show_progress = false))
    @test nsolutions(start_res) == 4
    S4 = solve(
        F,
        solutions(start_res),
        [1, 1, 1],
        Monodromy(; permutations = true, seed = UInt32(0xbe03), show_progress = false),
        Serial(),
    )
    C = permutations(S4)
    @test size(C, 1) == 4
end

@testset "linear subspaces (dim/codim, trace test)" begin
    Random.seed!(0x00011e01)
    @polyvar x[1:4]
    f1 = rand_poly(x, 6)
    F = System([f1]; variables = x)
    res = solve(
        F,
        Monodromy(; dim = 3, seed = UInt32(0x00011e02), show_progress = false),
        Serial(),
    )
    @test nsolutions(res) == 6
    @test trace(res) < 1.0e-6
    @test is_success(res)

    res = solve(
        F,
        Monodromy(;
            dim = 3, trace_test = false, seed = UInt32(0x00011e03), show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(res) == 6
    @test is_heuristic_stop(res)

    res = solve(
        F,
        Monodromy(;
            dim = 3, trace_test = true, trace_test_tol = 1.0e-50, seed = UInt32(0x00011e04),
            show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(res) == 6
    @test is_heuristic_stop(res)

    # homogeneous hypersurface in P^3
    f1h = rand_poly(x, 6; homogeneous = true)
    Fh = System([f1h]; variables = x)
    res = solve(
        Fh,
        Monodromy(; dim = 2, seed = UInt32(0x00011e05), show_progress = false),
        Serial(),
    )
    @test nsolutions(res) == 6
    @test trace(res) < 1.0e-6
    @test is_success(res)

    # curve of degree 6*3*4 = 72 in C^4
    Fc = System([f1, rand_poly(x, 3), rand_poly(x, 4)]; variables = x)
    res = solve(
        Fc,
        Monodromy(; codim = 3, seed = UInt32(0x00011e06), show_progress = false),
        Serial(),
    )
    @test nsolutions(res) == 72
    @test trace(res) < 1.0e-6
    @test is_success(res)
end

@testset "parameter homotopy from a monodromy result" begin
    F, uv = toric_ed_system()
    mres = solve(
        F,
        Monodromy(;
            target_solutions_count = 21, max_loops_no_progress = 20, seed = UInt32(0xf001),
            show_progress = false,
        ),
        Serial(),
    )
    Random.seed!(0xf002)
    r = solve(F, mres, randn(ComplexF64, 4), Continuation(; show_progress = false))
    @test nsolutions(r) == 21
end

@testset "verify_solution_completeness (circle + line)" begin
    @polyvar x y a b c
    f = x^2 + y^2 - 1
    l = a * x + b * y + c
    sys = System([f, l]; variables = [x, y], parameters = [a, b, c])
    res = solve(
        sys,
        [[-0.6 - 0.8im, -1.2 + 0.4im]],
        [1.0 + 0im, 2.0 + 0im, 3.0 + 0im],
        [1.0 + 0im, 2.0 + 0im, 3.0 + 0im],
        Continuation(; seed = UInt32(0xce01), show_progress = false),
    )
    @test nsolutions(res) == 1
    # complete the witness set by monodromy, then verify
    mres = solve(
        sys,
        solutions(res),
        [1.0 + 0im, 2.0 + 0im, 3.0 + 0im],
        Monodromy(;
            target_solutions_count = 2, seed = UInt32(0xce02), show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(mres) == 2
    sols = solutions(mres)
    p123 = [1.0 + 0im, 2.0 + 0im, 3.0 + 0im]
    @test verify_solution_completeness(sys, sols, p123, Monodromy(; show_progress = false)) === true
    @test verify_solution_completeness(sys, sols[1:1], p123, Monodromy(; show_progress = false)) in (false, nothing)
    # an impossibly strict trace tolerance rejects even the full set
    @test verify_solution_completeness(
        sys,
        sols,
        p123,
        Monodromy(; show_progress = false);
        trace_tol = 1.0e-60
    ) in (false, nothing)
end

@testset "unique_points tolerances" begin
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)
    # an absurdly large rtol identifies the two solutions with each other
    r = solve(
        F,
        Monodromy(;
            unique_points_rtol = 10.0, max_loops_no_progress = 3, seed = UInt32(0xd001),
            show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(r) == 1
end

# The symmetroid family of https://www.juliahomotopycontinuation.org/examples/symmetroids/:
# `f` sends a pencil of symmetric matrices to the coefficients of its
# determinant, `L₁` parameterizes the fiber directions and `L₂` cuts the image
# with a hyperplane whose normal is the parameter.
@testset "symmetroids: composition, custom distance, unique_points tolerances" begin
    Random.seed!(0x5717)
    n = 4
    d = 3
    M = binomial(d + 1, 2)
    D = (n - 1) * M + 3
    N = binomial(n - 1 + d, d)

    @polyvar xs[0:(n - 1)] as[1:D]
    blocks = map(0:(n - 2)) do ℓ
        columns = map(1:d) do i
            k = ℓ * M + sum(d - j for j in 0:(i - 1))
            return [zeros(Int, i - 1); as[(k - (d - i)):k]]
        end
        B = hcat(columns...)
        return (B + transpose(B)) ./ 2
    end
    A₀ = [as[D - 2] 0 0; 0 as[D - 1] 0; 0 0 as[D]]
    μ = xs[1] .* A₀ + sum(xs[i + 1] .* blocks[i] for i in 1:(n - 1))
    detμ = det(μ)
    f = System([coefficient(detμ, m, xs) for m in monomials(xs, d)]; variables = as)
    @test size(f) == (N, D)

    evaluate_at(F, x, p = ComplexF64[]) = begin
        u = FSVec{ComplexF64}(zeros(ComplexF64, size(F)[1]))
        evaluate!(
            u, F.evaluator, FSVec{ComplexF64}(collect(ComplexF64, x)),
            FSVec{ComplexF64}(collect(ComplexF64, p))
        )
        collect(u)
    end

    u₀ = FSVec{ComplexF64}(zeros(ComplexF64, N))
    J₀ = FSMat{ComplexF64}(zeros(ComplexF64, N, D))
    evaluate_and_jacobian!(
        u₀, J₀, f.evaluator, FSVec{ComplexF64}(randn(ComplexF64, D)),
        FSVec{ComplexF64}(ComplexF64[]),
    )
    dimQ = rank(collect(J₀))
    @test dimQ == 16

    Q = Matrix(qr(randn(ComplexF64, D, D)).Q)
    @var bs[1:dimQ] ks[1:(N + 1)] f₀[1:N]
    L₁ = System(Q[:, 1:dimQ] * collect(bs) + Q[:, dimQ + 1]; variables = collect(bs))

    b₁ = randn(ComplexF64, dimQ)
    f₁ = evaluate_at(f, evaluate_at(L₁, b₁))
    R₁ = Matrix(qr(randn(ComplexF64, N, N)).Q)
    R = [transpose(collect(ks[1:N])); R₁[1:(dimQ - 1), :]]
    r = [ks[N + 1]; R₁[1:(dimQ - 1), :] * f₁]
    L₂ = System(R * collect(f₀) - r; variables = collect(f₀), parameters = collect(ks))

    C = L₂ ∘ f ∘ L₁
    @test size(C) == (dimQ, dimQ)
    @test nparameters(C) == N + 1

    p₁ = randn(ComplexF64, N)
    params = [p₁; transpose(p₁) * f₁]
    @test maximum(abs, evaluate_at(C, b₁, params)) < 1.0e-10

    # the fibers of `f ∘ L₁` are positive-dimensional, so points sharing an
    # image are the same solution
    fL₁ = f ∘ L₁
    distance(x, y) = maximum(abs, evaluate_at(fL₁, x) .- evaluate_at(fL₁, y))

    points = solve(
        C,
        [b₁],
        params,
        Monodromy(;
            distance = distance, unique_points_rtol = 1.0e-8, unique_points_atol = 1.0e-14,
            target_solutions_count = 305, max_loops_no_progress = 20, show_progress = false,
            seed = UInt32(0x5717),
        ),
        Serial(),
    )
    @test is_success(points)
    @test nsolutions(points) == 305
    @test length(
        unique_points(
            solutions(points); distance = distance, rtol = 1.0e-8, atol = 1.0e-14,
        ),
    ) == 305
end
