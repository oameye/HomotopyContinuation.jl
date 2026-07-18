using Test, Random
using LinearAlgebra
using HomotopyContinuationNext
using HomotopyContinuationNext: find_start_pair, monodromy_solve, permutations,
    is_heuristic_stop, verify_solution_completeness, parameters, trace,
    SymmetricGroup, multiplicities, InfNorm
using DynamicPolynomials: @polyvar, subs, differentiate, monomials

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

@testset "toric ED: monodromy_solve options" begin
    F, uv = toric_ed_system()

    # out of the box: heuristic stop after finding all 21 solutions
    r0 = monodromy_solve(F; seed = UInt32(0x008b8683), threading = false, show_progress = false)
    @test is_heuristic_stop(r0)
    @test nsolutions(r0) == 21

    r = monodromy_solve(
        F; target_solutions_count = 21, max_loops_no_progress = 50,
        threading = false, seed = UInt32(0x008b8684), show_progress = false,
    )
    @test is_success(r)
    @test nsolutions(r) == 21
    @test isempty(multiplicities(solutions(r)))
    @test isempty(sprint(show, r)) == false

    # seed reproducibility: identical loop counts
    r2 = monodromy_solve(
        F; target_solutions_count = 21, max_loops_no_progress = 50,
        threading = false, seed = r.seed, show_progress = false,
    )
    @test r2.statistics.tracked_loops[] == r.statistics.tracked_loops[]

    # threading
    rt = monodromy_solve(
        F; target_solutions_count = 21, max_loops_no_progress = 50,
        threading = true, seed = UInt32(0x008b8685), show_progress = false,
    )
    @test is_success(rt)
    @test nsolutions(rt) == 21

    # timeout stops the run before all solutions are found
    r_timeout = monodromy_solve(
        F; target_solutions_count = 21, timeout = 1.0e-12,
        seed = UInt32(0x008b8686), threading = false, show_progress = false,
    )
    @test nsolutions(r_timeout) < 21
end

@testset "toric ED: start pairs, distances, dedup options" begin
    F, uv = toric_ed_system()
    Random.seed!(0x008b8687)
    x₀, p₀ = find_start_pair(F)

    # input of length > 1 (duplicated start solutions)
    r = monodromy_solve(
        F, [x₀ for _ in 1:30], p₀;
        target_solutions_count = 21, max_loops_no_progress = 50,
        seed = UInt32(0x008b8688), threading = false, show_progress = false,
    )
    @test nsolutions(r) == 21

    # raw polynomial input with explicit parameters
    r = monodromy_solve(
        collect(F.polys), [x₀], p₀;
        parameters = collect(uv),
        target_solutions_count = 21, max_loops_no_progress = 50,
        seed = UInt32(0x008b8689), threading = false, show_progress = false,
    )
    @test nsolutions(r) == 21

    # degenerate distance: everything is identified
    r = monodromy_solve(
        F, [x₀], p₀; distance = (x, y) -> 0.0,
        seed = UInt32(0x008b868a), threading = false, show_progress = false,
    )
    @test nsolutions(r) == 1

    # distance without the triangle inequality (squared Euclidean)
    r = monodromy_solve(
        F, [x₀], p₀; distance = (x, y) -> norm(x - y, 2)^2,
        target_solutions_count = 21, max_loops_no_progress = 50,
        seed = UInt32(0x008b868b), threading = false, show_progress = false,
    )
    @test nsolutions(r) == 21

    # explicit triangle_inequality choices
    for ti in (false, true)
        r = monodromy_solve(
            F, [x₀], p₀; triangle_inequality = ti,
            target_solutions_count = 21, max_loops_no_progress = 50,
            seed = UInt32(0x008b868c), threading = false, show_progress = false,
        )
        @test nsolutions(r) == 21
    end

    # heuristic stop without a target count
    r = monodromy_solve(
        F, [x₀], p₀;
        seed = UInt32(0x008b868d), threading = false, show_progress = false,
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
    r = monodromy_solve(
        F, [x₀], p₀;
        target_solutions_count = 21, max_loops_no_progress = 100,
        equivalence_classes = false, group_action = roots_of_unity,
        seed = UInt32(0x008b868f), threading = false, show_progress = false,
    )
    @test nsolutions(r) == 21

    # equivalence classes: 21 solutions collapse into 7 orbits
    r = monodromy_solve(
        F, [x₀], p₀;
        equivalence_classes = true, target_solutions_count = 7,
        max_loops_no_progress = 50, group_actions = roots_of_unity,
        seed = UInt32(0x008b8690), threading = false, show_progress = false,
    )
    @test nresults(r) == 7

    # equivalence classes are on by default when a group action is given
    r = monodromy_solve(
        F, [x₀], p₀;
        group_action = roots_of_unity, max_loops_no_progress = 50,
        seed = UInt32(0x008b8691), threading = false, show_progress = false,
    )
    @test nsolutions(r) == 7

    for rl in (:all, :random, :none)
        r = monodromy_solve(
            F, [x₀], p₀;
            group_action = roots_of_unity, target_solutions_count = 7,
            reuse_loops = rl, max_loops_no_progress = 200,
            seed = UInt32(0x008b8692), threading = false, show_progress = false,
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

    R = monodromy_solve(
        f; group_action = relabeling, show_progress = false,
        max_loops_no_progress = 1, seed = UInt32(0x6d31), threading = false,
    )
    @test nsolutions(R) ≤ 225

    for threading in (true, false)
        R = monodromy_solve(
            f; group_action = relabeling, show_progress = false,
            max_loops_no_progress = 20, target_solutions_count = 225,
            threading = threading, seed = UInt32(0x6d32),
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

    R = monodromy_solve(
        M2; group_action = relabeling, show_progress = false,
        threading = false, max_loops_no_progress = 10, seed = UInt32(0x9a01),
    )
    @test nsolutions(R) == 9
end

@testset "permutations (circle pair)" begin
    @polyvar w[1:2] a b c
    c₁ = (w[1] - 2)^2 + w[2]^2 - 1
    c₂ = (w[1] + 2)^2 + w[2]^2 - 1

    F = System([c₁ * c₂, a * w[1] + b * w[2] - c]; variables = w, parameters = [a, b, c])
    S = monodromy_solve(
        F, [[1.0, 0.0]], [1, 1, 1];
        permutations = true, max_loops_no_progress = 20,
        seed = UInt32(0xbe01), threading = false, show_progress = false,
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
    start_res = solve(F₁; show_progress = false)
    @test nsolutions(start_res) == 4
    S4 = monodromy_solve(
        F, solutions(start_res), [1, 1, 1];
        permutations = true, seed = UInt32(0xbe03),
        threading = false, show_progress = false,
    )
    C = permutations(S4)
    @test size(C, 1) == 4
end

@testset "linear subspaces (dim/codim, trace test)" begin
    Random.seed!(0x00011e01)
    @polyvar x[1:4]
    f1 = rand_poly(x, 6)
    F = System([f1]; variables = x)
    res = monodromy_solve(F; dim = 3, threading = false, seed = UInt32(0x00011e02), show_progress = false)
    @test nsolutions(res) == 6
    @test trace(res) < 1.0e-6
    @test is_success(res)

    res = monodromy_solve(
        F; dim = 3, trace_test = false, threading = false,
        seed = UInt32(0x00011e03), show_progress = false,
    )
    @test nsolutions(res) == 6
    @test is_heuristic_stop(res)

    res = monodromy_solve(
        F; dim = 3, trace_test = true, trace_test_tol = 1.0e-50,
        threading = false, seed = UInt32(0x00011e04), show_progress = false,
    )
    @test nsolutions(res) == 6
    @test is_heuristic_stop(res)

    # homogeneous hypersurface in P^3
    f1h = rand_poly(x, 6; homogeneous = true)
    Fh = System([f1h]; variables = x)
    res = monodromy_solve(Fh; dim = 2, seed = UInt32(0x00011e05), threading = false, show_progress = false)
    @test nsolutions(res) == 6
    @test trace(res) < 1.0e-6
    @test is_success(res)

    # curve of degree 6*3*4 = 72 in C^4
    Fc = System([f1, rand_poly(x, 3), rand_poly(x, 4)]; variables = x)
    res = monodromy_solve(Fc; codim = 3, seed = UInt32(0x00011e06), threading = false, show_progress = false)
    @test nsolutions(res) == 72
    @test trace(res) < 1.0e-6
    @test is_success(res)
end

@testset "parameter homotopy from a monodromy result" begin
    F, uv = toric_ed_system()
    mres = monodromy_solve(
        F; target_solutions_count = 21, max_loops_no_progress = 20,
        threading = false, seed = UInt32(0xf001), show_progress = false,
    )
    Random.seed!(0xf002)
    r = solve(F, mres; target_parameters = randn(ComplexF64, 4), show_progress = false)
    @test nsolutions(r) == 21
end

@testset "verify_solution_completeness (circle + line)" begin
    @polyvar x y a b c
    f = x^2 + y^2 - 1
    l = a * x + b * y + c
    sys = System([f, l]; variables = [x, y], parameters = [a, b, c])
    res = solve(
        sys, [[-0.6 - 0.8im, -1.2 + 0.4im]];
        start_parameters = [1.0 + 0im, 2.0 + 0im, 3.0 + 0im],
        target_parameters = [1.0 + 0im, 2.0 + 0im, 3.0 + 0im],
        seed = UInt32(0xce01), show_progress = false,
    )
    @test nsolutions(res) == 1
    # complete the witness set by monodromy, then verify
    mres = monodromy_solve(
        sys, solutions(res), [1.0 + 0im, 2.0 + 0im, 3.0 + 0im];
        target_solutions_count = 2, seed = UInt32(0xce02),
        threading = false, show_progress = false,
    )
    @test nsolutions(mres) == 2
    sols = solutions(mres)
    p123 = [1.0 + 0im, 2.0 + 0im, 3.0 + 0im]
    @test verify_solution_completeness(sys, sols, p123; show_progress = false) === true
    @test verify_solution_completeness(sys, sols[1:1], p123; show_progress = false) in (false, nothing)
    # an impossibly strict trace tolerance rejects even the full set
    @test verify_solution_completeness(
        sys, sols, p123; trace_tol = 1.0e-60, show_progress = false,
    ) in (false, nothing)
end

@testset "unique_points tolerances" begin
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)
    # an absurdly large rtol identifies the two solutions with each other
    r = monodromy_solve(
        F; unique_points_rtol = 10.0, max_loops_no_progress = 3,
        seed = UInt32(0xd001), threading = false, show_progress = false,
    )
    @test nsolutions(r) == 1
end
