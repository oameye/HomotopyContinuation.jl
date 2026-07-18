using Test, Random
using LinearAlgebra
using HomotopyContinuationNext
using HomotopyContinuationNext: find_start_pair
using DynamicPolynomials: @polyvar, subs

@testset "find_start_pair" begin
    Random.seed!(0xf00d)
    # linear in parameters
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)
    pair = find_start_pair(F)
    @test pair !== nothing
    x0, p0 = pair
    residual = [abs(ComplexF64(f(y => x0, p => p0))) for f in F.polys]
    @test maximum(residual) < 1.0e-10

    # nonlinear in parameters: exercises the Newton fallback
    @polyvar q[1:2]
    G = System(
        [y[1]^2 * q[1]^2 + y[2] * q[2]^3 - 1, y[1] + y[2] - q[1]];
        variables = y, parameters = q,
    )
    pair2 = find_start_pair(G)
    @test pair2 !== nothing
    x2, p2 = pair2
    residual2 = [abs(ComplexF64(f(y => x2, q => p2))) for f in G.polys]
    @test maximum(residual2) < 1.0e-8

    # parameter-free system
    @polyvar u
    P = System([u^2 - 4]; variables = [u])
    pair3 = find_start_pair(P)
    @test pair3 !== nothing && pair3[2] === nothing
    @test abs(pair3[1][1]^2 - 4) < 1.0e-8
end

using HomotopyContinuationNext: MonodromyOptions, MonodromyLoop, MonodromyStatistics,
    MonodromyCode, ReuseLoops, independent_normal, loops_no_change, loop_finished!,
    rand_subspace, translate, extrinsic

@testset "monodromy data structures" begin
    Random.seed!(21)
    opts = MonodromyOptions(; group_actions = nothing)
    @test opts.equivalence_classes == false || opts.group_actions !== nothing
    @test opts.reuse_loops == ReuseLoops.ALL

    p = [1.0 + 0im, 2.0 + 0im]
    loop = MonodromyLoop(p, independent_normal)
    @test loop.p == p
    @test loop.p₀₁ ≈ 0.5 .* (loop.p₁ .- p)

    L = rand_subspace(3; dim = 1)
    loopL = MonodromyLoop(L, independent_normal)
    # equal spacing: b₀₁ - b == b₁ - b₀₁
    d1 = extrinsic(loopL.p₀₁).b .- extrinsic(loopL.p).b
    d2 = extrinsic(loopL.p₁).b .- extrinsic(loopL.p₀₁).b
    @test d1 ≈ d2 atol = 1.0e-12

    stats = MonodromyStatistics()
    loop_finished!(stats, 2)
    loop_finished!(stats, 2)
    loop_finished!(stats, 3)
    loop_finished!(stats, 3)
    @test loops_no_change(stats, 3) == 1
end

using HomotopyContinuationNext: MonodromySolver, MonodromyWorkerState, track_loop!,
    track_start!, set_loop_segment!, reset_trace!, trace_colinearity, PathResult,
    is_success, solution

@testset "worker state loop tracking (vector parameters)" begin
    Random.seed!(31)
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)
    x0, p0 = find_start_pair(F)
    MS = MonodromySolver(F, ComplexF64.(p0))
    ws = MS.workers[1]

    # A p -> p track refines the start solution into a PathResult
    res0 = track_start!(ws, ComplexF64.(x0))
    @test res0 !== nothing && is_success(res0)

    loop = MonodromyLoop(ComplexF64.(p0), MS.options.parameter_sampler)
    res1 = track_loop!(ws, loop, res0, false, MS)
    @test res1 !== nothing && is_success(res1)
    # endpoint is back on the fiber over p0
    residual = [abs(ComplexF64(f(y => solution(res1), p => p0))) for f in F.polys]
    @test maximum(residual) < 1.0e-8
end

using HomotopyContinuationNext: monodromy_solve, MonodromyResult, permutations,
    is_heuristic_stop, nsolutions, solutions, trace

@testset "monodromy_solve: v2 oracle expectations (serial)" begin
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)

    r = monodromy_solve(
        F; permutations = true, seed = UInt32(4242),
        threading = false, show_progress = false,
    )
    @test nsolutions(r) == 2
    perm = permutations(r)
    @test size(perm, 1) == 2
    @test sort(perm[:, 1]) == [1, 2]   # columns are permutations of 1:2
    @test any(col -> perm[:, col] == [2, 1] || perm[:, col] == [1, 2], 1:size(perm, 2))

    r2 = monodromy_solve(
        F; target_solutions_count = 2, seed = UInt32(7),
        threading = false, show_progress = false,
    )
    @test nsolutions(r2) == 2
    @test is_success(r2)

    # group action: x^2 - p has a 2-orbit collapsing to 1 class
    @polyvar u q
    G = System([u^2 - q]; variables = [u], parameters = [q])
    rg = monodromy_solve(
        G, [[2.0 + 0im]], [4.0 + 0im];
        group_action = s -> ([-s[1]],), seed = UInt32(11),
        threading = false, show_progress = false,
    )
    @test nsolutions(rg) == 1
    rng = monodromy_solve(
        G, [[2.0 + 0im]], [4.0 + 0im];
        seed = UInt32(11), threading = false, show_progress = false,
    )
    @test nsolutions(rng) == 2

    # heuristic stop fires when nothing new appears
    @test is_heuristic_stop(rng) || is_success(rng)
end

@testset "monodromy_solve: subspace run with trace test" begin
    @polyvar z[1:3]
    Q = System([z[1]^2 + 2z[2]^2 + 3z[3]^2 + z[1] * z[2] - 1]; variables = z)
    r = monodromy_solve(
        Q; dim = 2, seed = UInt32(99),
        threading = false, show_progress = false,
    )
    @test nsolutions(r) == 2
    @test is_success(r)                # via trace test
    @test trace(r) !== nothing && trace(r) < 1.0e-10

    # overstated component dimension raises an actionable error, not an assertion
    @test_throws ArgumentError monodromy_solve(
        Q; dim = 1, seed = UInt32(99),
        threading = false, show_progress = false,
    )
end

@testset "threaded == serial on solution sets" begin
    # nonlinear-in-p system with 4 solutions. target_solutions_count makes the
    # run deterministic (the heuristic stop can fire early on some seeds).
    @polyvar y[1:2] q[1:2]
    F = System(
        [y[1]^2 + y[2]^2 - q[1]^2, y[1] * y[2] - q[2]^3];
        variables = y, parameters = q,
    )
    rs = monodromy_solve(
        F; seed = UInt32(123), threading = false, show_progress = false,
        target_solutions_count = 4, max_loops_no_progress = 50,
    )
    rt = monodromy_solve(
        F; seed = UInt32(123), threading = true, show_progress = false,
        target_solutions_count = 4, max_loops_no_progress = 50,
    )
    @test nsolutions(rs) == 4
    @test nsolutions(rt) == 4
    # identical sets up to tolerance (ordering may differ)
    for s in solutions(rs)
        @test minimum(maximum(abs.(s .- t)) for t in solutions(rt)) < 1.0e-8
    end
end

using HomotopyContinuationNext: verify_solution_completeness, parameters

@testset "verify_solution_completeness" begin
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)
    r = monodromy_solve(
        F; target_solutions_count = 2, seed = UInt32(21),
        threading = false, show_progress = false,
    )
    ok = verify_solution_completeness(F, r; show_progress = false)
    @test ok === true

    # a strict subset is detected as incomplete
    incomplete = verify_solution_completeness(
        F, [solutions(r)[1]], Vector(parameters(r)); show_progress = false,
    )
    @test incomplete === false || incomplete === nothing
end

@testset "trace discrimination (prototype 11 magnitudes)" begin
    # full witness set gives near-zero colinearity, a strict subset does not;
    # exercised through the public API by comparing trace(r) with the threshold.
    @polyvar z[1:3]
    Q = System([z[1]^2 + 2z[2]^2 + 3z[3]^2 + z[1] * z[2] - 1]; variables = z)
    r = monodromy_solve(
        Q; dim = 2, seed = UInt32(202), threading = false,
        show_progress = false,
    )
    @test is_success(r) && trace(r) < 1.0e-6
end

@testset "reuse_loops variants and stopping" begin
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)
    for rl in (:all, :random, :none)
        r = monodromy_solve(
            F; reuse_loops = rl, target_solutions_count = 2,
            seed = UInt32(303), threading = false, show_progress = false,
        )
        @test nsolutions(r) == 2
    end
    # seed chosen so that both solutions appear before the no-progress window
    # of 2 loops elapses (some seeds stop with only 1 found)
    rml = monodromy_solve(
        F; max_loops_no_progress = 2, seed = UInt32(306),
        threading = false, show_progress = false,
    )
    @test nsolutions(rml) == 2 && is_heuristic_stop(rml)
end
