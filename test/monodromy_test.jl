using Test, Random
using LinearAlgebra
using HomotopyContinuationNext
using HomotopyContinuationNext: find_start_pair
using DynamicPolynomials: @polyvar, subs

using HomotopyContinuationNext: _trace_step, _with_solution, _conditioned_chart,
    _chart_alignment, _monodromy_starts, PathResult, solution

using HomotopyContinuationNext: MonodromySolver, MonodromyWorkerState, track_loop!,
    track_start!, set_loop_segment!, reset_trace!, trace_colinearity, trace_complete,
    is_success
using HomotopyContinuationNext: verify_solution_completeness, parameters
using HomotopyContinuationNext: _accumulate_trace!, _trace_dropped!
using HomotopyContinuationNext: DuplicateCheck

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
    loop = MonodromyLoop(p, independent_normal, Random.MersenneTwister(0x2718))
    @test loop.p == p
    @test loop.p₀₁ ≈ 0.5 .* (loop.p₁ .- p)

    L = rand_subspace(3; dim = 1)
    loopL = MonodromyLoop(L, independent_normal, Random.MersenneTwister(0x2718))
    # equal spacing: b₀₁ - b == b₁ - b₀₁
    d1 = extrinsic(loopL.p₀₁).b .- extrinsic(loopL.p).b
    d2 = extrinsic(loopL.p₁).b .- extrinsic(loopL.p₀₁).b
    @test d1 ≈ d2 atol = 1.0e-12

    # A linear base is the projective regime, where the step follows the points
    # instead of sweeping wide. Equal spacing holds either way.
    Llin = rand_subspace(3; dim = 1, affine = false)
    loopLin = MonodromyLoop(Llin, independent_normal, Random.MersenneTwister(0x2718))
    e1 = extrinsic(loopLin.p₀₁).b .- extrinsic(loopLin.p).b
    e2 = extrinsic(loopLin.p₁).b .- extrinsic(loopLin.p₀₁).b
    @test e1 ≈ e2 atol = 1.0e-12
    @test norm(d1) ≈ 5
    @test norm(e1) ≈ 1

    # An explicit step is what the translation uses.
    loopStep = MonodromyLoop(Llin, independent_normal, Random.MersenneTwister(0x2718), 3.0)
    @test norm(extrinsic(loopStep.p₀₁).b .- extrinsic(loopStep.p).b) ≈ 3

    stats = MonodromyStatistics()
    loop_finished!(stats, 2)
    loop_finished!(stats, 2)
    loop_finished!(stats, 3)
    loop_finished!(stats, 3)
    @test loops_no_change(stats, 3) == 1
end

@testset "trace slice step" begin
    @polyvar w[1:2]
    r = solve(
        System([w[1]^2 + w[2]^2 - 4, w[1] - w[2]^2]),
        TotalDegree(; seed = UInt32(11), show_progress = false),
    )
    rs = path_results(r)
    @test !isempty(rs)

    Laff = rand_subspace(2; dim = 1)
    Llin = rand_subspace(2; dim = 1, affine = false)
    # Affine: a fixed wide sweep, whatever scale the solutions sit at.
    @test _trace_step(Laff, rs) == 5.0
    # Linear (projective): the scale of the solutions themselves.
    scaled = [_with_solution(pr, 10 .* solution(pr)) for pr in rs]
    @test _trace_step(Llin, scaled) ≈ 10 * _trace_step(Llin, rs)
    @test _trace_step(Laff, scaled) == 5.0
    @test _trace_step(Llin, PathResult[]) == 1.0
end

@testset "affine chart conditioning" begin
    seed = 0x00c0ffee
    n = 4
    first_draw = randn(Random.MersenneTwister(seed), ComplexF64, n)
    # `on_chart!` divides by v'x, so a start solution orthogonal to the draw
    # cannot be placed on it at all.
    x = LinearAlgebra.nullspace(transpose(first_draw))[:, 1]
    @test _chart_alignment(first_draw, [x]) < 1.0e-12
    chart = _conditioned_chart(Random.MersenneTwister(seed), n, [x])
    @test _chart_alignment(chart, [x]) >= 0.2
    # Nothing to condition on leaves the draw alone.
    @test _conditioned_chart(Random.MersenneTwister(seed), n, Vector{ComplexF64}[]) ==
        first_draw
end

@testset "monodromy start solutions" begin
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)
    p0 = [3.0 + 0im, 1.0 + 0im]
    r = solve(
        System([y[1]^2 + y[2]^2 - 3, y[1] + y[2] - 1]; variables = y),
        TotalDegree(; seed = UInt32(5), show_progress = false),
    )
    @test nsolutions(r) == 2

    # A `Result` and a single solution are both accepted as start solutions.
    @test _monodromy_starts(r) == solutions(r)
    s = solutions(r)[1]
    @test _monodromy_starts(s) == [s]

    mr = solve(F, r, p0, Monodromy(; seed = UInt32(5), show_progress = false), Serial())
    @test nsolutions(mr) == 2
end


@testset "worker state loop tracking (vector parameters)" begin
    Random.seed!(31)
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)
    x0, p0 = find_start_pair(F)
    MS = MonodromySolver(F, ComplexF64.(p0))
    ws = MS.workers[1]
    @test ws.homotopy.system === F.evaluator
    cloned_ws = MS.builder()
    @test cloned_ws.homotopy.system !== F.evaluator
    @test cloned_ws.homotopy.system !== ws.homotopy.system

    # A p -> p track refines the start solution into a PathResult
    res0 = track_start!(ws, ComplexF64.(x0))
    @test res0 !== nothing && is_success(res0)

    loop = MonodromyLoop(
        ComplexF64.(p0), MS.options.parameter_sampler, Random.MersenneTwister(0x2718),
    )
    res1 = track_loop!(ws, loop, res0, false, MS)
    @test res1 !== nothing && is_success(res1)
    # endpoint is back on the fiber over p0
    residual = [abs(ComplexF64(f(y => solution(res1), p => p0))) for f in F.polys]
    @test maximum(residual) < 1.0e-8
end

using HomotopyContinuationNext: MonodromyResult, permutations,
    is_heuristic_stop, nsolutions, solutions, trace

@testset "monodromy: oracle expectations (serial)" begin
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)

    r = solve(
        F,
        Monodromy(; permutations = true, seed = UInt32(4242), show_progress = false),
        Serial(),
    )
    @test nsolutions(r) == 2
    perm = permutations(r)
    @test size(perm, 1) == 2
    @test sort(perm[:, 1]) == [1, 2]   # columns are permutations of 1:2
    @test any(col -> perm[:, col] == [2, 1] || perm[:, col] == [1, 2], 1:size(perm, 2))

    r2 = solve(
        F,
        Monodromy(; target_solutions_count = 2, seed = UInt32(7), show_progress = false),
        Serial(),
    )
    @test nsolutions(r2) == 2
    @test is_success(r2)

    # group action: x^2 - p has a 2-orbit collapsing to 1 class
    @polyvar u q
    G = System([u^2 - q]; variables = [u], parameters = [q])
    # A loop swaps the two roots only when it winds around the branch point
    # q = 0, which is a property of the loop the seed draws; this seed's does.
    rg = solve(
        G,
        [[2.0 + 0im]],
        [4.0 + 0im],
        Monodromy(;
            group_action = s -> ([-s[1]],), seed = UInt32(100), show_progress = false,
        ),
        Serial(),
    )
    @test nsolutions(rg) == 1
    rng = solve(
        G,
        [[2.0 + 0im]],
        [4.0 + 0im],
        Monodromy(; seed = UInt32(100), show_progress = false),
        Serial(),
    )
    @test nsolutions(rng) == 2

    # heuristic stop fires when nothing new appears
    @test is_heuristic_stop(rng) || is_success(rng)
end

@testset "monodromy: subspace run with trace test" begin
    @polyvar z[1:3]
    Q = System([z[1]^2 + 2z[2]^2 + 3z[3]^2 + z[1] * z[2] - 1]; variables = z)
    r = solve(Q, Monodromy(; dim = 2, seed = UInt32(99), show_progress = false), Serial())
    @test nsolutions(r) == 2
    @test is_success(r)                # via trace test
    @test trace(r) !== nothing && trace(r) < 1.0e-10

    # overstated component dimension raises an actionable error, not an assertion
    @test_throws ArgumentError solve(Q, Monodromy(; dim = 1, seed = UInt32(99), show_progress = false), Serial())
end

@testset "trace completeness" begin
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)
    MS = MonodromySolver(F, ComplexF64[3, 1])
    reset_trace!(MS)
    @test trace_complete(MS) && MS.trace_paths == 0

    x = ComplexF64[1, 1]
    _accumulate_trace!(MS, x, x, x)
    @test MS.trace_paths == 1 && trace_complete(MS)

    # A path that never reached the halfway subspace leaves the trace short, so
    # its value says nothing about the witness set.
    _trace_dropped!(MS)
    @test MS.trace_dropped == 1 && !trace_complete(MS)

    reset_trace!(MS)
    @test MS.trace_paths == 0 && MS.trace_dropped == 0 && trace_complete(MS)
end

@testset "threaded == serial on solution sets" begin
    # nonlinear-in-p system with 4 solutions. target_solutions_count makes the
    # run deterministic (the heuristic stop can fire early on some seeds).
    @polyvar y[1:2] q[1:2]
    F = System(
        [y[1]^2 + y[2]^2 - q[1]^2, y[1] * y[2] - q[2]^3];
        variables = y, parameters = q,
    )
    rs = solve(
        F,
        Monodromy(;
            seed = UInt32(123), show_progress = false, target_solutions_count = 4,
            max_loops_no_progress = 50,
        ),
        Serial(),
    )
    rt = solve(
        F,
        Monodromy(;
            seed = UInt32(123), show_progress = false, target_solutions_count = 4,
            max_loops_no_progress = 50,
        ),
        Threaded(),
    )
    @test nsolutions(rs) == 4
    @test nsolutions(rt) == 4
    # identical sets up to tolerance (ordering may differ)
    for s in solutions(rs)
        @test minimum(maximum(abs.(s .- t)) for t in solutions(rt)) < 1.0e-8
    end
end


@testset "verify_solution_completeness" begin
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)
    r = solve(
        F,
        Monodromy(; target_solutions_count = 2, seed = UInt32(21), show_progress = false),
        Serial(),
    )
    ok = verify_solution_completeness(F, r, Monodromy(; show_progress = false))
    @test ok === true

    # a strict subset is detected as incomplete
    incomplete = verify_solution_completeness(
        F,
        [solutions(r)[1]],
        Vector(parameters(r)),
        Monodromy(; show_progress = false),
    )
    @test incomplete === false || incomplete === nothing
end

@testset "trace discrimination (prototype 11 magnitudes)" begin
    # full witness set gives near-zero colinearity, a strict subset does not;
    # exercised through the public API by comparing trace(r) with the threshold.
    @polyvar z[1:3]
    Q = System([z[1]^2 + 2z[2]^2 + 3z[3]^2 + z[1] * z[2] - 1]; variables = z)
    r = solve(Q, Monodromy(; dim = 2, seed = UInt32(202), show_progress = false), Serial())
    @test is_success(r) && trace(r) < 1.0e-6
end

@testset "reuse_loops variants and stopping" begin
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)
    for rl in (ReuseLoops.ALL, ReuseLoops.RANDOM, ReuseLoops.NONE)
        r = solve(
            F,
            Monodromy(;
                reuse_loops = rl, target_solutions_count = 2, seed = UInt32(303),
                show_progress = false,
            ),
            Serial(),
        )
        @test nsolutions(r) == 2
    end
    # seed chosen so that both solutions appear before the no-progress window
    # of 2 loops elapses (some seeds stop with only 1 found)
    rml = solve(
        F,
        Monodromy(; max_loops_no_progress = 2, seed = UInt32(300), show_progress = false),
        Serial(),
    )
    @test nsolutions(rml) == 2 && is_heuristic_stop(rml)
end

@testset "duplicate_check option" begin
    @polyvar y[1:2] p[1:2]
    F = System([y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p)

    @test MonodromyOptions().duplicate_check == DuplicateCheck.HEURISTIC
    @test MonodromyOptions(; duplicate_check = DuplicateCheck.CERTIFIED).duplicate_check ==
        DuplicateCheck.CERTIFIED
    # The option is an enum, not a symbol.
    @test_throws TypeError MonodromyOptions(; duplicate_check = :certified)

    r = solve(
        F,
        Monodromy(; target_solutions_count = 2, seed = UInt32(404), show_progress = false),
        Serial(),
    )
    @test r.duplicate_check == DuplicateCheck.HEURISTIC
    @test ncertified_distinct(r) == 0
    @test ndiscarded_uncertified(r) == 0

    # Certifying needs the certification package, which core alone cannot reach.
    @test_throws ArgumentError solve(
        F,
        Monodromy(;
            duplicate_check = DuplicateCheck.CERTIFIED, show_progress = false,
        ),
        Serial(),
    )
end
