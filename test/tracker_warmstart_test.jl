using Test, Random
using HomotopyContinuationNext
using HomotopyContinuationNext: HomotopyEvaluator, ParameterHomotopy, Tracker,
    TrackerCode, track!, EndgameCode
using DynamicPolynomials: @polyvar

@testset "warm-started track!" begin
    Random.seed!(0x5eed)
    @polyvar y q
    G = System([y^2 - q]; variables = [y], parameters = [q])
    H = ParameterHomotopy(G.evaluator, [1.0 + 0im], [9.0 + 0im])
    tracker = Tracker(HomotopyEvaluator(H))

    # Cold start
    code = track!(tracker, [1.0 + 0.0im])
    @test code == TrackerCode.TRACKER_SUCCESS
    ω = tracker.state.ω
    μ = tracker.state.μ
    endpoint = copy(Vector(tracker.state.x))

    # Warm start with the stored certificates reproduces the endpoint
    code2 = track!(tracker, [1.0 + 0.0im]; ω = ω, μ = μ)
    @test code2 == TrackerCode.TRACKER_SUCCESS
    @test Vector(tracker.state.x) ≈ endpoint atol = 1.0e-12

    # extended_precision kwarg is accepted and sets the state flag on init
    code3 = track!(tracker, [1.0 + 0.0im]; ω = ω, μ = μ, extended_precision = true)
    @test code3 == TrackerCode.TRACKER_SUCCESS

    # EndgameTracker track! accepts the same kwargs
    eg = EndgameTracker(Tracker(HomotopyEvaluator(H)), EndgameOptions(; endgame_start = 0.0))
    egcode = track!(eg, [1.0 + 0.0im]; ω = ω, μ = μ, extended_precision = false)
    @test egcode == EndgameCode.SUCCESS
    @test eg.state.solution[1] ≈ 3.0 atol = 1.0e-10
end

@testset "PathResult carries ω and μ" begin
    @polyvar y q
    G = System([y^2 - q]; variables = [y], parameters = [q])
    res = solve(
        G, [[1.0 + 0.0im]], [1.0 + 0im], [9.0 + 0im], Serial();
        seed = UInt32(1), show_progress = false,
    )
    r = first(path_results(res))
    @test r.ω > 0.0 && isfinite(r.ω)
    @test r.μ > 0.0 && isfinite(r.μ)
end
