using Test
using HomotopyContinuationNext: System, ParameterHomotopy, StraightLineHomotopy,
    HomotopyEvaluator, Tracker, TrackerOptions, TrackerCode, @var,
    iterator, path_info, PathInfo, PathStep, path_table, track!, solution,
    steps, accepted_steps, rejected_steps

function parameter_tracker(; options::TrackerOptions = TrackerOptions())
    @var x y a b
    F = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
    H = ParameterHomotopy(F, [1.0, 0.0], [2.0, 4.0])
    return Tracker(HomotopyEvaluator(H); options = options)
end

@testset "path iterator" begin
    tracker = parameter_tracker()
    @test eltype(iterator(tracker, [1.0, 1.0], 1.0, 0.0)) ===
        Tuple{Vector{ComplexF64}, Float64}
    @test first(iterator(tracker, [1.0, 1.0], 1.0, 0.0)) isa
        Tuple{Vector{ComplexF64}, Float64}

    # Every yielded point is an accepted step, and the last one sits at the target.
    steps = collect(iterator(tracker, [1.0, 1.0], 1.0, 0.0))
    @test tracker.state.code == TrackerCode.TRACKER_SUCCESS
    @test last(steps)[2] == 0.0
    @test first(steps)[2] == 1.0
    @test issorted([t for (_, t) in steps]; rev = true)
    # The path ends where `track!` puts it.
    @test maximum(abs.(last(steps)[1] .- collect(tracker.state.x))) == 0.0

    @testset "a smaller step size yields more points" begin
        fine = parameter_tracker(; options = TrackerOptions(; max_step_size = 0.01))
        @test length(collect(iterator(fine, [1.0, 1.0], 1.0, 0.0))) >= 101
    end

    @testset "linear path steps at the requested size" begin
        @var z c
        G = System([z - c]; variables = [z], parameters = [c])
        ct = Tracker(
            HomotopyEvaluator(ParameterHomotopy(G, [1.0], [2.0]));
            options = TrackerOptions(; max_step_size = 0.015625),
        )
        xs = Vector{ComplexF64}[]
        for (x, _) in iterator(ct, [1.0], 1.0, 0.0)
            push!(xs, x)
        end
        @test length(xs) >= length(1:0.015625:2)
    end

    @testset "complex endpoints yield complex t" begin
        tracker = parameter_tracker()
        iter = iterator(tracker, [1.0, 1.0], 1.0 + 0.0im, 0.0 + 0.0im)
        @test eltype(iter) === Tuple{Vector{ComplexF64}, ComplexF64}
        (_, t) = first(iter)
        @test t isa ComplexF64
    end
end

@testset "path_info" begin
    tracker = parameter_tracker()
    info = path_info(tracker, [1.0, 1.0], 1.0, 0.0)
    @test info isa PathInfo
    @test info isa AbstractVector{PathStep}
    @test info.return_code == TrackerCode.TRACKER_SUCCESS
    @test !isempty(info)
    @test steps(info) == length(info) == length(collect(info))
    @test accepted_steps(info) == tracker.state.accepted_steps
    @test rejected_steps(info) == tracker.state.rejected_steps
    @test accepted_steps(info) + rejected_steps(info) == steps(info)
    @test info.n_factorizations > 0
    @test info.n_ldivs > 0
    @test first(info).s == 1.0
    @test last(info) === info[end] === info[length(info)]
    # The condition estimate is a lower bound, so 0 is a legal value, NaN is not.
    @test all(step -> step.cond >= 0, info)

    @test sprint(show, info) ==
        "PathInfo($(steps(info)) steps, $(accepted_steps(info)) ✓ / " *
        "$(rejected_steps(info)) ✗, $(info.return_code))"

    out = sprint(show, MIME("text/plain"), info)
    @test occursin("PathInfo:", out)
    @test occursin("TRACKER_SUCCESS", out)
    # One header row plus one row per step, inside the box rules.
    @test count(==('│'), out) == 14 * (steps(info) + 1)
    @test !isempty(sprint(path_table, info))

    @testset "a limited display elides the middle rows" begin
        long = path_info(
            parameter_tracker(; options = TrackerOptions(; max_step_size = 0.01)),
            [1.0, 1.0], 1.0, 0.0,
        )
        @test steps(long) > 20
        out = sprint(
            show, MIME("text/plain"), long;
            context = (:limit => true, :displaysize => (14, 200)),
        )
        @test occursin("⋮", out)
        @test count(==('│'), out) < 14 * (steps(long) + 1)
        # The full table is what `path_table` prints.
        @test !occursin("⋮", sprint(path_table, long))
    end

    @testset "tracking the same path twice gives the same table" begin
        again = path_info(parameter_tracker(), [1.0, 1.0], 1.0, 0.0)
        @test map(step -> step.s, again) == map(step -> step.s, info)
        @test map(step -> step.accepted, again) == map(step -> step.accepted, info)
    end
end
