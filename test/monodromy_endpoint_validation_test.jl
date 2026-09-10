using Test
using HomotopyContinuationNext
using HomotopyContinuationNext:
    MonodromySolver, track_start!, _with_solution, add_tracked_result!, permutations
using DynamicPolynomials: @polyvar

@testset "Monodromy endpoint admission" begin
    @polyvar x p
    F = System([x^2 - p]; variables = [x], parameters = [p])
    MS = MonodromySolver(F, ComplexF64[1])
    ws = MS.workers[1]
    base = track_start!(ws, ComplexF64[1])
    @test base !== nothing

    # x = 0 is not a solution of x² - 1 and has a singular Newton derivative.
    # It must not become a stored monodromy start merely because it looks new.
    invalid = _with_solution(base, ComplexF64[0])
    id, added, _ = add_tracked_result!(MS, invalid, 1, nothing, 1)
    @test id == 0
    @test !added
    @test length(MS.unique_points) == 0

    # A genuinely new approximate endpoint is revalidated/refined at the base.
    approximate = _with_solution(base, ComplexF64[1.01])
    id, added, accepted = add_tracked_result!(MS, approximate, 1, nothing, 1)
    @test id == 1
    @test added
    @test abs(solution(accepted)[1]^2 - 1) < 1.0e-10
    @test length(MS.unique_points) == 1
end

@testset "Incomplete permutation histories are padded" begin
    @polyvar y[1:2] q[1:2]
    F = System(
        [y[1]^2 + y[2]^2 - q[1], y[1] + y[2] - q[2]];
        variables = y, parameters = q,
    )
    r = solve(
        F,
        Monodromy(; permutations = true, seed = UInt32(4242), show_progress = false),
        Serial(),
    )
    @test nsolutions(r) == 2
    empty!(r.statistics.permutations)
    push!(r.statistics.permutations, [2, 1])
    push!(r.statistics.permutations, [1])
    @test permutations(r; reduced = false) == [2 1; 1 0]
end
