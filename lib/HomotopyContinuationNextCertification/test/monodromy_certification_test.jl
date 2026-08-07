using Test
using HomotopyContinuationNext:
    @polyvar, System, Monodromy, MonodromyOptions, MonodromySolver, Serial, Threaded,
    solve, solutions, nsolutions, nresults, results, is_success, is_singular,
    winding_number, slice, parameters, LinearSubspace, PathResult, PathResultCode,
    DuplicateCheck, ncertified_distinct, ndiscarded_uncertified, add!,
    accuracy, residual, condition_jacobian, path_results
using HomotopyContinuationNextCertification:
    Certification, certify, ndistinct_certified
using DynamicPolynomials: differentiate

# The toric ED fixture and its group action are shared with the core suite.
include(joinpath(@__DIR__, "..", "..", "..", "test", "test_systems.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# `duplicate_check = DuplicateCheck.CERTIFIED`: a monodromy endpoint is kept only if it
# certifies as a solution distinct from every solution found so far, and what is
# stored is the midpoint of its certified interval.
# ─────────────────────────────────────────────────────────────────────────────

certified(; kwargs...) = Monodromy(;
    duplicate_check = DuplicateCheck.CERTIFIED, show_progress = false,
    max_loops_no_progress = 20, kwargs...,
)

@testset "certified duplicate checks" begin
    @polyvar x a
    F = System([x^2 - a]; variables = [x], parameters = [a])
    p = [1.0 + 0im]

    for (exec, starts) in (
            (Serial(), [[1.0 + 0im], [1.0 + 0im], [-1.0 + 0im], [-1.0 + 0im]]),
            (Threaded(), [[1.0 + 0im]]),
        )
        r = solve(
            F, starts, p, certified(; target_solutions_count = 2), exec,
        )
        cert = certify(F, r, Certification(; show_progress = false), Serial())
        @test is_success(r)
        @test r.duplicate_check == DuplicateCheck.CERTIFIED
        @test nsolutions(r) == 2
        @test length(solutions(r)) == nsolutions(r)
        @test nresults(r) == length(results(r))
        @test ncertified_distinct(r) == 2
        @test ndiscarded_uncertified(r) == 0
        @test ndistinct_certified(cert) == ncertified_distinct(r)
        # Every stored endpoint is the midpoint of a certified interval, and its
        # diagnostics were measured there.
        @test all(s -> abs(s[1]^2 - 1) < 1.0e-12, solutions(r))
        @test all(pr -> residual(pr) < 1.0e-12, path_results(r))
    end

    @testset "heuristic route is unchanged" begin
        r = solve(
            F, [[1.0 + 0im]], p,
            Monodromy(; show_progress = false, target_solutions_count = 2), Serial(),
        )
        @test r.duplicate_check == DuplicateCheck.HEURISTIC
        @test nsolutions(r) == 2
        @test ncertified_distinct(r) == 0
        @test ndiscarded_uncertified(r) == 0
        @test r.statistics.certification_attempts[] == 0
    end

    @testset "a certified endpoint is not singular" begin
        # A path that stopped 1e-7 short of the solution with a winding number
        # and an ill-conditioned Jacobian, and was therefore reported singular:
        # certification decides the point is a simple solution, so it is accepted
        # and stored as non-singular.
        singular_endpoint = PathResult(
            PathResultCode.PATH_SUCCESS, ComplexF64[1.0 + 1.0e-7], 0.0, 1.0e-6, 1.0,
            1.0e-6, 3.2e-7, 1.0e10, 2, true, 0, 0, 0, false, ComplexF64[1.0], 0.0, 1,
            ComplexF64[1.0], Float64[], 1,
        )
        MS = MonodromySolver(
            F, p;
            options = MonodromyOptions(; duplicate_check = DuplicateCheck.CERTIFIED),
        )
        id, added, accepted = add!(MS, singular_endpoint, 1)
        @test added
        @test id == 1
        @test !is_singular(accepted)
        # The winding number describes the path, so it survives.
        @test winding_number(accepted) == 2
        # The diagnostics describe the point that is reported, not the endpoint
        # it replaced: that one sat 1e-7 off the solution and was called
        # ill-conditioned, and its numbers said so.
        @test residual(accepted) < 1.0e-12 < residual(singular_endpoint)
        @test accuracy(accepted) < 1.0e-12 < accuracy(singular_endpoint)
        @test condition_jacobian(accepted) < 1.0e3 < condition_jacobian(singular_endpoint)
        # The same point is now a certified duplicate, not a second solution.
        _, added_again, _ = add!(MS, singular_endpoint, 2)
        @test !added_again
        @test MS.statistics.certified_duplicates[] == 1

        # A point that is no solution certifies as neither, and is discarded.
        # Refinement is off: Newton would pull it onto a solution first.
        far_off = PathResult(
            PathResultCode.PATH_SUCCESS, ComplexF64[100.0], 0.0, 0.0, 0.0, 0.0, 0.0,
            1.0, 1, false, 0, 0, 0, false, ComplexF64[100.0], 0.0, 2,
            ComplexF64[100.0], Float64[], 1,
        )
        MS_raw = MonodromySolver(
            F, p;
            options = MonodromyOptions(;
                duplicate_check = DuplicateCheck.CERTIFIED,
                certification_refine_solution = false,
            ),
        )
        id_far, added_far, _ = add!(MS_raw, far_off, 1)
        @test !added_far
        @test id_far == 0
        @test MS_raw.statistics.uncertified_discards[] == 1
    end

    @testset "equivalence classes" begin
        # The ED discriminant of the twisted cubic: 21 solutions in 7 orbits of
        # the cube roots of unity acting on the first two coordinates.
        toric, _ = toric_ed_system()
        r = solve(
            toric,
            certified(;
                group_action = toric_ed_roots_of_unity, target_solutions_count = 7,
                seed = UInt32(0x2f7c1a05),
            ),
            Serial(),
        )
        @test is_success(r)
        @test nsolutions(r) == 7
        @test length(solutions(r)) == nsolutions(r)
        @test ncertified_distinct(r) == 7
    end

    # Both subspace regimes: the solutions are certified in ambient coordinates,
    # which is where the run reports them whichever regime tracked them.
    @testset "linear subspace (intrinsic = $intrinsic)" for intrinsic in (false, true)
        @polyvar z[1:2]
        G = System([z[1]^2 + z[2]^2 - 1]; variables = z)
        L = LinearSubspace([1.0 + 0im 0.0 + 0im], [0.0 + 0im])
        r = solve(
            G, [[0.0 + 0im, 1.0 + 0im], [0.0 + 0im, -1.0 + 0im]], L,
            certified(; target_solutions_count = 2, intrinsic = intrinsic), Serial(),
        )
        sliced = certify(
            slice(G, parameters(r)), solutions(r),
            Certification(; show_progress = false), Serial(),
        )
        @test is_success(r)
        @test nsolutions(r) == 2
        @test ncertified_distinct(r) == 2
        @test ndistinct_certified(sliced) == 2
    end

end
