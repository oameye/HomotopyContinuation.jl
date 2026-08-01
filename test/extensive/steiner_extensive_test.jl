using Test
using HomotopyContinuationNext
using HomotopyContinuationNext: CONSERVATIVE_TRACKER_OPTIONS
using HomotopyContinuationNextCertification: certify, ncertified, ndistinct_certified,
    ndistinct_real_certified
using Random: MersenneTwister

isdefined(@__MODULE__, :steiner_system) || include("../test_systems.jl")

const STEINER_SEED = 0x9a8b7c6d

# Five conics whose 3264 tangent conics are all real.
const REAL_CONICS = [
    10124547 // 662488724,
    8554609 // 755781377,
    5860508 // 2798943247,
    -251402893 // 1016797750,
    -25443962 // 277938473,
    1 // 1,
    520811 // 1788018449,
    2183697 // 542440933,
    9030222 // 652429049,
    -12680955 // 370629407,
    -24872323 // 105706890,
    1 // 1,
    6537193 // 241535591,
    -7424602 // 363844915,
    6264373 // 1630169777,
    13097677 // 39806827,
    -29825861 // 240478169,
    1 // 1,
    13173269 // 2284890206,
    4510030 // 483147459,
    2224435 // 588965799,
    33318719 // 219393000,
    92891037 // 755709662,
    1 // 1,
    8275097 // 452566634,
    -19174153 // 408565940,
    5184916 // 172253855,
    -23713234 // 87670601,
    28246737 // 81404569,
    1 // 1,
]

@testset "3264 conics tangent to five conics" begin
    polys, vars, params = steiner_system()
    F = System(polys; variables = vars, parameters = params)
    p = randn(MersenneTwister(2), ComplexF64, 30)

    generic = solve(
        fix_parameters(F, p), Polyhedral(; seed = STEINER_SEED, show_progress = false),
    )
    @test ntracked(generic) == 27072
    @test nsolutions(generic) == 3264
    @test ndistinct_certified(certify(F, generic, p)) == 3264

    # Five of the 3264 conics sit at scaled condition ~1e14, three orders under
    # `sing_cond`; certification confirms all 3264 are regular, distinct and real.
    @testset "tracked to the real conics" begin
        real_res = solve(
            F, solutions(generic), p, REAL_CONICS,
            Continuation(;
                tracker_options = CONSERVATIVE_TRACKER_OPTIONS,
                seed = STEINER_SEED, show_progress = false,
            ),
        )
        endpoints = [solution(r) for r in results(real_res; only_nonsingular = false)]
        @test length(endpoints) == 3264
        @test nsolutions(real_res) == 3264
        @test nsingular(real_res) == 0

        cert = certify(F, endpoints, REAL_CONICS)
        @test ncertified(cert) == 3264
        @test ndistinct_certified(cert) == 3264
        @test ndistinct_real_certified(cert) == 3264
    end

    @testset "solved at the real conics directly" begin
        direct = solve(
            fix_parameters(F, REAL_CONICS),
            Polyhedral(; seed = STEINER_SEED, show_progress = false),
        )
        # The direct solve also returns non-solution endpoints at condition ≥ 1e17;
        # those are the ones reported singular, and certification rejects exactly them.
        endpoints = [solution(r) for r in results(direct; only_nonsingular = false)]
        @test length(endpoints) >= 3264
        @test ndistinct_real_certified(certify(F, endpoints, REAL_CONICS)) == 3264
        @test nresults(direct; only_nonsingular = true) == 3264
        nonsingular_sols = [solution(r) for r in results(direct; only_nonsingular = true)]
        @test ndistinct_real_certified(certify(F, nonsingular_sols, REAL_CONICS)) == 3264
    end
end
