using Test
import HomotopyContinuationNext as HCN
using HomotopyContinuationNext: @polyvar, System, Result, TotalDegree, solve, solutions,
    path_results
using HomotopyContinuationNextCertification: Certification, certify, ncandidates, ncertified

@testset "Result certification does not trust numerical singular labels" begin
    @polyvar x
    F = System([x - 1])
    solved = solve(F, TotalDegree(; show_progress = false))
    pr = only(path_results(solved))

    # Mimic a successful regular root that the numerical endpoint classifier marked
    # singular because of conditioning. `solutions` intentionally hides it, but the
    # rigorous Krawczyk layer must still get a chance to decide.
    marked = HCN._with_fields(pr, (singular = true,))
    result = Result([marked], 1, UInt32(0x51a9))

    @test isempty(solutions(result))

    cert = certify(F, result, Certification(; show_progress = false))
    @test ncandidates(cert) == 1
    @test ncertified(cert) == 1
end
