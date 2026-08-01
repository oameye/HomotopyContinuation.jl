# Regression test guarding the public export surface. The other tests reach
# certification names via explicit `using ...: name` imports, so they never
# notice when a public name is defined but not exported. This test asserts
# membership in `names(HomotopyContinuationNextCertification)` directly.

using Test
using HomotopyContinuationNextCertification: HomotopyContinuationNextCertification

@testset "Export surface" begin
    Cert = HomotopyContinuationNextCertification
    exported = Set(names(Cert))

    # The public certification API.
    certification_public = [
        :certify,
        :SolutionCertificate,
        :CertificationResult,
        :CertificationCache,
        :is_certified,
        :is_real,
        :is_complex,
        :is_positive,
        :solution_candidate,
        :certified_solution_interval,
        :certified_solution_interval_after_krawczyk,
        :certificate_index,
        :solution_approximation,
        :certificates,
        :distinct_certificates,
        :distinct_solutions,
        :ncertified,
        :nreal_certified,
        :ncomplex_certified,
        :ndistinct_certified,
        :ndistinct_real_certified,
        :ndistinct_complex_certified,
        :save,
        :DistinctCertifiedSolutions,
        :add_solution!,
        :distinct_certified_solutions,
        :distinct_certified_solutions!,
        :stats,
        :ncertified_distinct,
        :nprocessed,
        :nduplicates,
        :nnotcertified,
        :show_straight_line_program,
    ]

    @testset "certification: $name" for name in certification_public
        @test name in exported
    end

    # Every exported name must actually resolve (guards typos in the export
    # list; Aqua also checks this, kept here so the surface test is standalone).
    @testset "resolves: $name" for name in certification_public
        @test isdefined(Cert, name)
    end
end
