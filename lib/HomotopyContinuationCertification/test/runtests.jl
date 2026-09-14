using Test
using HomotopyContinuation
using HomotopyContinuationCertification

# Existing certification test bodies are retained during the package-identity
# rename and imported through test-only compatibility modules. CI explicitly
# loads the renamed production packages above before running them.
const LEGACY_TEST_DIR = normpath(joinpath(
    @__DIR__, "..", "..", "HomotopyContinuationNextCertification", "test",
))

@testset "HomotopyContinuationCertification" begin
    for file in (
        "interval_arithmetic_test.jl",
        "acb_interpreter_test.jl",
        "log_test.jl",
        "certification_test.jl",
        "result_candidate_certification_test.jl",
        "iterator_certification_test.jl",
        "monodromy_certification_test.jl",
        "export_surface_test.jl",
        "quality_test.jl",
    )
        include(joinpath(LEGACY_TEST_DIR, file))
    end
end
