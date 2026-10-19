using Test
using HomotopyContinuationCertification: HomotopyContinuationCertification as Cert
using HomotopyContinuation: monodromy_certified_solutions
using Aqua: Aqua

@testset "Aqua" begin
    Aqua.test_all(
        Cert;
        deps_compat = (; ignore = [:HomotopyContinuation]),
        piracies = (; treat_as_own = [monodromy_certified_solutions]),
        persistent_tasks = VERSION >= v"1.11",
    )
end
