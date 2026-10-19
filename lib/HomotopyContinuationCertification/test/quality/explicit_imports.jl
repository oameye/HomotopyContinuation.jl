using Test
using HomotopyContinuationCertification: HomotopyContinuationCertification as Cert
using ExplicitImports:
    check_no_implicit_imports, check_no_stale_explicit_imports,
    check_all_explicit_imports_via_owners, check_all_qualified_accesses_via_owners

@testset "ExplicitImports" begin
    # `@enumx` expands to a module, which the checks cannot see into.
    allow_unanalyzable = (Cert.IteratorCertificationPhase,)
    ignore = (:ExecInstruction,)
    @test check_no_implicit_imports(Cert; allow_unanalyzable) === nothing
    @test check_no_stale_explicit_imports(Cert; allow_unanalyzable, ignore) === nothing
    @test check_all_explicit_imports_via_owners(Cert) === nothing
    @test check_all_qualified_accesses_via_owners(Cert) === nothing
end
