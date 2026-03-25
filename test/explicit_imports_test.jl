using Test
using ExplicitImports
using HomotopyContinuationNext

@testset "ExplicitImports" begin
    using ExplicitImports
    allow_unanalyzable = (
    )

    @test check_no_implicit_imports(HomotopyContinuationNext; allow_unanalyzable) == nothing
    @test check_all_explicit_imports_via_owners(HomotopyContinuationNext) == nothing
    @test check_all_explicit_imports_are_public(HomotopyContinuationNext) == nothing
    @test check_no_stale_explicit_imports(HomotopyContinuationNext; allow_unanalyzable) == nothing
    @test check_all_qualified_accesses_via_owners(HomotopyContinuationNext) == nothing
    # Allow non-public but necessary accesses:
    # - Base.RefValue: used for mutable cache scalars in immutable structs
    # - Base.decompose: required for DoubleF64 <: AbstractFloat
    # - LinearAlgebra.qrfactUnblocked!: needed for custom QR (LAPACK qr! returns QRCompactWY)
    @test check_all_qualified_accesses_are_public(
        HomotopyContinuationNext;
        ignore = (:RefValue, :decompose, :qrfactUnblocked!),
    ) == nothing
    @test check_no_self_qualified_accesses(HomotopyContinuationNext) == nothing
end
