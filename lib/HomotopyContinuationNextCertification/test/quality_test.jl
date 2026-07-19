using Test
using HomotopyContinuationNextCertification: HomotopyContinuationNextCertification,
    Interval, IComplex, CertificationResult, DistinctSolutionCertificates,
    SolutionCertificate, ExtendedSolutionCertificate, DistinctCertifiedSolutions
using HomotopyContinuationNext: @polyvar, System
using Aqua: Aqua
using JET: JET
using ExplicitImports:
    check_no_implicit_imports, check_no_stale_explicit_imports,
    check_all_explicit_imports_via_owners, check_all_qualified_accesses_via_owners
using CheckConcreteStructs: all_concrete

const Cert = HomotopyContinuationNextCertification

@testset "Quality" begin
    @testset "Aqua" begin
        # HomotopyContinuationNext is an intra-repo dev dependency with no
        # [compat] entry (its version is a prerelease), so skip it in the compat
        # check. Everything else is checked.
        Aqua.test_all(
            Cert;
            deps_compat = (; ignore = [:HomotopyContinuationNext]),
        )
    end

    @testset "CheckConcreteStructs" begin
        # Non-parametric structs defined in the certification package.
        for name in names(Cert; all = true)
            isdefined(Cert, name) || continue
            T = getfield(Cert, name)
            T isa Type || continue
            (isabstracttype(T) || T isa UnionAll) && continue
            isstructtype(T) || continue
            parentmodule(T) === Cert || continue
            @testset "$name" begin
                @test all_concrete(T; verbose = false)
            end
        end
        # Parametric structs — check concrete instantiations.
        @testset "IComplex{Float64}" begin
            @test all_concrete(IComplex{Float64}; verbose = false)
        end
        @testset "Interval{Float64}" begin
            @test all_concrete(Interval{Float64}; verbose = false)
        end
        @testset "CertificationResult{SolutionCertificate}" begin
            @test all_concrete(CertificationResult{SolutionCertificate}; verbose = false)
        end
        @testset "DistinctSolutionCertificates{SolutionCertificate}" begin
            @test all_concrete(
                DistinctSolutionCertificates{SolutionCertificate}; verbose = false,
            )
        end
        @testset "DistinctCertifiedSolutions (concrete instantiation)" begin
            @polyvar x y
            F = System([x^2 - 2, y^2 - 3])
            d = DistinctCertifiedSolutions(F, nothing)
            @test all_concrete(typeof(d); verbose = false)
        end
    end

    @testset "ExplicitImports" begin
        @test check_no_implicit_imports(Cert) === nothing
        @test check_no_stale_explicit_imports(Cert) === nothing
        @test check_all_explicit_imports_via_owners(Cert) === nothing
        @test check_all_qualified_accesses_via_owners(Cert) === nothing
    end

    @testset "JET" begin
        rep = JET.report_package(Cert; target_modules = (Cert,))
        @test isempty(JET.get_reports(rep))
    end
end
