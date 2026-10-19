using Test
using HomotopyContinuationCertification: HomotopyContinuationCertification as Cert,
    Interval, IComplex, CertificationResult, DistinctSolutionCertificates,
    SolutionCertificate, DistinctCertifiedSolutions
using HomotopyContinuation: @polyvar, System
using CheckConcreteStructs: all_concrete

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
    # Parametric structs: check concrete instantiations.
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
