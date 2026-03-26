using Test
using JET
using HomotopyContinuationNext

@testset "JET.jl" begin
    @static if isempty(VERSION.prerelease)
        using JET
        rep = JET.report_package(
            HomotopyContinuationNext;
            target_modules = (HomotopyContinuationNext,),
        )
        reports = JET.get_reports(rep)
        # Filter out known false positives:
        # MP.variables is only defined on concrete DynamicPolynomials types,
        # not on abstract MP.AbstractPolynomialLike. Construction-time code
        # that calls it is correct at runtime but unresolvable by JET.
        real_reports = filter(reports) do r
            msg = string(r)
            !contains(msg, "variables") || !contains(msg, "AbstractPolynomialLike")
        end
        @test length(real_reports) == 0
    end
end
