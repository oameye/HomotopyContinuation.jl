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
            # Filter MP.variables false positive (abstract type dispatch at construction time)
            contains(msg, "variables") && contains(msg, "AbstractPolynomialLike") && return false
            # Filter Moshi @match generated variable warnings (false positives —
            # pattern matching variables are always defined in the matched branch)
            contains(msg, "may be undefined") && contains(msg, "##") && return false
            # Filter Moshi @derive Hash/Eq generated code (false positive union split)
            contains(msg, "variant_getfield") && return false
            return true
        end
        @test length(real_reports) == 0
    end
end
