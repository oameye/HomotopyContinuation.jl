using Test
using JET
using HomotopyContinuation

@testset "JET.jl" begin
    @static if isempty(VERSION.prerelease)
        using JET
        rep = JET.report_package(
            HomotopyContinuation;
            target_modules = (HomotopyContinuation,),
        )
        reports = JET.get_reports(rep)
        # Filter out known false positives:
        # MP.variables is only defined on concrete DynamicPolynomials types,
        # not on abstract MP.AbstractPolynomialLike. Construction-time code
        # that calls it is correct at runtime but unresolvable by JET.
        real_reports = filter(reports) do r
            msg = string(r)
            contains(msg, "variables") && contains(msg, "AbstractPolynomialLike") && return false
            contains(msg, "may be undefined") && contains(msg, "##") && return false
            contains(msg, "variant_getfield") && return false
            contains(msg, "FixedParameterHomotopy") &&
                contains(msg, "AbstractHomotopy") && return false
            return true
        end
        @test length(real_reports) == 0
    end
end
