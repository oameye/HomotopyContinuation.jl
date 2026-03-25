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
        @show reports
        @test length(JET.get_reports(rep)) == 0
    end
end
