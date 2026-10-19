using Test
using HomotopyContinuationCertification: HomotopyContinuationCertification as Cert
using JET: JET

@testset "JET" begin
    rep = JET.report_package(Cert; target_modules = (Cert,))
    reports = filter(JET.get_reports(rep)) do r
        msg = string(r)
        !(contains(msg, "may be undefined") && contains(msg, "##"))
    end
    @test isempty(reports)
end
