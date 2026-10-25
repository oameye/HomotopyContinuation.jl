using Test
using JET: JET
using HomotopyContinuation: HomotopyContinuation
using HomotopyContinuationCertification: HomotopyContinuationCertification as Cert

include(joinpath(dirname(dirname(@__DIR__)), "workloads.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "test", "helpers", "jet_analysis.jl"))
include(joinpath(@__DIR__, "..", "..", "..", "..", "test", "quality", "exemptions.jl"))

const TARGETS = (Cert,)

# Certification call sites allowed to dispatch at runtime, as in
# `test/quality/exemptions.jl`, whose package entries cover the calls into
# HomotopyContinuation.
const INTERVAL_TREES = "IntervalTrees' B-tree nodes hold abstractly typed fields"

const CERTIFICATION_EXEMPTIONS = Dict(
    (:_certify_impl, "lib/HomotopyContinuationCertification/src/certification.jl") => PROGRESS,
    (:distinct_certified_solutions!, "lib/HomotopyContinuationCertification/src/certification.jl") => PROGRESS,
    (:_draw_progress!, "lib/HomotopyContinuationCertification/src/iterator_certification.jl") => PROGRESS,
    (:_finish_progress!, "lib/HomotopyContinuationCertification/src/iterator_certification.jl") => PROGRESS,
    (:add_certificate!, "lib/HomotopyContinuationCertification/src/certification.jl") => INTERVAL_TREES,
    (:guaranteed_duplicate_certificate, "lib/HomotopyContinuationCertification/src/certification.jl") => INTERVAL_TREES,
    (:show, "lib/HomotopyContinuationCertification/src/certification.jl") => "Arblib formats a ball through a keyword `string`",
    (:foreach_path, "src/solving/result_iterator.jl") => TASKS,
)

foreach(name -> CertificationWorkloads.run(Val(name)), CertificationWorkloads.workloads())
const SPECIALIZATIONS = compiled_specializations(TARGETS)

@static if isempty(VERSION.prerelease)
    @testset "JET report_package" begin
        reports = JET.get_reports(JET.report_package(Cert; target_modules = TARGETS))
        test_no_reports(filter(!moshi_binding, reports))
    end

    test_jet_workloads(
        CertificationWorkloads.run, CertificationWorkloads.workloads(), SPECIALIZATIONS;
        targets = TARGETS, exemptions = CERTIFICATION_EXEMPTIONS,
        inherited_owners = (HomotopyContinuation,), inherited = EXEMPTIONS,
    )
end
