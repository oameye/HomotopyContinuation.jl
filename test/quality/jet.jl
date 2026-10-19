using Test
using JET: JET
using HomotopyContinuation: HomotopyContinuation
using Distributed: Distributed
using SemialgebraicSets: SemialgebraicSets
using Serialization: Serialization

include(joinpath(dirname(dirname(@__DIR__)), "workloads", "workloads.jl"))
include(joinpath(dirname(@__DIR__), "helpers", "jet_analysis.jl"))
include(joinpath(@__DIR__, "exemptions.jl"))

const HC = HomotopyContinuation
const EXTENSIONS = (:DistributedExt, :SemialgebraicSetsExt)
const TARGETS = (HC, map(ext -> Base.get_extension(HC, ext), EXTENSIONS)...)

# `FixedParameterHomotopy` forwards to methods the wrapped user homotopy must define.
function extension_contract(report)
    report isa JET.MethodErrorReport || return false
    method = report.vst[end].linfo.def
    return method isa Method && occursin("FixedParameterHomotopy", string(method.sig))
end

errors(result) = filter(r -> !moshi_binding(r) && !extension_contract(r), JET.get_reports(result))

foreach(name -> Workloads.run(Val(name)), Workloads.workloads())
const SPECIALIZATIONS = compiled_specializations(TARGETS)

@static if isempty(VERSION.prerelease)
    @testset "JET report_package" begin
        test_no_reports(errors(JET.report_package(HC; target_modules = (HC,))))
        @testset "$ext" for ext in EXTENSIONS
            m = Base.get_extension(HC, ext)
            @test m isa Module
            test_no_reports(errors(JET.report_package(m; target_modules = (m,))))
        end
    end

    test_jet_workloads(
        Workloads.run, Workloads.workloads(), SPECIALIZATIONS;
        targets = TARGETS, exemptions = EXEMPTIONS, ignore = extension_contract,
    )
end
