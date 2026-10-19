using Test
using JET: JET
using HomotopyContinuation
using StrictMode: StrictMode
using StrictModeTest

include(joinpath(@__DIR__, "..", "..", "..", "workloads", "workloads.jl"))
include(joinpath(@__DIR__, "..", "..", "helpers", "jet_analysis.jl"))
include(joinpath(@__DIR__, "..", "exemptions.jl"))

const HC = HomotopyContinuation

for workload in Workloads.workloads()
    @info "StrictMode workload" workload
    Workloads.run(Val(workload))
end

const SPECIALIZATIONS = compiled_specializations((HC,))

# A specialization that always throws infers `Union{}`, the exact type of a call that
# never returns. Dispatch is accepted when JET charges all of it to `EXEMPTIONS`.
# A callable struct has no instance to analyse it by; `test/quality/jet.jl` covers it.
function typestable_failures(mi)
    f = mi.specTypes.parameters[1]
    isdefined(f, :instance) || return StrictMode.StrictFinding[]
    types = Tuple(mi.specTypes.parameters[2:end])
    findings = StrictModeTest.proof_findings(f.instance, types; guarantees = (:typestable,))
    return filter(findings) do finding
        finding.status === :fail || return false
        startswith(finding.reason, "return type Union{} ") && return false
        dispatch_only = startswith(finding.reason, "internal instability / runtime dispatch (") &&
            !occursin("unoptimized frame", finding.reason)
        return !(dispatch_only && isempty(unexempted_dispatch(mi.specTypes, (HC,), EXEMPTIONS)))
    end
end

@testset "every compiled specialization is typestable" begin
    @info "typestable checks $(length(SPECIALIZATIONS)) specialization(s)"
    @test !isempty(SPECIALIZATIONS)
    failures = mapreduce(typestable_failures, vcat, SPECIALIZATIONS)
    isempty(failures) ||
        @info join(("$(f.func)$(f.signature) at $(f.file):$(f.line): $(f.reason)" for f in failures), "\n")
    @test isempty(failures)
end

# The allocation-free kernels.
const STATIC_CORE = Set(
    [
        "core/abstract_types.jl",
        "core/linear_parameter_homotopy.jl",
        "core/symbolic_homotopy.jl",
        "core/system_evaluate.jl",
        "core/toric_homotopy.jl",
        "model_kit/taylor.jl",
        "primitives/double_f64.jl",
        "primitives/norms.jl",
    ]
)

function in_static_core(f)
    files = String[]
    for m in methods(f)
        file = String(m.file)
        i = findlast("/src/", file)
        i === nothing || push!(files, file[(last(i) + 1):end])
    end
    return !isempty(files) && all(in(STATIC_CORE), files)
end

@info "STATIC_CORE proved $(
    length(
        StrictModeTest.test_compiled(
            HC;
            guarantees = (:noalloc, :noboxing, :trim_compatible),
            only = in_static_core,
            exempt = collect(keys(NOT_KERNELS)),
        )
    )
) signature(s)"
