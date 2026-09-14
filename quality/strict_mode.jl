using HomotopyContinuation
using StrictModeTest

include(joinpath(@__DIR__, "..", "benchmark", "ttfx_workloads.jl"))

for workload in TTFXWorkloads.workloads()
    @info "StrictMode workload" workload
    TTFXWorkloads.run(Val(workload))
end

StrictModeTest.test_compiled(
    HomotopyContinuation;
    guarantees = (:typestable,),
)
