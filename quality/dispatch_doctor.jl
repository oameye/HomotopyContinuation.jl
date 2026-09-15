using Preferences: set_preferences!

set_preferences!(
    "HomotopyContinuation",
    "dispatch_doctor_mode" => "error",
    "dispatch_doctor_codegen_level" => "min";
    force = true,
)

using DispatchDoctor
using HomotopyContinuation
using Test

@test DispatchDoctor.JULIA_OK
@stable default_mode = "error" default_codegen_level = "min" function _dispatch_doctor_negative_control_quality(flag::Bool)
    return flag ? 1 : "unstable"
end

@test_throws TypeInstabilityError _dispatch_doctor_negative_control_quality(true)

include(joinpath(@__DIR__, "..", "benchmark", "ttfx_workloads.jl"))

for workload in TTFXWorkloads.workloads()
    @info "DispatchDoctor workload" workload
    TTFXWorkloads.run(Val(workload))
end
