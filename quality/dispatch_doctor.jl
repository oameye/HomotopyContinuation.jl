# Run with `--project=test/strict`, whose `LocalPreferences.toml` puts the
# package-wide `@stable` contract into hard-error mode. `test/strict/runtests.jl`
# covers the suite; this covers first-call paths the suite does not reach.
using DispatchDoctor
using HomotopyContinuation
using Preferences: load_preference
using Test

@test DispatchDoctor.JULIA_OK
@test load_preference(HomotopyContinuation, "dispatch_doctor_mode", "disable") == "error"

@stable default_mode = "error" default_codegen_level = "min" function _dispatch_doctor_negative_control_quality(flag::Bool)
    return flag ? 1 : "unstable"
end

@test_throws TypeInstabilityError _dispatch_doctor_negative_control_quality(true)

include(joinpath(@__DIR__, "..", "benchmark", "ttfx_workloads.jl"))

for workload in TTFXWorkloads.workloads()
    @info "DispatchDoctor workload" workload
    TTFXWorkloads.run(Val(workload))
end
