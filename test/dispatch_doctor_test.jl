using Test
using DispatchDoctor

@stable default_mode = "error" default_codegen_level = "min" function _dispatch_doctor_negative_control(flag::Bool)
    return flag ? 1 : "unstable"
end

@testset "DispatchDoctor activation" begin
    if DispatchDoctor.JULIA_OK
        @test_throws TypeInstabilityError _dispatch_doctor_negative_control(true)
    else
        # DispatchDoctor v0.4.28 deliberately disables instrumentation on Julia 1.13+.
        # The Julia 1.10 Core job is the activation/negative-control gate; Julia 1.13
        # is covered independently by StrictMode.
        @test VERSION >= v"1.13.0-DEV.0"
    end
end
