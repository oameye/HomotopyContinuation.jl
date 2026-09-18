using Test
using DispatchDoctor

@stable default_mode = "error" default_codegen_level = "min" function _dispatch_doctor_negative_control(flag::Bool)
    return flag ? 1 : "unstable"
end

@testset "DispatchDoctor activation" begin
    if DispatchDoctor.JULIA_OK
        @test_throws TypeInstabilityError _dispatch_doctor_negative_control(true)
    else
        # DispatchDoctor lifted its `JULIA_OK` cap past 1.13 in 0.4.29, which compat
        # requires, so this branch is unreachable on a resolved environment.
        @test VERSION >= v"1.14.0-DEV.0"
    end
end
