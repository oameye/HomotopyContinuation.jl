using Test
using DispatchDoctor

@stable default_mode = "error" default_codegen_level = "min" function _dispatch_doctor_negative_control(flag::Bool)
    return flag ? 1 : "unstable"
end

@testset "DispatchDoctor activation" begin
    @test DispatchDoctor.JULIA_OK
    @test_throws TypeInstabilityError _dispatch_doctor_negative_control(true)
end
