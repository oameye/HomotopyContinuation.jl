using Test

@testset "HomotopyContinuationNextCertification" begin
    include("interval_arithmetic_test.jl")
    include("acb_interpreter_test.jl")
    include("certification_test.jl")
    include("iterator_certification_test.jl")
    include("monodromy_certification_test.jl")
    include("export_surface_test.jl")
    include("quality_test.jl")
end
