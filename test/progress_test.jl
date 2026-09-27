using Test
using HomotopyContinuation

# ProgressMeter writes to `stdout`, which `redirect_stdout` can only capture via
# an fd-backed stream (not an IOBuffer). Use a temp file and read it back.
function run_capture(f)
    return mktemp() do _path, io
        result = redirect_stdout(io) do
            f()
        end
        flush(io)
        seekstart(io)
        return (result, read(io, String))
    end
end

@testset "solve progress" begin
    @polyvar x y
    F = System([x^2 + y^2 - 1, x + y - 1])

    @testset "show_progress = false keeps solve silent" begin
        result, out = run_capture() do
            solve(F, TotalDegree(; seed = UInt32(1), show_progress = false), Serial())
        end
        @test nsolutions(result) == 2
        @test !occursin("Tracking", out)
    end

    @testset "show_progress = true is accepted by serial and threaded solve" begin
        serial, _ = run_capture() do
            solve(F, TotalDegree(; seed = UInt32(1), show_progress = true), Serial())
        end
        threaded, _ = run_capture() do
            solve(F, TotalDegree(; seed = UInt32(1), show_progress = true), Threaded())
        end
        @test nsolutions(serial) == 2
        @test nsolutions(threaded) == 2
    end
end
