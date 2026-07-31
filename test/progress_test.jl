using Test
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: solve, System, TotalDegree, Serial, path_results
using DynamicPolynomials: @polyvar

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

@testset "solve progress bar" begin
    @polyvar x y
    F = System([x^2 + y^2 - 1, x + y - 1])   # 2 total-degree paths

    @testset "make_progress renders the tracking bar to stdout" begin
        # Force a print (mid-progress, so ProgressMeter's completion guard is
        # satisfied) to check the bar text deterministically, independent of the
        # 0.3s startup delay that suppresses the bar for fast solves.
        _, out = run_capture() do
            p = HC.make_progress(3, true)
            HC.ProgressMeter.update!(p, 2; force = true)
        end
        @test occursin("Tracking", out)
        @test occursin("paths", out)
    end

    @testset "make_progress(n, false) is a no-op sentinel" begin
        @test HC.make_progress(2, false) === nothing
        # update_progress! on the sentinel must be a silent no-op.
        r = first(path_results(solve(F, TotalDegree(; seed = UInt32(1), show_progress = false), Serial())))
        _, out = run_capture() do
            HC.update_progress!(HC.make_progress(2, false), 1, HC.ProgressStats(), r)
        end
        @test isempty(out)
    end

    @testset "record! tallies non-singular / singular / real endpoints" begin
        res = solve(F, TotalDegree(; seed = UInt32(1), show_progress = false), Serial())
        stats = HC.ProgressStats()
        for pr in path_results(res)
            HC.record!(stats, pr)
        end
        # F has two non-singular solutions; the two successful paths should be
        # counted as non-singular (both real here).
        @test stats.nonsingular == HC.nnonsingular(res)
        @test stats.singular == 0
        @test stats.nonsingular_real == HC.nreal(res)
    end

    @testset "showvalues includes the live solution counts" begin
        stats = HC.ProgressStats()
        stats.nonsingular = 3
        stats.nonsingular_real = 2
        stats.singular = 1
        stats.singular_real = 1
        sv = HC._showvalues(stats, 4)
        labels = join(first.(sv), " ")
        @test occursin("paths tracked", labels)
        @test occursin("non-singular", labels)
        @test occursin("singular endpoints", labels)
        @test occursin("total solutions", labels)
        # total line reflects 3+1 solutions, 2+1 real
        total_line = last(sv)
        @test occursin("4", total_line[2])
        @test occursin("3", total_line[2])
    end

    @testset "show_progress = false keeps solve silent" begin
        _, out = run_capture() do
            solve(F, TotalDegree(; seed = UInt32(1), show_progress = false), Serial())
        end
        @test !occursin("Tracking", out)
    end

    @testset "show_progress = true is accepted by serial and threaded solve" begin
        res_s, _ = run_capture() do
            solve(F, TotalDegree(; seed = UInt32(1), show_progress = true), Serial())
        end
        @test HC.nsolutions(res_s) >= 1

        res_t, _ = run_capture() do
            solve(F, TotalDegree(; seed = UInt32(1), show_progress = true))
        end
        @test HC.nsolutions(res_t) >= 1
    end
end
