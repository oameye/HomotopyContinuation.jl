# Measure the first execution of each workload: building its problem and
# solving it once, in a fresh Julia process per sample. Package precompilation
# and loading happen before the timer starts and are not reported here.
#
# Usage:
#   julia --project=benchmark benchmark/ttfx/runttfx.jl
#   julia --project=benchmark benchmark/ttfx/runttfx.jl monodromy_serial
#   julia --project=benchmark benchmark/ttfx/runttfx.jl monodromy geometry
#   TTFX_THREADS=1 TTFX_REPEATS=3 julia --project=benchmark benchmark/ttfx/runttfx.jl

const WORKLOAD_FILE = normpath(joinpath(@__DIR__, "..", "..", "workloads", "workloads.jl"))
const RESULT_PREFIX = "TTFX_RESULT"
const OUTPUT_FILE = normpath(joinpath(@__DIR__, "..", "..", "ttfx_output.json"))

if !isempty(ARGS) && first(ARGS) == "--child"
    using HomotopyContinuation
    include(WORKLOAD_FILE)
    name = Symbol(ARGS[2])
    first_call = @elapsed Workloads.run(name)
    println(RESULT_PREFIX, " ", first_call)
    exit(0)
end

include(WORKLOAD_FILE)

const TTFX_THREADS = parse(Int, get(ENV, "TTFX_THREADS", "1"))
const TTFX_REPEATS = parse(Int, get(ENV, "TTFX_REPEATS", "1"))
TTFX_THREADS > 0 || error("TTFX_THREADS must be positive")
TTFX_REPEATS > 0 || error("TTFX_REPEATS must be positive")

# Arguments are workload or category names; no arguments runs every workload.
function requested_workloads()::Vector{Symbol}
    isempty(ARGS) && return collect(Workloads.workloads())
    known = Workloads.workloads()
    names = Symbol[]
    for arg in Symbol.(ARGS)
        if haskey(Workloads.CATEGORIES, arg)
            append!(names, Workloads.CATEGORIES[arg])
        elseif arg in known
            push!(names, arg)
        else
            error(
                "unknown workload or category $arg; categories: ",
                join(keys(Workloads.CATEGORIES), ", "),
            )
        end
    end
    return unique(names)
end

function median_value(values::Vector{Float64})::Float64
    isempty(values) && return NaN
    sorted = sort(values)
    n = length(sorted)
    isodd(n) && return sorted[(n + 1) ÷ 2]
    return (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
end

function measure_once(name::Symbol)::Float64
    project = dirname(Base.active_project())
    cmd = `$(Base.julia_cmd()) --project=$(project) -t $(TTFX_THREADS) $(@__FILE__) --child $(name)`
    io = IOBuffer()
    process = run(pipeline(ignorestatus(cmd); stdout = io, stderr = io))
    output = String(take!(io))
    if success(process)
        for line in eachline(IOBuffer(output))
            startswith(line, RESULT_PREFIX) && return parse(Float64, split(line)[2])
        end
    end
    println(stderr, "\nTTFX workload failed: $name")
    print(stderr, output)
    return NaN
end

function write_json(path::String, rows::Vector{Tuple{Symbol, Vector{Float64}}})::Nothing
    successful = filter(row -> !isempty(row[2]), rows)
    open(path, "w") do io
        println(io, "[")
        for (i, (name, samples)) in enumerate(successful)
            comma = i == length(successful) ? "" : ","
            println(
                io, "  {\"name\": \"", name, "\", \"unit\": \"s\", \"value\": ",
                median_value(samples), "}", comma,
            )
        end
        println(io, "]")
    end
    return nothing
end

function main()
    names = requested_workloads()
    width = maximum(length ∘ string, names)
    rows = Tuple{Symbol, Vector{Float64}}[]
    failed = Symbol[]

    println(
        "TTFX first execution: fresh process per sample, -t ", TTFX_THREADS, ", ",
        TTFX_REPEATS, " sample(s), ", length(names), " workload(s)",
    )
    println()
    println(rpad("workload", width), "   first execution (s)")
    println("-"^(width + 22))

    for (i, name) in enumerate(names)
        status = string("(", i, "/", length(names), ") ", name)
        print(status)
        samples = Float64[]
        for _ in 1:TTFX_REPEATS
            sample = measure_once(name)
            isnan(sample) || push!(samples, sample)
        end
        push!(rows, (name, samples))
        length(samples) == TTFX_REPEATS || push!(failed, name)
        print("\r", " "^length(status), "\r")
        value = isempty(samples) ? "FAILED" : string(round(median_value(samples); digits = 3))
        println(rpad(name, width), "   ", lpad(value, 19))
    end

    write_json(OUTPUT_FILE, rows)
    println()
    println("wrote ", OUTPUT_FILE)

    return if !isempty(failed)
        println(stderr, "TTFX failed for ", length(failed), " workload(s): ", join(failed, ", "))
        exit(1)
    end
end

main()
