# Measure cold first-call latency with one fresh Julia process per sample.
#
# Usage:
#   julia --project=. benchmark/runttfx.jl
#   julia --project=. benchmark/runttfx.jl monodromy_serial
#   TTFX_THREADS=1 TTFX_REPEATS=3 julia --project=. benchmark/runttfx.jl
#
# The parent process only coordinates measurements. Each sample runs in a new
# Julia process so package loading and the first workload execution are cold.

const WORKLOAD_FILE = joinpath(@__DIR__, "ttfx_workloads.jl")
const RESULT_PREFIX = "TTFX_RESULT"
const OUTPUT_FILE = joinpath(dirname(@__DIR__), "ttfx_output.json")
const BENCHMARK_OUTPUT_FILE = joinpath(dirname(@__DIR__), "ttfx_benchmark.json")

# Child mode: measure package loading, workload-definition setup, and first call.
if !isempty(ARGS) && first(ARGS) == "--child"
    name = Symbol(ARGS[2])
    package_load = @elapsed @eval using HomotopyContinuationNext
    setup = @elapsed include(WORKLOAD_FILE)
    first_call = @elapsed TTFXWorkloads.run(name)
    println(RESULT_PREFIX, " ", name, " ", package_load, " ", setup, " ", first_call)
    exit(0)
end

# Parent mode: enumerate workloads and spawn independent child processes.
include(WORKLOAD_FILE)

const TTFX_THREADS = parse(Int, get(ENV, "TTFX_THREADS", "1"))
const TTFX_REPEATS = parse(Int, get(ENV, "TTFX_REPEATS", "1"))
TTFX_THREADS > 0 || error("TTFX_THREADS must be positive")
TTFX_REPEATS > 0 || error("TTFX_REPEATS must be positive")

struct WorkloadMeasurement
    name::Symbol
    package_load::Vector{Float64}
    setup::Vector{Float64}
    first_call::Vector{Float64}
    failures::Int
end

function requested_workloads()::Vector{Symbol}
    isempty(ARGS) && return collect(TTFXWorkloads.workloads())
    names = Symbol.(ARGS)
    known = TTFXWorkloads.workloads()
    for name in names
        name in known || error("unknown workload $name; known: $(join(known, ", "))")
    end
    return names
end

function median_value(values::Vector{Float64})::Float64
    isempty(values) && return NaN
    sorted = sort(values)
    n = length(sorted)
    isodd(n) && return sorted[(n + 1) ÷ 2]
    return (sorted[n ÷ 2] + sorted[n ÷ 2 + 1]) / 2
end

function measure_once(name::Symbol, repeat::Int)
    project = dirname(Base.active_project())
    cmd = `$(Base.julia_cmd()) --project=$(project) -t $(TTFX_THREADS) $(@__FILE__) --child $(name)`
    io = IOBuffer()
    process = run(pipeline(ignorestatus(cmd); stdout = io, stderr = io))
    output = String(take!(io))

    if !success(process)
        println(stderr, "\nTTFX workload failed: $name (sample $repeat/$TTFX_REPEATS)")
        print(stderr, output)
        return nothing
    end

    for line in eachline(IOBuffer(output))
        startswith(line, RESULT_PREFIX) || continue
        fields = split(line)
        length(fields) == 5 || break
        return (
            parse(Float64, fields[3]),
            parse(Float64, fields[4]),
            parse(Float64, fields[5]),
        )
    end

    println(
        stderr,
        "\nTTFX workload produced no result: $name (sample $repeat/$TTFX_REPEATS)",
    )
    print(stderr, output)
    return nothing
end

function measure(name::Symbol)::WorkloadMeasurement
    package_load = Float64[]
    setup = Float64[]
    first_call = Float64[]
    failures = 0

    for repeat in 1:TTFX_REPEATS
        sample = measure_once(name, repeat)
        if sample === nothing
            failures += 1
        else
            load_time, setup_time, first_call_time = sample
            push!(package_load, load_time)
            push!(setup, setup_time)
            push!(first_call, first_call_time)
        end
    end

    return WorkloadMeasurement(name, package_load, setup, first_call, failures)
end

function write_float_or_null(io::IO, value::Float64)
    return isnan(value) ? print(io, "null") : print(io, value)
end

function write_float_array(io::IO, values::Vector{Float64})
    print(io, "[")
    for (i, value) in enumerate(values)
        i > 1 && print(io, ", ")
        print(io, value)
    end
    return print(io, "]")
end

function write_detailed_json(path::String, rows::Vector{WorkloadMeasurement})::Nothing
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"threads\": ", TTFX_THREADS, ",")
        println(io, "  \"repeats\": ", TTFX_REPEATS, ",")
        println(io, "  \"julia\": \"", VERSION, "\",")
        println(io, "  \"workloads\": {")
        for (i, row) in enumerate(rows)
            comma = i == length(rows) ? "" : ","
            println(io, "    \"", row.name, "\": {")
            println(
                io,
                "      \"ok\": ",
                row.failures == 0 ? "true" : "false",
                ",",
            )
            println(io, "      \"failures\": ", row.failures, ",")
            print(io, "      \"package_load_samples\": ")
            write_float_array(io, row.package_load)
            println(io, ",")
            print(io, "      \"setup_samples\": ")
            write_float_array(io, row.setup)
            println(io, ",")
            print(io, "      \"first_call_samples\": ")
            write_float_array(io, row.first_call)
            println(io, ",")
            print(io, "      \"package_load_median\": ")
            write_float_or_null(io, median_value(row.package_load))
            println(io, ",")
            print(io, "      \"setup_median\": ")
            write_float_or_null(io, median_value(row.setup))
            println(io, ",")
            print(io, "      \"first_call_median\": ")
            write_float_or_null(io, median_value(row.first_call))
            println(io)
            println(io, "    }", comma)
        end
        println(io, "  }")
        println(io, "}")
    end
    return nothing
end

function write_benchmark_json(path::String, rows::Vector{WorkloadMeasurement})::Nothing
    successful = filter(row -> row.failures == 0, rows)
    package_loads = reduce(
        vcat, (row.package_load for row in successful); init = Float64[]
    )
    entries = Tuple{String, Float64}[]
    if !isempty(package_loads)
        push!(entries, ("package_load", median_value(package_loads)))
    end
    for row in successful
        push!(entries, ("first_call/$(row.name)", median_value(row.first_call)))
    end

    open(path, "w") do io
        println(io, "[")
        for (i, (name, value)) in enumerate(entries)
            comma = i == length(entries) ? "" : ","
            println(
                io,
                "  {\"name\": \"", name, "\", \"unit\": \"s\", \"value\": ", value, "}",
                comma,
            )
        end
        println(io, "]")
    end
    return nothing
end

function main()
    names = requested_workloads()
    rows = WorkloadMeasurement[]
    width = maximum(length ∘ string, names)

    println(
        "TTFX: fresh process per sample, -t ", TTFX_THREADS, ", ", TTFX_REPEATS,
        " sample(s), ", length(names), " workload(s)",
    )
    println()
    println(rpad("workload", width), "   package (s)   setup (s)   first call (s)   status")
    println("-"^(width + 51))

    for (i, name) in enumerate(names)
        status = string("(", i, "/", length(names), ") ", name)
        print(status)
        row = measure(name)
        push!(rows, row)
        print("\r", " "^length(status), "\r")

        package_load = median_value(row.package_load)
        setup = median_value(row.setup)
        first_call = median_value(row.first_call)
        state = row.failures == 0 ? "ok" : "FAILED ($(row.failures)/$TTFX_REPEATS)"
        println(
            rpad(name, width), "   ",
            lpad(round(package_load; digits = 3), 11), "   ",
            lpad(round(setup; digits = 3), 9), "   ",
            lpad(round(first_call; digits = 3), 14), "   ",
            state,
        )
    end

    write_detailed_json(OUTPUT_FILE, rows)
    write_benchmark_json(BENCHMARK_OUTPUT_FILE, rows)
    println()
    println("wrote ", OUTPUT_FILE)
    println("wrote ", BENCHMARK_OUTPUT_FILE)

    failed = filter(row -> row.failures > 0, rows)
    return if !isempty(failed)
        println(
            stderr,
            "TTFX failed for ",
            length(failed),
            " workload(s): ",
            join(getfield.(failed, :name), ", "),
        )
        exit(1)
    end
end

main()
