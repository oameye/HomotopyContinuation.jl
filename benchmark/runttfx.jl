# Measure first-call (TTFX) latency per workload, one fresh Julia process each.
#
# Usage:
#   julia --project=benchmark benchmark/runttfx.jl                    # all workloads
#   julia --project=benchmark benchmark/runttfx.jl monodromy_serial   # a subset
#   THREADS=1 julia --project=benchmark benchmark/runttfx.jl          # override thread count
#
# Steady-state numbers live in runbenchmarks.jl; this harness only measures cold
# latency, so every workload must run in a process that has never compiled it.

const WORKLOAD_FILE = joinpath(@__DIR__, "ttfx_workloads.jl")
const RESULT_PREFIX = "TTFX_RESULT"
const OUTPUT_FILE = joinpath(dirname(@__DIR__), "ttfx_output.json")

# ── Child mode: time package load and the first call, then report one line ────

if !isempty(ARGS) && first(ARGS) == "--child"
    name = Symbol(ARGS[2])
    # Top-level include so the `using` inside the workload module is what gets timed.
    t_load = @elapsed include(WORKLOAD_FILE)
    t_first = @elapsed TTFXWorkloads.run(name)
    println(RESULT_PREFIX, " ", name, " ", t_load, " ", t_first)
    exit(0)
end

# ── Parent mode: spawn one child per workload and collect ─────────────────────

include(WORKLOAD_FILE)

const THREADS = get(ENV, "THREADS", "4")

function requested_workloads()::Vector{Symbol}
    isempty(ARGS) && return collect(TTFXWorkloads.workloads())
    names = Symbol.(ARGS)
    known = TTFXWorkloads.workloads()
    for name in names
        name in known || error("unknown workload $name; known: $(join(known, ", "))")
    end
    return names
end

function measure(name::Symbol)::Tuple{Float64, Float64}
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -t $(THREADS) $(@__FILE__) --child $(name)`
    out = try
        read(cmd, String)
    catch err
        @warn "workload failed" workload = name exception = err
        return (NaN, NaN)
    end
    for line in eachline(IOBuffer(out))
        startswith(line, RESULT_PREFIX) || continue
        fields = split(line)
        return (parse(Float64, fields[3]), parse(Float64, fields[4]))
    end
    @warn "workload produced no result line" workload = name output = out
    return (NaN, NaN)
end

function write_json(path::String, rows::Vector{Tuple{Symbol, Float64, Float64}})::Nothing
    open(path, "w") do io
        println(io, "{")
        println(io, "  \"threads\": ", THREADS, ",")
        println(io, "  \"julia\": \"", VERSION, "\",")
        println(io, "  \"workloads\": {")
        for (i, (name, load, first_call)) in enumerate(rows)
            comma = i == length(rows) ? "" : ","
            println(
                io, "    \"", name, "\": {\"load\": ", load,
                ", \"first_call\": ", first_call, "}", comma,
            )
        end
        println(io, "  }")
        println(io, "}")
    end
    return nothing
end

names = requested_workloads()
rows = Tuple{Symbol, Float64, Float64}[]
width = maximum(length ∘ string, names)

println("TTFX: fresh process per workload, -t ", THREADS, ", ", length(names), " workloads")
println()
println(rpad("workload", width), "   load (s)   first call (s)")
println("-"^(width + 28))

for (i, name) in enumerate(names)
    status = string("(", i, "/", length(names), ") ", name)
    print(status)
    load, first_call = measure(name)
    push!(rows, (name, load, first_call))
    print("\r", " "^length(status), "\r")
    println(
        rpad(name, width), "   ",
        lpad(round(load; digits = 3), 8), "   ",
        lpad(round(first_call; digits = 3), 14),
    )
end

println("-"^(width + 28))
first_calls = filter(!isnan, [r[3] for r in rows])
println(
    rpad("total first call", width), "   ", lpad("", 8), "   ",
    lpad(round(sum(first_calls); digits = 3), 14),
)
println(
    rpad("slowest first call", width), "   ", lpad("", 8), "   ",
    lpad(round(maximum(first_calls); digits = 3), 14),
    "  (", rows[argmax([isnan(r[3]) ? -Inf : r[3] for r in rows])][1], ")",
)

write_json(OUTPUT_FILE, rows)
println()
println("wrote ", OUTPUT_FILE)
