# Run all benchmarks
# Usage: julia --project=benchmark benchmark/runbenchmarks.jl

using BenchmarkTools

const SUITE = BenchmarkGroup()

# Add benchmark files here as phases are implemented:
include("primitives.jl")
benchmark_primitives!(SUITE)

include("interpreter.jl")
benchmark_interpreter!(SUITE)

BenchmarkTools.tune!(SUITE)
results = BenchmarkTools.run(SUITE; verbose = true)
display(median(results))

BenchmarkTools.save("benchmarks_output.json", median(results))
