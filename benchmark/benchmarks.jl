using BenchmarkTools
using HomotopyContinuationNext

const SUITE = BenchmarkGroup()

# Add benchmark files here as features are implemented:
# include("solve.jl")
# benchmark_solve!(SUITE)

BenchmarkTools.tune!(SUITE)
results = BenchmarkTools.run(SUITE; verbose = true)
display(median(results))

BenchmarkTools.save("benchmarks_output.json", median(results))
