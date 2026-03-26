# Run all v2 comparison benchmarks.
# Usage: julia --project=benchmark benchmark/compare/runcompare.jl [category...]
#
# Categories: primitives, interpreter, tracking, all (default)
#
# Examples:
#   julia --project=benchmark benchmark/compare/runcompare.jl            # run all
#   julia --project=benchmark benchmark/compare/runcompare.jl tracking   # tracking only
#   julia --project=benchmark benchmark/compare/runcompare.jl primitives interpreter

include("common.jl")

categories = isempty(ARGS) ? ["all"] : ARGS

if "all" in categories
    categories = ["primitives", "interpreter", "tracking"]
end

for cat in categories
    include("$cat.jl")
end

println("\n" * "="^72)
println("  Done.")
println("="^72)
