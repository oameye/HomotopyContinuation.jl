using HomotopyContinuation
using StrictModeTest

include(joinpath(@__DIR__, "..", "benchmark", "ttfx_workloads.jl"))

for workload in TTFXWorkloads.workloads()
    @info "StrictMode workload" workload
    TTFXWorkloads.run(Val(workload))
end

const GUARANTEED = Set(
    [
        "core/abstract_types.jl",
        "core/symbolic_homotopy.jl",
        "core/system_evaluate.jl",
        "model_kit/cse.jl",
        "model_kit/instruction_sequence.jl",
        "model_kit/taylor.jl",
        "primitives/double_f64.jl",
        "solving/algorithm.jl",
        "solving/group_actions.jl",
        "solving/unique_points.jl",
        "solving/voronoi_tree.jl",
        "utils.jl",
    ]
)

function _package_files(f)
    files = String[]
    for m in methods(f)
        file = String(m.file)
        i = findlast("/src/", file)
        i === nothing && continue
        push!(files, file[(last(i) + 1):end])
    end
    return files
end

function in_guaranteed_layer(f)
    files = _package_files(f)
    return !isempty(files) && all(in(GUARANTEED), files)
end

StrictModeTest.test_compiled(
    HomotopyContinuation;
    guarantees = (:typestable,),
    only = in_guaranteed_layer,
)
