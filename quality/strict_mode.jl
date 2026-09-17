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
        "core/linear_subspace.jl",
        "core/symbolic_homotopy.jl",
        "core/system_evaluate.jl",
        "model_kit/cse.jl",
        "model_kit/expression.jl",
        "model_kit/instruction_sequence.jl",
        "model_kit/sexpr.jl",
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

const STATIC_CORE = Set(
    [
        "core/abstract_types.jl",
        "core/symbolic_homotopy.jl",
        "core/system_evaluate.jl",
        "model_kit/taylor.jl",
        "primitives/double_f64.jl",
        "primitives/norms.jl",
    ]
)

const CODEGEN_HELPERS = (
    HomotopyContinuation._cauchy_product_exprs,
    HomotopyContinuation._expr_sum,
    HomotopyContinuation._taylor_hyperbolic_stmts,
    HomotopyContinuation._taylor_pow_recurrence_expr,
    HomotopyContinuation._taylor_tangent_stmts,
)

function in_static_core(f)
    files = _package_files(f)
    return !isempty(files) && all(in(STATIC_CORE), files)
end

StrictModeTest.test_compiled(
    HomotopyContinuation;
    guarantees = (:noalloc, :trim_compatible),
    only = in_static_core,
    exempt = CODEGEN_HELPERS,
)
