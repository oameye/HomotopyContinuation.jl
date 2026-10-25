# Package call sites allowed to dispatch at runtime, keyed by `(function, path)`, each
# with its reason. A dispatch inside a dependency is charged to the package line that
# calls it (`call_site` in `test/helpers/jet_analysis.jl`). `test/quality/jet.jl`
# fails on runtime dispatch anywhere else and on an entry that is no longer reached;
# `test/quality/strict_mode/strict_mode.jl` accepts a `:typestable` failure only when
# all of its dispatch is charged to entries here. Every entry is a candidate for
# removal.

const FUNCTION_WRAPPER =
    "FunctionWrapper call; FunctionWrappers' cold `reinit_wrapper` branch for a null pointer dispatches"
const BOX = "RefValue{Any} box erases a factory or worker"
const CODEGEN = "builds a RuntimeGeneratedFunction from an expression and erases it behind a FunctionWrapper"
const PROGRESS = "ProgressMeter writes to `stdout`, an abstractly typed global, through untyped fields"
const MESSAGE = "formats an error or warning message"
const TASKS = "OhMyThreads `@tasks` checks its captures and builds its chunking keywords through runtime dispatch (JuliaFolds2/OhMyThreads.jl#179)"
const VARIABLE_NAME = "names a user variable through its own `string`/`Symbol`"

const EXEMPTIONS = Dict(
    (:evaluate!, "src/core/homotopy_evaluator.jl") => FUNCTION_WRAPPER,
    (:evaluate_and_jacobian!, "src/core/homotopy_evaluator.jl") => FUNCTION_WRAPPER,
    (:taylor!, "src/core/homotopy_evaluator.jl") => FUNCTION_WRAPPER,
    (:evaluate!, "src/core/system_evaluator.jl") => FUNCTION_WRAPPER,
    (:evaluate_and_jacobian!, "src/core/system_evaluator.jl") => FUNCTION_WRAPPER,
    (:taylor!, "src/core/system_evaluator.jl") => FUNCTION_WRAPPER,
    (:build_zero_parameter_taylor_fws, "src/core/system.jl") => FUNCTION_WRAPPER,
    (:weighted_stage_degrees, "src/core/composition_system.jl") => FUNCTION_WRAPPER,
    (:stage_equations, "src/core/composition_system.jl") => FUNCTION_WRAPPER,
    (:factorize!, "src/primitives/linear_algebra.jl") => FUNCTION_WRAPPER,
    (:ldiv!, "src/primitives/linear_algebra.jl") => FUNCTION_WRAPPER,
    (:track_path!, "src/solving/builder.jl") => FUNCTION_WRAPPER,
    (:track_paths, "src/solving/track_paths.jl") => FUNCTION_WRAPPER,
    (:build_compiled_evaluator, "src/model_kit/codegen.jl") => CODEGEN,
    (:build_taylor_fws, "src/model_kit/codegen.jl") => CODEGEN,
    (:clone_system_evaluator, "src/core/system_evaluator.jl") => BOX,
    (:fold_composition, "src/core/composition_system.jl") => BOX,
    (:PathBuilder, "src/solving/builder.jl") => BOX,
    (:retarget!, "src/solving/sweep.jl") => BOX,
    (:next_progress!, "src/solving/progress.jl") => PROGRESS,
    (:update_progress!, "src/solving/progress.jl") => PROGRESS,
    (:update_many_progress!, "src/solving/progress.jl") => PROGRESS,
    (:update_progress!, "src/solving/monodromy/solve.jl") => PROGRESS,
    (:solve, "src/solving/regeneration.jl") => PROGRESS,
    (:threaded_intersection!, "src/solving/regeneration.jl") => TASKS,
    (:support_mixed_volume, "src/solving/polyhedral.jl") => "MixedSubdivisions reports progress through ProgressMeter",
    (:compile_exec_instruction, "src/model_kit/interpreter.jl") => MESSAGE,
    (:monodromy_solve_body!, "src/solving/monodromy/solve.jl") => MESSAGE,
    (:polynomial_variable, "src/model_kit/expression.jl") => VARIABLE_NAME,
    (:variable_group_indices, "src/core/system.jl") => VARIABLE_NAME,
    (:compute_angular_part!, "src/solving/binomial_system.jl") => "`setprecision` stores the precision in a ScopedValue",
    (:log, "src/primitives/double_f64.jl") => "Base's generic `hypot` reaches a mixed-type `fma` error path",
)

# Functions in the allocation-free kernel files of
# `test/quality/strict_mode/strict_mode.jl` that are not kernels.
const CODE_GENERATION = "builds Taylor kernel expressions during code generation"
const ALLOCATING = "allocating convenience that returns a new array"

const NOT_KERNELS = Dict(
    :cauchy_product_exprs => CODE_GENERATION,
    :expr_sum => CODE_GENERATION,
    :taylor_hyperbolic_stmts => CODE_GENERATION,
    :taylor_pow_recurrence_expr => CODE_GENERATION,
    :taylor_tangent_stmts => CODE_GENERATION,
    :evaluate => ALLOCATING,
    :jacobian => ALLOCATING,
    :as_homotopy => "argument conversion for the allocating `evaluate` and `jacobian`",
    :evaluation_arguments => "argument conversion for the allocating `evaluate` and `jacobian`",
    :path_parameters => "argument conversion for the allocating `evaluate` and `jacobian`",
    :Homotopy => "the symbolic constructor builds its system through the dispatch exemptions",
)
