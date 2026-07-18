module HomotopyContinuationNext

using LinearAlgebra: LinearAlgebra
using Random: Random
using Printf: Printf
using ProgressMeter: ProgressMeter

using EnumX: @enumx
using Moshi.Data: @data, variant_storage, variant_storage_type
using Moshi.Derive: @derive
using MultivariatePolynomials: MultivariatePolynomials
using DynamicPolynomials: DynamicPolynomials, @polyvar
using FixedSizeArrays: FixedSizeArray
import FunctionWrappers: FunctionWrapper
using CommonSolve: CommonSolve
using MixedSubdivisions: MixedSubdivisions
using OhMyThreads: @tasks, @set, @local
using RuntimeGeneratedFunctions: RuntimeGeneratedFunctions, @RuntimeGeneratedFunction
RuntimeGeneratedFunctions.init(@__MODULE__)

@enumx CompileMode::Int8 begin
    INTERPRETED
    COMPILED
    COMPILED_ALL
end

export @polyvar, solve, System
export CompileMode
export solutions, real_solutions, nsolutions, nreal, nsingular, nnonsingular, nat_infinity
export nexcess_solutions, nfailed
export nresults, results, multiplicity
export is_success, is_singular, is_nonsingular, is_at_infinity, is_real, is_excess_solution
export is_failed, is_finite
export TotalDegree, Polyhedral, Result, PathResult
export path_results, seed, ntracked, failed, at_infinity, nonsingular, singular
export statistics, ResultStatistics
export solution, accuracy, residual, steps, accepted_steps, rejected_steps
export winding_number, condition_jacobian, last_path_point
export path_number, start_solution, valuation
export EndgameOptions, EndgameTracker
export ParameterHomotopy
export GroupActions, SymmetricGroup
export UniquePoints, search_in_radius, add!, multiplicities, unique_points
export LinearSubspace, ExtrinsicDescription, IntrinsicDescription, Intrinsic, Extrinsic
export intrinsic, extrinsic, is_linear, dim, codim, ambient_dim
export rand_subspace, translate, geodesic, geodesic_distance, coord_change
export IntrinsicSubspaceHomotopy, ExtrinsicSubspaceHomotopy, set_subspaces!
export AffineChartHomotopy, on_affine_chart, linear_subspace_homotopy
export find_start_pair, monodromy_solve, verify_solution_completeness
export MonodromyOptions, MonodromyResult, is_heuristic_stop, permutations, trace
export newton, NewtonResult, NewtonCache, NewtonReturnCode
export Serial, Threaded

const MP = MultivariatePolynomials
# Concrete type aliases — FixedSizeVector{T} alone is NOT concrete because
# the Mem parameter is free. On Julia 1.11+ the backing is Memory{T}.
const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

include("primitives/double_f64.jl")
include("utils.jl")
include("primitives/norms.jl")
include("primitives/linear_algebra.jl")

include("model_kit/taylor.jl")
include("model_kit/operations.jl")
include("model_kit/instruction_sequence.jl")

# model_kit/sexpr.jl — SExpr types, hash/==, canonicalization, poly_to_sexpr
include("model_kit/sexpr.jl")

# model_kit/cse.jl — SymEngine-style CSE algorithm (opt_cse + tree_cse)
include("model_kit/cse.jl")

# model_kit/tape_compiler.jl — TapeCompiler, compile_to_instructions
include("model_kit/tape_compiler.jl")

include("model_kit/polynomial_compiler.jl")
include("model_kit/symbolic_polynomial_compiler.jl")
include("model_kit/interpreter.jl")
include("model_kit/codegen.jl")
include("model_kit/polynomial_input.jl")

include("core/abstract_types.jl")
include("core/system_evaluator.jl")
include("solving/support.jl")
include("core/system.jl")
include("core/linear_subspace.jl")
include("core/randomized_system.jl")
include("core/homotopy_evaluator.jl")
include("core/straight_line_homotopy.jl")
include("core/linear_parameter_homotopy.jl")
include("core/subspace_homotopies.jl")
include("core/affine_chart.jl")
include("core/toric_homotopy.jl")

include("tracking/newton_corrector.jl")
include("tracking/newton.jl")
include("tracking/predictor.jl")
include("tracking/tracker.jl")
include("tracking/valuation.jl")
include("tracking/endgame_tracker.jl")
include("solving/group_actions.jl")
include("solving/voronoi_tree.jl")
include("solving/unique_points.jl")
include("solving/executor.jl")
include("solving/worker_state.jl")
include("solving/binomial_system.jl")
include("solving/path_result.jl")
include("solving/progress.jl")
include("solving/excess_solution.jl")
include("solving/total_degree.jl")
include("solving/builder.jl")
include("solving/polyhedral.jl")
include("solving/result.jl")
include("solving/solve.jl")
include("solving/monodromy.jl")
# include("precompile.jl")

end # module
