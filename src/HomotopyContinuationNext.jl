module HomotopyContinuationNext

using LinearAlgebra: LinearAlgebra
using Random: Random
using Printf: Printf
using ProgressMeter: ProgressMeter

using EnumX: @enumx
using Moshi.Data: @data, variant_storage, variant_storage_type
using Moshi.Derive: @derive
using MultivariatePolynomials: MultivariatePolynomials
import MultivariatePolynomials: coefficients, degree, differentiate, monomials
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
export CompositionSystem, compose
export FixedParameterSystem, fix_parameters
export @var, @unique_var, Expression, differentiate, subs, num_den
export expand, to_dict, horner, monomials, dense_poly, rand_poly
export coefficients, coeffs_as_dense_poly
export exponents_coefficients, poly_from_exponents_coefficients
export to_number, evaluate, jacobian
export CompileMode
export solutions, real_solutions, nsolutions, nreal, nsingular, nnonsingular, nat_infinity
export nexcess_solutions, nfailed
export nresults, results, multiplicity
export is_success, is_singular, is_nonsingular, is_at_infinity, is_real, is_excess_solution
export is_failed, is_finite
export AbstractResult, AbstractSolutionResult
export TotalDegree, Polyhedral, Result, PathResult, paths_to_track, mixed_volume
export Continuation, Sweep, Monodromy, Witness, Membership
export Regeneration, Intersection, Decomposition, EquationSorting
export CommonOptions, early_stop_callback, excess_residual_tol
export path_results, seed, ntracked, failed, at_infinity, nonsingular, singular
export recluster, clusters, cluster_of
export statistics, ResultStatistics
export solution, accuracy, residual, steps, accepted_steps, rejected_steps
export winding_number, condition_jacobian, last_path_point
export path_number, start_solution, valuation
export TrackerOptions, EndgameOptions, EndgameTracker
export iterator, path_info, PathInfo, PathStep, path_table
export ParameterHomotopy, Homotopy, expressions, equation_scales
export GroupActions, SymmetricGroup
export UniquePoints, search_in_radius, add!, multiplicities, unique_points
export satisfies_triangle_inequality, InfNorm, EuclideanNorm
export LinearSubspace, ExtrinsicDescription, IntrinsicDescription, Intrinsic, Extrinsic
export intrinsic, extrinsic, is_linear, dim, codim, ambient_dim
export rand_subspace, rand_subspace!, translate, geodesic, geodesic_distance, coord_change
export IntrinsicSubspaceHomotopy, ExtrinsicSubspaceHomotopy, set_subspaces!
export AffineChartHomotopy, on_affine_chart, linear_subspace_homotopy
export find_start_pair, verify_solution_completeness
export MonodromyOptions, MonodromyResult, is_heuristic_stop, permutations, trace
export ReuseLoops, DuplicateCheck, ncertified_distinct, ndiscarded_uncertified
export independent_normal, weighted_normal
# Witness sets / numerical irreducible decomposition
export slice
export ResultIterator, result_iterator, selection, restrict, start_solutions
export nstart_solutions
export total_degree_start_solutions
export WitnessSet, trace_test, membership
export system, linear_subspace, is_irreducible, Irreducibility, degree, points
export WitnessPoints
export NumericalIrreducibleDecomposition
export ncomponents, n_components, witness_sets, degrees
export newton, NewtonResult, NewtonCache, NewtonReturnCode
export Serial, Threaded, DistributedExecutor
export SemialgebraicSetsHCSolver
export write_solutions, read_solutions, write_parameters, read_parameters
# Certification (certify, SolutionCertificate, …) lives in the
# HomotopyContinuationNextCertification subpackage (lib/), which depends on
# Arblib. Keeping Arblib out of this core package is what makes core TTFX
# minimal; load the subpackage to certify.

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

# model_kit/expression.jl — user-facing symbolic Expression frontend
include("model_kit/expression.jl")
include("model_kit/symbolic_utils.jl")

# model_kit/cse.jl — SymEngine-style CSE algorithm (opt_cse + tree_cse)
include("model_kit/cse.jl")

# model_kit/tape_compiler.jl — TapeCompiler, compile_to_instructions
include("model_kit/tape_compiler.jl")

include("model_kit/polynomial_compiler.jl")
include("model_kit/symbolic_polynomial_compiler.jl")
include("model_kit/expression_compiler.jl")
include("model_kit/interpreter.jl")
include("model_kit/codegen.jl")
include("model_kit/polynomial_input.jl")

include("core/abstract_types.jl")
include("core/system_evaluator.jl")
include("solving/support.jl")
include("core/system.jl")
include("core/linear_subspace.jl")
include("core/randomized_system.jl")
include("core/composition_system.jl")
include("core/start_pair_system.jl")
include("core/homotopy_evaluator.jl")
include("core/straight_line_homotopy.jl")
include("core/linear_parameter_homotopy.jl")
include("core/subspace_homotopies.jl")
include("core/affine_chart.jl")
include("core/sliced_system.jl")
include("core/fixed_parameter_system.jl")
include("core/system_evaluate.jl")
include("core/symbolic_homotopy.jl")
include("core/toric_homotopy.jl")

include("tracking/newton_corrector.jl")
include("tracking/newton.jl")
include("tracking/predictor.jl")
include("tracking/tracker.jl")
include("tracking/path_info.jl")
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
include("solving/algorithm.jl")
include("solving/total_degree.jl")
include("solving/builder.jl")
include("solving/polyhedral.jl")
include("solving/result.jl")
include("solving/solution_files.jl")
include("solving/starts.jl")
include("solving/solve.jl")
include("solving/homotopy_solve.jl")
include("solving/slice.jl")
include("solving/multi_homogeneous.jl")
include("solving/subspace_solve.jl")
include("solving/sweep.jl")
include("solving/result_iterator.jl")
include("solving/monodromy.jl")
include("solving/witness_set.jl")
include("solving/regeneration.jl")
include("solving/nid.jl")
# include("precompile.jl")
include("precompile_signatures.jl")

end # module
