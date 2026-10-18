module HomotopyContinuationCertification

# Solution certification (Krawczyk / interval arithmetic + Arb fallback) for
# HomotopyContinuation. This is a separate package so the heavy Arblib
# binary dependency stays out of the core package (which keeps core TTFX
# minimal). Load this package to certify:
#
#     using HomotopyContinuation, HomotopyContinuationCertification
#     certify(F, solutions)

using Printf: Printf
using Random: Random
using ProgressMeter: ProgressMeter
using LinearAlgebra: LinearAlgebra

import MultivariatePolynomials as MP

using Arblib: Arblib, AcbMatrix, AcbRefVector, AcbRefMatrix, Mag
using EnumX: @enumx
using IntervalTrees: IntervalTrees
using OhMyThreads: @tasks, @set, @local
using Moshi.Match: @match

# Core HomotopyContinuation internals the certification code builds on.
# These are accessed via qualified imports; certification is intimately coupled
# to the tape interpreter, so it reaches non-exported names by design.
using HomotopyContinuation:
    HomotopyContinuation,
    System, SystemEvaluator, FSVec,
    nparameters, parameters, is_real,
    Interpreter, InstructionSequence, ExecInstruction, ExecInstructionT,
    OpType, op_call, arity, should_use_index_not_reference,
    instruction_op, instruction_output,
    _EXEC_INSTRUCTION_SPECS,
    _compile_exec_instructions, execute!,
    NewtonCache, _newton, _clone_system_evaluator, solution,
    _with_fields,
    Result, PathResult, MonodromyResult, CertifiedEndpoint, results,
    AbstractCertifiedSolutions, AbstractCertifiedCandidate,
    ResultIterator, selection, restrict, is_success,
    _foreach_path, _replay_ntasks, _path_workers,
    make_progress,
    AddSolutionCode,
    Serial, Threaded

import HomotopyContinuation: is_real, solutions, nstart_solutions, ntracked,
    ncertified_distinct

include("interval_arithmetic.jl")
include("interval_arblib.jl")
include("acb_interpreter.jl")
include("certification.jl")
include("certification_arb.jl")
include("iterator_certification.jl")
include("monodromy_certification.jl")

export certify, Certification, SolutionCertificate, ExtendedSolutionCertificate, CertificationResult,
    CertificationCache, is_certified, is_real, is_complex, is_positive,
    solution_candidate, certified_solution_interval,
    certified_solution_interval_after_krawczyk, certificate_index,
    solution_approximation, certificates, distinct_certificates, distinct_solutions,
    ncertified, nreal_certified, ncomplex_certified, ndistinct_certified,
    ndistinct_real_certified, ndistinct_complex_certified, solutions,
    save, DistinctCertifiedSolutions, add_solution!, AddSolutionCode, distinct_certified_solutions,
    distinct_certified_solutions!, stats, ncertified_distinct, nprocessed,
    nduplicates, nnotcertified, show_straight_line_program, ncandidates,
    IteratorCertification, IteratorCertificationResult, BSPPartition, bsp,
    nstart_solutions, ntracked, nleaves, max_leaf_size,
    oversized_leaves, unsplittable_leaves, nleaf_splits

end # module
