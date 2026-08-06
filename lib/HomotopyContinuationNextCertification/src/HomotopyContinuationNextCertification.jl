module HomotopyContinuationNextCertification

# Solution certification (Krawczyk / interval arithmetic + Arb fallback) for
# HomotopyContinuationNext. This is a separate package so the heavy Arblib
# binary dependency stays out of the core package (which keeps core TTFX
# minimal). Load this package to certify:
#
#     using HomotopyContinuationNext, HomotopyContinuationNextCertification
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
using Moshi.Data: variant_storage_type

# Core (HomotopyContinuationNext) internals the certification code builds on.
# These are accessed via qualified imports; certification is intimately coupled
# to the tape interpreter, so it reaches non-exported names by design.
using HomotopyContinuationNext:
    HomotopyContinuationNext,
    # systems / evaluators
    System, SystemEvaluator, FSVec,
    nparameters, parameters, is_real,
    # tape interpreter + ADT internals (for the Acb interpreter)
    Interpreter, InstructionSequence, ExecInstruction, ExecInstructionT,
    OpType, op_call, arity, should_use_index_not_reference,
    instruction_op, instruction_output,
    _EXEC_INSTRUCTION_SPECS, nested_ifs, exec_instruction_storage,
    _compile_exec_instructions, execute!,
    # newton refinement
    NewtonCache, _newton, _clone_system_evaluator, solution,
    # results consumed by the certify entry points
    Result, PathResult, MonodromyResult,
    # lazy path tracking, for certification of a `ResultIterator`
    ResultIterator, selection, restrict, is_success,
    _foreach_path, _replay_ntasks, _path_workers,
    # progress bar helper
    make_progress,
    # executors: certification is Serial/Threaded only, see `Certification`
    Serial, Threaded

# Core functions extended with methods on certification types. `nstart_solutions`
# and `ntracked` are also called on a `ResultIterator`, which an import allows.
import HomotopyContinuationNext: is_real, solutions, nstart_solutions, ntracked

include("interval_arithmetic.jl")
include("interval_arblib.jl")
include("acb_interpreter.jl")
include("certification.jl")
include("certification_arb.jl")
include("iterator_certification.jl")

export certify, Certification, SolutionCertificate, ExtendedSolutionCertificate, CertificationResult,
    CertificationCache, is_certified, is_real, is_complex, is_positive,
    solution_candidate, certified_solution_interval,
    certified_solution_interval_after_krawczyk, certificate_index,
    solution_approximation, certificates, distinct_certificates, distinct_solutions,
    ncertified, nreal_certified, ncomplex_certified, ndistinct_certified,
    ndistinct_real_certified, ndistinct_complex_certified, solutions,
    save, DistinctCertifiedSolutions, add_solution!, distinct_certified_solutions,
    distinct_certified_solutions!, stats, ncertified_distinct, nprocessed,
    nduplicates, nnotcertified, show_straight_line_program, ncandidates,
    IteratorCertification, IteratorCertificationResult, BSPPartition, bsp,
    nstart_solutions, ntracked, nleaves, max_leaf_size,
    oversized_leaves, unsplittable_leaves, nleaf_splits

end # module
