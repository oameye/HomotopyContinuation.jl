module HomotopyContinuationNext

using LinearAlgebra: LinearAlgebra
using Random: Random
using Printf: Printf

using EnumX: @enumx
using MultivariatePolynomials: MultivariatePolynomials
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray

export @polyvar

const MP = MultivariatePolynomials
# Concrete type aliases — FixedSizeVector{T} alone is NOT concrete because
# the Mem parameter is free. On Julia 1.11+ the backing is Memory{T}.
const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

# primitives/double_f64.jl
export DoubleF64, ComplexDF64
export wide_add, wide_sub, wide_mul, wide_div, wide_square, wide_sqrt

# utils.jl
export fast_abs

# primitives/norms.jl
export InfNorm, WeightedNorm, WeightedNormOptions
export inf_norm, inf_distance, weighted_norm, weighted_distance
export init!, update!

# primitives/linear_algebra.jl
export MatrixWorkspace, updated!, factorize!
export skeel_row_scaling!, apply_row_scaling!
export residual!
export mixed_precision_iterative_refinement!, fixed_precision_iterative_refinement!
export inverse_inf_norm_est, inf_norm_matrix
export Jacobian

include("primitives/double_f64.jl")
include("utils.jl")
include("primitives/norms.jl")
include("primitives/linear_algebra.jl")

# model_kit/taylor.jl
export TruncatedTaylorSeries, TaylorVector
export vectors
export taylor_op_identity, taylor_op_neg
export taylor_op_add, taylor_op_sub, taylor_op_mul, taylor_op_div
export taylor_op_inv, taylor_op_inv_not_zero, taylor_op_invsqr
export taylor_op_sqr, taylor_op_cb, taylor_op_sqrt
export taylor_op_pow_int
export taylor_op_sin, taylor_op_cos
export taylor_op_muladd, taylor_op_mulsub, taylor_op_submul
export taylor_op_add3, taylor_op_add4
export taylor_op_mul3, taylor_op_mul4
export taylor_op_mulmuladd, taylor_op_mulmulsub

include("model_kit/taylor.jl")

# model_kit/operations.jl
export OpType
export arity, op_call, should_use_index_not_reference
export op_stop
export op_cb, op_cos, op_identity, op_inv, op_inv_not_zero, op_invsqr
export op_neg, op_sin, op_sqr, op_sqrt
export op_add, op_div, op_mul, op_sub, op_pow_int
export op_add3, op_mul3, op_muladd, op_mulsub, op_submul
export op_add4, op_mul4, op_mulmuladd, op_mulmulsub

include("model_kit/operations.jl")

# model_kit/instruction_sequence.jl
export IRStatementRef, IRStatement, IRStatementArg
export IntermediateRepresentation
export Instruction, InstructionSequence
export build_instruction_sequence_from_ir

include("model_kit/instruction_sequence.jl")

# model_kit/cse.jl — SExpr types and SymEngine-style CSE algorithm
include("model_kit/cse.jl")

# model_kit/interpreter.jl
export Interpreter, execute!, execute_taylor!

include("model_kit/interpreter.jl")

# model_kit/polynomial_input.jl
export build_interpreter, build_jacobian_interpreter
export build_taylor_interpreter, build_df64_interpreter

include("model_kit/polynomial_input.jl")

end # module
