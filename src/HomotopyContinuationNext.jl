module HomotopyContinuationNext

using LinearAlgebra: LinearAlgebra
using Random: Random
using Printf: Printf

using EnumX: @enumx
using MultivariatePolynomials: MultivariatePolynomials
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray
import FunctionWrappers: FunctionWrapper

export @polyvar

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

include("model_kit/interpreter.jl")
include("model_kit/polynomial_input.jl")

include("core/abstract_types.jl")
include("core/system_evaluator.jl")
include("core/system_eval.jl")
include("core/homotopy_evaluator.jl")
include("core/straight_line_homotopy.jl")

end # module
