module HomotopyContinuationNext

using LinearAlgebra: LinearAlgebra
using Random: Random
using Printf: Printf

using MultivariatePolynomials: MultivariatePolynomials
using FixedSizeArrays: FixedSizeArray

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

end # module
