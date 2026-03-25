module HomotopyContinuationNext

using LinearAlgebra: LinearAlgebra
using Random: Random
using Printf: Printf

using MultivariatePolynomials: MultivariatePolynomials
using FixedSizeArrays: FixedSizeVector, FixedSizeMatrix

const MP = MultivariatePolynomials
const FSVec{T} = FixedSizeVector{T}
const FSMat{T} = FixedSizeMatrix{T}

# primitives/double_f64.jl
export DoubleF64, ComplexDF64

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
