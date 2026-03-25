module HomotopyContinuationNext

using LinearAlgebra: LinearAlgebra
using Random: Random
using Printf: Printf

using MultivariatePolynomials: MultivariatePolynomials
using FixedSizeArrays: FixedSizeVector, FixedSizeMatrix

const MP = MultivariatePolynomials
const FSVec{T} = FixedSizeVector{T}
const FSMat{T} = FixedSizeMatrix{T}

include("primitives/double_f64.jl")
include("utils.jl")

end # module
