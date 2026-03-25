# FixedSizeArrays.jl — Compatibility Notes

## Type Hierarchy

```julia
struct FixedSizeArray{T,N,Mem<:DenseVector{T}} <: DenseArray{T,N}
    mem::Mem
    size::NTuple{N,Int}
end
```

- `FixedSizeArray <: DenseArray <: AbstractArray`
- `IndexStyle = IndexLinear()` (contiguous, column-major)
- `pointer` / `unsafe_convert` implemented (delegates to parent Memory/Vector)
- `copy` implemented (copies parent storage)
- `Mem` defaults to `Memory{T}` on Julia 1.11+, `Vector{T}` on 1.10

## Verified Compatible

- Standard indexing (`getindex`, `setindex!`)
- Broadcasting (custom `FixedSizeArrayBroadcastStyle`)
- `copy`, `copyto!`, `similar`, `reshape`
- `parent` access to underlying memory
- Iteration

## Verification Needed (Phase 1)

Before using `FSMat` in `MatrixWorkspace`, verify these work:

```julia
using FixedSizeArrays, LinearAlgebra

A = FixedSizeMatrix{ComplexF64}(rand(ComplexF64, 4, 4))
b = FixedSizeVector{ComplexF64}(rand(ComplexF64, 4))

# Critical for MatrixWorkspace:
F = lu!(A)           # Does LAPACK.getrf! work on FSMat?
ldiv!(F, b)          # Does triangular solve work?
Q = qr!(A)           # Does LAPACK.geqrf! work?
mul!(b, A, b)        # Does BLAS.gemv! work?
strides(A)           # Does strides() return correct values?

# Critical for FunctionWrapper compatibility:
using FunctionWrappers
fw = FunctionWrapper{Nothing, Tuple{FixedSizeVector{Float64}}}(x -> nothing)
fw(FixedSizeVector{Float64}(ones(3)))  # Does FW accept FSVec?
```

If `lu!` or `qr!` fail on `FSMat`, fallback options:
1. Use `parent(A)` to get the underlying `Memory`/`Vector` and wrap in a view
2. Keep `MatrixWorkspace.A` as regular `Matrix{ComplexF64}` (only the scratch buffers use FSVec)
3. Implement `strides(::FixedSizeArray)` ourselves if missing

## Notes

- `strides()` is NOT explicitly defined in the source. Since `DenseArray` implies
  contiguous column-major storage, Julia may provide a default. But LAPACK routines
  often check `strides` explicitly. Must test.
- `BoundsErrorLight` instead of `BoundsError` for escape analysis optimization.
- Size is runtime value in `NTuple{N,Int}` field — NOT a type parameter.
