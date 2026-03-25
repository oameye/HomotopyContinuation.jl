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

## CRITICAL: Concrete Type Aliases

**`FixedSizeVector{T}` and `FixedSizeMatrix{T}` are NOT concrete types** — the `Mem`
type parameter is free. Using them as struct field types causes type instability
and ~30x performance degradation (getindex returns `Any`).

Always use the fully concrete aliases:

```julia
using FixedSizeArrays: FixedSizeArray
const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}   # concrete on Julia 1.11+
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}   # concrete on Julia 1.11+
```

Verification:
```julia
julia> isconcretetype(FixedSizeVector{Float64})
false   # ← BAD for struct fields

julia> isconcretetype(FixedSizeArray{Float64, 1, Memory{Float64}})
true    # ← GOOD
```

## Verified Compatible (Phase 1)

Tested in `test/fixedsizearrays_compat_test.jl`:

- `strides(A)` returns `(1, n)` — correct column-major layout
- `lu!(FSMat)` works — returns `LU{ComplexF64, FSMat{ComplexF64}, FSVec{Int64}}`
- `ldiv!(FSVec, LU, FSVec)` works
- `mul!(FSVec, FSMat, FSVec)` works
- `copyto!` between FSMat and Matrix works
- `qr!(FSMat)` returns `QRCompactWY` (LAPACK blocked), NOT `QR`
  → use `Matrix{ComplexF64}` with `LinearAlgebra.qrfactUnblocked!` for custom QR

## Notes

- `BoundsErrorLight` instead of `BoundsError` for escape analysis optimization.
- Size is runtime value in `NTuple{N,Int}` field — NOT a type parameter.
- LU ipiv type is `FSVec{Int64}` when factoring an `FSMat`, not `Vector{BlasInt}`.
