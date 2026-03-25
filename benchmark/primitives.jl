# Phase 1: Primitives benchmarks
# Only benchmarks that measure distinct codepaths or meaningful scaling.

using BenchmarkTools
using HomotopyContinuationNext
using HomotopyContinuationNext: DoubleF64, MatrixWorkspace, updated!, inf_norm
using FixedSizeArrays: FixedSizeArray
using LinearAlgebra: LinearAlgebra

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}

function benchmark_primitives!(SUITE::BenchmarkGroup)
    # --- DoubleF64: one representative op (mul) to detect regressions ---
    SUITE["double_f64"] = BenchmarkGroup()
    let a = DoubleF64(1.234567890123456789), b = DoubleF64(9.876543210987654321)
        SUITE["double_f64"]["mul"] = @benchmarkable $a * $b
    end

    # --- inf_norm: fast path, n=4 (hot) and n=16 (scaling) ---
    SUITE["norms"] = BenchmarkGroup()
    for n in [4, 16]
        x = FSVec{ComplexF64}(rand(ComplexF64, n))
        SUITE["norms"]["inf_norm_$n"] = @benchmarkable inf_norm($x)
    end

    # --- LU + ldiv!: O(n³) scaling across typical system sizes ---
    SUITE["linear_algebra"] = BenchmarkGroup()
    for n in [4, 8, 16]
        A_data = rand(ComplexF64, n, n) + 5.0 * LinearAlgebra.I
        b_data = rand(ComplexF64, n)
        WS = MatrixWorkspace(n, n)
        copyto!(WS.A, A_data)
        updated!(WS)
        x = FSVec{ComplexF64}(zeros(ComplexF64, n))
        b = FSVec{ComplexF64}(b_data)

        SUITE["linear_algebra"]["lu_ldiv_$n"] = @benchmarkable begin
            copyto!($WS.A, $A_data)
            updated!($WS)
            LinearAlgebra.ldiv!($x, $WS, $b)
        end
    end

    return SUITE
end
