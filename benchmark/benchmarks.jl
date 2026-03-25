using BenchmarkTools
using HomotopyContinuationNext
using HomotopyContinuationNext:
    DoubleF64, ComplexDF64,
    MatrixWorkspace, updated!, factorize!,
    WeightedNorm, weighted_norm, init!,
    inf_norm, fast_abs, Jacobian
using FixedSizeArrays: FixedSizeVector, FixedSizeMatrix
using LinearAlgebra: LinearAlgebra

const FSVec{T} = FixedSizeVector{T}
const FSMat{T} = FixedSizeMatrix{T}

const SUITE = BenchmarkGroup()

# --- DoubleF64 ---
SUITE["double_f64"] = BenchmarkGroup()
let a = DoubleF64(1.234567890123456789), b = DoubleF64(9.876543210987654321)
    SUITE["double_f64"]["add"] = @benchmarkable $a + $b
    SUITE["double_f64"]["mul"] = @benchmarkable $a * $b
    SUITE["double_f64"]["div"] = @benchmarkable $a / $b
    SUITE["double_f64"]["sqrt"] = @benchmarkable sqrt($a)
end

# --- Norms ---
SUITE["norms"] = BenchmarkGroup()
for n in [4, 16, 64]
    x = FSVec{ComplexF64}(rand(ComplexF64, n))
    SUITE["norms"]["inf_norm_$n"] = @benchmarkable inf_norm($x)
    w = WeightedNorm(n)
    init!(w, x)
    SUITE["norms"]["weighted_norm_$n"] = @benchmarkable weighted_norm($x, $w)
end

# --- Linear Algebra ---
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

BenchmarkTools.tune!(SUITE)
results = BenchmarkTools.run(SUITE; verbose = true)
display(median(results))

BenchmarkTools.save("benchmarks_output.json", median(results))
