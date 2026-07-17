# End-to-end compile mode comparison: INTERPRETED vs COMPILED vs COMPILED_ALL.
# Measures what the kernel benchmarks in compile_modes.jl cannot show: how much
# of the kernel speedup survives a full solve. Serial executor, progress off,
# so the numbers are free of threading and progress-bar noise.
# Run standalone: julia --project=benchmark benchmark/compile_modes_e2e.jl

using BenchmarkTools
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray
using HomotopyContinuationNext:
    System, solve, CompileMode, TotalDegree, Serial,
    StraightLineHomotopy, HomotopyEvaluator,
    evaluate!, evaluate_and_jacobian!, taylor!, TaylorVector
import HomotopyContinuationNext as HC

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2.0

function _katsura(vars, n)
    lin = vars[1] + sum(2vars[i] for i in 2:(n + 1)) - 1
    eqs = [lin]
    for l in 0:(n - 1)
        eq = -vars[l + 1]
        for i in (-n):n
            j = l - i
            abs(i) <= n && abs(j) <= n && (eq += vars[abs(i) + 1] * vars[abs(j) + 1])
        end
        push!(eqs, eq)
    end
    return eqs
end

# Kernel times as the tracker sees them: through the StraightLineHomotopy and
# the HomotopyEvaluator FunctionWrappers, so start-system evaluation and the
# γ blend are included. The raw SystemEvaluator speedups in compile_modes.jl
# are roughly halved here.
println("── HomotopyEvaluator kernel times ──")
for n in [3, 5, 7]
    @polyvar kv[1:(n + 1)]
    F = _katsura(kv, n)
    m = n + 1
    for (label, mode) in [("interp  ", CompileMode.INTERPRETED), ("compiled", CompileMode.COMPILED)]
        sys = System(F; compile = mode)
        se = HC._total_degree_startevaluator(HC.degrees(sys))
        H = StraightLineHomotopy(se, sys.evaluator; γ = 0.7 + 0.3im)
        he = HomotopyEvaluator(H)
        x = FSVec{ComplexF64}(ComplexF64.(randn(m)))
        u = FSVec{ComplexF64}(zeros(ComplexF64, m))
        U = FSMat{ComplexF64}(zeros(ComplexF64, m, m))
        t = ComplexF64(0.5)

        t_ev = @belapsed evaluate!($u, $he, $x, $t)
        t_jc = @belapsed evaluate_and_jacobian!($u, $U, $he, $x, $t)

        tx2 = TaylorVector{3, ComplexF64}(FSMat{ComplexF64}(randn(ComplexF64, 3, m)))
        tx3 = TaylorVector{4, ComplexF64}(FSMat{ComplexF64}(randn(ComplexF64, 4, m)))
        t_t1 = @belapsed taylor!($u, Val(1), $he, $x, $t)
        t_t2 = @belapsed taylor!($u, Val(2), $he, $tx2, $t, false)
        t_t3 = @belapsed taylor!($u, Val(3), $he, $tx3, $t, false)

        println(
            "  katsura-$n $label: eval=", round(t_ev * 1.0e9; digits = 0),
            "ns  jac=", round(t_jc * 1.0e9; digits = 0),
            "ns  taylor1=", round(t_t1 * 1.0e9; digits = 0),
            "ns  taylor2=", round(t_t2 * 1.0e9; digits = 0),
            "ns  taylor3=", round(t_t3 * 1.0e9; digits = 0), "ns"
        )
    end
end

println("\n── End-to-end serial solve (speedup vs INTERPRETED) ──")
for n in [3, 5, 7, 9]
    @polyvar kv[1:(n + 1)]
    F = _katsura(kv, n)
    sys_i = System(F; compile = CompileMode.INTERPRETED)
    sys_c = System(F; compile = CompileMode.COMPILED)
    sys_a = System(F; compile = CompileMode.COMPILED_ALL)
    for s in (sys_i, sys_c, sys_a)
        solve(s, TotalDegree(), Serial(); show_progress = false)
    end
    t_i = @belapsed solve($sys_i, TotalDegree(), Serial(); show_progress = false)
    t_c = @belapsed solve($sys_c, TotalDegree(), Serial(); show_progress = false)
    t_a = @belapsed solve($sys_a, TotalDegree(), Serial(); show_progress = false)
    println(
        "  katsura-$n: interp=", round(t_i * 1.0e3; digits = 2),
        "ms  compiled=", round(t_c * 1.0e3; digits = 2),
        "ms (", round(t_i / t_c; digits = 2),
        "x)  all=", round(t_a * 1.0e3; digits = 2),
        "ms (", round(t_i / t_a; digits = 2), "x)"
    )
end

println("\n── Build time per mode ──")
for n in [3, 5, 7, 9]
    @polyvar kv[1:(n + 1)]
    F = _katsura(kv, n)
    t_i = @belapsed System($F; compile = CompileMode.INTERPRETED)
    t_c = @belapsed System($F; compile = CompileMode.COMPILED)
    t_a = @belapsed System($F; compile = CompileMode.COMPILED_ALL)
    println(
        "  katsura-$n: interp=", round(t_i * 1.0e3; digits = 2),
        "ms  compiled=", round(t_c * 1.0e3; digits = 2),
        "ms  all=", round(t_a * 1.0e3; digits = 2), "ms"
    )
end
