# Compiled vs interpreted evaluation benchmarks.
# Integrated into the main suite via benchmark_compile_modes!(SUITE).
# Also runnable standalone: julia --project=benchmark benchmark/compile_modes.jl

using BenchmarkTools
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray
using HomotopyContinuationNext:
    System, evaluate!, evaluate_and_jacobian!, CompileMode

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

function benchmark_compile_modes!(SUITE::BenchmarkGroup)
    SUITE["compile_modes"] = BenchmarkGroup()

    @polyvar x0 x1 x2 x3
    F_k3 = [
        x0 + 2x1 + 2x2 + 2x3 - 1,
        x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
        2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
        x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
    ]

    sys_i = System(F_k3; compile = CompileMode.INTERPRETED)
    sys_c = System(F_k3; compile = CompileMode.COMPILED)

    xv = FSVec{ComplexF64}(ComplexF64.(randn(4)))
    pv = FSVec{ComplexF64}(ComplexF64[])
    u_i = FSVec{ComplexF64}(zeros(ComplexF64, 4))
    u_c = FSVec{ComplexF64}(zeros(ComplexF64, 4))
    U_i = FSMat{ComplexF64}(zeros(ComplexF64, 4, 4))
    U_c = FSMat{ComplexF64}(zeros(ComplexF64, 4, 4))

    SUITE["compile_modes"]["eval_interpreted_katsura3"] =
        @benchmarkable evaluate!($u_i, $(sys_i.evaluator), $xv, $pv)
    SUITE["compile_modes"]["eval_compiled_katsura3"] =
        @benchmarkable evaluate!($u_c, $(sys_c.evaluator), $xv, $pv)
    SUITE["compile_modes"]["jac_interpreted_katsura3"] =
        @benchmarkable evaluate_and_jacobian!($u_i, $U_i, $(sys_i.evaluator), $xv, $pv)
    SUITE["compile_modes"]["jac_compiled_katsura3"] =
        @benchmarkable evaluate_and_jacobian!($u_c, $U_c, $(sys_c.evaluator), $xv, $pv)
    SUITE["compile_modes"]["build_interpreted_katsura3"] =
        @benchmarkable System($F_k3; compile = CompileMode.INTERPRETED)
    SUITE["compile_modes"]["build_compiled_katsura3"] =
        @benchmarkable System($F_k3; compile = CompileMode.COMPILED)

    return SUITE
end

# Standalone mode: run directly with @belapsed for quick comparison output
if abspath(PROGRAM_FILE) == @__FILE__
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1.0

    println("="^72)
    println("  v3 Compile Modes: INTERPRETED vs COMPILED")
    println("="^72)

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

    println("\n── Eval through SystemEvaluator ──")
    for n in [3, 5, 7]
        @polyvar kv[1:(n + 1)]
        F = _katsura(kv, n)
        sys_i = System(F; compile = CompileMode.INTERPRETED)
        sys_c = System(F; compile = CompileMode.COMPILED)

        xv = FSVec{ComplexF64}(ComplexF64.(randn(n + 1)))
        pv = FSVec{ComplexF64}(ComplexF64[])
        u_i = FSVec{ComplexF64}(zeros(ComplexF64, n + 1))
        u_c = FSVec{ComplexF64}(zeros(ComplexF64, n + 1))

        evaluate!(u_i, sys_i.evaluator, xv, pv)
        evaluate!(u_c, sys_c.evaluator, xv, pv)
        @assert u_i ≈ u_c "Correctness check failed for katsura-$n"

        t_i = @belapsed evaluate!($u_i, $(sys_i.evaluator), $xv, $pv)
        t_c = @belapsed evaluate!($u_c, $(sys_c.evaluator), $xv, $pv)
        println("  katsura-$n: interp=$(round(t_i * 1.0e9; digits = 1))ns  compiled=$(round(t_c * 1.0e9; digits = 1))ns  speedup=$(round(t_i / t_c; digits = 2))x")
    end

    println("\n── Jacobian through SystemEvaluator ──")
    for n in [3, 5, 7]
        @polyvar kv[1:(n + 1)]
        F = _katsura(kv, n)
        sys_i = System(F; compile = CompileMode.INTERPRETED)
        sys_c = System(F; compile = CompileMode.COMPILED)

        m = n + 1
        xv = FSVec{ComplexF64}(ComplexF64.(randn(m)))
        pv = FSVec{ComplexF64}(ComplexF64[])
        u_i = FSVec{ComplexF64}(zeros(ComplexF64, m))
        u_c = FSVec{ComplexF64}(zeros(ComplexF64, m))
        U_i = FSMat{ComplexF64}(zeros(ComplexF64, m, m))
        U_c = FSMat{ComplexF64}(zeros(ComplexF64, m, m))

        evaluate_and_jacobian!(u_i, U_i, sys_i.evaluator, xv, pv)
        evaluate_and_jacobian!(u_c, U_c, sys_c.evaluator, xv, pv)
        @assert u_i ≈ u_c "Correctness check failed for katsura-$n jac"

        t_i = @belapsed evaluate_and_jacobian!($u_i, $U_i, $(sys_i.evaluator), $xv, $pv)
        t_c = @belapsed evaluate_and_jacobian!($u_c, $U_c, $(sys_c.evaluator), $xv, $pv)
        println("  katsura-$n: interp=$(round(t_i * 1.0e9; digits = 1))ns  compiled=$(round(t_c * 1.0e9; digits = 1))ns  speedup=$(round(t_i / t_c; digits = 2))x")
    end

    println("\n── Build time ──")
    for n in [3, 5, 7]
        @polyvar kv[1:(n + 1)]
        F = _katsura(kv, n)
        t_i = @belapsed System($F; compile = CompileMode.INTERPRETED)
        t_c = @belapsed System($F; compile = CompileMode.COMPILED)
        println("  katsura-$n: interp=$(round(t_i / 1.0e-6; digits = 1))us  compiled=$(round(t_c / 1.0e-6; digits = 1))us  overhead=$(round(t_c / t_i; digits = 2))x")
    end

    println("\n── End-to-end solve ──")
    using HomotopyContinuationNext: solve, nsolutions
    for n in [3, 4, 5]
        @polyvar kv[1:(n + 1)]
        F = _katsura(kv, n)
        sys_i = System(F; compile = CompileMode.INTERPRETED)
        sys_c = System(F; compile = CompileMode.COMPILED)
        solve(sys_i); solve(sys_c)
        t_i = @belapsed solve($sys_i)
        t_c = @belapsed solve($sys_c)
        println("  katsura-$n: interp=$(round(t_i * 1.0e3; digits = 2))ms  compiled=$(round(t_c * 1.0e3; digits = 2))ms  speedup=$(round(t_i / t_c; digits = 2))x")
    end

    println("\n── Taylor through SystemEvaluator ──")
    using HomotopyContinuationNext: taylor!, TaylorVector
    for n in [3, 5, 7]
        @polyvar kv[1:(n + 1)]
        F = _katsura(kv, n)
        sys_i = System(F; compile = CompileMode.INTERPRETED)
        sys_c = System(F; compile = CompileMode.COMPILED)
        sys_a = System(F; compile = CompileMode.COMPILED_ALL)

        m = n + 1
        pv = FSVec{ComplexF64}(ComplexF64[])
        for K in 1:3
            N = K + 1
            data = FSMat{ComplexF64}(randn(ComplexF64, N, m))
            tx = TaylorVector{N, ComplexF64}(data)
            u = FSVec{ComplexF64}(zeros(ComplexF64, m))

            taylor!(u, Val(K), sys_i.evaluator, tx, pv)
            taylor!(u, Val(K), sys_c.evaluator, tx, pv)
            taylor!(u, Val(K), sys_a.evaluator, tx, pv)

            t_i = @belapsed taylor!($u, Val($K), $(sys_i.evaluator), $tx, $pv)
            t_c = @belapsed taylor!($u, Val($K), $(sys_c.evaluator), $tx, $pv)
            t_a = @belapsed taylor!($u, Val($K), $(sys_a.evaluator), $tx, $pv)
            println("  katsura-$n taylor_$K: interp=$(round(t_i * 1.0e9; digits = 1))ns  compiled=$(round(t_c * 1.0e9; digits = 1))ns  all=$(round(t_a * 1.0e9; digits = 1))ns  speedup=$(round(t_i / t_a; digits = 2))x")
        end
    end

    println("\n── Taylor with TaylorVector parameters (production path) ──")
    for n in [3, 5, 7]
        @polyvar kv[1:(n + 1)]
        @polyvar params[1:(n + 1)]
        F_param = _katsura(kv, n)
        # Make it parametric by multiplying each eq by a parameter
        F_param = [params[i] * F_param[i] for i in eachindex(F_param)]
        sys_i = System(F_param; parameters = collect(params), compile = CompileMode.INTERPRETED)
        sys_c = System(F_param; parameters = collect(params), compile = CompileMode.COMPILED)
        sys_a = System(F_param; parameters = collect(params), compile = CompileMode.COMPILED_ALL)

        m = n + 1
        for K in 1:3
            N = K + 1
            data_x = FSMat{ComplexF64}(randn(ComplexF64, N, m))
            tx = TaylorVector{N, ComplexF64}(data_x)
            data_p = FSMat{ComplexF64}(randn(ComplexF64, N, m))
            tp = TaylorVector{N, ComplexF64}(data_p)
            u = FSVec{ComplexF64}(zeros(ComplexF64, m))

            taylor!(u, Val(K), sys_i.evaluator, tx, tp)
            taylor!(u, Val(K), sys_c.evaluator, tx, tp)
            taylor!(u, Val(K), sys_a.evaluator, tx, tp)

            t_i = @belapsed taylor!($u, Val($K), $(sys_i.evaluator), $tx, $tp)
            t_c = @belapsed taylor!($u, Val($K), $(sys_c.evaluator), $tx, $tp)
            t_a = @belapsed taylor!($u, Val($K), $(sys_a.evaluator), $tx, $tp)
            println("  katsura-$n taylor_$K(param): interp=$(round(t_i * 1.0e9; digits = 1))ns  compiled=$(round(t_c * 1.0e9; digits = 1))ns  all=$(round(t_a * 1.0e9; digits = 1))ns  speedup=$(round(t_i / t_a; digits = 2))x")
        end
    end

    println("\n── End-to-end solve (COMPILED_ALL) ──")
    for n in [3, 4, 5]
        @polyvar kv[1:(n + 1)]
        F = _katsura(kv, n)
        sys_c = System(F; compile = CompileMode.COMPILED)
        sys_a = System(F; compile = CompileMode.COMPILED_ALL)
        solve(sys_c); solve(sys_a)
        t_c = @belapsed solve($sys_c)
        t_a = @belapsed solve($sys_a)
        println("  katsura-$n: compiled=$(round(t_c * 1.0e3; digits = 2))ms  all=$(round(t_a * 1.0e3; digits = 2))ms  speedup=$(round(t_c / t_a; digits = 2))x")
    end

    println("\n── Build time (COMPILED_ALL) ──")
    for n in [3, 5, 7]
        @polyvar kv[1:(n + 1)]
        F = _katsura(kv, n)
        t_c = @belapsed System($F; compile = CompileMode.COMPILED)
        t_a = @belapsed System($F; compile = CompileMode.COMPILED_ALL)
        println("  katsura-$n: compiled=$(round(t_c / 1.0e-6; digits = 1))us  all=$(round(t_a / 1.0e-6; digits = 1))us  overhead=$(round(t_a / t_c; digits = 2))x")
    end
end
