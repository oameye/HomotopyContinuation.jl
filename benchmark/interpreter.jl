# Interpreter pipeline benchmarks
# Measures tape execution, Jacobian, Taylor, and build time.

using BenchmarkTools
using DynamicPolynomials: @polyvar
using HomotopyContinuationNext:
    System, Interpreter, InstructionSequence,
    execute!, execute_taylor!, TaylorVector,
    TruncatedTaylorSeries

function benchmark_interpreter!(SUITE::BenchmarkGroup)
    SUITE["interpreter"] = BenchmarkGroup()

    # --- Systems ---
    @polyvar x0 x1 x2 x3
    F_katsura = [
        x0 + 2x1 + 2x2 + 2x3 - 1,
        x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
        2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
        x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
    ]

    @polyvar c1 c2 c3 c4 c5 c6 c7
    cv = [c1, c2, c3, c4, c5, c6, c7]
    F_cyclic7 = [
        [sum(prod(cv[mod1(j + k, 7)] for k in 0:(d - 1)) for j in 1:7) for d in 1:6]
        [prod(cv) - 1]
    ]

    # --- Eval ---
    sys_katsura = System(F_katsura)
    I_katsura = sys_katsura._interp_f64
    u_katsura = zeros(ComplexF64, 4)
    x_katsura = ComplexF64.(randn(4))
    SUITE["interpreter"]["eval_katsura3"] = @benchmarkable execute!($u_katsura, $I_katsura, $x_katsura)

    sys_cyc7 = System(F_cyclic7)
    I_cyc7 = sys_cyc7._interp_f64
    u_cyc7 = zeros(ComplexF64, 7)
    x_cyc7 = ComplexF64.(randn(7))
    SUITE["interpreter"]["eval_cyclic7"] = @benchmarkable execute!($u_cyc7, $I_cyc7, $x_cyc7)

    # --- Jacobian ---
    I_jac = sys_katsura._interp_jac
    U_katsura = zeros(ComplexF64, 4, 4)
    SUITE["interpreter"]["jac_katsura3"] = @benchmarkable execute!($u_katsura, $U_katsura, $I_jac, $x_katsura)

    I_jac7 = sys_cyc7._interp_jac
    U_cyc7 = zeros(ComplexF64, 7, 7)
    SUITE["interpreter"]["jac_cyclic7"] = @benchmarkable execute!($u_cyc7, $U_cyc7, $I_jac7, $x_cyc7)

    # --- Taylor ---
    I_taylor = sys_katsura._interp_t3
    tx = TaylorVector{4, ComplexF64}(4)
    for i in 1:4
        tx[i] = ntuple(k -> randn(ComplexF64), Val(4))
    end
    u_taylor = zeros(ComplexF64, 4)
    p_empty = ComplexF64[]
    SUITE["interpreter"]["taylor_katsura3"] = @benchmarkable execute_taylor!($u_taylor, Val(3), $I_taylor, $tx, $p_empty)

    # --- Build time (System construction includes full interpreter pipeline) ---
    SUITE["interpreter"]["build_jac_katsura3"] = @benchmarkable System($F_katsura)

    return SUITE
end
