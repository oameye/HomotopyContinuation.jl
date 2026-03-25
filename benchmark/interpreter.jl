# Phase 2: Interpreter pipeline benchmarks
# Measures tape execution overhead and compilation cost.

using BenchmarkTools
using DynamicPolynomials: @polyvar
using HomotopyContinuationNext: build_interpreter, build_jacobian_interpreter, execute!

function benchmark_interpreter!(SUITE::BenchmarkGroup)
    SUITE["interpreter"] = BenchmarkGroup()

    # --- Execution: tape dispatch overhead at different system sizes ---
    # Small (2 vars, 2 polys): dominated by dispatch overhead per instruction
    # Medium (4 vars, 4 polys): Katsura-3, typical HC workload
    # Larger (5 vars, 5 polys): Cyclic-5, more instructions per evaluation

    @polyvar x y
    F_small = [x^2 + y, x * y - 1]
    I_small = build_interpreter(F_small)
    u_small = zeros(ComplexF64, 2)
    x_small = ComplexF64[1.5, -0.5]
    SUITE["interpreter"]["execute_2x2"] = @benchmarkable execute!($u_small, $I_small, $x_small, $(ComplexF64[]))

    @polyvar x0 x1 x2 x3
    F_katsura = [
        x0 + 2x1 + 2x2 + 2x3 - 1,
        x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
        2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
        x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
    ]
    I_katsura = build_interpreter(F_katsura)
    u_katsura = zeros(ComplexF64, 4)
    x_katsura = ComplexF64.(randn(4))
    SUITE["interpreter"]["execute_katsura3"] = @benchmarkable execute!($u_katsura, $I_katsura, $x_katsura, $(ComplexF64[]))

    @polyvar x1 x2 x3 x4 x5
    F_cyclic = [
        x1 + x2 + x3 + x4 + x5,
        x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1,
        x1 * x2 * x3 + x2 * x3 * x4 + x3 * x4 * x5 + x4 * x5 * x1 + x5 * x1 * x2,
        x1 * x2 * x3 * x4 + x2 * x3 * x4 * x5 + x3 * x4 * x5 * x1 +
            x4 * x5 * x1 * x2 + x5 * x1 * x2 * x3,
        x1 * x2 * x3 * x4 * x5 - 1,
    ]
    I_cyclic = build_interpreter(F_cyclic)
    u_cyclic = zeros(ComplexF64, 5)
    x_cyclic = ComplexF64.(randn(5))
    SUITE["interpreter"]["execute_cyclic5"] = @benchmarkable execute!($u_cyclic, $I_cyclic, $x_cyclic, $(ComplexF64[]))

    # --- Execution with Jacobian: more work per call ---
    I_jac = build_jacobian_interpreter(F_katsura)
    U_katsura = zeros(ComplexF64, 4, 4)
    SUITE["interpreter"]["execute_jac_katsura3"] = @benchmarkable execute!($u_katsura, $U_katsura, $I_jac, $x_katsura, $(ComplexF64[]))

    return SUITE
end
