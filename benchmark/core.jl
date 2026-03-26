# Phase 3: Core types benchmarks
# Measures FunctionWrapper overhead vs raw interpreter, homotopy evaluation,
# and scaling behavior across system sizes.

using BenchmarkTools
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray
using HomotopyContinuationNext:
    System, evaluate!, evaluate_and_jacobian!, taylor!,
    StraightLineHomotopy, HomotopyEvaluator,
    TaylorVector, TruncatedTaylorSeries,
    build_interpreter, build_jacobian_interpreter,
    execute!

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

function benchmark_core!(SUITE::BenchmarkGroup)
    SUITE["core"] = BenchmarkGroup()

    # ── Katsura-3 (4 eqs, 4 vars) ───────────────────────────────────────
    @polyvar x0 x1 x2 x3
    F_k3 = [
        x0 + 2x1 + 2x2 + 2x3 - 1,
        x0^2 + 2x1^2 + 2x2^2 + 2x3^2 - x0,
        2x0 * x1 + 2x1 * x2 + 2x2 * x3 - x1,
        x1^2 + 2x0 * x2 + 2x1 * x3 - x2,
    ]

    # FW-wrapped vs raw: quantify FunctionWrapper overhead
    sys_k3 = System(F_k3)
    seval_k3 = sys_k3.evaluator
    xv_k3 = FSVec{ComplexF64}(randn(ComplexF64, 4))
    p_k3 = FSVec{ComplexF64}(ComplexF64[])
    u_k3 = FSVec{ComplexF64}(zeros(ComplexF64, 4))
    U_k3 = FSMat{ComplexF64}(zeros(ComplexF64, 4, 4))

    I_eval_k3 = build_interpreter(F_k3)
    I_jac_k3 = build_jacobian_interpreter(F_k3)
    u_raw_k3 = zeros(ComplexF64, 4)
    U_raw_k3 = zeros(ComplexF64, 4, 4)
    x_raw_k3 = ComplexF64.(randn(4))

    SUITE["core"]["eval_raw_katsura3"] =
        @benchmarkable execute!($u_raw_k3, $I_eval_k3, $x_raw_k3, $(ComplexF64[]))
    SUITE["core"]["eval_fw_katsura3"] =
        @benchmarkable evaluate!($u_k3, $seval_k3, $xv_k3, $p_k3)
    SUITE["core"]["jac_raw_katsura3"] =
        @benchmarkable execute!($u_raw_k3, $U_raw_k3, $I_jac_k3, $x_raw_k3, $(ComplexF64[]))
    SUITE["core"]["jac_fw_katsura3"] =
        @benchmarkable evaluate_and_jacobian!($u_k3, $U_k3, $seval_k3, $xv_k3, $p_k3)

    # ── Cyclic-7 (7 eqs, 7 vars) — scaling check ────────────────────────
    @polyvar c1 c2 c3 c4 c5 c6 c7
    cv = [c1, c2, c3, c4, c5, c6, c7]
    F_c7 = [
        [sum(prod(cv[mod1(j + k, 7)] for k in 0:(d - 1)) for j in 1:7) for d in 1:6]
        [prod(cv) - 1]
    ]

    sys_c7 = System(F_c7)
    seval_c7 = sys_c7.evaluator
    xv_c7 = FSVec{ComplexF64}(randn(ComplexF64, 7))
    p_c7 = FSVec{ComplexF64}(ComplexF64[])
    u_c7 = FSVec{ComplexF64}(zeros(ComplexF64, 7))
    U_c7 = FSMat{ComplexF64}(zeros(ComplexF64, 7, 7))

    I_eval_c7 = build_interpreter(F_c7)
    I_jac_c7 = build_jacobian_interpreter(F_c7)
    u_raw_c7 = zeros(ComplexF64, 7)
    U_raw_c7 = zeros(ComplexF64, 7, 7)
    x_raw_c7 = ComplexF64.(randn(7))

    SUITE["core"]["eval_raw_cyclic7"] =
        @benchmarkable execute!($u_raw_c7, $I_eval_c7, $x_raw_c7, $(ComplexF64[]))
    SUITE["core"]["eval_fw_cyclic7"] =
        @benchmarkable evaluate!($u_c7, $seval_c7, $xv_c7, $p_c7)
    SUITE["core"]["jac_raw_cyclic7"] =
        @benchmarkable execute!($u_raw_c7, $U_raw_c7, $I_jac_c7, $x_raw_c7, $(ComplexF64[]))
    SUITE["core"]["jac_fw_cyclic7"] =
        @benchmarkable evaluate_and_jacobian!($u_c7, $U_c7, $seval_c7, $xv_c7, $p_c7)

    # ── StraightLineHomotopy (katsura-3) — end-to-end homotopy cost ─────
    @polyvar y0 y1 y2 y3
    G_k3 = [y0 - 1, y1^2 - 1, y2^2 - 1, y3^2 - 1]
    sys_G_k3 = System(G_k3)
    sys_F_k3 = System(F_k3)

    H = StraightLineHomotopy(sys_G_k3.evaluator, sys_F_k3.evaluator)
    heval = HomotopyEvaluator(H)
    u_hom = FSVec{ComplexF64}(zeros(ComplexF64, 4))
    U_hom = FSMat{ComplexF64}(zeros(ComplexF64, 4, 4))
    x_hom = FSVec{ComplexF64}(randn(ComplexF64, 4))
    t = ComplexF64(0.5)

    SUITE["core"]["homotopy_eval_katsura3"] =
        @benchmarkable evaluate!($u_hom, $heval, $x_hom, $t)
    SUITE["core"]["homotopy_jac_katsura3"] =
        @benchmarkable evaluate_and_jacobian!($u_hom, $U_hom, $heval, $x_hom, $t)
    SUITE["core"]["homotopy_taylor1_katsura3"] =
        @benchmarkable taylor!($u_hom, Val(1), $heval, $x_hom, $t)

    # ── Build time ───────────────────────────────────────────────────────
    SUITE["core"]["build_katsura3"] =
        @benchmarkable System($F_k3)
    SUITE["core"]["build_cyclic7"] =
        @benchmarkable System($F_c7)

    return SUITE
end
