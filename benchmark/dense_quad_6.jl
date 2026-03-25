# Dense quadratic 6x6 tape comparison for CSE debugging
# Run:
#   julia --project=benchmark benchmark/dense_quad_6.jl

using DynamicPolynomials: @polyvar
using HomotopyContinuationNext
using HomotopyContinuation
using HomotopyContinuation.ModelKit: @var, System, InterpretedSystem

const Next = HomotopyContinuationNext
const HC = HomotopyContinuation

function dense_quad_6_next()
    @polyvar x1 x2 x3 x4 x5 x6
    xs = [x1, x2, x3, x4, x5, x6]
    return [sum((i + j) * xs[i] * xs[j] for i in 1:6 for j in i:6) + xs[k] - k for k in 1:6]
end

function dense_quad_6_hc()
    @var y1 y2 y3 y4 y5 y6
    ys = [y1, y2, y3, y4, y5, y6]
    return System([sum((i + j) * ys[i] * ys[j] for i in 1:6 for j in i:6) + ys[k] - k for k in 1:6])
end

function op_histogram(seq)
    counts = Dict{Any, Int}()
    for instr in seq.instructions
        counts[instr.op] = get(counts, instr.op, 0) + 1
    end
    return sort!(collect(counts); by = x -> string(x.first))
end

function print_histogram(label, seq)
    println(label)
    for (op, count) in op_histogram(seq)
        println("  ", lpad(string(op), 18), " : ", count)
    end
end

F_next = dense_quad_6_next()
I_next = Next.build_interpreter(F_next)
IJ_next = Next.build_jacobian_interpreter(F_next)

F_hc = dense_quad_6_hc()
IS_hc = InterpretedSystem(F_hc)

next_eval = I_next.sequence
next_jac = IJ_next.sequence
hc_eval = IS_hc.eval_ComplexF64.sequence
hc_jac = IS_hc.jac_ComplexF64.sequence

next_eval_count = length(next_eval.instructions) - 1
next_jac_count = length(next_jac.instructions) - 1
hc_eval_count = length(hc_eval.instructions) - 1
hc_jac_count = length(hc_jac.instructions) - 1

println("="^72)
println("  dense_quad_6 tape comparison")
println("="^72)
println()
println("Instruction counts")
println("  eval: Next=", next_eval_count, "  HC=", hc_eval_count, "  gap=", next_eval_count - hc_eval_count)
println("  jac : Next=", next_jac_count, "  HC=", hc_jac_count, "  gap=", next_jac_count - hc_jac_count)
println()
println("Selected op counts")
println(
    "  eval OP_MUL3: Next=",
    count(instr -> instr.op == Next.OpType.OP_MUL3, next_eval.instructions),
    "  HC=",
    count(instr -> string(instr.op) == "MUL3", hc_eval.instructions),
)
println(
    "  jac  OP_MUL3: Next=",
    count(instr -> instr.op == Next.OpType.OP_MUL3, next_jac.instructions),
    "  HC=",
    count(instr -> string(instr.op) == "MUL3", hc_jac.instructions),
)
println()
print_histogram("Next eval histogram", next_eval)
println()
print_histogram("HC eval histogram", hc_eval)
println()
print_histogram("Next jac histogram", next_jac)
println()
print_histogram("HC jac histogram", hc_jac)
