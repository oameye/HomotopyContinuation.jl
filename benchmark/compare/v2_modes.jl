# Compare v2 evaluation modes: :none (interpreted) vs :all (compiled).
# Standalone: julia --project=benchmark benchmark/compare/v2_modes.jl
#
# This verifies the claim that v2's interpreted mode roughly matches compiled.
# The comparison is done inside v2 only — no v3 code involved.

if !@isdefined(print_header)
    include(joinpath(@__DIR__, "common.jl"))
end

using HomotopyContinuation.ModelKit: InterpretedSystem, CompiledSystem

print_header("v2 Internal: :none (interpreted) vs :all (compiled)")
println()
println("  This measures v2's own InterpretedSystem vs CompiledSystem.")
println("  If interpreted is close to compiled, the interpreter-only")
println("  approach in v3 does not sacrifice meaningful performance.")
println()

function compare_v2_modes(name, hc_polys)
    sys = System(hc_polys)
    n = length(HomotopyContinuation.ModelKit.variables(sys))
    m = length(hc_polys)

    sys_interp = InterpretedSystem(sys)
    sys_compiled = CompiledSystem(sys)

    x = ComplexF64.(randn(n))
    u_interp = zeros(ComplexF64, m)
    u_compiled = zeros(ComplexF64, m)
    U_interp = zeros(ComplexF64, m, n)
    U_compiled = zeros(ComplexF64, m, n)

    # Warmup
    HC.ModelKit.evaluate!(u_interp, sys_interp, x)
    HC.ModelKit.evaluate!(u_compiled, sys_compiled, x)
    HC.ModelKit.evaluate_and_jacobian!(u_interp, U_interp, sys_interp, x)
    HC.ModelKit.evaluate_and_jacobian!(u_compiled, U_compiled, sys_compiled, x)

    t_interp_eval = @belapsed HC.ModelKit.evaluate!($u_interp, $sys_interp, $x)
    t_compiled_eval = @belapsed HC.ModelKit.evaluate!($u_compiled, $sys_compiled, $x)
    t_interp_jac = @belapsed HC.ModelKit.evaluate_and_jacobian!($u_interp, $U_interp, $sys_interp, $x)
    t_compiled_jac = @belapsed HC.ModelKit.evaluate_and_jacobian!($u_compiled, $U_compiled, $sys_compiled, $x)

    eval_ratio = t_interp_eval / t_compiled_eval
    jac_ratio = t_interp_jac / t_compiled_jac

    println(
        "  ", rpad(name, 16),
        "eval: interp/compiled = ", rpad(string(round(eval_ratio; digits = 2)), 5), "x  ",
        "jac: interp/compiled = ", rpad(string(round(jac_ratio; digits = 2)), 5), "x",
    )
    return (; name, eval_ratio, jac_ratio)
end

results = NamedTuple[]

# Katsura systems
for n in [3, 4, 5]
    @var kv[1:(n + 1)]
    hc_F = _katsura_polys(kv, n)
    push!(results, compare_v2_modes("katsura$n", hc_F))
end

# Cyclic systems
for n in [5, 6, 7]
    @var cv[1:n]
    hc_F = let vars = cv
        eqs = Any[]
        for k in 1:(n - 1)
            eq = sum(prod(vars[mod1(j + i, n)] for i in 0:(k - 1)) for j in 1:n)
            push!(eqs, eq)
        end
        push!(eqs, prod(vars) - 1)
        eqs
    end
    push!(results, compare_v2_modes("cyclic$n", hc_F))
end

println()

eval_ratios = [r.eval_ratio for r in results]
jac_ratios = [r.jac_ratio for r in results]

println("  eval overhead:  min=$(round(minimum(eval_ratios); digits=2))x  median=$(round(sort(eval_ratios)[cld(length(eval_ratios),2)]; digits=2))x  max=$(round(maximum(eval_ratios); digits=2))x")
println("  jac overhead:   min=$(round(minimum(jac_ratios); digits=2))x  median=$(round(sort(jac_ratios)[cld(length(jac_ratios),2)]; digits=2))x  max=$(round(maximum(jac_ratios); digits=2))x")
println()
println("  ratio = 1.0 means identical speed. ratio = 1.05 means interpreted is 5% slower.")
println("  If median < 1.10, the claim holds: interpreted ≈ compiled.")
