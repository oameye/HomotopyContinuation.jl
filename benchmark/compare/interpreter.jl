# Compare interpreter: eval, jacobian, build time across system families.
# Standalone: julia --project=benchmark benchmark/compare/interpreter.jl

if !@isdefined(print_header)
    include(joinpath(@__DIR__, "common.jl"))
end

using HomotopyContinuationNext: build_interpreter, build_jacobian_interpreter

print_header("Interpreter: Next vs HC v2")

function compare_case(name::String, specs)
    n = length(first(first(specs))[2])
    next_polys = [_poly(NEXT_VARS[1:n], spec) for spec in specs]
    hc_polys = [_poly(HC_VARS[1:n], spec) for spec in specs]

    I_next = build_interpreter(next_polys)
    J_next = build_jacobian_interpreter(next_polys)
    IS_hc = InterpretedSystem(System(hc_polys))

    m = length(specs)
    x = ComplexF64.(randn(n))
    u_next = zeros(ComplexF64, m)
    u_hc = zeros(ComplexF64, m)
    U_next = zeros(ComplexF64, m, n)
    U_hc = zeros(ComplexF64, m, n)

    t_next_eval = @belapsed Next.execute!($u_next, $I_next, $x)
    t_hc_eval = @belapsed HC.ModelKit.evaluate!($u_hc, $IS_hc, $x)
    t_next_jac = @belapsed Next.execute!($u_next, $U_next, $J_next, $x)
    t_hc_jac = @belapsed HC.ModelKit.evaluate_and_jacobian!($u_hc, $U_hc, $IS_hc, $x)
    t_next_build = @belapsed build_interpreter($next_polys)
    t_hc_build = @belapsed InterpretedSystem(System($hc_polys))

    println(
        "  ", rpad(name, 16),
        "eval=", rpad(string(round(t_hc_eval / t_next_eval; digits = 2)), 5), "x  ",
        "jac=", rpad(string(round(t_hc_jac / t_next_jac; digits = 2)), 5), "x  ",
        "build=", rpad(string(round(t_hc_build / t_next_build; digits = 1)), 5), "x",
    )
    return (;
        name,
        eval_ratio = t_hc_eval / t_next_eval,
        jac_ratio = t_hc_jac / t_next_jac,
        build_ratio = t_hc_build / t_next_build,
    )
end

interp_results = NamedTuple[]
println()
for n in [5, 6, 7]
    push!(interp_results, compare_case("cyclic$n", _cyclic_specs(n)))
end
for n in [5, 6, 7]
    push!(interp_results, compare_case("chain$n", _chain_specs(n)))
end
for seed in 1:2
    push!(interp_results, compare_case("sparse6_$seed", _random_sparse_specs(MersenneTwister(seed), 6, 6)))
end

println()
for field in (:eval_ratio, :jac_ratio, :build_ratio)
    vals = getfield.(interp_results, field)
    println(
        "  ", rpad(string(field), 14),
        "min=", rpad(string(round(minimum(vals); digits = 2)), 5),
        " median=", rpad(string(round(sort(vals)[cld(length(vals), 2)]; digits = 2)), 5),
        " max=", round(maximum(vals); digits = 2),
    )
end
println("  ratio > 1.0 means Next is faster")
