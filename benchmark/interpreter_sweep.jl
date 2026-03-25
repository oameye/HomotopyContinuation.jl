using Random
using BenchmarkTools
using DynamicPolynomials: @polyvar
using HomotopyContinuationNext
using HomotopyContinuation
using HomotopyContinuationNext: build_interpreter, build_jacobian_interpreter
using HomotopyContinuation.ModelKit: @var, System, InterpretedSystem

const Next = HomotopyContinuationNext
const HC = HomotopyContinuation

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.06
BenchmarkTools.DEFAULT_PARAMETERS.samples = 4000

@polyvar nx1 nx2 nx3 nx4 nx5 nx6 nx7 nx8
const NEXT_VARS = [nx1, nx2, nx3, nx4, nx5, nx6, nx7, nx8]
@var hx1 hx2 hx3 hx4 hx5 hx6 hx7 hx8
const HC_VARS = [hx1, hx2, hx3, hx4, hx5, hx6, hx7, hx8]

function term_from_spec(vars, coeff, exps)
    term = coeff
    @inbounds for i in eachindex(exps)
        e = exps[i]
        if e == 1
            term *= vars[i]
        elseif e != 0
            term *= vars[i]^e
        end
    end
    return term
end

poly_from_spec(vars, spec) = reduce(+, (term_from_spec(vars, c, exps) for (c, exps) in spec))

function cyclic_specs(n)
    specs = Vector{Vector{Tuple{ComplexF64, Vector{Int}}}}()
    for k in 1:(n - 1)
        poly = Tuple{ComplexF64, Vector{Int}}[]
        for start in 1:n
            exps = zeros(Int, n)
            for off in 0:(k - 1)
                exps[mod1(start + off, n)] += 1
            end
            push!(poly, (1.0 + 0im, exps))
        end
        push!(specs, poly)
    end
    push!(specs, [(1.0 + 0im, ones(Int, n)), (-1.0 + 0im, zeros(Int, n))])
    return specs
end

function chain_specs(n)
    specs = Vector{Vector{Tuple{ComplexF64, Vector{Int}}}}(undef, n)
    for j in 1:n
        jp = mod1(j + 1, n)
        jm = mod1(j - 1, n)
        poly = Tuple{ComplexF64, Vector{Int}}[]

        e = zeros(Int, n)
        e[j] = 2
        push!(poly, (1.0 + 0im, copy(e)))

        fill!(e, 0)
        e[jp] = 2
        push!(poly, (1.0 + 0im, copy(e)))

        fill!(e, 0)
        e[j] = 1
        e[jp] = 1
        push!(poly, (2.0 + 0im, copy(e)))

        fill!(e, 0)
        e[j] = 1
        e[jm] = 1
        push!(poly, (-1.0 + 0im, copy(e)))

        fill!(e, 0)
        e[j] = 1
        push!(poly, (1.0 + 0im, copy(e)))

        push!(poly, (-1.0 + 0im, zeros(Int, n)))
        specs[j] = poly
    end
    return specs
end

function dense_quadratic_specs(n)
    specs = Vector{Vector{Tuple{ComplexF64, Vector{Int}}}}(undef, n)
    for j in 1:n
        poly = Tuple{ComplexF64, Vector{Int}}[]
        for i in 1:n
            e = zeros(Int, n)
            e[i] = 1
            c = ComplexF64(((i + 2j) % 5) - 2, ((j - i) % 3) - 1)
            c == 0 && continue
            push!(poly, (c, e))
        end
        for i in 1:n
            for k in i:n
                e = zeros(Int, n)
                e[i] += 1
                e[k] += 1
                c = ComplexF64(((i + 3k + j) % 7) - 3, ((2i + k + j) % 3) - 1)
                c == 0 && continue
                push!(poly, (c, e))
            end
        end
        push!(poly, (ComplexF64((j % 5) - 2, (j % 3) - 1), zeros(Int, n)))
        specs[j] = poly
    end
    return specs
end

function random_sparse_specs(rng, n, m; terms_per_poly = 10)
    specs = Vector{Vector{Tuple{ComplexF64, Vector{Int}}}}(undef, m)
    coeffs_r = [-2, -1, 1, 2]
    coeffs_i = [-1, 0, 1]
    for j in 1:m
        poly = Tuple{ComplexF64, Vector{Int}}[]
        for _ in 1:terms_per_poly
            deg = rand(rng, 2:4)
            exps = zeros(Int, n)
            for _ in 1:deg
                exps[rand(rng, 1:n)] += 1
            end
            push!(poly, (ComplexF64(rand(rng, coeffs_r), rand(rng, coeffs_i)), exps))
        end
        push!(poly, (ComplexF64(rand(rng, -2:2), rand(rng, coeffs_i)), zeros(Int, n)))
        specs[j] = poly
    end
    return specs
end

function compare_case(name, specs)
    n = length(first(first(specs))[2])
    next_polys = [poly_from_spec(NEXT_VARS[1:n], spec) for spec in specs]
    hc_polys = [poly_from_spec(HC_VARS[1:n], spec) for spec in specs]

    I_next = build_interpreter(next_polys)
    J_next = build_jacobian_interpreter(next_polys)
    IS_hc = InterpretedSystem(System(hc_polys))

    m = length(specs)
    x = ComplexF64.(randn(n))
    u_next = zeros(ComplexF64, m)
    u_hc = zeros(ComplexF64, m)
    U_next = zeros(ComplexF64, m, n)
    U_hc = zeros(ComplexF64, m, n)

    Next.execute!(u_next, I_next, x)
    HC.ModelKit.evaluate!(u_hc, IS_hc, x)
    eval_err = maximum(abs.(u_next .- u_hc))

    Next.execute!(u_next, U_next, J_next, x)
    HC.ModelKit.evaluate_and_jacobian!(u_hc, U_hc, IS_hc, x)
    jac_err = maximum(abs.(U_next .- U_hc))

    t_next_eval = @belapsed Next.execute!($u_next, $I_next, $x)
    t_hc_eval = @belapsed HC.ModelKit.evaluate!($u_hc, $IS_hc, $x)
    t_next_jac = @belapsed Next.execute!($u_next, $U_next, $J_next, $x)
    t_hc_jac = @belapsed HC.ModelKit.evaluate_and_jacobian!($u_hc, $U_hc, $IS_hc, $x)

    println(
        name,
        ": eval_ratio=", round(t_hc_eval / t_next_eval; digits = 2),
        " jac_ratio=", round(t_hc_jac / t_next_jac; digits = 2),
        " eval_ns=", round(t_next_eval * 1.0e9; digits = 1), "/", round(t_hc_eval * 1.0e9; digits = 1),
        " jac_ns=", round(t_next_jac * 1.0e9; digits = 1), "/", round(t_hc_jac * 1.0e9; digits = 1),
        " err=", max(eval_err, jac_err),
    )
    return (eval_ratio = t_hc_eval / t_next_eval, jac_ratio = t_hc_jac / t_next_jac)
end

function summarize(results, field)
    vals = getfield.(results, field)
    ord = sortperm(vals)
    sorted = sort(vals)
    println(
        "summary ", field,
        ": min=", round(vals[ord[1]]; digits = 2),
        " median=", round(sorted[cld(length(sorted), 2)]; digits = 2),
        " max=", round(vals[ord[end]]; digits = 2),
    )
    for idx in ord[1:min(3, length(ord))]
        println("  worst ", results[idx].name, " -> ", round(vals[idx]; digits = 2), "x")
    end
    return
end

println("="^72)
println("  Interpreter Sweep: HomotopyContinuationNext vs HomotopyContinuation")
println("="^72)

results = NamedTuple[]
push!(results, (; name = "cyclic6", compare_case("cyclic6", cyclic_specs(6))...))
push!(results, (; name = "cyclic7", compare_case("cyclic7", cyclic_specs(7))...))
push!(results, (; name = "chain6", compare_case("chain6", chain_specs(6))...))
push!(results, (; name = "densequad6", compare_case("densequad6", dense_quadratic_specs(6))...))
for seed in 1:6
    rng = MersenneTwister(seed)
    name = "sparse6_$(seed)"
    push!(results, (; name, compare_case(name, random_sparse_specs(rng, 6, 6; terms_per_poly = 10))...))
end

println("\n" * "="^72)
summarize(results, :eval_ratio)
summarize(results, :jac_ratio)
println("="^72)
println("ratio > 1.0 means Next is faster")
println("="^72)
