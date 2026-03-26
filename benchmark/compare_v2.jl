# Compare HomotopyContinuationNext vs HomotopyContinuation (v2)
# Run: make compare
#   or: julia --project=benchmark benchmark/compare_v2.jl

using Random: MersenneTwister
using BenchmarkTools
using LinearAlgebra: LinearAlgebra, I, ldiv!, diagm

using HomotopyContinuationNext
using HomotopyContinuation
using DynamicPolynomials: @polyvar
using HomotopyContinuationNext: build_interpreter, build_jacobian_interpreter
using HomotopyContinuation.ModelKit: @var, System, InterpretedSystem
using FixedSizeArrays: FixedSizeArray

const Next = HomotopyContinuationNext
const HC = HomotopyContinuation
const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}

BenchmarkTools.DEFAULT_PARAMETERS.seconds = 0.06
BenchmarkTools.DEFAULT_PARAMETERS.samples = 4000

function print_row(name::String, t_next::Float64, t_hc::Float64)
    ratio = t_hc / t_next
    return println(
        "  ", rpad(name, 28),
        "Next=", lpad(string(round(t_next * 1.0e9; digits = 1)), 8), "ns  ",
        "HC=", lpad(string(round(t_hc * 1.0e9; digits = 1)), 8), "ns  ",
        "ratio=", round(ratio; digits = 2), "x",
    )
end

## ── Primitives ────────────────────────────────────────────────────────────

println("="^72)
println("  Primitives: HomotopyContinuationNext vs HomotopyContinuation (v2)")
println("="^72)

println("\n── inf_norm ──")
for n in [4, 16, 64]
    x_fs = FSVec{ComplexF64}(rand(ComplexF64, n))
    x_vec = Vector{ComplexF64}(collect(x_fs))
    t_next = @belapsed Next.inf_norm($x_fs)
    hc_inf = HC.InfNorm()
    t_hc = @belapsed $hc_inf($x_vec)
    print_row("inf_norm n=$n", t_next, t_hc)
end

println("\n── LU + ldiv! (well-conditioned) ──")
for n in [4, 8, 16]
    A_data = rand(ComplexF64, n, n) + 5.0I
    b_data = rand(ComplexF64, n)

    WS_next = Next.MatrixWorkspace(n, n)
    copyto!(WS_next.A, A_data); Next.updated!(WS_next)
    x_next = FSVec{ComplexF64}(zeros(ComplexF64, n))
    b_next = FSVec{ComplexF64}(b_data)
    t_next = @belapsed begin
        copyto!($WS_next.A, $A_data); Next.updated!($WS_next); ldiv!($x_next, $WS_next, $b_next)
    end

    WS_hc = HC.MatrixWorkspace(n, n)
    copyto!(WS_hc.A, A_data); HC.updated!(WS_hc)
    x_hc = zeros(ComplexF64, n)
    b_hc = Vector{ComplexF64}(b_data)
    t_hc = @belapsed begin
        copyto!($WS_hc.A, $A_data); HC.updated!($WS_hc); ldiv!($x_hc, $WS_hc, $b_hc)
    end
    print_row("lu_ldiv n=$n", t_next, t_hc)
end

## ── Interpreter sweep ─────────────────────────────────────────────────────

println("\n" * "="^72)
println("  Interpreter: HomotopyContinuationNext vs HomotopyContinuation (v2)")
println("="^72)

# Polynomial system generators from exponent specs
@polyvar nx1 nx2 nx3 nx4 nx5 nx6 nx7 nx8
const NEXT_VARS = [nx1, nx2, nx3, nx4, nx5, nx6, nx7, nx8]
@var hx1 hx2 hx3 hx4 hx5 hx6 hx7 hx8
const HC_VARS = [hx1, hx2, hx3, hx4, hx5, hx6, hx7, hx8]

function _term(vars, coeff, exps)
    t = coeff
    @inbounds for i in eachindex(exps)
        e = exps[i]
        e == 0 && continue
        t *= e == 1 ? vars[i] : vars[i]^e
    end
    return t
end
_poly(vars, spec) = reduce(+, (_term(vars, c, e) for (c, e) in spec))

function _cyclic_specs(n)
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

function _chain_specs(n)
    specs = Vector{Vector{Tuple{ComplexF64, Vector{Int}}}}(undef, n)
    for j in 1:n
        jp = mod1(j + 1, n); jm = mod1(j - 1, n)
        poly = Tuple{ComplexF64, Vector{Int}}[]
        for (coeff, idxs) in [
                (1.0, [(j, 2)]), (1.0, [(jp, 2)]), (2.0, [(j, 1), (jp, 1)]),
                (-1.0, [(j, 1), (jm, 1)]), (1.0, [(j, 1)]), (-1.0, Int[]),
            ]
            e = zeros(Int, n)
            for (idx, exp) in idxs
                e[idx] = exp
            end
            push!(poly, (ComplexF64(coeff), e))
        end
        specs[j] = poly
    end
    return specs
end

function _random_sparse_specs(rng, n, m; terms_per_poly = 10)
    specs = Vector{Vector{Tuple{ComplexF64, Vector{Int}}}}(undef, m)
    for j in 1:m
        poly = Tuple{ComplexF64, Vector{Int}}[]
        for _ in 1:terms_per_poly
            exps = zeros(Int, n)
            for _ in 1:rand(rng, 2:4)
                exps[rand(rng, 1:n)] += 1
            end
            push!(poly, (ComplexF64(rand(rng, [-2, -1, 1, 2]), rand(rng, [-1, 0, 1])), exps))
        end
        push!(poly, (ComplexF64(rand(rng, -2:2), rand(rng, [-1, 0, 1])), zeros(Int, n)))
        specs[j] = poly
    end
    return specs
end

function compare_case(name::String, specs)
    n = length(first(first(specs))[2])
    next_polys = [_poly(NEXT_VARS[1:n], spec) for spec in specs]
    hc_polys = [_poly(HC_VARS[1:n], spec) for spec in specs]

    I_next = build_interpreter(next_polys)
    J_next = build_jacobian_interpreter(next_polys)
    IS_hc = InterpretedSystem(System(hc_polys))

    m = length(specs)
    x = ComplexF64.(randn(n))
    u_next = zeros(ComplexF64, m); u_hc = zeros(ComplexF64, m)
    U_next = zeros(ComplexF64, m, n); U_hc = zeros(ComplexF64, m, n)

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

results = NamedTuple[]
println()
for n in [5, 6, 7]
    push!(results, compare_case("cyclic$n", _cyclic_specs(n)))
end
for n in [5, 6, 7]
    push!(results, compare_case("chain$n", _chain_specs(n)))
end
for seed in 1:2
    push!(results, compare_case("sparse6_$seed", _random_sparse_specs(MersenneTwister(seed), 6, 6)))
end

println("\n" * "="^72)
for field in (:eval_ratio, :jac_ratio, :build_ratio)
    vals = getfield.(results, field)
    println(
        "  ", rpad(string(field), 14),
        "min=", rpad(string(round(minimum(vals); digits = 2)), 5),
        " median=", rpad(string(round(sort(vals)[cld(length(vals), 2)]; digits = 2)), 5),
        " max=", round(maximum(vals); digits = 2),
    )
end
println("  ratio > 1.0 means Next is faster")
println("="^72)
