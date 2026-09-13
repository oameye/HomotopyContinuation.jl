## Instruction count regression + correctness test
#
# For every system:
# 1. Numerical correctness: eval and Jacobian match ground truth (direct MP evaluation)
# 2. Instruction count: must not regress beyond recorded maximum

using Test
using Random: MersenneTwister
import HomotopyContinuationNext as Next
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: differentiate as mp_diff

# ── Polynomial system generators ─────────────────────────────────────────────

@polyvar _nx[1:8]

function _term(vars, coeff, exps)
    t = coeff
    for i in eachindex(exps)
        e = exps[i]
        e == 1 && (t *= vars[i])
        e > 1 && (t *= vars[i]^e)
    end
    return t
end
_poly(vars, spec) = reduce(+, (_term(vars, c, exps) for (c, exps) in spec))

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
        jp = mod1(j + 1, n)
        jm = mod1(j - 1, n)
        poly = Tuple{ComplexF64, Vector{Int}}[]
        e = zeros(Int, n); e[j] = 2; push!(poly, (1.0 + 0im, copy(e)))
        fill!(e, 0); e[jp] = 2; push!(poly, (1.0 + 0im, copy(e)))
        fill!(e, 0); e[j] = 1; e[jp] = 1; push!(poly, (2.0 + 0im, copy(e)))
        fill!(e, 0); e[j] = 1; e[jm] = 1; push!(poly, (-1.0 + 0im, copy(e)))
        fill!(e, 0); e[j] = 1; push!(poly, (1.0 + 0im, copy(e)))
        push!(poly, (-1.0 + 0im, zeros(Int, n)))
        specs[j] = poly
    end
    return specs
end

function _dense_quadratic_specs(n)
    specs = Vector{Vector{Tuple{ComplexF64, Vector{Int}}}}(undef, n)
    for j in 1:n
        poly = Tuple{ComplexF64, Vector{Int}}[]
        for i in 1:n
            e = zeros(Int, n); e[i] = 1
            c = ComplexF64(((i + 2j) % 5) - 2, ((j - i) % 3) - 1)
            c == 0 && continue
            push!(poly, (c, e))
        end
        for i in 1:n, k in i:n
            e = zeros(Int, n); e[i] += 1; e[k] += 1
            c = ComplexF64(((i + 3k + j) % 7) - 3, ((2i + k + j) % 3) - 1)
            c == 0 && continue
            push!(poly, (c, e))
        end
        push!(poly, (ComplexF64((j % 5) - 2, (j % 3) - 1), zeros(Int, n)))
        specs[j] = poly
    end
    return specs
end

function _random_sparse_specs(rng, n, m; terms_per_poly = 10)
    specs = Vector{Vector{Tuple{ComplexF64, Vector{Int}}}}(undef, m)
    for j in 1:m
        poly = Tuple{ComplexF64, Vector{Int}}[]
        for _ in 1:terms_per_poly
            deg = rand(rng, 2:4)
            exps = zeros(Int, n)
            for _ in 1:deg
                exps[rand(rng, 1:n)] += 1
            end
            push!(poly, (ComplexF64(rand(rng, [-2, -1, 1, 2]), rand(rng, [-1, 0, 1])), exps))
        end
        push!(poly, (ComplexF64(rand(rng, -2:2), rand(rng, [-1, 0, 1])), zeros(Int, n)))
        specs[j] = poly
    end
    return specs
end

# ── Test infrastructure ──────────────────────────────────────────────────────

function _test_system(name::String, specs; max_eval_instrs::Int)
    n = length(first(first(specs))[2])
    nv = collect(_nx[1:n])
    next_polys = [_poly(nv, s) for s in specs]
    m = length(specs)

    sys = Next.System(next_polys)
    I_eval = sys._interp_f64
    I_jac = sys._interp_jac

    x = ComplexF64.(randn(n))
    u = zeros(ComplexF64, m)
    U = zeros(ComplexF64, m, n)

    # ── Eval correctness vs ground truth (MP evaluation) ─────────────────
    Next.execute!(u, I_eval, x)
    u_mp = ComplexF64[p(nv => x) for p in next_polys]
    @test maximum(abs.(u .- u_mp)) < 1.0e-10

    # ── Jacobian correctness vs ground truth (MP differentiation) ────────
    Next.execute!(u, U, I_jac, x)
    J_mp = zeros(ComplexF64, m, n)
    for j in 1:n, i in 1:m
        dp = mp_diff(next_polys[i], nv[j])
        J_mp[i, j] = dp(nv => x)
    end
    @test maximum(abs.(u .- u_mp)) < 1.0e-10
    @test maximum(abs.(U .- J_mp)) < 1.0e-10

    # ── Instruction count regression check ───────────────────────────────
    ne = length(I_eval.sequence.instructions)
    return @test ne <= max_eval_instrs
end

# ── Recorded instruction counts ──────────────────────────────────────────────
# These are the current eval instruction counts. Tests fail if we regress.
# If you improve the pipeline, lower these numbers.
#
# Each number is the largest count over the Julia versions we support, because
# the count is not the same on all of them. `_eadd` and `_emul` put a sum or
# product in canonical order with `sort!(collected; lt = _expr_lt)`, and
# `_expr_lt` compares `hash`. `hash(::ComplexF64)` changed between 1.11 and
# 1.13, so systems with complex coefficients reach CSE with their terms in a
# different order and come out a few instructions apart: 13 of the 26 below
# differ across those two versions, in both directions.
#
# So lower a number only once the smaller count holds on the oldest supported
# Julia, not just on the newest. Making `_expr_lt` order on something stable
# across Julia versions would remove the split, and is worth doing separately.

const _MAX_EVAL_INSTRS = Dict(
    # Cyclic (real integer coefficients)
    "cyclic_3" => 6,
    "cyclic_4" => 11,
    "cyclic_5" => 19,
    "cyclic_6" => 28,
    "cyclic_7" => 38,
    # Chain (real integer coefficients, shared structure)
    "chain_3" => 17,
    "chain_4" => 23,
    "chain_5" => 28,
    "chain_6" => 34,
    "chain_7" => 39,
    # Dense quadratic (complex coefficients)
    "dense_quad_3" => 28,
    "dense_quad_4" => 53,
    "dense_quad_5" => 92,
    "dense_quad_6" => 140,
    # Random sparse 6×6 (complex coefficients, degree 2–4)
    "sparse6_1" => 95,
    "sparse6_2" => 96,
    "sparse6_3" => 91,
    "sparse6_4" => 85,
    "sparse6_5" => 86,
    "sparse6_6" => 96,
    "sparse6_7" => 90,
    "sparse6_8" => 93,
    # Random sparse 8×8 (complex coefficients, degree 2–4)
    "sparse8_1" => 170,
    "sparse8_2" => 170,
    "sparse8_3" => 170,
    "sparse8_4" => 170,
)

# ── Test suite ───────────────────────────────────────────────────────────────

@testset "Instruction count + correctness" begin
    @testset "Cyclic systems" begin
        for n in 3:7
            @testset "cyclic-$n" begin
                name = "cyclic_$n"
                _test_system(
                    name, _cyclic_specs(n);
                    max_eval_instrs = _MAX_EVAL_INSTRS[name]
                )
            end
        end
    end

    @testset "Chain systems" begin
        for n in 3:7
            @testset "chain-$n" begin
                name = "chain_$n"
                _test_system(
                    name, _chain_specs(n);
                    max_eval_instrs = _MAX_EVAL_INSTRS[name]
                )
            end
        end
    end

    @testset "Dense quadratic systems" begin
        for n in 3:6
            @testset "dense_quad-$n" begin
                name = "dense_quad_$n"
                _test_system(
                    name, _dense_quadratic_specs(n);
                    max_eval_instrs = _MAX_EVAL_INSTRS[name]
                )
            end
        end
    end

    @testset "Random sparse 6×6" begin
        for seed in 1:8
            @testset "sparse6_$seed" begin
                name = "sparse6_$seed"
                _test_system(
                    name,
                    _random_sparse_specs(MersenneTwister(seed), 6, 6; terms_per_poly = 10);
                    max_eval_instrs = _MAX_EVAL_INSTRS[name]
                )
            end
        end
    end

    @testset "Random sparse 8×8" begin
        for seed in 1:4
            @testset "sparse8_$seed" begin
                name = "sparse8_$seed"
                _test_system(
                    name,
                    _random_sparse_specs(MersenneTwister(100 + seed), 8, 8; terms_per_poly = 15);
                    max_eval_instrs = _MAX_EVAL_INSTRS[name]
                )
            end
        end
    end
end
