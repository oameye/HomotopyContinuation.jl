## CSE refactor correctness tests
#
# These tests verify numerical correctness of eval + jacobian across
# all benchmark systems. They must pass before AND after each refactor step.

using Test
using Random: MersenneTwister
import HomotopyContinuationNext as Next
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: differentiate as mp_diff

@polyvar _rx[1:8]

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

function _verify_system(name, specs)
    n = length(first(first(specs))[2])
    nv = collect(_rx[1:n])
    polys = [_poly(nv, s) for s in specs]
    m = length(specs)
    x = ComplexF64.(randn(n))

    I_eval = Next.build_interpreter(polys)
    I_jac = Next.build_jacobian_interpreter(polys)

    u = zeros(ComplexF64, m)
    U = zeros(ComplexF64, m, n)

    Next.execute!(u, I_eval, x)
    u_mp = ComplexF64[p(nv => x) for p in polys]
    @test maximum(abs.(u .- u_mp)) < 1.0e-10

    Next.execute!(u, U, I_jac, x)
    J_mp = zeros(ComplexF64, m, n)
    for j in 1:n, i in 1:m
        dp = mp_diff(polys[i], nv[j])
        J_mp[i, j] = dp(nv => x)
    end
    @test maximum(abs.(u .- u_mp)) < 1.0e-10
    return @test maximum(abs.(U .- J_mp)) < 1.0e-10
end

@testset "CSE refactor correctness" begin
    @testset "cyclic-$n" for n in 3:7
        _verify_system("cyclic_$n", _cyclic_specs(n))
    end
    @testset "chain-$n" for n in 3:7
        _verify_system("chain_$n", _chain_specs(n))
    end
    @testset "sparse6_$seed" for seed in 1:8
        _verify_system("sparse6_$seed", _random_sparse_specs(MersenneTwister(seed), 6, 6; terms_per_poly = 10))
    end
end
