# Shared imports and utilities for v2 comparison benchmarks.

using Random: MersenneTwister
using BenchmarkTools
using LinearAlgebra: LinearAlgebra, I, ldiv!, diagm

using HomotopyContinuationNext
using HomotopyContinuation
using DynamicPolynomials: @polyvar
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

function print_header(title::String)
    println("\n" * "="^72)
    println("  ", title)
    return println("="^72)
end

## ── Polynomial system generators (shared by interpreter + tracking) ──────

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
        jp = mod1(j + 1, n)
        jm = mod1(j - 1, n)
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

## ── Katsura system generator ─────────────────────────────────────────────

function _katsura_polys(vars, n)
    # Build linear equation: v1 + 2v2 + 2v3 + ... - 1
    lin = vars[1] + sum(2vars[i] for i in 2:(n + 1)) - 1
    eqs = [lin]
    # Build quadratic convolution equations
    for l in 0:(n - 1)
        eq = -vars[l + 1]
        for i in (-n):n
            j = l - i
            if abs(i) <= n && abs(j) <= n
                eq += vars[abs(i) + 1] * vars[abs(j) + 1]
            end
        end
        push!(eqs, eq)
    end
    return eqs
end

## ── Total-degree start solutions ─────────────────────────────────────────

function _td_starts(degrees)
    td = Iterators.product((cis.(2π .* (0:(d - 1)) ./ d) for d in degrees)...)
    return [ComplexF64[s...] for s in td]
end
