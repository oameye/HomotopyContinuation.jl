using Test, Random
using HomotopyContinuationNext
using HomotopyContinuationNext: VoronoiTree, search_in_radius, add!, InfNorm, inf_distance

# Brute-force oracle: first id within tol, or nothing.
function brute_search(pts::Vector{Vector{ComplexF64}}, ids::Vector{Int}, v, tol)
    for (p, id) in zip(pts, ids)
        inf_distance(p, v) < tol && return id
    end
    return nothing
end

@testset "VoronoiTree vs brute-force oracle" begin
    Random.seed!(1234)
    d, tol = 4, 1.0e-4
    tree = VoronoiTree{ComplexF64}(d)
    pts = Vector{Vector{ComplexF64}}()
    ids = Int[]
    next_id = 1
    mismatches = 0
    for k in 1:3000
        # Mix of fresh random points and near-duplicates straddling the tolerance
        v = if k % 3 == 0 && !isempty(pts)
            base = pts[rand(1:length(pts))]
            offset = randn(ComplexF64, d)
            base .+ (tol * (0.1 + 2.9 * rand())) .* offset ./ inf_distance(offset, zeros(ComplexF64, d))
        else
            randn(ComplexF64, d)
        end
        expected = brute_search(pts, ids, v, tol)
        got = search_in_radius(tree, v, tol)
        got == expected || (mismatches += 1)
        if expected === nothing
            add!(tree, v, next_id, tol)
            push!(pts, v)
            push!(ids, next_id)
            next_id += 1
        end
    end
    @test mismatches == 0
    @test length(tree) == length(pts)
end

@testset "10k points: insert, collect, radius search" begin
    Random.seed!(0xabc0)
    data = [randn(ComplexF64, 12) for i in 1:10_000]
    tree = VoronoiTree{ComplexF64}(12)
    for (i, d) in enumerate(data)
        insert!(tree, d, i)
    end
    @test length(tree) == length(data)
    @test sort!(collect(tree)) == 1:10_000
    @test all(i -> search_in_radius(tree, data[i], 1.0e-12) == i, 1:length(data))

    # add! finds each already-inserted point instead of re-inserting
    let tree2 = VoronoiTree{ComplexF64}(12)
        @test all(enumerate(data)) do (i, d)
            add!(tree2, d, i, 1.0e-12) == (i, true) &&
                search_in_radius(tree2, d, 1.0e-12) == i
        end
    end

    d = data[9342] .+ 1.0e-5
    @test search_in_radius(tree, d, 1.0e-4) == 9342
    @test search_in_radius(tree, d, 1.0e-6) === nothing
end

@testset "many points at nearly identical distance" begin
    Random.seed!(0xabc1)
    # 100 points on the unit circle: all have almost the same distance to the
    # origin, stressing the case-3 sorted sweep of the search.
    p = shuffle!([[cis(k / 100 * 2π)] for k in 0:99])
    tree = VoronoiTree{ComplexF64}(1)
    for (i, pi) in enumerate(p)
        insert!(tree, pi, i)
    end
    @test isnothing(search_in_radius(tree, [0.0 + 0im], 1.0e-5))
    insert!(tree, [1.0e-5 + 0im], 101)
    @test isnothing(search_in_radius(tree, [0.0 + 0im], 1.0e-6))
    @test search_in_radius(tree, [0.0 + 0im], 1.0e-4) == 101
end

@testset "empty! and collect" begin
    tree = VoronoiTree{ComplexF64}(2)
    add!(tree, [1.0 + 0im, 2.0 + 0im], 1, 1.0e-6)
    add!(tree, [3.0 + 0im, 4.0 + 0im], 2, 1.0e-6)
    @test length(tree) == 2
    @test sort(collect(tree)) == [1, 2]
    empty!(tree)
    @test length(tree) == 0
    @test search_in_radius(tree, [1.0 + 0im, 2.0 + 0im], 1.0e-6) === nothing
end
