using Test
using HomotopyContinuationNext
using HomotopyContinuationNextCertification:
    IteratorCertification, certify, ncertified, ndistinct_certified, nnotcertified,
    ntracked, nleaves, max_leaf_size, oversized_leaves, unsplittable_leaves,
    ncandidates
using Random: MersenneTwister

isdefined(@__MODULE__, :steiner_system) || include("../test_systems.jl")

# Certification of a `ResultIterator` at a size where it is the point: 27072 paths
# and 3264 solutions, certified in leaves of at most 200 without ever holding more
# than one leaf's certificates. The counts must match what certifying the
# collected solutions gives, and the tracking must stay linear in the path count.
@testset "3264 conics through the iterator route" begin
    polys, vars, params = steiner_system()
    F = System(polys; variables = vars, parameters = params)
    p = randn(MersenneTwister(2), ComplexF64, 30)

    ri = result_iterator(
        fix_parameters(F, p), Polyhedral(; seed = 0x9a8b7c6d, show_progress = false),
    )
    res = certify(
        F, ri, p,
        IteratorCertification(;
            leaf_size_bound = 200, boundaries = -10:0.5:10, show_progress = false,
        ),
    )

    @test ncertified(res) == 3264
    @test ndistinct_certified(res) == 3264
    # The polyhedral solve also ends on endpoints that are not solutions; they
    # reach the certifier here, where the eagerly collected route never sees them,
    # and each one is reported uncertified rather than certified or thrown on.
    @test ncandidates(res) == 3264 + nnotcertified(res)
    @test max_leaf_size(res) <= 200
    @test oversized_leaves(res) == 0
    @test unsplittable_leaves(res) == 0

    # One pass over the 27072 paths places the solutions, then each terminal leaf
    # is tracked once more, which together cover the certified ones exactly.
    # Refinement adds no pass of its own, so anything approaching a multiple of
    # 27072 here means the partition is being rebuilt by re-tracking.
    @test ntracked(res) == 27072 + ncertified(res)
    # Balanced cuts keep the tree shallow: 3264 enclosures at 200 per leaf need
    # tens of leaves, not thousands.
    @test nleaves(res) <= 100
end
