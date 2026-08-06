using Test
using HomotopyContinuationNext:
    @var, System, Polyhedral, Serial, Threaded, fix_parameters, result_iterator,
    restrict, selection, is_success, solve, solutions, slice, rand_subspace,
    ParameterHomotopy
using HomotopyContinuationNextCertification:
    IteratorCertification, IteratorCertificationResult, BSPPartition,
    Certification, certify, bsp, ntracked, nstart_solutions,
    ncandidates, ncertified,
    nreal_certified, ncomplex_certified, nnotcertified, ndistinct_certified,
    ndistinct_real_certified, ndistinct_complex_certified, nleaves, max_leaf_size,
    oversized_leaves, unsplittable_leaves, nleaf_splits

# ─────────────────────────────────────────────────────────────────────────────
# Certification of a `ResultIterator`: the solutions are filed into a binary
# partition of one coordinate's real line and certified one leaf at a time, so
# the certificates are never all held at once.
# ─────────────────────────────────────────────────────────────────────────────

# Options with the progress meter silenced.
quiet(;
    leaf_size_bound::Int = 50_000,
    coordinate::Int = 1,
    boundaries::AbstractVector{<:Real} = -100:0.1:100,
    max_depth::Int = 500,
    certify_oversized_leaves::Bool = false,
) = IteratorCertification(;
    show_progress = false, leaf_size_bound, coordinate, boundaries, max_depth,
    certify_oversized_leaves,
)

@testset "iterator certification: parameter-free" begin
    @var x y
    F = System([x^2 - 1, y - 1]; variables = [x, y])
    ri = result_iterator(F)

    res = certify(F, ri, nothing, quiet(; leaf_size_bound = 2, boundaries = -3:3))

    @test res isa IteratorCertificationResult
    @test bsp(res) isa BSPPartition
    @test ncertified(res) == 2
    @test ndistinct_certified(res) == 2
    @test nnotcertified(res) == 0
    @test nreal_certified(res) == 2
    @test ndistinct_real_certified(res) == 2
    @test ndistinct_complex_certified(res) == 0
    @test nstart_solutions(res) == 2
    @test ncandidates(res) == 2
    @test ntracked(res) >= length(ri)
    # -3:3 cuts the line into eight leaves, but the roots are x = ±1, which lie
    # on two of the cuts: each enclosure straddles one, and the two leaves
    # sharing it merge.
    @test nleaves(res) == 6
    @test max_leaf_size(res) == 1
    @test nleaf_splits(res) == 0
    @test oversized_leaves(res) == 0
    @test unsplittable_leaves(res) == 0
end

# Two conics sharing a factor: 7 paths by polyhedral, 3 solutions.
@var x y a[1:6]
PARAMETRIC = System(
    [
        (a[1] * x^2 + a[2] * y) * (a[3] * x + a[4] * y) + 1,
        (a[1] * x^2 + a[2] * y) * (a[5] * x + a[6] * y) + 1,
    ];
    variables = [x, y], parameters = a,
)
PARAMS = [0.257, -0.139, -1.73, -0.199, 1.79, -1.32]

@testset "iterator certification: parametric" begin
    ri = result_iterator(fix_parameters(PARAMETRIC, PARAMS), Polyhedral())
    # Certified against the parametric system at an enclosure of `PARAMS`, which
    # is a stronger statement than certifying the substituted system.
    res = certify(PARAMETRIC, ri, PARAMS, quiet())

    @test nstart_solutions(res) == 7
    @test ntracked(res) >= length(ri)
    @test ncandidates(res) == 3
    @test ncertified(res) == 3
    @test ndistinct_certified(res) == 3
end

@testset "iterator certification: restricted iterator" begin
    ri = result_iterator(fix_parameters(PARAMETRIC, PARAMS), Polyhedral())
    finite = restrict(ri, selection(is_success, ri))
    res = certify(
        PARAMETRIC, finite, PARAMS,
        quiet(; coordinate = 2, certify_oversized_leaves = true),
    )

    @test nstart_solutions(res) == 7
    @test ntracked(res) >= length(finite)
    @test ncandidates(res) == 3
    @test ncertified(res) == 3
    @test ndistinct_certified(res) == 3
end

@testset "iterator certification: lazily filtered iterator" begin
    ri = result_iterator(fix_parameters(PARAMETRIC, PARAMS), Polyhedral())
    res = certify(
        PARAMETRIC, Iterators.filter(is_success, ri), PARAMS, quiet(),
    )

    @test ncandidates(res) == 3
    @test ndistinct_certified(res) == 3
    # A predicate no path passes leaves nothing to certify.
    none = certify(PARAMETRIC, Iterators.filter(Returns(false), ri), PARAMS, quiet())
    @test ncandidates(none) == 0
    @test ncertified(none) == 0
    @test ndistinct_certified(none) == 0
    @test ntracked(none) == 7
end

@testset "iterator certification: iterator as start solutions" begin
    @var u v p
    F = System([v - u^2 + p, v - u^3 - p]; variables = [u, v], parameters = [p])

    first_iter = result_iterator(F, [[1, 1], [-1, 1]], [0], [-1])
    second_iter = result_iterator(F, first_iter, [-1], [-2])

    res = certify(F, second_iter, [-2], quiet(; max_depth = 0))
    results = collect(second_iter)
    nsuccess = count(is_success, results)

    # v3 materializes the successful endpoints of `first_iter` when the second
    # iterator is built, so the start count is what got through, not
    # `length(first_iter)`.
    @test nstart_solutions(res) == count(is_success, first_iter)
    @test ntracked(res) >= length(results)
    @test ncandidates(res) == nsuccess
    @test ndistinct_certified(res) == ncertified(res)
    @test ncertified(res) <= nsuccess
    # `max_depth = 0` forbids every split.
    @test nleaf_splits(res) == 0
end

@testset "iterator certification: subspace-move iterator" begin
    # The endpoints of a subspace move are ambient, so what is certified is the
    # sliced system they solve. Covers the third kind of solve cache a
    # `ResultIterator` can carry.
    @var x y
    F = System([x^2 + y^2 - 4]; variables = [x, y])
    L₀ = rand_subspace(2; codim = 1)
    L₁ = rand_subspace(2; codim = 1)
    ri = result_iterator(F, solutions(solve(F, L₀)), L₀, L₁)

    res = certify(slice(F, L₁), ri, nothing, quiet())

    @test ncertified(res) == 2
    @test ndistinct_certified(res) == 2
end

@testset "iterator certification: leaf splitting" begin
    # One leaf holding everything, and a bound of one, so the partition has to be
    # refined until every solution is separated.
    @var x y
    F = System([x^2 + y^2 - 4, x * y - 1]; variables = [x, y])
    ri = result_iterator(F)
    res = certify(
        F, ri, nothing,
        quiet(; leaf_size_bound = 1, boundaries = Float64[]),
    )

    @test ncertified(res) == 4
    @test ndistinct_certified(res) == 4
    @test nleaf_splits(res) == 3
    @test max_leaf_size(res) == 1
    @test oversized_leaves(res) == 0
    @test unsplittable_leaves(res) == 0
    @test nleaves(res) == 4
    # Refinement re-tracks nothing: one pass places the four solutions, and one
    # pass per terminal leaf certifies it.
    @test ntracked(res) == 8
end

@testset "iterator certification: split balance and pass count" begin
    # 25 solutions in one leaf with a bound of 3. A cut that peels one enclosure
    # off at a time would leave ~22 leaves and pay a pass for each; balanced cuts
    # leave ~9, and refinement pays no pass at all. The bound below is what
    # separates the two: linear against quadratic in the solution count.
    @var x y
    F = System(
        [x^5 + 2y^5 + x^2 * y - 3, 2x^5 - y^5 + x * y^2 + 1]; variables = [x, y],
    )
    ri = result_iterator(F)
    res = certify(
        F, ri, nothing, quiet(; leaf_size_bound = 3, boundaries = Float64[]),
    )

    @test ncertified(res) == 25
    @test ndistinct_certified(res) == 25
    @test max_leaf_size(res) <= 3
    @test oversized_leaves(res) == 0
    # One pass places all 25, then each terminal leaf is tracked once more.
    @test ntracked(res) == 50
    # A balanced cut splits by count, so the tree stays shallow: 25 enclosures at
    # 3 per leaf need at most a handful of leaves, not one per solution.
    @test nleaves(res) <= 12
end

@testset "iterator certification: merged leaves" begin
    # Every root lies on a cut, so each enclosure straddles one and the whole
    # partition collapses into a single leaf holding all three. Refining that
    # leaf then has to work from a merged, unordered entry list.
    @var x y
    F = System([x^3 - x, y - 1]; variables = [x, y])
    res = certify(
        F, result_iterator(F), nothing,
        quiet(; leaf_size_bound = 1, boundaries = -1:1),
    )

    @test ncertified(res) == 3
    @test ndistinct_certified(res) == 3
    @test nleaves(res) == 3
    @test nleaf_splits(res) == 2
    @test max_leaf_size(res) == 1
    @test oversized_leaves(res) == 0
end

@testset "iterator certification: threaded agrees with serial" begin
    @var x y
    F = System([x^2 + y^2 - 4, x * y - 1]; variables = [x, y])
    alg = quiet(; leaf_size_bound = 1, boundaries = Float64[])
    # The same iterator both times: a fresh one would draw its own seed, and
    # which solution the iterator yields first is what picks the split points.
    ri = result_iterator(F)

    serial = certify(F, ri, nothing, alg, Serial())
    threaded = certify(F, ri, nothing, alg, Threaded(min(4, Threads.nthreads())))

    for f in (
            ncertified, nreal_certified, ncomplex_certified, nnotcertified,
            ndistinct_certified, ndistinct_real_certified, ndistinct_complex_certified,
            ncandidates, ntracked, nleaves, max_leaf_size, nleaf_splits,
            oversized_leaves, unsplittable_leaves,
        )
        @test f(serial) == f(threaded)
    end
end

@testset "iterator certification: caller's homotopy stays on one task" begin
    # A cache built around a caller's homotopy cannot hand out independent
    # workers, so this route tracks serially whatever `exec` asks for. Certifying
    # it on several tasks would otherwise share one evaluator's tapes.
    @var x y a
    # At a = 1 the roots are (±1, 0); at a = 4 they are (±2, ±3/2), separated in
    # the first coordinate.
    F = System([x^2 - a, x * y - a + 1]; variables = [x, y], parameters = [a])
    H = ParameterHomotopy(F, [1.0], [4.0])
    G = fix_parameters(F, [4.0])
    alg = quiet(; leaf_size_bound = 1, boundaries = Float64[])
    ri = result_iterator(H, [[1.0, 0.0], [-1.0, 0.0]])

    serial = certify(G, ri, nothing, alg, Serial())
    threaded = certify(G, ri, nothing, alg, Threaded(4))

    @test ncertified(serial) == 2
    for f in (ncertified, ndistinct_certified, ntracked, nleaves, max_leaf_size)
        @test f(serial) == f(threaded)
    end
end

@testset "iterator certification: oversized leaves" begin
    @var x y
    F = System([x^2 + y^2 - 4, x * y - 1]; variables = [x, y])
    ri = result_iterator(F)
    # No depth to split with, so the single leaf stays oversized. It is reported
    # either way, and only certified when asked for. Leaving it out of the distinct
    # counts is warned about: the count is otherwise a silent undercount.
    skipped = @test_logs (:warn,) match_mode = :any certify(
        F, ri, nothing,
        quiet(; leaf_size_bound = 1, boundaries = Float64[], max_depth = 0),
    )
    @test oversized_leaves(skipped) == 1
    @test max_leaf_size(skipped) == 4
    @test ncertified(skipped) == 4
    @test ndistinct_certified(skipped) == 0

    certified = certify(
        F, ri, nothing,
        quiet(;
            leaf_size_bound = 1, boundaries = Float64[], max_depth = 0,
            certify_oversized_leaves = true,
        ),
    )
    @test oversized_leaves(certified) == 1
    @test ndistinct_certified(certified) == 4
end

@testset "iterator certification: option and input errors" begin
    @var x y
    F = System([x^2 - 1, y - 1]; variables = [x, y])
    ri = result_iterator(F)

    @test_throws ArgumentError IteratorCertification(; leaf_size_bound = 0)
    @test_throws ArgumentError IteratorCertification(; coordinate = 0)
    @test_throws ArgumentError IteratorCertification(; ε = 0.0)
    @test_throws ArgumentError IteratorCertification(; ε = Inf)
    @test_throws ArgumentError IteratorCertification(; max_depth = -1)
    @test_throws ArgumentError IteratorCertification(; boundaries = [0.0, Inf])
    # A coordinate the system does not have is only knowable here.
    @test_throws ArgumentError certify(F, ri, nothing, quiet(; coordinate = 3))
    # An overdetermined system cannot be certified.
    G = System([x^2 - 1, y - 1, x + y - 2]; variables = [x, y])
    @test_throws ArgumentError certify(G, ri, nothing, quiet())
    # A filter nest has to bottom out at a `ResultIterator`, at any depth.
    @test_throws ArgumentError certify(
        F, Iterators.filter(is_success, [1, 2]), nothing, quiet(),
    )
    @test_throws ArgumentError certify(
        F, Iterators.filter(is_success, Iterators.filter(is_success, [1, 2])),
        nothing, quiet(),
    )
    # The eager algorithm would hold every certificate at once, which is the one
    # thing this route exists to avoid.
    @test_throws ArgumentError certify(F, ri, Certification())
    @test_throws ArgumentError certify(F, ri, nothing, Certification())
    @test_throws ArgumentError certify(F, Iterators.filter(is_success, ri), Certification())
end

@testset "iterator certification: progress meter" begin
    @var x y
    F = System([x^2 - 1, y - 1]; variables = [x, y])
    res = redirect_stdout(devnull) do
        certify(
            F, result_iterator(F), nothing,
            IteratorCertification(; boundaries = -3:3),
        )
    end
    @test ndistinct_certified(res) == 2
end

@testset "iterator certification: show" begin
    @var x y
    F = System([x^2 - 1, y - 1]; variables = [x, y])
    res = certify(F, result_iterator(F), nothing, quiet(; boundaries = -3:3))

    @test occursin("2 distinct certified", sprint(show, res))
    plain = sprint((io, r) -> show(io, MIME("text/plain"), r), res)
    @test occursin("IteratorCertificationResult", plain)
    @test occursin("2 distinct certified enclosures", plain)
    @test occursin("BSPPartition", sprint(show, bsp(res)))
    plain_bsp = sprint((io, b) -> show(io, MIME("text/plain"), b), bsp(res))
    @test occursin("BSPPartition", plain_bsp)
    @test occursin("=> 1", plain_bsp)
end

@testset "iterator certification: partition accessors" begin
    @var x y
    F = System([x^2 - 1, y - 1]; variables = [x, y])
    res = certify(F, result_iterator(F), nothing, quiet(; boundaries = -3:3))
    partition = bsp(res)

    @test length(partition) == nleaves(res)
    ls = collect(partition)
    @test length(ls) == nleaves(res)
    @test eltype(ls) == eltype(BSPPartition)
    # The leaves tile the line in increasing order, from -Inf to Inf.
    @test first(ls).lo == -Inf
    @test last(ls).hi == Inf
    @test all(ls[i].hi == ls[i + 1].lo for i in 1:(length(ls) - 1))
    @test sum(leaf -> leaf.nenclosures, ls) == ncertified(res)
    @test count(leaf -> leaf.unsplittable, ls) == unsplittable_leaves(res)
end
