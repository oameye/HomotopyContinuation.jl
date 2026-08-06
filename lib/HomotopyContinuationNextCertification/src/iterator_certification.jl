## Low-memory certification of a `ResultIterator`.
##
## `certify` of a `ResultIterator` never holds every certificate at once. It
## streams the iterator, certifies each successful endpoint, and files the
## enclosure of one coordinate's real part into a binary partition of the real
## line. Leaves are refined until each holds at most `leaf_size_bound`
## enclosures; only then is a leaf certified jointly and deduplicated, one leaf
## at a time. Two enclosures in different leaves are separated in that
## coordinate, so they cannot enclose the same solution and never have to be
## compared. What bounds the memory is that a pass keeps one interval per path
## and drops the certificate it came from.
##
## Tracking is therefore one pass over the iterator to place the solutions, plus
## one pass per terminal leaf to certify it. Refining the partition needs no pass
## of its own: the cut and both child memberships follow from the intervals the
## first pass already collected.
##
## Follows Breiding, Brysiewicz and Johnson, "Low-Memory Numerical
## Certification" (arXiv:2604.16623).

# ─────────────────────────────────────────────────────────────────────────────
# Binary partition of the real line
# ─────────────────────────────────────────────────────────────────────────────

# What a pass keeps for one certified solution: where the path sits in the
# iterator it was tracked from, and its enclosure projected to the chosen
# coordinate.
struct BSPLeafEntry
    index::Int
    lo::Float64
    hi::Float64
end

"""
    BSPPartition

The binary partition of the real line that [`certify`](@ref) builds for a
`ResultIterator`. Each leaf is an interval carrying the number of certified
enclosures assigned to it. Iterate it, or `collect` it, to read the leaves in
increasing order; `length` counts them.
"""
struct BSPPartition
    # Leaf `i` is the interval `(cuts[i], cuts[i + 1])`, so the leaves tile the
    # real line in increasing order and are addressed by index. The cuts run from
    # -Inf to Inf, which is what puts every finite enclosure in some leaf. All
    # three vectors mutate in place: a leaf splits when it holds too much
    # (inserting a cut), and neighbouring leaves merge when an enclosure straddles
    # their boundary (deleting a run of cuts).
    cuts::Vector{Float64}
    counts::Vector{Int}
    # Leaves that stayed oversized because the projection left no safe cut.
    unsplittable::BitVector
end

_nleaves(bsp::BSPPartition)::Int = length(bsp.counts)
_leaf_lo(bsp::BSPPartition, i::Int)::Float64 = bsp.cuts[i]
_leaf_hi(bsp::BSPPartition, i::Int)::Float64 = bsp.cuts[i + 1]

function _build_partition(boundaries::Vector{Float64})::BSPPartition
    cuts = [-Inf; boundaries; Inf]
    n = length(cuts) - 1
    return BSPPartition(cuts, zeros(Int, n), falses(n))
end

# The leaf containing `[lo, hi]`, or `nothing` when it straddles a cut. A leaf owns
# both its endpoints, so an enclosure touching one is inside rather than straddling.
function _find_leaf(bsp::BSPPartition, lo::Float64, hi::Float64)
    i = min(searchsortedlast(bsp.cuts, lo), _nleaves(bsp))
    return hi <= _leaf_hi(bsp, i) ? i : nothing
end

# Merge the run of leaves `[lo, hi]` touches into one, so it fits in a single leaf,
# and absorb their entry lists into the merged leaf's. Keyed by lower cut, which is
# the merged leaf's own key.
function _merge_covering_leaves!(
        bsp::BSPPartition, by_leaf::Dict{Float64, Vector{BSPLeafEntry}},
        lo::Float64, hi::Float64,
    )::Int
    left = max(searchsortedfirst(bsp.cuts, lo) - 1, 1)
    right = min(searchsortedlast(bsp.cuts, hi), _nleaves(bsp))
    right <= left && return left

    merged = get!(() -> BSPLeafEntry[], by_leaf, _leaf_lo(bsp, left))
    for i in (left + 1):right
        bsp.counts[left] += bsp.counts[i]
        absorbed = pop!(by_leaf, _leaf_lo(bsp, i), nothing)
        absorbed === nothing || append!(merged, absorbed)
    end
    deleteat!(bsp.cuts, (left + 1):right)
    deleteat!(bsp.counts, (left + 1):right)
    deleteat!(bsp.unsplittable, (left + 1):right)
    bsp.unsplittable[left] = false
    return left
end

# A merge always yields a leaf containing the enclosure, so one retry suffices.
function _ensure_leaf!(
        bsp::BSPPartition, by_leaf::Dict{Float64, Vector{BSPLeafEntry}},
        lo::Float64, hi::Float64,
    )::Int
    i = _find_leaf(bsp, lo, hi)
    i === nothing || return i
    return _merge_covering_leaves!(bsp, by_leaf, lo, hi)
end

# Split leaf `i` at `cut`, leaving the halves at `i` and `i + 1`.
function _split_leaf!(bsp::BSPPartition, i::Int, cut::Float64)::Nothing
    insert!(bsp.cuts, i + 1, cut)
    bsp.counts[i] = 0
    insert!(bsp.counts, i + 1, 0)
    bsp.unsplittable[i] = false
    insert!(bsp.unsplittable, i + 1, false)
    return nothing
end

_leaf_stats(bsp::BSPPartition, leaf_size_bound::Int) =
    (maximum(bsp.counts; init = 0), count(>(leaf_size_bound), bsp.counts))

const BSPLeafInfo = @NamedTuple{
    lo::Float64, hi::Float64, nenclosures::Int, unsplittable::Bool,
}

# Iterating a partition yields its leaves in increasing order, so `collect(bsp)`
# lists them and no generic name has to be exported for it. Each leaf is a named
# tuple: `lo` and `hi` bounds, the number `nenclosures` of certified enclosures
# filed into it, and whether it is `unsplittable`, meaning the partitioned
# coordinate left no cut clear of every enclosure in it.
Base.eltype(::Type{BSPPartition}) = BSPLeafInfo
Base.length(bsp::BSPPartition)::Int = _nleaves(bsp)

function Base.iterate(bsp::BSPPartition, i::Int = 1)
    i > _nleaves(bsp) && return nothing
    leaf = (
        lo = _leaf_lo(bsp, i), hi = _leaf_hi(bsp, i),
        nenclosures = bsp.counts[i], unsplittable = bsp.unsplittable[i],
    )
    return leaf, i + 1
end

Base.show(io::IO, bsp::BSPPartition) =
    print(io, "BSPPartition with ", _nleaves(bsp), " leaves")

function Base.show(io::IO, ::MIME"text/plain", bsp::BSPPartition)
    println(io, "BSPPartition")
    println(io, "============")
    occupied = findall(>(0), bsp.counts)
    for i in Iterators.take(occupied, 10)
        println(io, "• (", _leaf_lo(bsp, i), ", ", _leaf_hi(bsp, i), ") => ", bsp.counts[i])
    end
    length(occupied) > 10 && println(io, " ⋮ (", length(occupied) - 10, " more)")
    return
end

# ─────────────────────────────────────────────────────────────────────────────
# Options
# ─────────────────────────────────────────────────────────────────────────────

"""
    IteratorCertification(; certification, leaf_size_bound, coordinate,
                          boundaries, ε, max_depth, certify_oversized_leaves)

Options for [`certify`](@ref) of a `ResultIterator`, which certifies without
holding every certificate at once. `certification` carries the per-solution
options and its keywords (`max_precision`, `refine_solution`,
`extended_certificate`, `show_progress`) may be given here directly; the rest
describe the partition of the real line the solutions are filed into.

- `leaf_size_bound = 50_000`: how many enclosures a leaf may hold before it is
  split. Only a single leaf's certificates are held at once, so this bounds the
  memory. A leaf cannot be refined below the number of enclosures that overlap in
  `coordinate`, which is at least the number of paths reaching the same solution,
  so a bound under that leaves an oversized leaf.
- `coordinate = 1`: the solution coordinate whose real part is partitioned.
- `boundaries = -100:0.1:100`: the initial cut points, which must be finite.
- `ε = 1.0e-4`: how far a proposed cut is placed from the enclosure it is
  derived from.
- `max_depth = 500`: how deep the refinement may go. `0` splits nothing.
- `certify_oversized_leaves = false`: whether a leaf that stayed oversized is
  certified anyway. Its size is reported either way.
"""
struct IteratorCertification
    certification::Certification
    leaf_size_bound::Int
    coordinate::Int
    boundaries::Vector{Float64}
    ε::Float64
    max_depth::Int
    certify_oversized_leaves::Bool
end

function IteratorCertification(;
        certification::Certification = Certification(),
        leaf_size_bound::Int = 50_000,
        coordinate::Int = 1,
        boundaries::AbstractVector{<:Real} = -100:0.1:100,
        ε::Float64 = 1.0e-4,
        max_depth::Int = 500,
        certify_oversized_leaves::Bool = false,
        max_precision::Int = certification.max_precision,
        refine_solution::Bool = certification.refine_solution,
        extended_certificate::Bool = certification.extended_certificate,
        show_progress::Bool = certification.show_progress,
    )
    leaf_size_bound > 0 ||
        throw(ArgumentError("`leaf_size_bound` must be positive, got $leaf_size_bound."))
    coordinate > 0 ||
        throw(ArgumentError("`coordinate` must be positive, got $coordinate."))
    ε > 0 && isfinite(ε) ||
        throw(ArgumentError("`ε` must be positive and finite, got $ε."))
    max_depth >= 0 ||
        throw(ArgumentError("`max_depth` must be nonnegative, got $max_depth."))
    cuts = unique!(sort!(Vector{Float64}(boundaries)))
    all(isfinite, cuts) || throw(ArgumentError("`boundaries` must be finite."))
    return IteratorCertification(
        Certification(
            max_precision, refine_solution, extended_certificate, show_progress,
        ),
        leaf_size_bound, coordinate, cuts, ε, max_depth, certify_oversized_leaves,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Running tally
# ─────────────────────────────────────────────────────────────────────────────

@enumx IteratorCertificationPhase assignment refinement finished

# Every count the run accumulates, in one place: the certified ones are written from
# every task under the pass's lock, the leaf ones from the serial walk. Mutable
# because that is what it is; no field is a buffer, so none can be `const`.
mutable struct IteratorCertificationStats
    phase::IteratorCertificationPhase.T
    # Path trackings over every pass, and the assignment pass's own counts, which
    # are the ones that mean "every solution once" — later passes recertify inside
    # one leaf.
    ntracked::Int
    ncandidates::Int
    certified::Int
    certified_real::Int
    certified_complex::Int
    not_certified::Int
    distinct::Int
    distinct_real::Int
    distinct_complex::Int
    splits::Int
    processed_leaves::Int
    leaves::Int
end

IteratorCertificationStats() = IteratorCertificationStats(
    IteratorCertificationPhase.assignment, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)

# How many passes the refinement needs is not known in advance, so the meter counts
# paths tracked rather than a fraction of a known total.
function _iterator_progress(delay::Float64 = 0.3)
    meter = ProgressMeter.ProgressUnknown(;
        dt = 0.2, desc = "Certifying iterator... ", output = stdout, spinner = true,
    )
    meter.tlast += delay
    return meter
end

_showvalues(s::IteratorCertificationStats) = (
    ("phase", string(s.phase)),
    ("# paths tracked", s.ntracked),
    ("# certified (not certified)", string(s.certified, " (", s.not_certified, ")")),
    ("# distinct certified", s.distinct),
    ("# leaves (processed)", string(s.leaves, " (", s.processed_leaves, ")")),
    ("# leaf splits", s.splits),
)

# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────

"""
    IteratorCertificationResult

The result of [`certify`](@ref) for a `ResultIterator`. It reports counts and the
[`BSPPartition`](@ref) they were obtained from rather than the certificates: the
point of this route is that the certificates are never all held at once.
"""
struct IteratorCertificationResult
    # The leaf counts are read off `bsp` rather than stored, so they cannot drift
    # from the partition they describe; `options` is what `oversized_leaves` needs.
    bsp::BSPPartition
    options::IteratorCertification
    nstart_solutions::Int
    # The run's own tally, taken over when it finishes. Nothing mutates it after.
    stats::IteratorCertificationStats
end

"""
    bsp(R::IteratorCertificationResult)

Return the [`BSPPartition`](@ref) of the real line the solutions were filed into.
"""
bsp(R::IteratorCertificationResult)::BSPPartition = R.bsp

"""
    nstart_solutions(R::IteratorCertificationResult)

Return the number of start solutions of the underlying solve. A restricted
iterator tracks fewer paths than this.
"""
nstart_solutions(R::IteratorCertificationResult)::Int = R.nstart_solutions

# `ntracked` counts path trackings, so every pass over the iterator adds to it;
# the eager routes track nothing and have no method.
ntracked(R::IteratorCertificationResult)::Int = R.stats.ntracked

# These count the same things as for a `CertificationResult`, where they are
# documented; the distinct ones are summed over the leaves.
ncandidates(R::IteratorCertificationResult)::Int = R.stats.ncandidates
ncertified(R::IteratorCertificationResult)::Int = R.stats.certified
nreal_certified(R::IteratorCertificationResult)::Int = R.stats.certified_real
ncomplex_certified(R::IteratorCertificationResult)::Int = R.stats.certified_complex
nnotcertified(R::IteratorCertificationResult)::Int = R.stats.not_certified
ndistinct_certified(R::IteratorCertificationResult)::Int = R.stats.distinct
ndistinct_real_certified(R::IteratorCertificationResult)::Int = R.stats.distinct_real
ndistinct_complex_certified(R::IteratorCertificationResult)::Int = R.stats.distinct_complex

"""
    nleaves(R::IteratorCertificationResult)

Return the number of leaves in the final partition.
"""
nleaves(R::IteratorCertificationResult)::Int = length(R.bsp)

"""
    max_leaf_size(R::IteratorCertificationResult)

Return the number of enclosures in the largest leaf.
"""
max_leaf_size(R::IteratorCertificationResult)::Int = maximum(R.bsp.counts; init = 0)

"""
    oversized_leaves(R::IteratorCertificationResult)

Return the number of leaves holding more than `leaf_size_bound` enclosures.
"""
oversized_leaves(R::IteratorCertificationResult)::Int =
    count(>(R.options.leaf_size_bound), R.bsp.counts)

"""
    unsplittable_leaves(R::IteratorCertificationResult)

Return the number of leaves that could not be refined further in the chosen
coordinate.
"""
unsplittable_leaves(R::IteratorCertificationResult)::Int = count(R.bsp.unsplittable)

"""
    nleaf_splits(R::IteratorCertificationResult)

Return the number of leaf splits performed.
"""
nleaf_splits(R::IteratorCertificationResult)::Int = R.stats.splits

Base.show(io::IO, R::IteratorCertificationResult) = print(
    io, "IteratorCertificationResult with ", R.stats.distinct,
    " distinct certified solutions",
)

function Base.show(io::IO, ::MIME"text/plain", R::IteratorCertificationResult)
    s = R.stats
    println(io, "IteratorCertificationResult")
    println(io, "===========================")
    println(
        io, "• ", s.certified, " certified enclosures (", s.certified_real,
        " real, ", s.certified_complex, " complex)",
    )
    println(io, "• ", s.not_certified, " not certified")
    println(
        io, "• ", s.distinct, " distinct certified enclosures (",
        s.distinct_real, " real, ", s.distinct_complex, " complex)",
    )
    println(io, "• ", s.ntracked, " paths tracked over all passes")
    println(io, "• max leaf size: ", max_leaf_size(R))
    println(io, "• ", nleaves(R), " leaves")
    oversized = oversized_leaves(R)
    if oversized > 0
        println(
            io, "• ", oversized, " oversized leaves remain (",
            unsplittable_leaves(R), " unsplittable in this coordinate)",
        )
    end
    return
end

# ─────────────────────────────────────────────────────────────────────────────
# One pass over an iterator
# ─────────────────────────────────────────────────────────────────────────────

_payload_index(entry::BSPLeafEntry)::Int = entry.index
_payload_index(cert::AbstractSolutionCertificate)::Int = certificate_index(cert)

struct IteratorCertificationContext{
        S <: System,
        P <: Union{Nothing, CertificationParameters},
        Pred,
        CertT <: AbstractSolutionCertificate,
        W,
    }
    system::S
    cert_params::P
    is_real_system::Bool
    # Applied to every path result; a lazily filtered iterator is certified by
    # carrying its predicate here rather than by materializing the filter.
    predicate::Pred
    certificate_type::Type{CertT}
    options::IteratorCertification
    # One reference point for the whole run: it only decides how the per-leaf
    # interval tree buckets certificates, never whether two of them overlap.
    reference_point::Vector{ComplexF64}
    # One certification workspace and one tracking worker per task, handed out per
    # pass. Both are expensive to build and this route makes many passes, so they
    # are built once for the whole run. Workers are keyed to the cache, not the
    # iterator, so the same ones serve every leaf's `restrict` of it.
    caches::Vector{CertificationCache}
    workers::Vector{W}
    # The parent iterator's selected start indices, so a leaf's mask costs one
    # write per entry rather than a walk over every start solution.
    selected::Vector{Int}
    stats::IteratorCertificationStats
    meter::Union{Nothing, ProgressMeter.ProgressUnknown}
end

# Track every path the iterator selects, certify the successful endpoints, and
# collect `keep(k, certificate)` for the certified ones in iterator order. The
# certificate is dropped as soon as `keep` returns, so a pass that keeps only an
# enclosure holds nothing per path but that enclosure.
function _certified_pass(
        keep::K,
        ::Type{Payload},
        ri::ResultIterator,
        ctx::IteratorCertificationContext{S, P, Pred, CertT},
        exec::Union{Serial, Threaded},
    ) where {K, Payload, S, P, Pred, CertT}
    payloads = Payload[]
    lk = ReentrantLock()
    caches = _cache_pool(ctx.caches)
    alg = ctx.options.certification
    stats = ctx.stats
    # Only the assignment pass certifies every solution once, so it is the one whose
    # certified counts mean anything; later passes recertify inside one leaf.
    counting = stats.phase === IteratorCertificationPhase.assignment

    _foreach_path(() -> take!(caches), ri, exec, ctx.workers) do cache, k, path_result
        if !(ctx.predicate(path_result)::Bool && is_success(path_result))
            Base.@lock lk begin
                stats.ntracked += 1
                _draw_progress!(ctx, true)
            end
            return nothing
        end
        cert = certify_solution(
            ctx.system, solution(path_result), ctx.cert_params, cache, k,
            ctx.is_real_system, CertT, alg.max_precision, alg.refine_solution,
        )
        Base.@lock lk begin
            stats.ntracked += 1
            certified = is_certified(cert)
            certified && push!(payloads, keep(k, cert))
            if counting
                stats.ncandidates += 1
                if certified
                    stats.certified += 1
                    is_real(cert) && (stats.certified_real += 1)
                    is_complex(cert) && (stats.certified_complex += 1)
                else
                    stats.not_certified += 1
                end
            end
            _draw_progress!(ctx, true)
        end
        return nothing
    end
    close(caches)

    # Tasks finish out of order, so the payloads are put back into iterator order:
    # per-leaf dedup picks a representative, and which one it is decides the real /
    # complex counts, so the order has to be the task count's business.
    sort!(payloads; by = _payload_index)
    return payloads
end

# `throttled` is for the per-path calls, which run inside the pass's lock:
# `showvalues` allocates and `update!` redraws at most every `dt`, so the tuple is
# built only when a redraw is actually due.
function _draw_progress!(
        ctx::IteratorCertificationContext, throttled::Bool = false,
    )::Nothing
    meter = ctx.meter
    meter === nothing && return nothing
    throttled && time() < meter.tlast + meter.dt && return nothing
    ProgressMeter.update!(meter; showvalues = _showvalues(ctx.stats))
    return nothing
end

function _finish_progress!(ctx::IteratorCertificationContext)::Nothing
    ctx.stats.phase = IteratorCertificationPhase.finished
    meter = ctx.meter
    meter === nothing && return nothing
    ProgressMeter.finish!(meter; showvalues = _showvalues(ctx.stats))
    return nothing
end

# The enclosure of one coordinate's real part, which is what a leaf holds.
function _project(cert::AbstractSolutionCertificate, coordinate::Int)
    I = certified_solution_interval(cert)::AcbMatrix
    re = real(IComplexF64(Arblib.ref(I, coordinate, 1)))
    return re.lo, re.hi
end

function _entry_pass(
        ri::ResultIterator, ctx::IteratorCertificationContext,
        exec::Union{Serial, Threaded},
    )
    coordinate = ctx.options.coordinate
    return _certified_pass(
        (k, cert) -> BSPLeafEntry(k, _project(cert, coordinate)...),
        BSPLeafEntry, ri, ctx, exec,
    )
end

_certificate_pass(
    ri::ResultIterator,
    ctx::IteratorCertificationContext{S, P, Pred, CertT},
    exec::Union{Serial, Threaded},
) where {S, P, Pred, CertT} =
    _certified_pass((_, cert) -> cert, CertT, ri, ctx, exec)

# ─────────────────────────────────────────────────────────────────────────────
# Leaf iterators
# ─────────────────────────────────────────────────────────────────────────────

# The iterator over exactly the paths `entries` came from. An entry's index is its
# position in `ri`'s selection, so `selected` maps it straight to a start index.
function _leaf_iterator(
        ri::ResultIterator, selected::Vector{Int}, entries::Vector{BSPLeafEntry},
    )::ResultIterator
    mask = falses(nstart_solutions(ri))
    for entry in entries
        mask[selected[entry.index]] = true
    end
    return restrict(ri, mask)
end

# ─────────────────────────────────────────────────────────────────────────────
# Leaf refinement
# ─────────────────────────────────────────────────────────────────────────────

_inside(entry::BSPLeafEntry, lo::Float64, hi::Float64)::Bool =
    lo <= entry.lo && entry.hi <= hi

# The most balanced cut the leaf admits: a point at least `ε` clear of every
# enclosure in it, splitting them as evenly as it can. `nothing` when they leave
# no room, which is what makes an oversized leaf terminal.
#
# Sorting by upper endpoint makes the candidates the gaps after each enclosure,
# and a candidate is admissible exactly when no enclosure further along starts
# at or before it, which the running suffix minimum answers in one pass.
function _best_split(
        entries::Vector{BSPLeafEntry}, lo::Float64, hi::Float64, ε::Float64,
    )
    n = length(entries)
    n > 1 || return nothing
    order = sortperm(entries; by = entry -> entry.hi)
    lower_bound_after = fill(Inf, n + 1)
    for k in n:-1:1
        lower_bound_after[k] = min(lower_bound_after[k + 1], entries[order[k]].lo)
    end

    best = nothing
    imbalance = typemax(Int)
    for k in 1:(n - 1)
        cut = entries[order[k]].hi + ε
        cut < lower_bound_after[k + 1] || continue
        lo < cut < hi || continue
        # |k on the left, n - k on the right|, smallest wins.
        score = abs(2k - n)
        if score < imbalance
            imbalance = score
            best = cut
        end
    end
    return best
end

# Split a leaf's enclosures at a cut `_best_split` returned. Accepting the cut is
# itself the separation certificate: every enclosure lies strictly on one side,
# so solutions in different children cannot coincide and are never compared.
function _partition_entries(entries::Vector{BSPLeafEntry}, cut::Float64)
    left = BSPLeafEntry[]
    right = BSPLeafEntry[]
    for entry in entries
        if entry.hi < cut
            push!(left, entry)
        elseif cut < entry.lo
            push!(right, entry)
        else
            error(
                "Internal error: the cut $cut crosses the enclosure " *
                    "($(entry.lo), $(entry.hi)).",
            )
        end
    end
    return left, right
end

"""
    _process_leaf!(bsp, i, ri, entries, depth, ctx, exec) -> next

Refine leaf `i` and certify what stays in it, depth first, so only the current
branch of the partition is ever held. `entries` are the enclosures filed into it,
indexed by position in `ri`'s selection. Distinct-solution counts and splits go into
`ctx.stats`; the return value is the index just past the leaves the subtree left
behind, which is where the walk over the partition resumes.

Refinement is arithmetic on `entries`: the cut and both child memberships follow
from the enclosures already in hand, so nothing is re-tracked until a leaf is
terminal.
"""
function _process_leaf!(
        bsp::BSPPartition, i::Int, ri::ResultIterator,
        entries::Vector{BSPLeafEntry}, depth::Int,
        ctx::IteratorCertificationContext, exec::Union{Serial, Threaded},
    )::Int
    options = ctx.options
    lo, hi = _leaf_lo(bsp, i), _leaf_hi(bsp, i)
    _record_leaf_progress!(ctx, bsp)

    if length(entries) > options.leaf_size_bound && depth < options.max_depth
        cut = _best_split(entries, lo, hi, options.ε)
        if cut !== nothing
            return _split_and_recurse!(bsp, i, cut, entries, ri, depth, ctx, exec)
        end
        # No cut this coordinate admits separates the leaf.
        bsp.unsplittable[i] = true
    end

    # From here the leaf is terminal, either small enough, out of depth, or
    # unsplittable in this coordinate.
    bsp.counts[i] = length(entries)
    ctx.stats.processed_leaves += 1
    if isempty(entries) ||
            (length(entries) > options.leaf_size_bound && !options.certify_oversized_leaves)
        _record_leaf_progress!(ctx, bsp)
        return i + 1
    end

    _certify_leaf!(
        _leaf_iterator(ri, ctx.selected, entries), lo, hi, length(entries), ctx, exec,
    )
    _record_leaf_progress!(ctx, bsp)
    return i + 1
end

function _split_and_recurse!(
        bsp::BSPPartition, i::Int, cut::Float64, entries::Vector{BSPLeafEntry},
        ri::ResultIterator, depth::Int,
        ctx::IteratorCertificationContext, exec::Union{Serial, Threaded},
    )::Int
    left_entries, right_entries = _partition_entries(entries, cut)
    _split_leaf!(bsp, i, cut)
    ctx.stats.splits += 1
    _record_leaf_progress!(ctx, bsp)

    next = _process_leaf!(bsp, i, ri, left_entries, depth + 1, ctx, exec)
    return _process_leaf!(bsp, next, ri, right_entries, depth + 1, ctx, exec)
end

function _record_leaf_progress!(
        ctx::IteratorCertificationContext, bsp::BSPPartition,
    )::Nothing
    ctx.stats.leaves = _nleaves(bsp)
    return _draw_progress!(ctx)
end

# Certify one terminal leaf jointly and add its distinct solutions to the tally.
# Only this leaf's certificates are held, and they are released when it returns.
function _certify_leaf!(
        leaf_iter::ResultIterator, lo::Float64, hi::Float64, expected::Int,
        ctx::IteratorCertificationContext{S, P, Pred, CertT},
        exec::Union{Serial, Threaded},
    )::Nothing where {S, P, Pred, CertT}
    certs = _certificate_pass(leaf_iter, ctx, exec)
    # The partition rests on the enclosures being reproducible: they were computed
    # once to place the solutions and are recomputed here to compare them. Both
    # checks below are that assumption, since a leaf that came back with fewer
    # enclosures, or with one reaching outside itself, would leave a solution
    # unaccounted for or unseparated from its neighbour.
    length(certs) == expected || error(
        "Re-tracking a leaf certified $(length(certs)) of $expected enclosures.",
    )
    distinct = DistinctSolutionCertificates(
        ctx.reference_point;
        extended_certificate = (CertT === ExtendedSolutionCertificate),
    )
    stats = ctx.stats
    for cert in certs
        entry = BSPLeafEntry(0, _project(cert, ctx.options.coordinate)...)
        _inside(entry, lo, hi) || error(
            "Re-tracking a leaf enclosed a solution in ($(entry.lo), $(entry.hi)), " *
                "which reaches outside the leaf ($lo, $hi) it was filed into.",
        )
        added, _ = add_certificate!(distinct, cert)
        added || continue
        stats.distinct += 1
        is_real(cert) && (stats.distinct_real += 1)
        is_complex(cert) && (stats.distinct_complex += 1)
    end
    return nothing
end


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

# First pass: stream the whole iterator and file every certified enclosure into
# the coarse partition, merging leaves where one straddles a cut. Entry lists are
# keyed by their leaf's lower cut, which survives a merge (the merged leaf keeps
# the leftmost one) where an index would not.
function _assign_initial_leaves!(
        bsp::BSPPartition, ri::ResultIterator,
        ctx::IteratorCertificationContext, exec::Union{Serial, Threaded},
    )
    entries = _entry_pass(ri, ctx, exec)
    by_leaf = Dict{Float64, Vector{BSPLeafEntry}}()
    for entry in entries
        i = _ensure_leaf!(bsp, by_leaf, entry.lo, entry.hi)
        push!(get!(() -> BSPLeafEntry[], by_leaf, _leaf_lo(bsp, i)), entry)
        bsp.counts[i] += 1
    end
    return by_leaf
end

function _certify_iterator(
        ri::ResultIterator,
        F::System,
        predicate::Pred,
        cert_params::Union{Nothing, CertificationParameters},
        cache::CertificationCache,
        ::Type{CertT},
        alg::IteratorCertification,
        exec::Union{Serial, Threaded},
    )::IteratorCertificationResult where {Pred, CertT <: AbstractSolutionCertificate}
    m, n = size(F)
    m == n || throw(ArgumentError("We can only certify solutions to square systems."))
    if isnothing(cert_params) && nparameters(F) > 0
        throw(ArgumentError("The given system expects parameters but none are given."))
    end
    alg.coordinate <= n || throw(
        ArgumentError(
            "`coordinate` must be at most the number of variables ($n), got " *
                "$(alg.coordinate).",
        ),
    )

    # Only as many caches and workers as tasks will actually run: `_replay_ntasks`
    # drops to one for a cache that cannot hand out independent workers, and
    # neither a `CertificationCache` nor a worker is cheap enough to build one per
    # idle task.
    selected = findall(selection(ri))
    ntasks = min(_replay_ntasks(ri, exec), max(length(selected), 1))
    ctx = IteratorCertificationContext(
        F, cert_params, is_real(F), predicate, CertT, alg, randn(ComplexF64, n),
        _cache_set(F, cache, ntasks), _path_workers(ri, ntasks), selected,
        IteratorCertificationStats(),
        alg.certification.show_progress ? _iterator_progress() : nothing,
    )
    stats = ctx.stats

    bsp = _build_partition(alg.boundaries)
    start_length = nstart_solutions(ri)
    by_leaf = _assign_initial_leaves!(bsp, ri, ctx, exec)
    stats.phase = IteratorCertificationPhase.refinement
    _draw_progress!(ctx)

    i = 1
    while i <= _nleaves(bsp)
        # Walk the partition left to right. Processing one leaf may split it into
        # a whole subtree, so the walk resumes past the leaves that subtree left.
        entries = get(by_leaf, _leaf_lo(bsp, i), nothing)
        if entries === nothing || isempty(entries)
            stats.processed_leaves += 1
            i += 1
            continue
        end
        i = _process_leaf!(bsp, i, ri, entries, 0, ctx, exec)
    end

    max_size, oversized = _leaf_stats(bsp, alg.leaf_size_bound)
    _finish_progress!(ctx)
    if oversized > 0 && !alg.certify_oversized_leaves
        # Their solutions are certified but never deduplicated, so they are missing
        # from the distinct counts. Saying so is the difference between a partial
        # answer and a wrong one.
        @warn "$oversized leaf/leaves hold more than leaf_size_bound = " *
            "$(alg.leaf_size_bound) enclosures (largest: $max_size) and were left " *
            "uncertified, so the distinct counts omit them. Raise leaf_size_bound, " *
            "raise max_depth, pick another coordinate, or pass " *
            "certify_oversized_leaves = true."
    end
    return IteratorCertificationResult(bsp, alg, start_length, stats)
end

# ─────────────────────────────────────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────────────────────────────────────

# A lazily filtered iterator is certified by carrying its predicate rather than
# by materializing the filter, so `certify` still sees every path exactly once
# per pass. `restrict(ri, selection(f, ri))` is the eager spelling and arrives
# here as a plain `ResultIterator`.
_iterator_and_predicate(ri::ResultIterator) = (ri, Returns(true))
function _iterator_and_predicate(it::Iterators.Filter)
    inner, predicate = _iterator_and_predicate(it.itr)
    return inner, x -> predicate(x) && it.flt(x)::Bool
end
_iterator_and_predicate(it) = throw(
    ArgumentError(
        "`certify` takes a `ResultIterator`, or a `Iterators.filter` of one, got " *
            "$(typeof(it)).",
    ),
)

# No type spells "a filter nest bottoming out at a `ResultIterator`", so this route
# accepts every `Iterators.Filter` and `_iterator_and_predicate` is the single place
# that decides. Narrowing the union here instead would only move some of the same
# rejections from its message to a `MethodError`.
const IteratorLike = Union{ResultIterator, Iterators.Filter}

"""
    certify(F, ri::ResultIterator, [p], alg = IteratorCertification(), exec = Threaded())

Certify the solutions a `ResultIterator` tracks without ever holding every
certificate at once, and return an [`IteratorCertificationResult`](@ref).

The certified solutions are filed into a partition of one coordinate's real line
by their enclosure, refined until each part holds at most `leaf_size_bound` of
them, and certified one part at a time; two solutions in different parts are
separated in that coordinate, so neither has to be compared against the other.
See [`IteratorCertification`](@ref) for the options, and note that the iterator is
tracked several times over: once to place the solutions, and again per part.

Only the endpoints with `is_success` are certified, singular ones included. This
differs from `certify(F, path_results)`, which certifies every endpoint given, and
from `certify(F, result)`, which takes the nonsingular ones.

A lazily filtered iterator (`Iterators.filter(f, ri)`) is accepted and certifies
the successful results passing `f`.

The counts returned are of certified enclosures, not of paths: `p` is enclosed
and what is certified is `F` at that enclosure, exactly as for the other `certify`
methods.
"""
function certify(
        F::System,
        it::IteratorLike,
        p::Union{Nothing, AbstractArray} = nothing,
        alg::IteratorCertification = IteratorCertification(),
        exec::Union{Serial, Threaded} = Threaded();
        cache::CertificationCache = CertificationCache(F),
    )::IteratorCertificationResult
    ri, predicate = _iterator_and_predicate(it)
    cert_params = certification_parameters(p; prec = alg.certification.max_precision)
    CertT = alg.certification.extended_certificate ? ExtendedSolutionCertificate :
        SolutionCertificate
    return _certify_iterator(
        ri, F, predicate, cert_params, cache, CertT, alg, exec,
    )
end

certify(
    F::System, it::IteratorLike, alg::IteratorCertification,
    exec::Union{Serial, Threaded} = Threaded();
    cache::CertificationCache = CertificationCache(F),
) = certify(F, it, nothing, alg, exec; cache = cache)

# `Certification` is the eager algorithm, which would have to hold every
# certificate at once. Saying so beats a `MethodError` on the most natural wrong
# call, and beats silently collecting the iterator.
_eager_alg_error() = throw(
    ArgumentError(
        "certifying a `ResultIterator` takes an `IteratorCertification`, not a " *
            "`Certification`. Pass `IteratorCertification(; ...)`, whose keywords " *
            "include the `Certification` ones, or `certify(F, collect(ri), ...)` to " *
            "certify eagerly and hold every certificate.",
    ),
)

# `cache` is accepted so that supplying it reaches the message rather than a
# `MethodError`, and defaulted to `nothing` so nothing is built to be thrown away.
certify(
    ::System, ::IteratorLike, ::Union{Nothing, AbstractArray}, ::Certification,
    ::Union{Serial, Threaded} = Threaded();
    cache::Union{Nothing, CertificationCache} = nothing,
) = _eager_alg_error()

certify(
    ::System, ::IteratorLike, ::Certification,
    ::Union{Serial, Threaded} = Threaded();
    cache::Union{Nothing, CertificationCache} = nothing,
) = _eager_alg_error()
