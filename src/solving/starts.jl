## Start-solution input types.
#
# `ResultIterator` lives here rather than beside its iteration methods because
# `StartsLike` names it, and every route that takes start solutions is compiled
# before those methods are.

"""
    ResultIterator(cache[, mask])

Lazily track the paths of a solve cache, yielding one [`PathResult`](@ref) per
start solution. `mask` selects which start solutions are tracked (all by
default); see [`bitmask_filter`](@ref).

Build one with [`result_iterator`](@ref) rather than from a cache directly.
"""
struct ResultIterator{C}
    cache::C
    mask::BitVector

    # `length(ri)` counts the mask, so a mask of the wrong length would make the
    # advertised length disagree with what iteration yields.
    function ResultIterator{C}(cache::C, mask::BitVector) where {C}
        n = length(cache.start_solutions)
        length(mask) == n || throw(
            ArgumentError("The mask has length $(length(mask)), expected $n."),
        )
        return new{C}(cache, mask)
    end
end

ResultIterator(cache::C, mask::BitVector) where {C} = ResultIterator{C}(cache, mask)
ResultIterator(cache::C) where {C} =
    ResultIterator{C}(cache, trues(length(cache.start_solutions)))

"""
    StartsLike

The start-solution inputs every route accepts: a vector of solution vectors, a
[`Result`](@ref), or a [`ResultIterator`](@ref).
"""
const StartsLike = Union{
    AbstractVector{<:AbstractVector{<:Number}}, Result, ResultIterator,
}

# Monodromy additionally takes a single solution, which it wraps. Kept out of
# `StartsLike` so a flat vector stays an error on the routes that never accepted
# one, instead of silently becoming a one-element start set.
const SolutionsLike = Union{StartsLike, AbstractVector{<:Number}}

# Typing the `starts` slot puts anything else out of reach of dispatch, so each
# route carries a less specific method that lands here. One message, one place.
@noinline function _bad_starts(x)
    Base.@nospecialize x
    throw(
        ArgumentError(
            "start solutions must be a vector of solution vectors, a `Result`, or a " *
                "`ResultIterator`, got $(typeof(x)).",
        ),
    )
end
