## GroupActions and SymmetricGroup: composed symmetry-group actions for dedup.
#
# Cold path over user closures: no FSVec, no EnumX. The recursion in
# _apply_actions passes only the REMAINING actions (Base.tail) to each produced
# image; re-applying the full tuple would recurse forever.

"""
    GroupActions(actions::Function...)

Store a bunch of group actions `(f1, f2, f3, ...)`.
Each action has to return a tuple.
The actions are applied in the following sense
1) f1 is applied on the original solution `s`
2) f2 is applied on `s` and the results of 1
3) f3 is applied on `s` and the results of 1) and 2)
and so on

## Example
```julia-repl
julia> f1(s) = (s * s,);

julia> f2(s) = (2s, -s, 5s);

julia> f3(s) = (s + 1,);

julia> GroupActions(f1)(3)
2-element Vector{Int64}: contains 3 and 9

julia> length(GroupActions(f1, f2)(3))
8
```
"""
struct GroupActions{T <: Tuple}
    actions::T
end
GroupActions(::Nothing) = GroupActions(())
GroupActions(actions::GroupActions) = actions
GroupActions(actions::Function...) = GroupActions(actions)
GroupActions(actions) = GroupActions(actions...)

function (actions::GroupActions)(s)
    S = [s]
    T = typeof(s)
    apply_actions(actions, s) do sᵢ
        sⱼ = convert(T, sᵢ)
        if sⱼ != s
            push!(S, sⱼ)
        end
        false
    end
    return S
end

apply_actions(cb, action::GroupActions, s) = _apply_actions(action.actions, s, cb)
@inline function _apply_actions(actions::Tuple, x, cb::F) where {F}
    f, rest = first(actions), Base.tail(actions)
    y = f(x)
    if isa(x, AbstractVector{<:Number}) && isa(y, AbstractVector{<:Number})
        cb(y) && return true
        if _apply_actions(rest, y, cb)
            return true
        end
    else
        # `y` is a tuple/iterable of images: iterate it directly rather than
        # recomputing `f(x)` (avoids a second evaluation of `f`, which is wasteful
        # and would disagree with `y` for a nondeterministic action).
        for yᵢ in y
            cb(yᵢ) && return true
            if _apply_actions(rest, yᵢ, cb)
                return true
            end
        end
    end
    return _apply_actions(rest, x, cb)
end
@inline _apply_actions(::Tuple{}, s, cb) = false

apply_actions(cb::G, actions::F, s) where {G, F} = actions(cb, s)

# Implemented group actions

"""
    SymmetricGroup(n)

Group action of the symmetric group S(n).
"""
struct SymmetricGroup
    permutations::Vector{Vector{Int}}
end
SymmetricGroup(N::Int) = SymmetricGroup(permutations(N))
function permutations(N::Int)
    N == 0 && return [Int[]]
    s = Vector(1:N)
    perms = [copy(s)]
    while true
        i = N - 1
        while i >= 1 && s[i] >= s[i + 1]
            i -= 1
        end
        if i > 0
            j = N
            while j > i && s[i] >= s[j]
                j -= 1
            end
            s[i], s[j] = s[j], s[i]
            reverse!(s, i + 1)
        else
            s[1] = N + 1
        end

        s[1] > N && break
        push!(perms, copy(s))
    end
    return perms
end

Base.eltype(::Type{SymmetricGroup}) = Vector{Int}
Base.length(p::SymmetricGroup) = length(p.permutations)
Base.iterate(p::SymmetricGroup) = iterate(p.permutations)
Base.iterate(p::SymmetricGroup, s) = iterate(p.permutations, s)
