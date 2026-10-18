# Shared helper for comparing solution sets. Defines no tests.

"""
    same_solution_set(a, b; atol = 1.0e-8)

`true` when the points of `a` and `b` can be paired one to one with every pair
within `atol` in the maximum norm. Multiplicity counts: `[p, p, q]` and
`[p, q, q]` are different sets.
"""
function same_solution_set(a, b; atol::Float64 = 1.0e-8)::Bool
    length(a) == length(b) || return false
    n = length(a)
    close = [maximum(abs.(a[i] .- b[j])) < atol for i in 1:n, j in 1:n]
    match_of_b = zeros(Int, n)
    function augment!(i, seen)
        for j in 1:n
            (close[i, j] && !seen[j]) || continue
            seen[j] = true
            if match_of_b[j] == 0 || augment!(match_of_b[j], seen)
                match_of_b[j] = i
                return true
            end
        end
        return false
    end
    return all(i -> augment!(i, falses(n)), 1:n)
end
