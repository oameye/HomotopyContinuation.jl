## Multi-homogeneous total degree: one degree per variable group instead of one
## total degree.
#
# Group j of nⱼ variables contributes kⱼ affine coordinates: nⱼ - 1 when the system
# is multi-homogeneous (its last variable homogenizes the group), nⱼ otherwise.

_group_dims(groups::Vector{Vector{Int}}, homogeneous::Bool)::Vector{Int} =
    [length(group) - homogeneous for group in groups]

# Every map from equations to groups giving group j exactly k[j] equations and a
# positive degree at every equation; a zero degree contributes no paths.
function _bezout_assignments(D::Matrix{Int}, k::Vector{Int})::Vector{Vector{Int}}
    out = Vector{Vector{Int}}()
    _collect_assignments!(out, Vector{Int}(undef, size(D, 2)), copy(k), D, 1)
    return out
end

function _collect_assignments!(
        out::Vector{Vector{Int}}, assignment::Vector{Int},
        remaining::Vector{Int}, D::Matrix{Int}, i::Int,
    )::Nothing
    if i > length(assignment)
        push!(out, copy(assignment))
        return nothing
    end
    for j in eachindex(remaining)
        (remaining[j] > 0 && D[j, i] > 0) || continue
        remaining[j] -= 1
        assignment[i] = j
        _collect_assignments!(out, assignment, remaining, D, i + 1)
        remaining[j] += 1
    end
    return nothing
end

_assignment_paths(D::Matrix{Int}, assignment::Vector{Int})::Int =
    prod(i -> D[assignment[i], i], eachindex(assignment); init = 1)

# Multi-homogeneous Bezout number: `D` has one row per group, `k[j]` is the
# number of affine coordinates of group j.
_multi_bezout_count(D::Matrix{Int}, k::Vector{Int})::Int =
    sum(a -> _assignment_paths(D, a), _bezout_assignments(D, k); init = 0)

# `C[j][l, i]` multiplies the l-th affine coordinate of group j in the linear form
# equation i uses. The leading identity block makes a single group reproduce xᵢ^dᵢ - 1.
function _multi_start_coefficients(
        rng::Random.MersenneTwister, k::Vector{Int}, N::Int,
    )::Vector{Matrix{ComplexF64}}
    return map(k) do kⱼ
        C = randn(rng, ComplexF64, kⱼ, N)
        for i in 1:min(kⱼ, N), l in 1:kⱼ
            C[l, i] = l == i
        end
        C
    end
end

function _multi_linear_form(
        C::Matrix{ComplexF64}, i::Int, group::Vector{Int},
        kⱼ::Int, z::Vector{Expression},
    )::Expression
    b = zero(Expression)
    for l in 1:kⱼ
        c = C[l, i]
        iszero(c) && continue
        b += isone(c) ? z[group[l]] : c * z[group[l]]
    end
    return b
end

# Equation i is the product over the groups it has positive degree in of
# `bᵢⱼ^dᵢⱼ - wⱼ^dᵢⱼ`, with `bᵢⱼ` a linear form in the affine coordinates of group j
# and `wⱼ` its homogenizing coordinate (`1` when not homogeneous). Homogeneous input
# gets one row `wⱼ - 1` per group, matching the target's chart rows.
function _multi_start_system(
        D::Matrix{Int}, k::Vector{Int}, groups::Vector{Vector{Int}},
        C::Vector{Matrix{ComplexF64}}, homogeneous::Bool, n::Int,
    )::System
    z = variable_array(:z, 1:n)
    N = size(D, 2)
    eqs = Vector{Expression}(undef, homogeneous ? N + length(groups) : N)
    for i in 1:N
        gᵢ = one(Expression)
        for j in eachindex(groups)
            d = D[j, i]
            d == 0 && continue
            b = _multi_linear_form(C[j], i, groups[j], k[j], z)
            gᵢ *= homogeneous ? b^d - z[last(groups[j])]^d : b^d - 1
        end
        eqs[i] = gᵢ
    end
    if homogeneous
        for j in eachindex(groups)
            eqs[N + j] = z[last(groups[j])] - 1
        end
    end
    return System(eqs; variables = z)
end

# The solutions of `_multi_start_system`. Each of group j's k[j] equations fixes its
# linear form to a root of unity, and the k[j] × k[j] system this poses is the same
# for every choice of roots, so it is factorized once per assignment.
function _multi_start_solutions(
        D::Matrix{Int}, k::Vector{Int}, groups::Vector{Vector{Int}},
        C::Vector{Matrix{ComplexF64}}, homogeneous::Bool, n::Int,
        assignments::Vector{Vector{Int}},
    )::Vector{Vector{ComplexF64}}
    N = size(D, 2)
    M = length(groups)
    out = Vector{Vector{ComplexF64}}()
    rows = [Int[] for _ in 1:M]
    row_of = Vector{Int}(undef, N)
    rhs = [Vector{ComplexF64}(undef, kⱼ) for kⱼ in k]
    for assignment in assignments
        foreach(empty!, rows)
        for i in 1:N
            push!(rows[assignment[i]], i)
            row_of[i] = length(rows[assignment[i]])
        end
        factorizations = map(1:M) do j
            A = Matrix{ComplexF64}(undef, k[j], k[j])
            for (s, i) in enumerate(rows[j]), l in 1:k[j]
                A[s, l] = C[j][l, i]
            end
            LA.lu!(A)
        end
        for path in 0:(_assignment_paths(D, assignment) - 1)
            q = path
            for i in 1:N
                d = D[assignment[i], i]
                q, r = divrem(q, d)
                rhs[assignment[i]][row_of[i]] = cis(2π * r / d)
            end
            x = zeros(ComplexF64, n)
            for j in 1:M
                y = copy(rhs[j])
                LA.ldiv!(factorizations[j], y)
                for l in 1:k[j]
                    x[groups[j][l]] = y[l]
                end
                homogeneous && (x[last(groups[j])] = 1)
            end
            push!(out, x)
        end
    end
    return out
end

# One chart row `cⱼ·x_Gⱼ = 1` per group, meeting the cone of every multi-projective
# solution in a single point. A slice, so the sliced machinery appends the rows.
function _multi_chart_subspace(
        rng::Random.MersenneTwister, groups::Vector{Vector{Int}}, n::Int,
    )::LinearSubspace{ComplexF64}
    A = zeros(ComplexF64, length(groups), n)
    for (j, group) in enumerate(groups), i in group
        A[j, i] = randn(rng, ComplexF64)
    end
    return LinearSubspace(A, ones(ComplexF64, length(groups)))
end

# Square-up data for `[F; chart rows]`, folded to its first `N` equations plus the
# chart rows. The chart rows stay identity rows (their block of `A` is zero), so they
# still cut a representative out of every solution cone. Returns the fold, the
# permutation of the charted system, and that of `F` alone.
function _multi_square_up(
        rng::Random.MersenneTwister, degs::Vector{Int}, m::Int, N::Int, nchart::Int,
    )::Tuple{FSMat{ComplexF64}, Vector{Int}, Vector{Int}}
    perm_equations = _randomization_permutation(degs)
    perm = [
        perm_equations[1:N]; (m + 1):(m + nchart); perm_equations[(N + 1):m]
    ]
    A = zeros(ComplexF64, N + nchart, m - N)
    A[1:N, :] .= randn(rng, ComplexF64, N, m - N)
    return FSMat{ComplexF64}(A), perm, perm_equations
end

# Degrees of `[I A]·(F∘perm)`: equation i folds in every equation past the first
# `N`, so its degree in a group is the largest of theirs.
function _folded_group_degrees(D::Matrix{Int}, perm::Vector{Int}, N::Int)::Matrix{Int}
    out = Matrix{Int}(undef, size(D, 1), N)
    for j in axes(D, 1)
        excess = 0
        for i in (N + 1):length(perm)
            excess = max(excess, D[j, perm[i]])
        end
        for i in 1:N
            out[j, i] = max(D[j, perm[i]], excess)
        end
    end
    return out
end

function _check_multi_determined(F::System, N::Int, M::Int)::Nothing
    m, n = size(F)
    m >= N || throw(
        ArgumentError(
            "`TotalDegree` puts the homogeneous system on one affine chart per " *
                "variable group, so $m equation(s) in $n variables and $M group(s) " *
                "leave $(N - m) too few. The projective solution set is " *
                "positive-dimensional; only finitely many solutions are supported.",
        ),
    )
    return nothing
end

function _init_multi_homogeneous(
        F::System, alg::TotalDegree, exec::AbstractExecutor,
    )
    show_progress = _show_progress(alg)
    groups = variable_groups(F)
    homogeneous = is_homogeneous(F)
    k = _group_dims(groups, homogeneous)
    M = length(groups)
    m, n = size(F)
    N = sum(k)
    if homogeneous
        _check_multi_determined(F, N, M)
    else
        _check_square_or_overdetermined(F)
    end

    rng = Random.MersenneTwister(_seed(alg))
    γ = _random_gamma(rng)

    D = multi_degrees(F)
    L = homogeneous ? _multi_chart_subspace(rng, groups, n) : _full_subspace(n)

    A = FSMat{ComplexF64}(zeros(ComplexF64, 0, 0))
    perm = Int[]
    checker = nothing
    if m > N
        A, perm, perm_equations = _multi_square_up(
            rng, degrees(F), m, N, homogeneous ? M : 0,
        )
        # The excess check runs Newton on the charted but unfolded system, which
        # the folded homotopy does not expose.
        charted = homogeneous ?
            _sliced_evaluator(F.evaluator, L, ComplexF64[]) : F.evaluator
        checker = ExcessSolutionChecker(charted, A, perm)
        D = _folded_group_degrees(D, perm_equations, N)
    end

    C = _multi_start_coefficients(rng, k, N)
    start = _multi_start_system(D, k, groups, C, homogeneous, n)
    starts = _multi_start_solutions(
        D, k, groups, C, homogeneous, n, _bezout_assignments(D, k),
    )
    builder = MultiHomogeneousBuilder(
        start, F, A, perm, L, γ, _tracker_options(alg), _endgame_options(alg),
    )
    return _solve_cache(
        exec, builder, starts, _seed(alg), checker, show_progress,
        early_stop_callback(alg),
    )
end
