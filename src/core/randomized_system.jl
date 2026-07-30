## RandomizedSystem: square-up wrapper for overdetermined systems.
#
# For an m×n system F with m > n equations, represents the square system
#   G_i(x) = F_perm[i](x) + Σ_j A[i,j]·F_perm[n+j](x),   i = 1..n
# i.e. G = [I A]·(F∘perm) with an identity block on the first n (permuted)
# equations and a random block A folding in the remaining m-n equations.
# With perm sorting equations by descending degree, deg(G_i) = deg(F_perm[i]),
# so the top-n degrees of F are the degrees of G.
#
# The wrapper works on the type-erased SystemEvaluator, so any compile mode of
# the inner system (INTERPRETED/COMPILED/COMPILED_ALL) is reused unchanged.
# Since x ↦ A·x is linear and constant in t, every Taylor coefficient of G is
# the randomization of the corresponding Taylor coefficient of F.

struct RandomizedSystem <: AbstractSystem
    system::SystemEvaluator
    A::FSMat{ComplexF64}
    perm::Vector{Int}
    # Scratch buffers (contents mutated, references fixed)
    u_full::FSVec{ComplexF64}
    U_full::FSMat{ComplexF64}
    ū_full::FSVec{ComplexDF64}
end

function RandomizedSystem(
        system::SystemEvaluator, A::FSMat{ComplexF64}, perm::Vector{Int},
    )::RandomizedSystem
    m, n = size(system)
    m > n || throw(ArgumentError("RandomizedSystem requires an overdetermined system, got size ($m, $n)"))
    size(A) == (n, m - n) ||
        throw(ArgumentError("Randomization block must have size ($n, $(m - n)), got $(size(A))"))
    length(perm) == m ||
        throw(ArgumentError("Equation permutation must have length $m, got $(length(perm))"))
    # _randomize! indexes v[perm[i]] under @inbounds; a malformed perm would be
    # memory-unsafe, so validate once at construction time.
    isperm(perm) ||
        throw(ArgumentError("Equation permutation must be a permutation of 1:$m"))
    return RandomizedSystem(
        system, A, perm,
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSMat{ComplexF64}(zeros(ComplexF64, m, n)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, m)),
    )
end

Base.size(R::RandomizedSystem)::Tuple{Int, Int} = (size(R.system)[2], size(R.system)[2])
nparameters(R::RandomizedSystem)::Int = nparameters(R.system)

# Canonical square-up wrap: every consumer (solve init, worker builders) goes
# through this so the evaluator construction cannot drift between call sites.
_randomized_evaluator(
    inner::SystemEvaluator, A::FSMat{ComplexF64}, perm::Vector{Int},
)::SystemEvaluator = SystemEvaluator(RandomizedSystem(inner, A, perm))

_clone_system(R::RandomizedSystem)::RandomizedSystem =
    RandomizedSystem(_clone_system_evaluator(R.system), R.A, R.perm)

# u[i] = v[perm[i]] + Σ_j A[i,j]·v[perm[n+j]]
function _randomize!(
        u::FSVec{ComplexF64}, A::FSMat{ComplexF64}, perm::Vector{Int},
        v::FSVec{ComplexF64},
    )::Nothing
    n = length(u)
    @inbounds for i in 1:n
        u[i] = v[perm[i]]
    end
    @inbounds for j in axes(A, 2)
        v_j = v[perm[n + j]]
        for i in 1:n
            u[i] += A[i, j] * v_j
        end
    end
    return nothing
end

# DF64 fold variants: near a solution of G the identity term and the A fold
# cancel, so accumulate in extended precision (ComplexF64 · ComplexDF64
# promotes) and round only on store into the output.
function _randomize!(
        u::FSVec{ComplexF64}, A::FSMat{ComplexF64}, perm::Vector{Int},
        v::FSVec{ComplexDF64},
    )::Nothing
    n = length(u)
    @inbounds for i in 1:n
        acc = v[perm[i]]
        for j in axes(A, 2)
            acc += A[i, j] * v[perm[n + j]]
        end
        u[i] = ComplexF64(acc)
    end
    return nothing
end

function _randomize!(
        u::FSVec{ComplexDF64}, A::FSMat{ComplexF64}, perm::Vector{Int},
        v::FSVec{ComplexDF64},
    )::Nothing
    n = length(u)
    @inbounds for i in 1:n
        acc = v[perm[i]]
        for j in axes(A, 2)
            acc += A[i, j] * v[perm[n + j]]
        end
        u[i] = acc
    end
    return nothing
end

# U[i,k] = V[perm[i],k] + Σ_j A[i,j]·V[perm[n+j],k]
function _randomize_jacobian!(
        U::FSMat{ComplexF64}, A::FSMat{ComplexF64}, perm::Vector{Int},
        V::FSMat{ComplexF64},
    )::Nothing
    n = size(U, 1)
    @inbounds for k in axes(U, 2)
        for i in 1:n
            U[i, k] = V[perm[i], k]
        end
        for j in axes(A, 2)
            v_jk = V[perm[n + j], k]
            for i in 1:n
                U[i, k] += A[i, j] * v_jk
            end
        end
    end
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, R::RandomizedSystem,
        x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate!(R.u_full, R.system, x, p)
    _randomize!(u, R.A, R.perm, R.u_full)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, R::RandomizedSystem,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate!(R.ū_full, R.system, x, p)
    _randomize!(u, R.A, R.perm, R.ū_full)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexDF64}, R::RandomizedSystem,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate!(R.ū_full, R.system, x, p)
    _randomize!(u, R.A, R.perm, R.ū_full)
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64},
        R::RandomizedSystem, x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    evaluate_and_jacobian!(R.u_full, R.U_full, R.system, x, p)
    _randomize!(u, R.A, R.perm, R.u_full)
    _randomize_jacobian!(U, R.A, R.perm, R.U_full)
    return nothing
end

# Taylor: order-K coefficient of A·F is A·(order-K coefficient of F). Shared
# implementation for all taylor! dispatch variants.
@inline function _randomized_taylor!(
        u::FSVec{ComplexF64}, v::Val{K}, R::RandomizedSystem,
        tx::TaylorVector{M, ComplexF64},
        p::Union{FSVec{ComplexF64}, TaylorVector{M, ComplexF64}},
    )::Nothing where {K, M}
    taylor!(R.u_full, v, R.system, tx, p)
    _randomize!(u, R.A, R.perm, R.u_full)
    return nothing
end

# FSVec parameter variants (used by StraightLineHomotopy, etc.)
taylor!(
    u::FSVec{ComplexF64}, v::Val{1}, R::RandomizedSystem,
    tx::TaylorVector{2, ComplexF64}, p::FSVec{ComplexF64},
)::Nothing = _randomized_taylor!(u, v, R, tx, p)
taylor!(
    u::FSVec{ComplexF64}, v::Val{2}, R::RandomizedSystem,
    tx::TaylorVector{3, ComplexF64}, p::FSVec{ComplexF64},
)::Nothing = _randomized_taylor!(u, v, R, tx, p)
taylor!(
    u::FSVec{ComplexF64}, v::Val{3}, R::RandomizedSystem,
    tx::TaylorVector{4, ComplexF64}, p::FSVec{ComplexF64},
)::Nothing = _randomized_taylor!(u, v, R, tx, p)

# TaylorVector parameter variants (used by CoefficientHomotopy/ToricHomotopy Cauchy product path)
taylor!(
    u::FSVec{ComplexF64}, v::Val{1}, R::RandomizedSystem,
    tx::TaylorVector{2, ComplexF64}, tp::TaylorVector{2, ComplexF64},
)::Nothing = _randomized_taylor!(u, v, R, tx, tp)
taylor!(
    u::FSVec{ComplexF64}, v::Val{2}, R::RandomizedSystem,
    tx::TaylorVector{3, ComplexF64}, tp::TaylorVector{3, ComplexF64},
)::Nothing = _randomized_taylor!(u, v, R, tx, tp)
taylor!(
    u::FSVec{ComplexF64}, v::Val{3}, R::RandomizedSystem,
    tx::TaylorVector{4, ComplexF64}, tp::TaylorVector{4, ComplexF64},
)::Nothing = _randomized_taylor!(u, v, R, tx, tp)
