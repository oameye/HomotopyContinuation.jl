## Excess-solution filtering for overdetermined solves.
#
# The squared-up system G = [I A]·F has solutions that need not solve the
# original overdetermined F. Every successful path is re-checked against F:
# - nonsingular endpoints: Newton on F (least-squares via QR, extended-precision
#   residuals) must converge from the endpoint within tolerances scaled to the
#   path accuracy, otherwise the endpoint is an excess solution.
# - singular endpoints: Newton cannot converge quadratically, so compare the
#   F-residual against the G-residual; an excess solution solves G but not F.

struct ExcessSolutionChecker
    system::SystemEvaluator          # original m×n system
    A::FSMat{ComplexF64}             # randomization block of G = [I A]·(F∘perm)
    perm::Vector{Int}
    workspace::MatrixWorkspace       # m×n QR workspace
    x::FSVec{ComplexF64}
    x_ext::FSVec{ComplexDF64}
    r::FSVec{ComplexF64}             # m-dim residual of F
    r_rand::FSVec{ComplexF64}        # n-dim residual of G
    Δx::FSVec{ComplexF64}
    p::FSVec{ComplexF64}             # empty parameter vector
end

function ExcessSolutionChecker(
        system::SystemEvaluator, A::FSMat{ComplexF64}, perm::Vector{Int},
    )::ExcessSolutionChecker
    m, n = size(system)
    return ExcessSolutionChecker(
        system, A, perm,
        MatrixWorkspace(m, n),
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, n)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexF64}(ComplexF64[]),
    )
end

# Descending-degree equation permutation, so the identity block of the
# randomization keeps the highest-degree equations and deg(G_i) = degrees[perm[i]].
_randomization_permutation(degrees::Vector{Int})::Vector{Int} =
    sortperm(degrees; rev = true)

"""
    _square_up(rng, F) -> (A, perm, checker)

Randomization data for squaring up an overdetermined system: the random fold
block `A`, the descending-degree equation permutation, and the excess-solution
checker for the post-tracking filter. Total-degree and polyhedral init both go
through this single helper so the RNG draw and the square-up construction
cannot drift apart.
"""
function _square_up(
        rng::Random.MersenneTwister, F::System,
    )::Tuple{FSMat{ComplexF64}, Vector{Int}, ExcessSolutionChecker}
    m, n = size(F.evaluator)
    perm = _randomization_permutation(F.degrees)
    A = FSMat{ComplexF64}(randn(rng, ComplexF64, n, m - n))
    return A, perm, ExcessSolutionChecker(F.evaluator, A, perm)
end

const EXCESS_NEWTON_MAX_ITERS = 10

"""
    _newton_refines(c, solution, accuracy) -> Bool

Newton's method on the original overdetermined system, starting from a path
endpoint. Returns `true` if it converges within tolerances scaled to the path
accuracy (`atol = rtol = 1e3·accuracy`, first update ≤ `100·√accuracy`).
"""
function _newton_refines(
        c::ExcessSolutionChecker, solution::Vector{ComplexF64}, accuracy::Float64,
    )::Bool
    acc = isfinite(accuracy) ? max(accuracy, eps()) : eps()
    atol = 1.0e3 * acc
    rtol = 1.0e3 * acc
    max_norm_first_update = 100.0 * sqrt(acc)

    copyto!(c.x, solution)
    norm_Δx_prev = Inf
    for k in 1:EXCESS_NEWTON_MAX_ITERS
        evaluate_and_jacobian!(c.r, c.workspace.A, c.system, c.x, c.p)
        updated!(c.workspace)
        # Extended-precision residual: the residual is dominated by cancellation
        # near a true solution, exactly where Float64 evaluation loses all digits.
        _copy_df64!(c.x_ext, c.x)
        evaluate!(c.r, c.system, c.x_ext, c.p)
        LA.ldiv!(c.Δx, c.workspace, c.r)

        # inf_norm's max comparison silently drops NaN entries, so check every
        # component explicitly instead of the norm.
        all(isfinite, c.Δx) || return false
        norm_Δx = inf_norm(c.Δx)

        @inbounds for i in eachindex(c.x, c.Δx)
            c.x[i] -= c.Δx[i]
        end

        k == 1 && norm_Δx > max_norm_first_update && return false
        norm_Δx <= max(atol, rtol * inf_norm(c.x)) && return true
        # Newton on a true regular solution contracts quadratically; a stalling
        # update sequence means the least-squares residual is not going to zero.
        k > 1 && norm_Δx > 0.5 * norm_Δx_prev && return false
        norm_Δx_prev = norm_Δx
    end
    return false
end

"""
    _residual_comparable(c, solution) -> Bool

For singular endpoints: `true` if the residual of the original system is within
a factor 100 of the residual of the squared-up system at the endpoint. The
G-residual is the randomization fold of the F-residual, so one evaluation of F
suffices.
"""
function _residual_comparable(
        c::ExcessSolutionChecker, solution::Vector{ComplexF64},
    )::Bool
    copyto!(c.x, solution)
    evaluate!(c.r, c.system, c.x, c.p)
    res_F = inf_norm(c.r)
    _randomize!(c.r_rand, c.A, c.perm, c.r)
    res_G = inf_norm(c.r_rand)
    return res_F <= 100.0 * max(res_G, eps())
end

"""
    check_excess_solution(c, r::PathResult) -> PathResult

Reclassify a successful path result as `PATH_EXCESS_SOLUTION` if its endpoint
does not solve the original overdetermined system.
"""
function check_excess_solution(c::ExcessSolutionChecker, r::PathResult)::PathResult
    is_success(r) || return r
    genuine = if r.singular
        _residual_comparable(c, r.solution)
    else
        _newton_refines(c, r.solution, r.accuracy)
    end
    genuine && return r
    return _with_return_code(r, PathResultCode.PATH_EXCESS_SOLUTION)
end

# Post-pass over all path results. `nothing` checker (square systems) is a no-op.
_check_excess_solutions!(::Vector{PathResult}, ::Nothing)::Nothing = nothing

function _check_excess_solutions!(
        results::Vector{PathResult}, c::ExcessSolutionChecker,
    )::Nothing
    for i in eachindex(results)
        results[i] = check_excess_solution(c, results[i])
    end
    return nothing
end
