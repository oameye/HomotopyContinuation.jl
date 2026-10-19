## `F(x; p)` seen as a system in the joint unknown `[x; p]`, which is how a
## monodromy start pair is found for a system with no symbolic equations. The
## parameter block of the Jacobian is the order-1 coefficient of `F(x; p + eⱼt)`.

struct StartPairSystem <: AbstractSystem
    F::SystemEvaluator
    nvars::Int
    nparams::Int
    # Scratch buffers
    x::FSVec{ComplexF64}
    x̄::FSVec{ComplexDF64}
    p::FSVec{ComplexF64}
    J::FSMat{ComplexF64}
    du::FSVec{ComplexF64}
    tx::TaylorVector{4, ComplexF64}
    tp::TaylorVector{4, ComplexF64}
end

function StartPairSystem(F::SystemEvaluator)::StartPairSystem
    m, n = size(F)
    np = nparameters(F)
    np > 0 || throw(
        ArgumentError("a start-pair system needs a system with parameters"),
    )
    return StartPairSystem(
        F, n, np,
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, n)),
        FSVec{ComplexF64}(zeros(ComplexF64, np)),
        FSMat{ComplexF64}(zeros(ComplexF64, m, n)),
        FSVec{ComplexF64}(zeros(ComplexF64, m)),
        TaylorVector{4, ComplexF64}(n),
        TaylorVector{4, ComplexF64}(np),
    )
end

clone_system(S::StartPairSystem)::StartPairSystem =
    StartPairSystem(clone_system_evaluator(S.F))

Base.size(S::StartPairSystem)::Tuple{Int, Int} =
    (size(S.F)[1], S.nvars + S.nparams)
nparameters(::StartPairSystem)::Int = 0

@inline function split_start_pair!(
        S::StartPairSystem, xp::FSVec{ComplexF64},
    )::Nothing
    n = S.nvars
    @inbounds for i in 1:n
        S.x[i] = xp[i]
    end
    @inbounds for j in 1:S.nparams
        S.p[j] = xp[n + j]
    end
    return nothing
end

# The evaluator takes `ComplexF64` parameters, so only the variable block keeps
# extended precision.
@inline function split_start_pair!(
        S::StartPairSystem, xp::FSVec{ComplexDF64},
    )::Nothing
    n = S.nvars
    @inbounds for i in 1:n
        S.x̄[i] = xp[i]
    end
    @inbounds for j in 1:S.nparams
        S.p[j] = ComplexF64(xp[n + j])
    end
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, S::StartPairSystem,
        xp::FSVec{ComplexF64}, ::FSVec{ComplexF64},
    )::Nothing
    split_start_pair!(S, xp)
    evaluate!(u, S.F, S.x, S.p)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, S::StartPairSystem,
        xp::FSVec{ComplexDF64}, ::FSVec{ComplexF64},
    )::Nothing
    split_start_pair!(S, xp)
    evaluate!(u, S.F, S.x̄, S.p)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexDF64}, S::StartPairSystem,
        xp::FSVec{ComplexDF64}, ::FSVec{ComplexF64},
    )::Nothing
    split_start_pair!(S, xp)
    evaluate!(u, S.F, S.x̄, S.p)
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64}, S::StartPairSystem,
        xp::FSVec{ComplexF64}, ::FSVec{ComplexF64},
    )::Nothing
    split_start_pair!(S, xp)
    evaluate_and_jacobian!(u, S.J, S.F, S.x, S.p)
    @inbounds for j in 1:S.nvars, i in axes(U, 1)
        U[i, j] = S.J[i, j]
    end
    parameter_jacobian!(U, S)
    return nothing
end

# Column `n + j` is `∂F/∂pⱼ`.
function parameter_jacobian!(
        U::FSMat{ComplexF64}, S::StartPairSystem,
    )::Nothing
    n = S.nvars
    dx = S.tx.data
    dp = S.tp.data
    @inbounds for i in 1:n
        dx[1, i] = S.x[i]
        dx[2, i] = zero(ComplexF64)
    end
    @inbounds for j in 1:S.nparams
        dp[1, j] = S.p[j]
        dp[2, j] = zero(ComplexF64)
    end
    tx = TaylorVector{2, ComplexF64}(dx)
    tp = TaylorVector{2, ComplexF64}(dp)
    @inbounds for j in 1:S.nparams
        dp[2, j] = one(ComplexF64)
        taylor!(S.du, Val(1), S.F, tx, tp)
        dp[2, j] = zero(ComplexF64)
        for i in axes(U, 1)
            U[i, n + j] = S.du[i]
        end
    end
    return nothing
end

@inline function split_start_pair_series!(
        S::StartPairSystem, ::Val{K}, txp::TaylorVector{N, ComplexF64},
    )::Nothing where {K, N}
    n = S.nvars
    d = txp.data
    dx = S.tx.data
    dp = S.tp.data
    @inbounds for i in 1:n, r in 1:(K + 1)
        dx[r, i] = d[r, i]
    end
    @inbounds for j in 1:S.nparams, r in 1:(K + 1)
        dp[r, j] = d[r, n + j]
    end
    return nothing
end

# The joint system has no parameters of its own, so a parameter series is empty.
function taylor!(
        u::FSVec{ComplexF64}, v::Val{K}, S::StartPairSystem,
        txp::TaylorVector{N, ComplexF64},
        ::Union{FSVec{ComplexF64}, TaylorVector{<:Any, ComplexF64}},
    )::Nothing where {K, N}
    split_start_pair_series!(S, v, txp)
    taylor!(
        u, v, S.F,
        TaylorVector{K + 1, ComplexF64}(S.tx.data),
        TaylorVector{K + 1, ComplexF64}(S.tp.data),
    )
    return nothing
end
