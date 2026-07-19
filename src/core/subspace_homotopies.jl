## Subspace homotopies.
#
# The computation is performed using Stiefel coordinates for the Grassmannian.
# Two homotopies move a linear space V (start, t = 1) to W (target, t = 0):
# * ExtrinsicSubspaceHomotopy
#       for V = {x | Ax - a = 0} and W = {x | Bx - b = 0}
#       moves [F; Ax - a] to [F; Bx - b] by taking a geodesic in the
#       Grassmannian from A to B and interpolating a and b linearly
# * IntrinsicSubspaceHomotopy
#       for V = {x = Av + a} and W = {x = Bv + b}
#       moves F(Av + a) to F(Bv + b) by taking a geodesic in the
#       Grassmannian from A to B and interpolating a and b linearly
#
# Implementation notes:
# * Cache invalidation uses complex(NaN), never complex(0.0): a 0.0 sentinel
#   serves a stale γ when the first query after a retarget is exactly t = 0.
# * Taylor methods go up to Val(3) (the predictor's maximum order). The
#   formulas drop the highest-order tx term; that is correct because the
#   predictor zeroes that row before each taylor! call (see
#   src/tracking/predictor.jl).
# * Solution conversion is not tracker business: the tracker-facing
#   set_solution!/get_solution! are identity copies; the monodromy solver calls
#   intrinsic_coordinates!/ambient_coordinates! on the concrete homotopy.
# * Geodesics are recomputed on retarget (cold path), not cached.

## Data structure for geodesics in the Grassmannian

struct GrassmannianGeodesic
    Q::Matrix{ComplexF64}
    Q_cos::Matrix{ComplexF64}
    Θ::Vector{Float64}
    U::Matrix{ComplexF64}
    γ1::Matrix{ComplexF64}
    B_start::Union{Nothing, Matrix{ComplexF64}} # for base change (extrinsic only)
    B_target::Union{Nothing, Matrix{ComplexF64}}
end

# γ(t) = Q_cos * cos(t Θ) + Q * sin(t Θ) columnwise: t = 0 gives the target
# frame Q_cos, t = 1 the start frame returned here.
function _geodesic_start_frame(
        Q_cos::Matrix{ComplexF64}, Q::Matrix{ComplexF64}, Θ::Vector{Float64},
    )::Matrix{ComplexF64}
    γ1 = similar(Q_cos)
    n, k = size(γ1)
    @inbounds for j in 1:k
        s, c = sincos(Θ[j])
        for i in 1:n
            γ1[i, j] = Q_cos[i, j] * c + Q[i, j] * s
        end
    end
    return γ1
end

# Two concrete constructors: `start` and `target` always share the same
# description kind (both extrinsic or both intrinsic). Keeping the argument
# types concrete (rather than a shared `Union`) means grassmannian_svd is only
# ever called on matching kinds, never a mixed extrinsic/intrinsic dispatch.
function GrassmannianGeodesic(
        start::ExtrinsicDescription{ComplexF64}, target::ExtrinsicDescription{ComplexF64},
    )
    Q, Θ, U = grassmannian_svd(target, start)
    Q_cos = transpose(target.A) * U
    γ1 = _geodesic_start_frame(Q_cos, Q, Θ)
    return GrassmannianGeodesic(
        Q, Q_cos, Θ, U, γ1, transpose(γ1) * start.A', transpose(Q_cos) * target.A',
    )
end

function GrassmannianGeodesic(
        start::IntrinsicDescription{ComplexF64}, target::IntrinsicDescription{ComplexF64},
    )
    Q, Θ, U = grassmannian_svd(target, start)
    Q_cos = target.X * U
    γ1 = _geodesic_start_frame(Q_cos, Q, Θ)
    return GrassmannianGeodesic(Q, Q_cos, Θ, U, γ1, nothing, nothing)
end

"""
    IntrinsicSubspaceHomotopy(system::SystemEvaluator, V::LinearSubspace, W::LinearSubspace)

Creates a homotopy ``H(v, t) = F(A(t)v + a(t))`` from ``V`` to ``W``. At
``t = 1`` we have ``H(v, 1) = F(Av + a)`` where ``V = \\{x = Av + a\\}`` and
``A`` is a Stiefel matrix; at ``t = 0`` similarly for ``W``. The matrix part
follows a geodesic in the Grassmannian, the offset interpolates linearly.
"""
mutable struct IntrinsicSubspaceHomotopy <: AbstractHomotopy
    # Mutable: start/target/path are reassigned by set_subspaces! between
    # monodromy loops. All buffer fields are const (contents mutated in place).
    const system::SystemEvaluator
    start::LinearSubspace{ComplexF64}
    target::LinearSubspace{ComplexF64}
    path::GrassmannianGeodesic

    # For the offset part (linear interpolation)
    const a_minus_b::FSVec{ComplexF64}
    const offset::FSVec{ComplexF64}

    # caches for t
    const t_cache::Base.RefValue{ComplexF64}
    const offset_t_cache::Base.RefValue{ComplexF64}

    # caches for Jacobian, ambient point x and its derivative ẋ
    const J::FSMat{ComplexF64}
    const x::FSVec{ComplexF64}
    const ẋ::FSVec{ComplexF64}
    const x_high::FSVec{ComplexDF64}

    # for taylor
    const taylor_t_cache::Base.RefValue{ComplexF64}
    const taylor_γ::NTuple{4, Matrix{ComplexF64}}
    const tx2::TaylorVector{3, ComplexF64}
    const tx3::TaylorVector{4, ComplexF64}

    # Genericity perturbation applied to the start subspace; reapplied on every
    # set_subspaces! retarget so all monodromy loops trace consistently
    # perturbed paths (|gamma| = 1, or exactly 1 when disabled).
    const gamma::ComplexF64
end

# Normalize a caller-supplied gamma to |gamma| = 1, or 1 when disabled (nothing).
_normalize_gamma(gamma::Nothing)::ComplexF64 = one(ComplexF64)
_normalize_gamma(gamma::ComplexF64)::ComplexF64 = gamma / abs(gamma)

# Apply the genericity perturbation to a start subspace (identity when g == 1).
function _apply_gamma(g::ComplexF64, L::LinearSubspace{ComplexF64})::LinearSubspace{ComplexF64}
    isone(g) && return L
    return LinearSubspace(g .* extrinsic(L).A, g .* extrinsic(L).b)
end

function IntrinsicSubspaceHomotopy(
        system::SystemEvaluator,
        start::LinearSubspace,
        target::LinearSubspace;
        gamma::Union{Nothing, ComplexF64} = cis(2 * pi * rand()),
    )
    g = _normalize_gamma(gamma)
    # multiply with random complex number to get generic paths
    start_c = _apply_gamma(g, convert(LinearSubspace{ComplexF64}, start))
    target_c = convert(LinearSubspace{ComplexF64}, target)

    path = GrassmannianGeodesic(intrinsic(start_c), intrinsic(target_c))
    Q = path.Q
    n = size(Q, 1)

    a = intrinsic(start_c).b
    b = intrinsic(target_c).b

    return IntrinsicSubspaceHomotopy(
        system,
        start_c,
        target_c,
        path,
        FSVec{ComplexF64}(a - b),
        FSVec{ComplexF64}(copy(b)),
        Ref(complex(NaN, NaN)),
        Ref(complex(NaN, NaN)),
        FSMat{ComplexF64}(zeros(ComplexF64, size(system))),
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexF64}(zeros(ComplexF64, n)),
        FSVec{ComplexDF64}(zeros(ComplexDF64, n)),
        Ref(complex(NaN, NaN)),
        ntuple(_ -> similar(Q), 4),
        TaylorVector{3, ComplexF64}(n),
        TaylorVector{4, ComplexF64}(n),
        g,
    )
end

function IntrinsicSubspaceHomotopy(
        F::System, start::LinearSubspace, target::LinearSubspace;
        gamma::Union{Nothing, ComplexF64} = cis(2 * pi * rand()),
    )
    return IntrinsicSubspaceHomotopy(F.evaluator, start, target; gamma = gamma)
end

Base.size(H::IntrinsicSubspaceHomotopy) = (size(H.system)[1], dim(H.start))

"""
    ExtrinsicSubspaceHomotopy(system::SystemEvaluator, V::LinearSubspace, W::LinearSubspace)

Creates a homotopy ``H(x, t) = [F(x); A(t)x - a(t)]`` from ``V`` to ``W``.
At ``t = 1`` we have ``H(x, 1) = [F(x); Ax - a]`` where ``V = \\{x | Ax = a\\}``
and ``A`` is a Stiefel matrix; at ``t = 0`` similarly for ``W``.
"""
mutable struct ExtrinsicSubspaceHomotopy <: AbstractHomotopy
    # Mutable: start/target/path are reassigned by set_subspaces! between
    # monodromy loops. All buffer fields are const (contents mutated in place).
    const system::SystemEvaluator
    start::LinearSubspace{ComplexF64}
    target::LinearSubspace{ComplexF64}
    path::GrassmannianGeodesic

    # Offsets in the Stiefel bases
    const a0::FSVec{ComplexF64}
    const b0::FSVec{ComplexF64}
    const a_minus_b::FSVec{ComplexF64}
    const offset::FSVec{ComplexF64}

    # caches for t
    const t_cache::Base.RefValue{ComplexF64}
    const offset_t_cache::Base.RefValue{ComplexF64}

    # for taylor
    const L::FSVec{ComplexF64}
    const taylor_t_cache::Base.RefValue{ComplexF64}
    const taylor_γ::NTuple{4, Matrix{ComplexF64}}

    # Genericity perturbation applied to the start subspace; reapplied on every
    # set_subspaces! retarget so all monodromy loops trace consistently
    # perturbed paths (|gamma| = 1, or exactly 1 when disabled).
    const gamma::ComplexF64
end

function ExtrinsicSubspaceHomotopy(
        system::SystemEvaluator,
        start::LinearSubspace,
        target::LinearSubspace;
        gamma::Union{Nothing, ComplexF64} = cis(2 * pi * rand()),
    )
    g = _normalize_gamma(gamma)
    start_c = _apply_gamma(g, convert(LinearSubspace{ComplexF64}, start))
    target_c = convert(LinearSubspace{ComplexF64}, target)

    path = GrassmannianGeodesic(extrinsic(start_c), extrinsic(target_c))
    # Get correct coordinates for a and b in the Stiefel homotopy:
    # extrinsic(start).A is replaced by transpose(path.γ1) and
    # extrinsic(target).A by transpose(path.Q_cos).
    a0 = something(path.B_start) * extrinsic(start_c).b
    b0 = something(path.B_target) * extrinsic(target_c).b
    k = size(path.γ1, 2)

    return ExtrinsicSubspaceHomotopy(
        system,
        start_c,
        target_c,
        path,
        FSVec{ComplexF64}(a0),
        FSVec{ComplexF64}(b0),
        FSVec{ComplexF64}(a0 - b0),
        FSVec{ComplexF64}(copy(b0)),
        Ref(complex(NaN, NaN)),
        Ref(complex(NaN, NaN)),
        FSVec{ComplexF64}(zeros(ComplexF64, k)),
        Ref(complex(NaN, NaN)),
        ntuple(_ -> similar(path.Q), 4),
        g,
    )
end

function ExtrinsicSubspaceHomotopy(
        F::System, start::LinearSubspace, target::LinearSubspace;
        gamma::Union{Nothing, ComplexF64} = cis(2 * pi * rand()),
    )
    return ExtrinsicSubspaceHomotopy(F.evaluator, start, target; gamma = gamma)
end

Base.size(H::ExtrinsicSubspaceHomotopy) =
    (size(H.system)[1] + size(H.path.γ1, 2), size(H.system)[2])

const SubspaceHomotopy = Union{ExtrinsicSubspaceHomotopy, IntrinsicSubspaceHomotopy}

"""
    set_subspaces!(H::SubspaceHomotopy, start::LinearSubspace, target::LinearSubspace)

Update the homotopy `H` to track from the linear subspace `start` to `target`.
All t-caches are invalidated with `complex(NaN)` (a 0.0 sentinel would serve
a stale γ if the first query after a retarget is exactly t = 0).
"""
function set_subspaces!(
        H::IntrinsicSubspaceHomotopy, start::LinearSubspace, target::LinearSubspace,
    )::Nothing
    H.start = _apply_gamma(H.gamma, convert(LinearSubspace{ComplexF64}, start))
    H.target = convert(LinearSubspace{ComplexF64}, target)
    H.path = GrassmannianGeodesic(intrinsic(H.start), intrinsic(H.target))
    H.a_minus_b .= intrinsic(H.start).b .- intrinsic(H.target).b
    H.offset .= intrinsic(H.target).b
    H.t_cache[] = complex(NaN)
    H.offset_t_cache[] = complex(NaN)
    H.taylor_t_cache[] = complex(NaN)
    return nothing
end

function set_subspaces!(
        H::ExtrinsicSubspaceHomotopy, start::LinearSubspace, target::LinearSubspace,
    )::Nothing
    H.start = _apply_gamma(H.gamma, convert(LinearSubspace{ComplexF64}, start))
    H.target = convert(LinearSubspace{ComplexF64}, target)
    H.path = GrassmannianGeodesic(extrinsic(H.start), extrinsic(H.target))
    LA.mul!(H.a0, something(H.path.B_start), extrinsic(H.start).b)
    LA.mul!(H.b0, something(H.path.B_target), extrinsic(H.target).b)
    H.a_minus_b .= H.a0 .- H.b0
    H.offset .= H.b0
    H.t_cache[] = complex(NaN)
    H.offset_t_cache[] = complex(NaN)
    H.taylor_t_cache[] = complex(NaN)
    return nothing
end

start_parameters!(H::SubspaceHomotopy, p::LinearSubspace)::Nothing =
    set_subspaces!(H, convert(LinearSubspace{ComplexF64}, p), H.target)
target_parameters!(H::SubspaceHomotopy, q::LinearSubspace)::Nothing =
    set_subspaces!(H, H.start, convert(LinearSubspace{ComplexF64}, q))
parameters!(H::SubspaceHomotopy, p::LinearSubspace, q::LinearSubspace)::Nothing =
    set_subspaces!(
    H,
    convert(LinearSubspace{ComplexF64}, p),
    convert(LinearSubspace{ComplexF64}, q),
)

## Geodesic frame helpers

# γ(t) into first(H.taylor_γ), cached on t.
function γ!(H::SubspaceHomotopy, t::ComplexF64)::Matrix{ComplexF64}
    H.t_cache[] != t || return first(H.taylor_γ)
    if isreal(t)
        _γ!(H, real(t))
    else
        _γ!(H, t)
    end
    H.t_cache[] = t
    # The order-0 frame was just overwritten in place; the cached higher-order
    # frames no longer belong to the same t, so a later taylor_γ! at the old t
    # must recompute instead of trusting its cache.
    H.taylor_t_cache[] = complex(NaN, NaN)

    return first(H.taylor_γ)
end

@inline function _γ!(H::SubspaceHomotopy, t::Number)
    Q, Q_cos, Θ = H.path.Q, H.path.Q_cos, H.path.Θ
    γ = first(H.taylor_γ)
    n, k = size(γ)
    @inbounds for j in 1:k
        Θⱼ = Θ[j]
        s, c = sincos(t * Θⱼ)
        for i in 1:n
            γ[i, j] = Q_cos[i, j] * c + Q[i, j] * s
        end
    end
    return γ
end

function γ̇!(H::SubspaceHomotopy, t::ComplexF64)
    # Overwrites the order-1 frame in place; invalidate the taylor_γ! cache for
    # the same reason as in γ!.
    H.taylor_t_cache[] = complex(NaN, NaN)
    return isreal(t) ? _γ̇!(H, real(t)) : _γ̇!(H, t)
end

@inline function _γ̇!(H::SubspaceHomotopy, t::Number)
    Q, Q_cos, Θ = H.path.Q, H.path.Q_cos, H.path.Θ
    γ̇ = H.taylor_γ[2]
    n, k = size(γ̇)
    @inbounds for j in 1:k
        Θⱼ = Θ[j]
        s, c = sincos(t * Θⱼ)
        ċ = -s * Θⱼ
        ṡ = c * Θⱼ
        for i in 1:n
            γ̇[i, j] = Q_cos[i, j] * ċ + Q[i, j] * ṡ
        end
    end
    return γ̇
end

# Compute the offset at parameter t: offset(t) = t*a + (1-t)*b = t(a-b) + b
@inline function _compute_offset!(H::IntrinsicSubspaceHomotopy, t::ComplexF64)
    H.offset .= intrinsic(H.target).b
    LA.axpy!(t, H.a_minus_b, H.offset)
    return H.offset_t_cache[] = t
end
@inline function _compute_offset!(H::ExtrinsicSubspaceHomotopy, t::ComplexF64)
    H.offset .= H.b0
    LA.axpy!(t, H.a_minus_b, H.offset)
    return H.offset_t_cache[] = t
end

@inline function offset_at_t!(H::SubspaceHomotopy, t::ComplexF64)
    if H.offset_t_cache[] != t
        _compute_offset!(H, t)
    end
    return H.offset
end

# Taylor coefficients of γ(t) up to order 3, the predictor's maximum order.
function _taylor_γ!(H::SubspaceHomotopy, t::Number)
    Q, Q_cos, Θ = H.path.Q, H.path.Q_cos, H.path.Θ
    γ, γ¹, γ², γ³ = H.taylor_γ
    n, k = size(γ)
    @inbounds for j in 1:k
        Θⱼ = Θ[j]
        s, c = sincos(t * Θⱼ)
        c¹ = -s * Θⱼ
        s¹ = c * Θⱼ
        Θⱼ_2 = 0.5 * Θⱼ^2
        c² = -c * Θⱼ_2
        s² = -s * Θⱼ_2
        Θⱼ_3 = Θⱼ_2 * Θⱼ / 3
        c³ = s * Θⱼ_3
        s³ = -c * Θⱼ_3
        for i in 1:n
            γ[i, j] = Q_cos[i, j] * c + Q[i, j] * s
            γ¹[i, j] = Q_cos[i, j] * c¹ + Q[i, j] * s¹
            γ²[i, j] = Q_cos[i, j] * c² + Q[i, j] * s²
            γ³[i, j] = Q_cos[i, j] * c³ + Q[i, j] * s³
        end
    end
    return nothing
end

function taylor_γ!(H::SubspaceHomotopy, t::ComplexF64)
    H.taylor_t_cache[] != t || return H.taylor_γ

    if isreal(t)
        _taylor_γ!(H, real(t))
    else
        _taylor_γ!(H, t)
    end
    H.taylor_t_cache[] = t
    # The order-0 frame is now valid for γ! too.
    H.t_cache[] = t

    return H.taylor_γ
end

############################
## IntrinsicSubspaceHomotopy
############################

"""
    intrinsic_coordinates!(u, H::IntrinsicSubspaceHomotopy, x, t)

Convert the ambient point `x` into intrinsic coordinates `u` of the subspace
γ(t), using the exact frames `γ1` at t = 1 and `Q_cos` at t = 0.
"""
function intrinsic_coordinates!(
        u::AbstractVector{ComplexF64}, H::IntrinsicSubspaceHomotopy,
        x::AbstractVector{ComplexF64}, t::ComplexF64,
    )::Nothing
    length(x) == length(H.x) ||
        throw(ArgumentError("Cannot convert solution. Expected ambient coordinates."))
    offset_at_t!(H, t)
    H.x .= x .- H.offset
    if isone(t)
        LA.mul!(u, H.path.γ1', H.x)
    elseif iszero(t)
        LA.mul!(u, H.path.Q_cos', H.x)
    else
        LA.mul!(u, γ!(H, t)', H.x)
    end
    return nothing
end

"""
    ambient_coordinates!(x, H::IntrinsicSubspaceHomotopy, u, t)

Convert intrinsic coordinates `u` on the subspace γ(t) into the ambient point
`x = γ(t) u + offset(t)`.
"""
function ambient_coordinates!(
        x::AbstractVector{ComplexF64}, H::IntrinsicSubspaceHomotopy,
        u::AbstractVector{ComplexF64}, t::ComplexF64,
    )::Nothing
    offset_at_t!(H, t)
    if isone(t)
        LA.mul!(x, H.path.γ1, u)
    elseif iszero(t)
        LA.mul!(x, H.path.Q_cos, u)
    else
        LA.mul!(x, γ!(H, t), u)
    end
    x .+= H.offset
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, H::IntrinsicSubspaceHomotopy,
        v::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    γ = γ!(H, t)
    LA.mul!(H.x, γ, v)
    offset_at_t!(H, t)
    H.x .+= H.offset
    evaluate!(u, H.system, H.x, _EMPTY_PARAMS)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, H::IntrinsicSubspaceHomotopy,
        v::FSVec{ComplexDF64}, t::ComplexF64,
    )::Nothing
    γ = γ!(H, t)
    offset_at_t!(H, t)
    n, k = size(γ)
    @inbounds for i in 1:n
        acc = ComplexDF64(H.offset[i])
        for j in 1:k
            acc += γ[i, j] * v[j]
        end
        H.x_high[i] = acc
    end
    evaluate!(u, H.system, H.x_high, _EMPTY_PARAMS)
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64},
        H::IntrinsicSubspaceHomotopy, v::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    γ = γ!(H, t)
    LA.mul!(H.x, γ, v)
    offset_at_t!(H, t)
    H.x .+= H.offset
    evaluate_and_jacobian!(u, H.J, H.system, H.x, _EMPTY_PARAMS)
    LA.mul!(U, H.J, γ)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, H::IntrinsicSubspaceHomotopy,
        v::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    # d/dt H(v,t) = d/dt F(A(t)v + b(t)) = J_F(A(t)v + b(t)) * (Ȧ(t)v + ḃ(t))
    γ = γ!(H, t)
    γ̇ = γ̇!(H, t)

    LA.mul!(H.x, γ, v)
    LA.mul!(H.ẋ, γ̇, v)

    offset_at_t!(H, t)
    H.x .+= H.offset
    H.ẋ .+= H.a_minus_b  # a_minus_b = ḃ = derivative of the offset in t

    evaluate_and_jacobian!(u, H.J, H.system, H.x, _EMPTY_PARAMS)
    LA.mul!(u, H.J, H.ẋ)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, H::IntrinsicSubspaceHomotopy,
        tv::TaylorVector{3, ComplexF64}, t::ComplexF64,
    )::Nothing
    γ, γ¹, γ², _ = taylor_γ!(H, t)
    x, x¹, x² = vectors(H.tx2)
    v, v¹, _ = vectors(tv)

    LA.mul!(x, γ, v)
    LA.mul!(x¹, γ¹, v)
    offset_at_t!(H, t)
    x .+= H.offset
    x¹ .+= H.a_minus_b
    LA.mul!(x¹, γ, v¹, true, true)
    LA.mul!(x², γ², v)
    LA.mul!(x², γ¹, v¹, true, true)
    # No γ v² term: the caller zeroes the highest-order row of tv.

    taylor!(u, Val(2), H.system, H.tx2, _EMPTY_PARAMS)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, H::IntrinsicSubspaceHomotopy,
        tv::TaylorVector{4, ComplexF64}, t::ComplexF64,
    )::Nothing
    γ, γ¹, γ², γ³ = taylor_γ!(H, t)
    x, x¹, x², x³ = vectors(H.tx3)
    v, v¹, v², _ = vectors(tv)

    LA.mul!(x, γ, v)
    LA.mul!(x¹, γ¹, v)
    offset_at_t!(H, t)
    x .+= H.offset
    x¹ .+= H.a_minus_b
    LA.mul!(x¹, γ, v¹, true, true)
    LA.mul!(x², γ², v)
    LA.mul!(x², γ¹, v¹, true, true)
    LA.mul!(x², γ, v², true, true)
    LA.mul!(x³, γ³, v)
    LA.mul!(x³, γ², v¹, true, true)
    LA.mul!(x³, γ¹, v², true, true)
    # No γ v³ term: the caller zeroes the highest-order row of tv.

    taylor!(u, Val(3), H.system, H.tx3, _EMPTY_PARAMS)
    return nothing
end

# Tracker-facing solution transfer is the identity: coordinate conversion is
# the monodromy solver's job via intrinsic_coordinates!/ambient_coordinates!.
function set_solution!(
        x::FSVec{ComplexF64}, ::IntrinsicSubspaceHomotopy,
        y::FSVec{ComplexF64}, ::ComplexF64,
    )::Nothing
    copyto!(x, y)
    return nothing
end

function get_solution!(
        out::FSVec{ComplexF64}, ::IntrinsicSubspaceHomotopy,
        x::FSVec{ComplexF64}, ::ComplexF64,
    )::Nothing
    copyto!(out, x)
    return nothing
end

############################
## ExtrinsicSubspaceHomotopy
############################

function evaluate!(
        u::FSVec{ComplexF64}, H::ExtrinsicSubspaceHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    γ = γ!(H, t)
    offset_at_t!(H, t)
    n, k = size(γ) # linear equation is: transpose(γ) x - offset

    evaluate!(u, H.system, x, _EMPTY_PARAMS)
    m = size(H.system)[1]
    # transpose(γ) x - offset, one column of γ at a time (column-major).
    @inbounds for i in 1:k
        acc = -H.offset[i]
        for j in 1:n
            acc = muladd(γ[j, i], x[j], acc)
        end
        u[m + i] = acc
    end
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, H::ExtrinsicSubspaceHomotopy,
        x::FSVec{ComplexDF64}, t::ComplexF64,
    )::Nothing
    γ = γ!(H, t)
    offset_at_t!(H, t)
    n, k = size(γ)

    evaluate!(u, H.system, x, _EMPTY_PARAMS)
    m = size(H.system)[1]
    # transpose(γ) x - offset accumulated in DF64, one γ column at a time.
    @inbounds for i in 1:k
        acc = -ComplexDF64(H.offset[i])
        for j in 1:n
            acc += γ[j, i] * x[j]
        end
        u[m + i] = ComplexF64(acc)
    end
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64},
        H::ExtrinsicSubspaceHomotopy, x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    γ = γ!(H, t)
    offset_at_t!(H, t)
    n, k = size(γ)

    # The interpreter writes only the first m rows of u and U (Cartesian
    # indexing); the appended linear rows are written afterwards.
    evaluate_and_jacobian!(u, U, H.system, x, _EMPTY_PARAMS)
    m = size(H.system)[1]
    # transpose(γ) x - offset, one column of γ at a time (column-major).
    @inbounds for i in 1:k
        acc = -H.offset[i]
        for j in 1:n
            acc = muladd(γ[j, i], x[j], acc)
        end
        u[m + i] = acc
    end
    @inbounds for j in 1:n, i in 1:k
        U[m + i, j] = γ[j, i]
    end
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, H::ExtrinsicSubspaceHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    # d/dt [F(x); L(t)] = [0; L̇(t)] with L(t) = γ(t)'x - offset(t)
    γ̇ = γ̇!(H, t)
    m = size(H.system)[1]
    k = size(γ̇, 2)

    LA.mul!(H.L, transpose(γ̇), x)
    offset_at_t!(H, t)
    H.L .-= H.a_minus_b  # -a_minus_b = derivative of -offset(t)

    @inbounds for i in 1:m
        u[i] = zero(ComplexF64)
    end
    @inbounds for i in 1:k
        u[m + i] = H.L[i]
    end
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, H::ExtrinsicSubspaceHomotopy,
        tv::TaylorVector{3, ComplexF64}, t::ComplexF64,
    )::Nothing
    γ, γ¹, γ², _ = taylor_γ!(H, t)
    v, v¹, _ = vectors(tv)

    # L² = γ¹' v¹ + γ²' v (no γ' v² term: highest-order row of tv is zeroed).
    LA.mul!(H.L, transpose(γ¹), v¹)
    LA.mul!(H.L, transpose(γ²), v, true, true)

    taylor!(u, Val(2), H.system, tv, _EMPTY_PARAMS)
    m = size(H.system)[1]
    k = size(γ, 2)
    @inbounds for i in 1:k
        u[m + i] = H.L[i]
    end
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, H::ExtrinsicSubspaceHomotopy,
        tv::TaylorVector{4, ComplexF64}, t::ComplexF64,
    )::Nothing
    γ, γ¹, γ², γ³ = taylor_γ!(H, t)
    v, v¹, v², _ = vectors(tv)

    # L³ = γ¹' v² + γ²' v¹ + γ³' v (no γ' v³ term: top row of tv is zeroed).
    LA.mul!(H.L, transpose(γ¹), v²)
    LA.mul!(H.L, transpose(γ²), v¹, true, true)
    LA.mul!(H.L, transpose(γ³), v, true, true)

    taylor!(u, Val(3), H.system, tv, _EMPTY_PARAMS)
    m = size(H.system)[1]
    k = size(γ, 2)
    @inbounds for i in 1:k
        u[m + i] = H.L[i]
    end
    return nothing
end

function set_solution!(
        x::FSVec{ComplexF64}, ::ExtrinsicSubspaceHomotopy,
        y::FSVec{ComplexF64}, ::ComplexF64,
    )::Nothing
    copyto!(x, y)
    return nothing
end

function get_solution!(
        out::FSVec{ComplexF64}, ::ExtrinsicSubspaceHomotopy,
        x::FSVec{ComplexF64}, ::ComplexF64,
    )::Nothing
    copyto!(out, x)
    return nothing
end
