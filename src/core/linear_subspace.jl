## LinearSubspace machinery: intrinsic/extrinsic descriptions, Grassmannian geodesics.
#
# Cold path: all fields are plain Matrix/Vector (they feed LAPACK svd/qr).

# References
const _LKK19 = """Lim, Lek-Heng, Ken Sze-Wai Wong, and Ke Ye. "Numerical algorithms on the affine Grassmannian." SIAM Journal on Matrix Analysis and Applications 40.2 (2019): 371-393"""

abstract type AbstractSubspace{T} end

"""
    Coordinates

A type used for encoding the used coordinates and for performing coordinate changes.

Currently supported coordinates are:
- [`Intrinsic`](@ref)
- [`Extrinsic`](@ref)
"""
struct Coordinates{T} end
Base.broadcastable(C::Coordinates) = Ref(C)

"""
    Intrinsic

Indicates the use of the intrinsic description of an (affine) linear subspace.
See also [`IntrinsicDescription`](@ref).
"""
const Intrinsic = Coordinates{:Intrinsic}()

"""
    Extrinsic

Indicates the use of the extrinsic description of an (affine) linear subspace.
See also [`ExtrinsicDescription`](@ref).
"""
const Extrinsic = Coordinates{:Extrinsic}()

"""
    ExtrinsicDescription(A, b)

Extrinsic description of an ``m``-dimensional (affine) linear subspace ``L`` in
``n``-dimensional space. That is ``L = \\{ x | A x = b \\}``. Note that
internally `A` and `b` are stored such that the rows of `A` are orthonormal.
"""
struct ExtrinsicDescription{T}
    A::Matrix{T}
    b::Vector{T}

    function ExtrinsicDescription(
            A::Matrix{T},
            b::Vector{T};
            orthonormal::Bool = false,
        ) where {T}
        return if orthonormal || size(A, 1) == 0
            new{T}(A, b)
        else
            svd = LA.svd(A)
            new{T}(svd.Vt, (inv.(svd.S) .* (svd.U' * b)))
        end
    end
end
(A::ExtrinsicDescription)(x::AbstractVector) = A.A * x - A.b

# Identity when the eltype already matches: convert may return its argument
# (standard for convert(::Type{T}, ::T)), avoiding a needless orthonormal rebuild.
Base.convert(::Type{ExtrinsicDescription{T}}, A::ExtrinsicDescription{T}) where {T} = A
function Base.convert(::Type{ExtrinsicDescription{T}}, A::ExtrinsicDescription) where {T}
    return ExtrinsicDescription(
        convert(Matrix{T}, A.A),
        convert(Vector{T}, A.b);
        orthonormal = true,
    )
end

"""
    dim(A::ExtrinsicDescription)

Dimension of the (affine) linear subspace `A`.
"""
dim(A::ExtrinsicDescription) = size(A.A, 2) - size(A.A, 1)

"""
    codim(A::ExtrinsicDescription)

Codimension of the (affine) linear subspace `A`.
"""
codim(A::ExtrinsicDescription) = size(A.A, 1)

Base.:(==)(A::ExtrinsicDescription, B::ExtrinsicDescription) = A.A == B.A && A.b == B.b
function Base.copy!(A::ExtrinsicDescription, B::ExtrinsicDescription)
    copy!(A.A, B.A)
    copy!(A.b, B.b)
    return A
end
Base.copy(A::ExtrinsicDescription) =
    ExtrinsicDescription(copy(A.A), copy(A.b); orthonormal = true)

function Base.show(io::IO, A::ExtrinsicDescription{T}) where {T}
    println(io, "ExtrinsicDescription{$T}:")
    println(io, "A:")
    show(io, A.A)
    println(io, "\nb:")
    show(io, A.b)
    return
end

Base.broadcastable(A::ExtrinsicDescription) = Ref(A)

"""
    IntrinsicDescription(A, b)

Intrinsic description of an ``m``-dimensional (affine) linear subspace ``L`` in
``n``-dimensional space. That is ``L = \\{ A u + b \\}``. Here, ``A`` and ``b``
are in orthogonal coordinates: the columns of ``A`` are orthonormal and
``A' b = 0``.
"""
struct IntrinsicDescription{T}
    # orthogonal coordinates
    A::Matrix{T}
    b::Vector{T}
    # stiefel coordinates for A
    X::Matrix{T}
    # stiefel coordinates for [A b]
    Y::Matrix{T}
end
function IntrinsicDescription(A::Matrix{T}, b::Vector{T}) where {T}
    X = stiefel_coordinates_intrinsic(A)::Matrix{T}
    Y = stiefel_coordinates_intrinsic(A, b)::Matrix{T}
    return IntrinsicDescription{T}(A, b, X, Y)
end

# Identity when the eltype already matches: skips the Stiefel-coordinate SVD
# rebuild that the general method performs.
Base.convert(::Type{IntrinsicDescription{T}}, A::IntrinsicDescription{T}) where {T} = A
function Base.convert(::Type{IntrinsicDescription{T}}, A::IntrinsicDescription) where {T}
    return IntrinsicDescription(convert(Matrix{T}, A.A), convert(Vector{T}, A.b))
end

(A::IntrinsicDescription)(u::AbstractVector, ::Coordinates{:Intrinsic}) = A.A * u + A.b

function stiefel_coordinates_intrinsic(A::Matrix{T})::Matrix{T} where {T}
    SVD = LA.svd(A)
    return SVD.U
end
function stiefel_coordinates_intrinsic!(X, A::AbstractMatrix)
    X .= A
    SVD = LA.svd!(X)
    X .= SVD.U
    return X
end
function stiefel_coordinates_intrinsic(A::Matrix{T}, b::Vector{T})::Matrix{T} where {T}
    n, k = size(A)
    Y = zeros(T, n + 1, k + 1)
    stiefel_coordinates_intrinsic!(Y, A, b)
    return Y
end
function stiefel_coordinates_intrinsic!(Y, A::AbstractMatrix, b::AbstractVector)
    γ = sqrt(1 + sum(abs2, b))
    n, k = size(A)
    Y[1:n, 1:k] .= A
    Y[1:n, k + 1] .= b ./ γ
    Y[n + 1, k + 1] = 1 / γ
    SVD = LA.svd!(Y)
    Y .= SVD.U
    return Y
end

"""
    dim(A::IntrinsicDescription)

Dimension of the (affine) linear subspace `A`.
"""
dim(I::IntrinsicDescription) = size(I.A, 2)

"""
    codim(A::IntrinsicDescription)

Codimension of the (affine) linear subspace `A`.
"""
codim(I::IntrinsicDescription) = size(I.A, 1) - size(I.A, 2)

function Base.:(==)(A::IntrinsicDescription, B::IntrinsicDescription)
    return A.A == B.A && A.b == B.b && A.Y == B.Y
end

function Base.show(io::IO, A::IntrinsicDescription{T}) where {T}
    println(io, "IntrinsicDescription{$T}:")
    println(io, "A:")
    show(io, A.A)
    println(io, "\nb:")
    show(io, A.b)
    return
end

function Base.copy!(A::IntrinsicDescription, B::IntrinsicDescription)
    copy!(A.A, B.A)
    copy!(A.b, B.b)
    copy!(A.X, B.X)
    copy!(A.Y, B.Y)
    return A
end
Base.copy(A::IntrinsicDescription) =
    IntrinsicDescription(copy(A.A), copy(A.b), copy(A.X), copy(A.Y))

Base.broadcastable(A::IntrinsicDescription) = Ref(A)

function IntrinsicDescription(E::ExtrinsicDescription)
    svd = LA.svd(E.A; full = true)
    m, n = size(E.A)
    A = Matrix((@view svd.Vt[(m + 1):end, :])')
    b = if iszero(E.b)
        zeros(eltype(E.b), size(A, 1))
    else
        svd \ E.b
    end
    return IntrinsicDescription(A, b)
end

function ExtrinsicDescription(I::IntrinsicDescription)
    svd = LA.svd(I.A; full = true)
    m, n = size(I.A)
    A = Matrix((@view svd.U[:, (n + 1):end])')
    b = if iszero(I.b)
        zeros(eltype(A), size(A, 1))
    else
        A * I.b
    end
    return ExtrinsicDescription(A, b; orthonormal = true)
end

"""
    LinearSubspace(A, b)

An ``m``-dimensional (affine) linear subspace ``L`` in ``n``-dimensional space
given by the extrinsic description ``L = \\{ x | A x = b \\}``.

A `LinearSubspace` always holds both its [`extrinsic`](@ref) description, see
[`ExtrinsicDescription`](@ref), and its [`intrinsic`](@ref) description, see
[`IntrinsicDescription`](@ref). It can be evaluated with either
[`Intrinsic`](@ref) or [`Extrinsic`](@ref) coordinates; to change between them
use [`coord_change`](@ref).
"""
struct LinearSubspace{T} <: AbstractSubspace{T}
    extrinsic::ExtrinsicDescription{T}
    intrinsic::IntrinsicDescription{T}
end

LinearSubspace(I::IntrinsicDescription) = LinearSubspace(ExtrinsicDescription(I), I)
LinearSubspace(E::ExtrinsicDescription) = LinearSubspace(E, IntrinsicDescription(E))

function LinearSubspace(
        A::AbstractMatrix{T},
        b::AbstractVector{T} = zeros(eltype(A), size(A, 1)),
    ) where {T}
    size(A, 1) == length(b) || throw(ArgumentError("Size of A and b not compatible."))
    0 <= size(A, 1) <= size(A, 2) || throw(
        ArgumentError(
            "Affine subspace has to be given in extrinsic coordinates, i.e., by A x = b.",
        ),
    )

    return LinearSubspace(ExtrinsicDescription(Matrix(float.(A)), Vector(float.(b))))
end

# Identity when the eltype already matches. This is the hot case for monodromy:
# set_subspaces! converts the loop's LinearSubspace{ComplexF64} on every
# retarget, and without this it would rebuild both Stiefel frames via SVD each
# time (~40% of set_subspaces!). convert(::Type{T}, ::T) returning its argument
# is standard; the homotopy only reads the stored subspaces.
Base.convert(::Type{LinearSubspace{T}}, A::LinearSubspace{T}) where {T} = A
function Base.convert(::Type{LinearSubspace{T}}, A::LinearSubspace) where {T}
    return LinearSubspace(
        convert(ExtrinsicDescription{T}, A.extrinsic),
        convert(IntrinsicDescription{T}, A.intrinsic),
    )
end

Base.broadcastable(A::LinearSubspace) = Ref(A)

"""
    dim(A::LinearSubspace)

Dimension of the (affine) linear subspace `A`.
"""
dim(A::LinearSubspace) = dim(A.intrinsic)

"""
    codim(A::LinearSubspace)

Codimension of the (affine) linear subspace `A`.
"""
codim(A::LinearSubspace) = codim(A.intrinsic)

_default_intrinsic(A::LinearSubspace)::Bool = dim(A) <= codim(A)

"""
    ambient_dim(A::LinearSubspace)

Dimension of the ambient space of the (affine) linear subspace `A`.
"""
ambient_dim(A::LinearSubspace) = dim(A) + codim(A)

"""
    is_linear(L::LinearSubspace)

Returns `true` if the space is a proper linear subspace, i.e., described by
``L = \\{ x | Ax = 0 \\}``.
"""
is_linear(A::LinearSubspace) = iszero(extrinsic(A).b)

function Base.show(io::IO, A::LinearSubspace{T}) where {T}
    if is_linear(A)
        println(io, "$(dim(A))-dim. linear subspace {x | Ax=0} with eltype $T:")
        println(io, "A:")
        show(io, A.extrinsic.A)
    else
        println(io, "$(dim(A))-dim. affine linear subspace {x | Ax=b} with eltype $T:")
        println(io, "A:")
        show(io, A.extrinsic.A)
        println(io, "\nb:")
        show(io, A.extrinsic.b)
    end
    return
end

"""
    intrinsic(A::LinearSubspace)

Obtain the intrinsic description of `A`, see also [`IntrinsicDescription`](@ref).
"""
intrinsic(A::LinearSubspace) = A.intrinsic

"""
    extrinsic(A::LinearSubspace)

Obtain the extrinsic description of `A`, see also [`ExtrinsicDescription`](@ref).
"""
extrinsic(A::LinearSubspace) = A.extrinsic

function Base.copy!(A::LinearSubspace, B::LinearSubspace)
    copy!(A.intrinsic, B.intrinsic)
    copy!(A.extrinsic, B.extrinsic)
    return A
end
Base.copy(A::LinearSubspace) = LinearSubspace(copy(A.extrinsic), copy(A.intrinsic))

function Base.:(==)(A::LinearSubspace, B::LinearSubspace)
    return intrinsic(A) == intrinsic(B) && extrinsic(A) == extrinsic(B)
end
Base.isequal(A::LinearSubspace, B::LinearSubspace) = A === B

function (A::LinearSubspace)(x::AbstractVector, ::Coordinates{:Intrinsic})
    return intrinsic(A)(x, Intrinsic)
end
function (A::LinearSubspace)(x::AbstractVector, ::Coordinates{:Extrinsic} = Extrinsic)
    return extrinsic(A)(x)
end

"""
    rand_subspace([rng], n::Integer; dim | codim, affine = true, real = false)

Generate a random [`LinearSubspace`](@ref) with given dimension `dim` or
codimension `codim` (one of them has to be provided) in ambient space of
dimension `n`. If `real` is `true`, then the extrinsic description is real.
If `affine`, then an affine linear subspace is generated. The matrix `A` of the
extrinsic description is drawn independently from a normal distribution using
`randn`.

    rand_subspace([rng], x::AbstractVector; dim | codim, affine = true)

Generate a random [`LinearSubspace`](@ref) with given dimension `dim` or
codimension `codim` in ambient space of dimension `length(x)` going through the
given point `x`.

As with `rand` and `randn`, pass a random number generator `rng` as the first
argument to draw from it instead of the global one.
"""
function rand_subspace(
        rng::Random.AbstractRNG, n::Integer;
        dim::Union{Nothing, Integer} = nothing,
        codim::Union{Nothing, Integer} = nothing,
        real::Bool = false,
        affine::Bool = true,
    )
    dim !== nothing ||
        codim !== nothing ||
        throw(ArgumentError("Neither `dim` nor `codim` specified."))

    if dim !== nothing
        0 < dim < n || throw(ArgumentError("`dim` has to be between 0 and `n`."))
        k = dim
    else
        0 < codim < n || throw(ArgumentError("`codim` has to be between 0 and `n`."))
        k = n - codim
    end
    T = real ? Float64 : ComplexF64
    A = randn(rng, T, n - k, n)
    return if affine
        LinearSubspace(A, randn(rng, T, n - k))
    else
        LinearSubspace(A)
    end
end
rand_subspace(
    n::Integer;
    dim::Union{Nothing, Integer} = nothing,
    codim::Union{Nothing, Integer} = nothing,
    real::Bool = false,
    affine::Bool = true,
) = rand_subspace(
    Random.default_rng(), n; dim = dim, codim = codim, real = real, affine = affine,
)
rand_subspace(
    x::AbstractVector{<:MP.AbstractVariable};
    dim::Union{Nothing, Integer} = nothing,
    codim::Union{Nothing, Integer} = nothing,
    real::Bool = false,
    affine::Bool = true,
) = rand_subspace(
    length(x); dim = dim, codim = codim, real = real, affine = affine,
)
rand_subspace(
    rng::Random.AbstractRNG,
    x::AbstractVector{<:MP.AbstractVariable};
    dim::Union{Nothing, Integer} = nothing,
    codim::Union{Nothing, Integer} = nothing,
    real::Bool = false,
    affine::Bool = true,
) = rand_subspace(
    rng, length(x); dim = dim, codim = codim, real = real, affine = affine,
)
function rand_subspace(
        rng::Random.AbstractRNG, x::AbstractVector;
        dim::Union{Nothing, Integer} = nothing,
        codim::Union{Nothing, Integer} = nothing,
        affine::Bool = true,
    )
    n = length(x)
    dim !== nothing ||
        codim !== nothing ||
        throw(ArgumentError("Neither `dim` nor `codim` specified."))

    if dim !== nothing
        0 < dim < n || throw(ArgumentError("`dim` has to be between 0 and `n`."))
        k = dim
    else
        0 < codim < n || throw(ArgumentError("`codim` has to be between 0 and `n`."))
        k = n - codim
    end

    return if affine
        A = randn(rng, eltype(x), n - k, n)
        b = A * x
        LinearSubspace(A, b)
    else
        N = LA.nullspace(Matrix(x'))'
        A = randn(rng, eltype(N), n - k, size(N, 1)) * N
        LinearSubspace(A)
    end
end
rand_subspace(
    x::AbstractVector;
    dim::Union{Nothing, Integer} = nothing,
    codim::Union{Nothing, Integer} = nothing,
    affine::Bool = true,
) = rand_subspace(
    Random.default_rng(), x; dim = dim, codim = codim, affine = affine,
)

# Coordinate changes
"""
    coord_change(A::LinearSubspace, C₁::Coordinates, C₂::Coordinates, p)

Given an (affine) linear subspace `A` and a point `p` in coordinates `C₁`
compute the point `x` describing `p` in coordinates `C₂`.
"""
coord_change(A::LinearSubspace, ::C, ::C, x) where {C <: Coordinates} = x
coord_change(A::LinearSubspace, ::Coordinates{:Intrinsic}, ::Coordinates{:Extrinsic}, u) =
    A(u, Intrinsic)
coord_change(A::LinearSubspace, ::Coordinates{:Extrinsic}, ::Coordinates{:Intrinsic}, x) =
    A.intrinsic.A' * (x - A.intrinsic.b)

"""
    geodesic_distance(V::LinearSubspace, W::LinearSubspace)

Compute the geodesic distance between `V = {x | Ax = a}` and `W = {x | Bx = b}`
as `sqrt(d^2 + ||a-b||^2)`, where `d` is the distance from the column span of
`A` to the column span of `B` in the Grassmannian, following [^LKK19].

[^LKK19]: $_LKK19.
"""
geodesic_distance(A::LinearSubspace, B::LinearSubspace) =
    geodesic_distance(A.intrinsic, B.intrinsic)
function geodesic_distance(A::IntrinsicDescription, B::IntrinsicDescription)
    return sqrt(sum(σᵢ -> acos(min(σᵢ, 1.0))^2, LA.svdvals(A.Y' * B.Y)))
end

# geodesic
"""
    grassmannian_svd(A::LinearSubspace, B::LinearSubspace)

Computes the factors ``Q``, ``Θ`` and ``U`` from Corollary 4.3 in [^LKK19].
These values are necessary to construct the geodesic path from `A` to `B`
(parameter 0 at `A`, parameter 1 at `B`).

[^LKK19]: $_LKK19
"""
grassmannian_svd(A::LinearSubspace, B::LinearSubspace) =
    grassmannian_svd(extrinsic(A), extrinsic(B))
grassmannian_svd(A::E1, B::E2) where {E1 <: ExtrinsicDescription, E2 <: ExtrinsicDescription} =
    grassmannian_svd(transpose(A.A), transpose(B.A))
function grassmannian_svd(
        A::I1,
        B::I2;
        embedded_projective::Bool = false,
    ) where {I1 <: IntrinsicDescription, I2 <: IntrinsicDescription}
    return if !embedded_projective
        grassmannian_svd(A.X, B.X)
    else
        grassmannian_svd(A.Y, B.Y)
    end
end

function grassmannian_svd(A::AbstractMatrix, B::AbstractMatrix)
    # Here A and B are assumed to be Stiefel matrices representing linear spaces.
    n, k = size(A)
    U, Σ, V = LA.svd!(A' * B)
    # M = (LA.I - A * A') * B
    # Have to compute an SVD of M s.t. M = Q sinΘ V'
    # Equivalently M * V = Q sin(Θ)
    # We can achieve this by using a *pivoted* QR
    # since then R will be a diagonal matrix s.t. the absolute value of R_ii is θ_{k-i}
    MV = (LA.I - A * A') * B * V

    # Θ = acos.(min.(Σ, 1.0))
    # We have acos(1-eps()) = 2.1073424255447017e-8
    # So this is numerically super unstable if the singular value is off by only eps()
    # We try to correct this fact by treating σ >= 1 - 2eps() as 1.0
    Θ = map(σ -> σ + 2 * eps() > 1.0 ? 0.0 : acos(σ), Σ)

    F = LA.qr!(MV, LA.ColumnNorm())
    Q, R = F.Q, F.R
    # correct signs and ordering of Q
    Q′ = Q[:, k:-1:1]
    for j in 1:k
        # Look if we need to flip signs
        real(R[k - j + 1, k - j + 1]) < 0 || continue
        for i in 1:n
            Q′[i, j] = -Q′[i, j]
        end
    end

    return Q′, Θ, U
end

"""
    geodesic(V::LinearSubspace, W::LinearSubspace)

Returns the geodesic ``γ(t)`` connecting `V = {x | Ax = a}` and
`W = {x | Bx = b}`. `A` and `B` are interpolated in the Grassmannian using
Stiefel coordinates (Corollary 4.3 in [^LKK19]); `a` and `b` are interpolated
linearly.

Convention: `γ(1) == V` and `γ(0) == W` for BOTH the direction and the
offset, matching the homotopy convention where tracking runs from `t = 1`
(start) to `t = 0` (target).

[^LKK19]: $_LKK19
"""
geodesic(A::LinearSubspace, B::LinearSubspace) = geodesic(A.intrinsic, B.intrinsic)
function geodesic(A::IntrinsicDescription, B::IntrinsicDescription)
    # Direction from B (t = 0) to A (t = 1), offset t*a + (1-t)*b.
    Q, Θ, U = grassmannian_svd(B, A)
    return t -> LinearSubspace(
        IntrinsicDescription(
            B.X * U * LA.diagm(cos.(t .* Θ)) * U' + Q * LA.diagm(sin.(t .* Θ)) * U',
            t .* A.b + (1 - t) .* B.b,
        ),
    )
end

"""
    translate(L::LinearSubspace, δb, ::Coordinates = Extrinsic)

Translate the (affine) linear subspace `L` by `δb`.
"""
function translate(L::LinearSubspace, δb, ::Coordinates{:Extrinsic} = Extrinsic)
    return translate!(copy(L), δb, Extrinsic)
end
function translate!(L::LinearSubspace, δb, ::Coordinates{:Extrinsic} = Extrinsic)
    ext = extrinsic(L)
    ext.b .+= δb
    int = intrinsic(L)
    LA.mul!(int.b, ext.A', δb, true, true)
    stiefel_coordinates_intrinsic!(int.Y, int.A, int.b)
    return L
end

"""
    Base.intersect(L₁::LinearSubspace, L₂::LinearSubspace)

Intersect the two given linear subspaces. Throws an `ErrorException` if the sum
of the codimensions is larger than the ambient dimension.
"""
function Base.intersect(L₁::LinearSubspace, L₂::LinearSubspace)
    ambient_dim(L₁) == ambient_dim(L₂) || error("Ambient dimensions don't match.")
    codim(L₁) + codim(L₂) <= ambient_dim(L₁) ||
        error("Sum of codimensions larger than ambient dimension.")
    ext₁ = extrinsic(L₁)
    ext₂ = extrinsic(L₂)
    return LinearSubspace([ext₁.A; ext₂.A], [ext₁.b; ext₂.b])
end
