# Interval arithmetic for solution certification following Mayer, "Interval Analysis".
#
# Provides `Interval{T}` (a directed-rounding real interval) and `IComplex{T}`
# (a rectangular complex interval). These back the Float64 Krawczyk operator in
# `certify`. The Arb/Acb bridge lives with the extended-precision interpreter.
#
# The arithmetic follows IntervalArithmetic.jl
# (https://github.com/JuliaIntervals/IntervalArithmetic.jl), MIT "Expat" licensed.

# ---------------------------------------------------------------------------
# Interval{T}
# ---------------------------------------------------------------------------

"""
    Interval{T<:Real}

A real interval `[lo, hi]` with directed-rounding arithmetic.
"""
struct Interval{T <: Real}
    lo::T
    hi::T
end

Interval{T}(a::Real) where {T <: Real} = Interval(convert(T, a))
Interval{T}(a::Rational) where {T <: Real} =
    Interval(convert(T, numerator(a))) / Interval(convert(T, denominator(a)))
Interval{T}(a::Interval{S}) where {T <: Real, S <: Real} =
    Interval(convert(T, a.lo), convert(T, a.hi))
Interval(a::T) where {T <: Real} = Interval(a, a)
Interval(a::Rational) = Interval{Float64}(a)
Interval(a::T, b::S) where {T <: Real, S <: Real} = Interval(promote(a, b)...)
Interval(a::T, b::T) where {T <: Integer} = Interval(float(a), float(b))

Base.convert(::Type{Interval{T}}, a::Interval{S}) where {T, S} =
    Interval(convert(T, a.lo), convert(T, a.hi))
Base.convert(::Type{Interval{T}}, a::Real) where {T} = Interval{T}(a)

"""
    interval(a::Real, b::Real)

Construct `Interval(a, b)` after checking that `a ≤ b` and both are finite.
"""
function interval(a::Real, b::Real)
    is_valid_interval(a, b) || invalid_interval_error(a, b)
    return Interval(a, b)
end
is_valid_interval(a::Real, b::Real) = isfinite(a) && isfinite(b) && a ≤ b
@noinline function invalid_interval_error(a, b)
    throw(
        ArgumentError(
            "`[$a, $b]` is not a valid interval. Need `a ≤ b` to construct `interval(a, b)`.",
        ),
    )
end

Base.hash(x::Interval, h::UInt) = hash(x.hi, hash(x.lo, h))
Base.:(==)(a::Interval, b::Interval) = a.lo == b.lo && a.hi == b.hi
Base.eltype(::Interval{T}) where {T} = T

Base.promote_rule(::Type{Interval{T}}, ::Type{Interval{S}}) where {T <: Real, S <: Real} =
    Interval{promote_type(T, S)}
Base.promote_rule(::Type{Interval{T}}, ::Type{S}) where {T <: Real, S <: Real} =
    Interval{promote_type(T, S)}

round_up(a::AbstractFloat) = nextfloat(a)
round_down(a::AbstractFloat) = prevfloat(a)
macro round(a, b)
    return :(Interval(round_down($(esc(a))), round_up($(esc(b)))))
end

mid(a::Interval) = (a.lo + a.hi) / 2
diam(a::Interval) = round_up(a.hi - a.lo)
function rad(a::Interval)
    m = mid(a)
    return round_up(max(m - a.lo, a.hi - m))
end
mag(a::Interval) = max(abs(a.lo), abs(a.hi))
function mig(a::Interval{T}) where {T}
    zero(T) ∈ a && return zero(T)
    return min(abs(a.lo), abs(a.hi))
end
hull(a::Interval, b::Interval) = Interval(min(a.lo, b.lo), max(a.hi, b.hi))

Base.issubset(a::Interval, b::Interval) = (a.lo ≥ b.lo) && (a.hi ≤ b.hi)
isinterior(a::Interval, b::Interval) = (a.lo > b.lo) && (a.hi < b.hi)
Base.isdisjoint(a::Interval, b::Interval) = (b.hi < a.lo) || (a.hi < b.lo)
function Base.intersect(a::Interval, b::Interval)
    c = Interval(max(a.lo, b.lo), min(a.hi, b.hi))
    c.lo ≤ c.hi || return Interval(convert(eltype(c), NaN))
    return c
end
Base.isempty(a::Interval) = isnan(a.lo) || isnan(a.hi)
Base.in(x::Number, a::Interval) = a.lo ≤ x ≤ a.hi

function Base.show(io::IO, a::Interval)
    print(io, mid(a), " ± ")
    return Printf.@printf(io, "%.5g", rad(a))
end
Base.print(io::IO, a::Interval) = print(io, mid(a), " ± ", rad(a))

# arithmetic

Base.zero(::Interval{T}) where {T} = Interval(zero(T))
Base.zero(::Type{Interval{T}}) where {T} = Interval(zero(T))
Base.one(::Interval{T}) where {T} = Interval(one(T))
Base.one(::Type{Interval{T}}) where {T} = Interval(one(T))

## Addition and subtraction
Base.:+(a::Interval) = a
Base.:+(a::Interval, b::Real) = @round(a.lo + b, a.hi + b)
Base.:+(b::Real, a::Interval) = a + b
Base.:+(a::Interval, b::Interval) = @round(a.lo + b.lo, a.hi + b.hi)

Base.:-(a::Interval) = Interval(-a.hi, -a.lo)
Base.:-(a::Interval, b::Real) = @round(a.lo - b, a.hi - b)
Base.:-(b::Real, a::Interval) = @round(b - a.hi, b - a.lo)
Base.:-(a::Interval, b::Interval) = @round(a.lo - b.hi, a.hi - b.lo)

## Multiplication
function Base.:*(x::Real, a::Interval)
    # Return the zero *interval* [0,0], not the scalar zero: `0 * a` is an
    # `Interval`, so the scalar path would break type stability.
    (iszero(a) || iszero(x)) && return Interval(zero(a.lo * x))
    if x ≥ 0.0
        return @round(a.lo * x, a.hi * x)
    else
        return @round(a.hi * x, a.lo * x)
    end
end
Base.:*(a::Interval, x::Real) = x * a
function Base.:*(a::Interval, b::Interval)
    if b.lo >= zero(b.lo)
        a.lo >= zero(a.lo) && return @round(a.lo * b.lo, a.hi * b.hi)
        a.hi <= zero(a.hi) && return @round(a.lo * b.hi, a.hi * b.lo)
        return @round(a.lo * b.hi, a.hi * b.hi)   # zero(T) ∈ a
    elseif b.hi <= zero(b.hi)
        a.lo >= zero(a.lo) && return @round(a.hi * b.lo, a.lo * b.hi)
        a.hi <= zero(a.hi) && return @round(a.hi * b.hi, a.lo * b.lo)
        return @round(a.hi * b.lo, a.lo * b.lo)   # zero(T) ∈ a
    else
        a.lo > zero(a.lo) && return @round(a.hi * b.lo, a.hi * b.hi)
        a.hi < zero(a.hi) && return @round(a.lo * b.hi, a.lo * b.lo)
        return @round(min(a.lo * b.hi, a.hi * b.lo), max(a.lo * b.lo, a.hi * b.hi))
    end
end

function Base.muladd(a::Interval, b::Interval, c::Interval)
    lo = let
        lo1 = muladd(a.lo, b.lo, c.lo)
        lo2 = muladd(a.lo, b.hi, c.lo)
        lo3 = muladd(a.hi, b.lo, c.lo)
        lo4 = muladd(a.hi, b.hi, c.lo)
        round_down(Base.FastMath.min_fast(lo1, lo2, lo3, lo4))
    end
    hi = let
        hi1 = muladd(a.lo, b.lo, c.hi)
        hi2 = muladd(a.lo, b.hi, c.hi)
        hi3 = muladd(a.hi, b.lo, c.hi)
        hi4 = muladd(a.hi, b.hi, c.hi)
        round_up(Base.FastMath.max_fast(hi1, hi2, hi3, hi4))
    end
    return Interval(lo, hi)
end
function Base.muladd(a::Interval{T}, b::T, c::Interval) where {T}
    lo = let
        lo1 = muladd(a.lo, b, c.lo)
        lo2 = muladd(a.hi, b, c.lo)
        round_down(Base.FastMath.min_fast(lo1, lo2))
    end
    hi = let
        hi1 = muladd(a.lo, b, c.hi)
        hi2 = muladd(a.hi, b, c.hi)
        round_up(Base.FastMath.max_fast(hi1, hi2))
    end
    return Interval(lo, hi)
end
function Base.muladd(a::T, b::Interval{T}, c::Interval) where {T}
    lo = let
        lo1 = muladd(a, b.lo, c.lo)
        lo2 = muladd(a, b.hi, c.lo)
        round_down(Base.FastMath.min_fast(lo1, lo2))
    end
    hi = let
        hi1 = muladd(a, b.lo, c.hi)
        hi2 = muladd(a, b.hi, c.hi)
        round_up(Base.FastMath.max_fast(hi1, hi2))
    end
    return Interval(lo, hi)
end

function Base.inv(a::Interval{T}) where {T <: Real}
    zero(T) ∈ a && return Interval(convert(T, NaN))
    return @round(inv(a.hi), inv(a.lo))
end

function Base.:/(a::Interval{T}, x::T) where {T <: Real}
    iszero(a) && return zero(Interval{T})
    if x ≥ 0.0
        return @round(a.lo / x, a.hi / x)
    else
        return @round(a.hi / x, a.lo / x)
    end
end
function Base.:/(a::Interval{T}, b::Interval{T}) where {T <: Real}
    S = typeof(a.lo / b.lo)
    # `0 ∈ b` (including `b == [0,0]`) makes `a / b` indeterminate, so the
    # zero-numerator fast path applies only when `b` is bounded away from zero.
    if b.lo > zero(T) # b strictly positive
        iszero(a) && return zero(Interval{S})
        a.lo >= zero(T) && return @round(a.lo / b.hi, a.hi / b.lo)
        a.hi <= zero(T) && return @round(a.lo / b.lo, a.hi / b.hi)
        return @round(a.lo / b.lo, a.hi / b.lo)  # zero(T) ∈ a
    elseif b.hi < zero(T) # b strictly negative
        iszero(a) && return zero(Interval{S})
        a.lo >= zero(T) && return @round(a.hi / b.hi, a.lo / b.lo)
        a.hi <= zero(T) && return @round(a.hi / b.lo, a.lo / b.hi)
        return @round(a.hi / b.hi, a.lo / b.hi)  # zero(T) ∈ a
    else   # b contains zero → indeterminate (this covers b == [0,0])
        return Interval(convert(S, NaN))
    end
end

function sqr(a::Interval{T}) where {T <: Real}
    if a.lo ≥ zero(T)
        return @round(a.lo^2, a.hi^2)
    elseif a.hi ≤ zero(T)
        return @round(a.hi^2, a.lo^2)
    end
    return @round(mig(a)^2, mag(a)^2)
end
Base.literal_pow(::typeof(^), a::Interval, ::Val{2}) = sqr(a)

function Base.:^(x::Interval, n::Integer)  # fast integer power
    n == 0 && return one(x)   # x^0 ≡ 1 for every x (including 0 ∈ x)
    n < 0 && return inv(x^(-n))
    isempty(x) && return x
    if iseven(n) && 0 ∈ x
        return hull(
            zero(x),
            hull(
                Base.power_by_squaring(Interval(mig(x)), n),
                Base.power_by_squaring(Interval(mag(x)), n),
            ),
        )
    else
        return hull(
            Base.power_by_squaring(Interval(x.lo), n),
            Base.power_by_squaring(Interval(x.hi), n),
        )
    end
end

# ---------------------------------------------------------------------------
# IComplex{T}
# ---------------------------------------------------------------------------

"""
    IComplex{T}

A rectangular complex interval `re + im*i` with `re, im :: Interval{T}`.
"""
struct IComplex{T} <: Number
    re::Interval{T}
    im::Interval{T}
end
IComplex(x::Real, y::Interval) = IComplex(promote(x, y)...)
IComplex(x::Interval, y::Real) = IComplex(promote(x, y)...)
function IComplex(x::Real, y::Real)
    ix, iy = promote(x, y)
    return IComplex(Interval(ix), Interval(iy))
end
IComplex(re::Interval{T}, im::Interval{S}) where {T <: Real, S <: Real} =
    IComplex{promote_type(T, S)}(promote(re, im)...)
IComplex(x::Union{Interval, Real}) = IComplex(x, zero(x))
IComplex(z::Complex) = IComplex(real(z), imag(z))
IComplex(z::IComplex) = z
Base.complex(x::Interval, y::Real) = IComplex(x, y)
Base.complex(x::Real, y::Interval) = IComplex(x, y)
Base.complex(x::Interval, y::Interval) = IComplex(x, y)
Base.complex(x::IComplex) = x

const IComplexF64 = IComplex{Float64}
IComplex{T}(x::Complex{S}) where {T <: Real, S <: Real} =
    IComplex(Interval{T}(real(x)), Interval{T}(imag(x)))
IComplex{T}(x::IComplex{S}) where {T <: Real, S <: Real} =
    IComplex(Interval{T}(real(x)), Interval{T}(imag(x)))
# Constrained to `Real`/`Interval` (rather than an untyped `x`) so it does not
# clash with Base's `TwicePrecision`/`AbstractChar` `Number` constructors.
IComplex{T}(x::Union{Real, Interval}) where {T <: Real} = IComplex(Interval{T}(x))

Base.zero(::IComplex{T}) where {T} = IComplex(zero(Interval{T}))
Base.zero(::Type{IComplex{T}}) where {T} = IComplex(zero(Interval{T}))
Base.one(::IComplex{T}) where {T} = IComplex(one(Interval{T}))
Base.one(::Type{IComplex{T}}) where {T} = IComplex(one(Interval{T}))

function Base.show(io::IO, c::IComplex)
    print(io, "(")
    show(io, c.re)
    print(io, ") + (")
    show(io, c.im)
    return print(io, ")im")
end
Base.print(io::IO, c::IComplex) = print(io, "(", c.re, ") + (", c.im, ")im")
Base.broadcastable(z::IComplex) = z
Base.promote_rule(::Type{IComplex{T}}, ::Type{S}) where {T, S <: Real} =
    IComplex{promote_type(T, S)}
Base.promote_rule(::Type{IComplex{T}}, ::Type{Complex{S}}) where {T, S} =
    IComplex{promote_type(T, S)}
Base.promote_rule(::Type{IComplex{T}}, ::Type{Interval{S}}) where {T, S <: Real} =
    IComplex{promote_type(T, S)}
Base.promote_rule(::Type{IComplex{T}}, ::Type{IComplex{S}}) where {T, S} =
    IComplex{promote_type(T, S)}
Base.convert(::Type{IComplex{T}}, x::IComplex) where {T} = IComplex{T}(x)
Base.convert(::Type{IComplex{T}}, x::Interval) where {T} = IComplex{T}(x)

Base.widen(::Type{IComplex{T}}) where {T} = IComplex{widen(T)}

Base.real(z::IComplex) = z.re
Base.real(::Type{IComplex{T}}) where {T} = Interval{T}
Base.imag(z::IComplex) = z.im
Base.imag(::Type{IComplex{T}}) where {T} = Interval{T}
Base.reim(z::IComplex) = (real(z), imag(z))
Base.conj(z::IComplex) = IComplex(real(z), -imag(z))

Base.:+(z::IComplex) = z
Base.:+(x::Union{Interval, Real}, z::IComplex) = IComplex(x + real(z), imag(z))
Base.:+(z::IComplex, x::Union{Interval, Real}) = IComplex(x + real(z), imag(z))
Base.:+(z::IComplex, w::IComplex) = IComplex(real(z) + real(w), imag(z) + imag(w))
Base.:-(z::IComplex) = IComplex(-real(z), -imag(z))
Base.:-(z::IComplex, w::IComplex) = IComplex(real(z) - real(w), imag(z) - imag(w))
Base.:-(x::Union{Interval, Real}, z::IComplex) = IComplex(x - real(z), -imag(z))
Base.:-(z::IComplex, x::Union{Interval, Real}) = IComplex(real(z) - x, imag(z))

Base.:*(z::IComplex, w::IComplex) =
    IComplex(real(z) * real(w) - imag(z) * imag(w), real(z) * imag(w) + imag(z) * real(w))
Base.:*(x::Union{Interval, Real}, z::IComplex) = IComplex(x * real(z), x * imag(z))
Base.:*(z::IComplex, x::Union{Interval, Real}) = IComplex(x * real(z), x * imag(z))

Base.muladd(z::Union{Complex, IComplex}, w::Union{Complex, IComplex}, x::IComplex) =
    IComplex(
    muladd(real(z), real(w), real(x)) - imag(z) * imag(w),
    muladd(real(z), imag(w), muladd(imag(z), real(w), imag(x))),
)

function Base.inv(b::IComplex{T}) where {T}
    bre, bim = reim(b)
    denom = bre^2 + bim^2
    return IComplex(bre / denom, (-bim) / denom)
end
Base.:/(a::R, z::S) where {R <: Real, S <: IComplex} = (T = promote_type(R, S); a * inv(T(z)))
Base.:/(z::IComplex, x::Union{Interval, Real}) = IComplex(real(z) / x, imag(z) / x)
function Base.:/(a::IComplex{T}, b::IComplex{T}) where {T}
    are, aim = reim(a)
    bre, bim = reim(b)
    denom = bre^2 + bim^2
    return IComplex((are * bre + aim * bim) / denom, (aim * bre - are * bim) / denom)
end

# ---------------------------------------------------------------------------
# sqrt, sin and cos
# ---------------------------------------------------------------------------

# `Float64(π)` rounds down, so `π` lies between it and its successor.
const _PI = Interval(Float64(π), nextfloat(Float64(π)))
const _HALF_PI = _PI / 2.0
const _TWO_PI = 2.0 * _PI

# Base's libm wrappers are accurate to well under one ulp, so two ulps on each
# side is a rigorous enclosure.
_enclose(x::Float64) = Interval(prevfloat(x, 2), nextfloat(x, 2))

"""
    sqrt(a::Interval{Float64})

Enclosure of `√` over the nonnegative part of `a`. A wholly negative `a` is
empty (`NaN`); a straddling `a` is restricted to its nonnegative part.
"""
function Base.sqrt(a::Interval{Float64})::Interval{Float64}
    (isempty(a) || a.hi < 0.0) && return Interval(NaN)
    lo = a.lo ≤ 0.0 ? 0.0 : max(_enclose(sqrt(a.lo)).lo, 0.0)
    return Interval(lo, _enclose(sqrt(a.hi)).hi)
end

Base.sinh(a::Interval{Float64})::Interval{Float64} =
    Interval(_enclose(sinh(a.lo)).lo, _enclose(sinh(a.hi)).hi)

function Base.cosh(a::Interval{Float64})::Interval{Float64}
    a.lo ≥ 0.0 && return Interval(_enclose(cosh(a.lo)).lo, _enclose(cosh(a.hi)).hi)
    a.hi ≤ 0.0 && return Interval(_enclose(cosh(a.hi)).lo, _enclose(cosh(a.lo)).hi)
    # `cosh ≥ 1` everywhere, with equality at 0 ∈ a.
    return Interval(1.0, max(_enclose(cosh(a.lo)).hi, _enclose(cosh(a.hi)).hi))
end

# True when some `offset + k*period`, `k ∈ ℤ`, can meet `[lo, hi]`. `offset` and
# `period` are enclosures, so a "maybe" answers true.
function _meets_lattice(
        lo::Float64, hi::Float64, offset::Interval{Float64}, period::Interval{Float64},
    )::Bool
    k = floor((lo - mid(offset)) / mid(period))
    for j in (k - 1.0):(k + 2.0)
        c = offset + j * period
        c.hi ≥ lo && c.lo ≤ hi && return true
    end
    return false
end

function Base.sin(a::Interval{Float64})::Interval{Float64}
    isempty(a) && return a
    (!isfinite(a.lo) || !isfinite(a.hi)) && return Interval(-1.0, 1.0)
    diam(a) ≥ _TWO_PI.lo && return Interval(-1.0, 1.0)
    ends = hull(_enclose(sin(a.lo)), _enclose(sin(a.hi)))
    lo = _meets_lattice(a.lo, a.hi, -_HALF_PI, _TWO_PI) ? -1.0 : max(ends.lo, -1.0)
    hi = _meets_lattice(a.lo, a.hi, _HALF_PI, _TWO_PI) ? 1.0 : min(ends.hi, 1.0)
    return Interval(lo, hi)
end

function Base.cos(a::Interval{Float64})::Interval{Float64}
    isempty(a) && return a
    (!isfinite(a.lo) || !isfinite(a.hi)) && return Interval(-1.0, 1.0)
    diam(a) ≥ _TWO_PI.lo && return Interval(-1.0, 1.0)
    ends = hull(_enclose(cos(a.lo)), _enclose(cos(a.hi)))
    lo = _meets_lattice(a.lo, a.hi, _PI, _TWO_PI) ? -1.0 : max(ends.lo, -1.0)
    hi = _meets_lattice(a.lo, a.hi, zero(Interval{Float64}), _TWO_PI) ?
        1.0 : min(ends.hi, 1.0)
    return Interval(lo, hi)
end

"""
    sqrt(z::IComplex{Float64})

Enclosure of the principal square root `√z = u + i·sign(Im z)·v`, where
`u = √((|z| + Re z)/2)` and `v = √((|z| - Re z)/2)`. A box meeting the branch cut
along the negative real axis is returned empty.

Only the well-conditioned one of the two differences is evaluated; the other root
comes from `2uv = Im z`.
"""
function Base.sqrt(z::IComplex{Float64})::IComplex{Float64}
    x, y = real(z), imag(z)
    (x.lo < 0.0 && 0.0 ∈ y) && return IComplex(Interval(NaN), Interval(NaN))
    r = sqrt(sqr(x) + sqr(y))
    if x.lo ≥ 0.0
        u = sqrt((r + x) / 2.0)
        u.lo > 0.0 && return IComplex(u, y / (2.0 * u))
    else
        v = sqrt((r - x) / 2.0)
        if v.lo > 0.0
            y.lo ≥ 0.0 && return IComplex(y / (2.0 * v), v)
            return IComplex((-y) / (2.0 * v), -v)
        end
    end
    # Neither difference stays positive: no well-conditioned root to divide by.
    u = sqrt((r + x) / 2.0)
    v = sqrt((r - x) / 2.0)
    y.lo ≥ 0.0 && return IComplex(u, v)
    y.hi ≤ 0.0 && return IComplex(u, -v)
    return IComplex(u, Interval(-v.hi, v.hi))
end

Base.sin(z::IComplex{Float64})::IComplex{Float64} =
    IComplex(sin(real(z)) * cosh(imag(z)), cos(real(z)) * sinh(imag(z)))

Base.cos(z::IComplex{Float64})::IComplex{Float64} =
    IComplex(cos(real(z)) * cosh(imag(z)), -(sin(real(z)) * sinh(imag(z))))

mid(z::IComplex) = Complex(mid(real(z)), mid(imag(z)))
diam(z::IComplex) = max(diam(real(z)), diam(imag(z)))
rad(z::IComplex) = max(rad(real(z)), rad(imag(z)))
mag(z::IComplex) = max(mag(real(z)), mag(imag(z)))
isinterior(a::IComplex, b::IComplex) =
    isinterior(real(a), real(b)) && isinterior(imag(a), imag(b))
Base.issubset(a::IComplex, b::IComplex) = real(a) ⊆ real(b) && imag(a) ⊆ imag(b)
Base.isdisjoint(a::IComplex, b::IComplex) =
    isdisjoint(real(a), real(b)) || isdisjoint(imag(a), imag(b))
Base.intersect(a::IComplex, b::IComplex) =
    IComplex(intersect(real(a), real(b)), intersect(imag(a), imag(b)))
Base.in(x::Number, a::IComplex) = in(real(x), real(a)) && in(imag(x), imag(a))

"""
    inf_norm_bound(A::AbstractMatrix{<:IComplex})

Upper bound on the infinity operator norm of the interval matrix `A`.
"""
function inf_norm_bound(A::AbstractMatrix{IComplex{T}}) where {T}
    # ||A|| = maxᵢ ∑ⱼ|Aᵢⱼ| ≤ √2 * maxᵢ ∑ⱼ max(|real(Aᵢⱼ)|,|imag(Aᵢⱼ)|)
    bound = zero(T)
    for i in 1:size(A, 1)
        # use again an interval to account for accumulation error
        bᵢ = Interval(zero(T))
        for j in 1:size(A, 2)
            bᵢ += mag(A[i, j])
        end
        bound = max(bound, mag(bᵢ))
    end
    # Upper bound √2 by 1.41422, this also accounts for the multiplication error in √2 * bound
    return 1.41422 * bound
end
