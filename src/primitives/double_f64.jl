# DoubleF64 — extended-precision arithmetic using error-free transformations.
# The transcendental surface is limited to what the tape can execute: `exp`,
# `sin`, `cos`, `sincos`, `sinh` and `cosh`.

# ---------------------------------------------------------------------------
# Error-free transformations
# ---------------------------------------------------------------------------

"""
    quick_two_sum(a, b)

Computes `s = fl(a+b)` and `e = err(a+b)`. Assumes `|a| ≥ |b|`.
"""
@inline function quick_two_sum(a::Float64, b::Float64)
    s = a + b
    e = b - (s - a)
    return s, e
end

"""
    two_sum(a, b)

Computes `s = fl(a+b)` and `e = err(a+b)`.
"""
@inline function two_sum(a::Float64, b::Float64)
    s = a + b
    v = s - a
    e = (a - (s - v)) + (b - v)
    return s, e
end

"""
    two_diff(a, b)

Computes `s = fl(a-b)` and `e = err(a-b)`.
"""
@inline function two_diff(a::Float64, b::Float64)
    s = a - b
    v = s - a
    e = (a - (s - v)) - (b + v)
    return s, e
end

"""
    two_prod(a, b)

Computes `s = fl(a*b)` and `e = err(a*b)`.
"""
@inline function two_prod(a::Float64, b::Float64)
    p = a * b
    e = fma(a, b, -p)
    return p, e
end

"""
    two_square(a)

Computes `s = fl(a*a)` and `e = err(a*a)`. Faster than [`two_prod(a, a)`](@ref).
"""
@inline function two_square(a::Float64)
    p = a * a
    e = fma(a, a, -p)
    return p, e
end

# ---------------------------------------------------------------------------
# DoubleF64 type
# ---------------------------------------------------------------------------

"""
    DoubleF64 <: AbstractFloat

Double-precision floating-point number represented as the unevaluated sum of
two `Float64` values (`hi` and `lo`), giving approximately 31 decimal digits
of precision.
"""
struct DoubleF64 <: AbstractFloat
    hi::Float64
    lo::Float64
end

const ComplexDF64 = Complex{DoubleF64}

# Disambiguate with (::Type{T})(x::Real, r::RoundingMode) from Base.Rounding
DoubleF64(x::Real, r::RoundingMode) = DoubleF64(round(Float64(x), r))

# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

DoubleF64(x::DoubleF64) = DoubleF64(x.hi, x.lo)

function DoubleF64(x::Float64)
    return DoubleF64(x, isinf(x) ? Inf : 0.0)
end

function DoubleF64(x::Float32)
    return DoubleF64(convert(Float64, x), isinf(x) ? Inf : 0.0)
end

function DoubleF64(x::Float16)
    return DoubleF64(convert(Float64, x), isinf(x) ? Inf : 0.0)
end

function DoubleF64(x::Integer)
    return DoubleF64(convert(Float64, x), 0.0)
end

function DoubleF64(x::BigFloat)
    z = convert(Float64, x)
    return DoubleF64(z, convert(Float64, x - z))
end

function DoubleF64(x::Irrational)
    return DoubleF64(big(x))
end

function DoubleF64(x::Rational)
    return DoubleF64(numerator(x)) / DoubleF64(denominator(x))
end

# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

hi(x::DoubleF64) = x.hi
lo(x::DoubleF64) = x.lo

# ---------------------------------------------------------------------------
# Special values
# ---------------------------------------------------------------------------

Base.zero(::DoubleF64) = DoubleF64(0.0, 0.0)
Base.zero(::Type{DoubleF64}) = DoubleF64(0.0, 0.0)

Base.one(::DoubleF64) = DoubleF64(1.0, 0.0)
Base.one(::Type{DoubleF64}) = DoubleF64(1.0, 0.0)

# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

# Conversion design:
#
# - Conversions *to* `DoubleF64` are cheap and safe to define broadly because
#   the target type is fixed. Those methods are important for generic
#   `promote`/mixed arithmetic with ordinary Julia numbers.
# - Conversions *from* `DoubleF64` are intentionally kept on concrete
#   constructors such as `Float64(x)` and `Int64(x)`, rather than broad
#   `convert(::Type{T}, ::DoubleF64) where T <: Number` methods. The broad form
#   supersedes Base's generic number-conversion methods and triggers a large
#   invalidation cascade.
#   Measured with SnoopCompile on a minimal reproduction, the broad method
#   shape causes 419 unique invalidations; the narrowed shape used here causes 15.
# - We still provide the practical interop surface used by the package and by
#   generic Julia code: common signed/unsigned integer targets, common float
#   targets, and abstract `Integer`/`Signed`/`Unsigned` entry points.

# Concrete result-type constructors preserve the expected user-facing behavior
# without overriding Base's generic conversion fallback for every numeric type.
Base.Float64(a::DoubleF64) = a.hi
Base.Float32(a::DoubleF64) = Float32(a.hi)
Base.Float16(a::DoubleF64) = Float16(a.hi)
Base.BigFloat(a::DoubleF64) = big(a.hi) + big(a.lo)

# Integer-valued targets stay on constructors for the same reason: broad
# `convert(::Type{T}, ::DoubleF64)` definitions are the main invalidation trap.
Base.Int8(a::DoubleF64) = Int8(a.hi)
Base.Int16(a::DoubleF64) = Int16(a.hi)
Base.Int64(a::DoubleF64) = Int64(a.hi)
Base.Int32(a::DoubleF64) = Int32(a.hi)
Base.Int128(a::DoubleF64) = Int128(a.hi)
Base.UInt8(a::DoubleF64) = UInt8(a.hi)
Base.UInt16(a::DoubleF64) = UInt16(a.hi)
Base.UInt32(a::DoubleF64) = UInt32(a.hi)
Base.UInt64(a::DoubleF64) = UInt64(a.hi)
Base.UInt128(a::DoubleF64) = UInt128(a.hi)
Base.BigInt(a::DoubleF64) = BigInt(big(a.hi) + big(a.lo))
Base.Integer(a::DoubleF64) = Int64(a)
Base.Signed(a::DoubleF64) = Int64(a)
Base.Unsigned(a::DoubleF64) = UInt64(a)

# Broad conversions into `DoubleF64` are safe because the target type is fixed.
# This is what keeps `promote` and mixed arithmetic ergonomic with ordinary
# integers and floats.
Base.convert(::Type{DoubleF64}, x::DoubleF64) = x
Base.convert(::Type{DoubleF64}, x::AbstractFloat) = DoubleF64(x)
Base.convert(::Type{DoubleF64}, x::Irrational) = DoubleF64(x)
Base.convert(::Type{DoubleF64}, x::Integer) = DoubleF64(x)

# The promotion rules follow the same split: ordinary integers/floats promote to
# `DoubleF64`, while `BigInt`/`BigFloat` stay in the BigFloat tower.
Base.promote_rule(::Type{DoubleF64}, ::Type{<:Integer}) = DoubleF64
Base.promote_rule(::Type{DoubleF64}, ::Type{BigInt}) = BigFloat
Base.promote_rule(::Type{DoubleF64}, ::Type{BigFloat}) = BigFloat
Base.promote_rule(::Type{DoubleF64}, ::Type{<:AbstractFloat}) = DoubleF64

Base.big(x::DoubleF64) = big(x.hi) + big(x.lo)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

const double_π = DoubleF64(pi)
const double_2pi = DoubleF64(2 * BigFloat(Base.pi))
const double_pi = DoubleF64(BigFloat(Base.pi))
const double_pi2 = DoubleF64(BigFloat(Base.pi) * 0.5)
const double_pi4 = DoubleF64(BigFloat(Base.pi) * 0.25)
const double_pi16 = DoubleF64(BigFloat(Base.pi) * (1 / 16))
const double_3pi4 = DoubleF64(BigFloat(Base.pi) * 0.75)
const double_nan = DoubleF64(NaN, NaN)
const double_inf = DoubleF64(Inf)

const double_eps = 5.048709793414476e-29 # 2^-94 — conservative for basic arithmetic

Base.nextfloat(x::DoubleF64) = DoubleF64(x.hi, x.lo + double_eps * abs(x.hi))
Base.prevfloat(x::DoubleF64) = DoubleF64(x.hi, x.lo - double_eps * abs(x.hi))

# ---------------------------------------------------------------------------
# Addition
# ---------------------------------------------------------------------------

"""
    wide_add(a::Float64, b::Float64)

Add two `Float64`s with `DoubleF64` precision.
"""
@inline function wide_add(a::Float64, b::Float64)
    hi, lo = two_sum(a, b)
    return DoubleF64(hi, lo)
end

@inline function Base.:+(a::DoubleF64, b::Float64)
    hi, lo = two_sum(a.hi, b)
    lo += a.lo
    hi, lo = quick_two_sum(hi, lo)
    return DoubleF64(hi, lo)
end

Base.:+(a::Float64, b::DoubleF64) = b + a
Base.:+(a::Integer, b::DoubleF64) = b + float(a)
Base.:+(a::Bool, b::DoubleF64) = b + float(a)
Base.:+(a::DoubleF64, b::Integer) = a + float(b)
Base.:+(a::DoubleF64, b::Bool) = a + float(b)

@inline function Base.:+(a::DoubleF64, b::DoubleF64)
    hi, lo = two_sum(a.hi, b.hi)
    lo += (a.lo + b.lo)
    hi, lo = quick_two_sum(hi, lo)
    return DoubleF64(hi, lo)
end

# ---------------------------------------------------------------------------
# Subtraction
# ---------------------------------------------------------------------------

"""
    wide_sub(a::Float64, b::Float64)

Subtract two `Float64`s with `DoubleF64` precision.
"""
@inline function wide_sub(a::Float64, b::Float64)
    hi, lo = two_diff(a, b)
    return DoubleF64(hi, lo)
end

@inline function Base.:-(a::DoubleF64, b::Float64)
    hi, lo = two_diff(a.hi, b)
    lo += a.lo
    hi, lo = quick_two_sum(hi, lo)
    return DoubleF64(hi, lo)
end

@inline function Base.:-(a::Float64, b::DoubleF64)
    hi, lo = two_diff(a, b.hi)
    lo -= b.lo
    hi, lo = quick_two_sum(hi, lo)
    return DoubleF64(hi, lo)
end

@inline function Base.:-(a::DoubleF64, b::DoubleF64)
    hi, lo = two_diff(a.hi, b.hi)
    lo += a.lo
    lo -= b.lo
    hi, lo = quick_two_sum(hi, lo)
    return DoubleF64(hi, lo)
end

Base.:-(a::DoubleF64) = DoubleF64(-a.hi, -a.lo)

# ---------------------------------------------------------------------------
# Multiplication
# ---------------------------------------------------------------------------

"""
    wide_mul(a::Float64, b::Float64)

Multiply two `Float64`s with `DoubleF64` precision.
"""
@inline function wide_mul(a::Float64, b::Float64)
    hi, lo = two_prod(a, b)
    return DoubleF64(hi, lo)
end

@inline function Base.:*(a::DoubleF64, b::Float64)
    p1, p2 = two_prod(a.hi, b)
    p2 += a.lo * b
    p1, p2 = quick_two_sum(p1, p2)
    return DoubleF64(p1, p2)
end

Base.:*(a::Float64, b::DoubleF64) = b * a
Base.:*(a::Integer, b::DoubleF64) = b * float(a)
Base.:*(a::Bool, b::DoubleF64) = b * convert(Float64, a)
Base.:*(a::DoubleF64, b::Bool) = b ? a : zero(a)
Base.:*(a::DoubleF64, b::Integer) = a * float(b)

@inline function Base.:*(a::DoubleF64, b::DoubleF64)
    p1, p2 = two_prod(a.hi, b.hi)
    p2 += a.hi * b.lo + a.lo * b.hi
    p1, p2 = quick_two_sum(p1, p2)
    return DoubleF64(p1, p2)
end

# ---------------------------------------------------------------------------
# Division
# ---------------------------------------------------------------------------

"""
    wide_div(a::Float64, b::Float64)

Divide two `Float64`s with `DoubleF64` precision.
"""
@inline function wide_div(a::Float64, b::Float64)
    q1 = a / b
    # Compute a - q1 * b
    p1, p2 = two_prod(q1, b)
    s, e = two_diff(a, p1)
    e -= p2
    # get next approximation
    q2 = (s + e) / b
    s, e = quick_two_sum(q1, q2)
    return DoubleF64(s, e)
end

@inline function Base.:/(a::DoubleF64, b::Float64)
    q1 = a.hi / b
    # Compute this - q1 * d
    p1, p2 = two_prod(q1, b)
    s, e = two_diff(a.hi, p1)
    e += a.lo
    e -= p2
    # get next approximation
    q2 = (s + e) / b
    # renormalize
    hi, lo = quick_two_sum(q1, q2)
    return DoubleF64(hi, lo)
end

@inline function Base.:/(a::DoubleF64, b::DoubleF64)
    q1 = a.hi / b.hi  # approximate quotient
    # compute this - q1 * dd
    r = b * q1
    s1, s2 = two_diff(a.hi, r.hi)
    s2 -= r.lo
    s2 += a.lo
    # get next approximation
    q2 = (s1 + s2) / b.hi
    # renormalize
    hi, lo = quick_two_sum(q1, q2)
    return DoubleF64(hi, lo)
end

Base.:/(a::Float64, b::DoubleF64) = DoubleF64(a) / b
Base.:/(a::Integer, b::DoubleF64) = float(a) / b
Base.:/(a::DoubleF64, b::Integer) = a / float(b)

Base.inv(a::DoubleF64) = 1.0 / a

# ---------------------------------------------------------------------------
# Remainder, mod, divrem
# ---------------------------------------------------------------------------

Base.rem(
    a::DoubleF64,
    b::Union{Float64, DoubleF64},
    r::RoundingMode{:ToZero} = RoundToZero,
) = a - round(a / b, r) * b
Base.rem(a::DoubleF64, b::Union{Float64, DoubleF64}, r::RoundingMode{:Nearest}) =
    a - round(a / b, r) * b
Base.rem(a::DoubleF64, b::Union{Float64, DoubleF64}, r::RoundingMode{:Up}) =
    a - round(a / b, r) * b
Base.rem(a::DoubleF64, b::Union{Float64, DoubleF64}, r::RoundingMode{:Down}) =
    a - round(a / b, r) * b
Base.rem(a::DoubleF64, b::Union{Float64, DoubleF64}, r::RoundingMode{:FromZero}) =
    a - round(a / b, r) * b
Base.rem(a::DoubleF64, b::Union{Float64, DoubleF64}, r::RoundingMode) =
    a - round(a / b, r) * b

@inline function Base.divrem(a::DoubleF64, b::DoubleF64)
    n = round(a / b)
    return n, a - n * b
end

function Base.mod(x::DoubleF64, y::DoubleF64)
    n = round(x / y)
    return (x - y * n)
end

# ---------------------------------------------------------------------------
# Powers and square root
# ---------------------------------------------------------------------------

"""
    square(x::DoubleF64)

Compute `x * x` more efficiently than the generic product of two `DoubleF64`s.
"""
@inline function square(a::DoubleF64)
    p1, p2 = two_square(a.hi)
    p2 += 2.0 * a.hi * a.lo
    p2 += a.lo * a.lo
    hi, lo = quick_two_sum(p1, p2)
    return DoubleF64(hi, lo)
end

square(a::Float64) = wide_square(a)

"""
    wide_square(x::Float64)

Convert `x` to a `DoubleF64` and compute `x*x`.
"""
@inline function wide_square(a::Float64)
    hi, lo = two_square(a)
    return DoubleF64(hi, lo)
end

# Implementation adapted from Base
@inline function power_by_squaring(x::DoubleF64, p::Integer)
    if p == 1
        return copy(x)
    elseif p == 0
        return one(x)
    elseif p == 2
        return square(x)
    end
    P = abs(p)
    t = trailing_zeros(P) + 1
    P >>= t
    while (t -= 1) > 0
        x = square(x)
    end
    y = x
    while P > 0
        t = trailing_zeros(P) + 1
        P >>= t
        while (t -= 1) >= 0
            x = square(x)
        end
        y *= x
    end
    if p < 0
        return 1.0 / y
    end
    return y
end

Base.:^(a::DoubleF64, p::Integer) = power_by_squaring(a, p)

@inline function Base.sqrt(a::DoubleF64)
    # Strategy: Use Karp's trick.  If x is an approximation to sqrt(a), then
    #
    #   sqrt(a) = a*x + [a - (a*x)^2] * x / 2   (approx)
    #
    # The approximation is accurate to twice the accuracy of x.
    if a.hi < 0
        throw(
            DomainError(
                "sqrt will only return a complex result if called with a complex argument.",
            ),
        )
    end
    if iszero(a)
        return zero(a)
    end
    x = inv(sqrt(a.hi))
    ax = a.hi * x
    return wide_add(ax, (a - square(ax)).hi * (x * 0.5))
end

"""
    wide_sqrt(x::Float64)

Convert `x` to a `DoubleF64` and compute `sqrt(x)`.
"""
wide_sqrt(a::Float64) = sqrt(DoubleF64(a))

# ---------------------------------------------------------------------------
# Comparison and equality
# ---------------------------------------------------------------------------

Base.:<(a::DoubleF64, b::DoubleF64) = a.hi + a.lo < b.hi + b.lo
Base.:<(a::DoubleF64, b::Float64) = a.hi < b || (a.hi == b) && a.lo < 0.0
Base.:<(a::Float64, b::DoubleF64) = a < b.hi || (a == b.hi) && b.lo > 0.0

Base.isless(a::DoubleF64, b::DoubleF64) = isless(a.hi + a.lo, b.hi + b.lo)

Base.:<=(a::DoubleF64, b::DoubleF64) = !(b < a)
Base.:<=(a::DoubleF64, b::Float64) = !(b < a)
Base.:<=(a::Float64, b::DoubleF64) = !(b < a)

Base.:(==)(a::DoubleF64, b::Float64) = a.hi == b && a.lo == 0.0
Base.:(==)(a::Float64, b::DoubleF64) = b == a

# ---------------------------------------------------------------------------
# Predicates and special value queries
# ---------------------------------------------------------------------------

Base.iszero(a::DoubleF64) = a.hi == 0.0
Base.isone(a::DoubleF64) = a.hi == 1.0 && a.lo == 0.0
Base.abs(a::DoubleF64) = a.hi < 0.0 ? -a : a

Base.eps(::DoubleF64) = double_eps
Base.eps(::Type{DoubleF64}) = double_eps

Base.floatmin(::Type{DoubleF64}) = 2.0041683600089728e-292  # = 2^(-1022 + 53)
Base.floatmax(::Type{DoubleF64}) =
    DoubleF64(1.79769313486231570815e+308, 9.97920154767359795037e+291)

Base.isnan(a::DoubleF64) = isnan(a.hi) || isnan(a.lo)
Base.isinf(a::DoubleF64) = isinf(a.hi)
Base.isfinite(a::DoubleF64) = isfinite(a.hi)

# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

# Disambiguate specific RoundingModes that Base defines methods for on AbstractFloat
Base.round(a::DoubleF64, ::RoundingMode{:NearestTiesAway}) = round(a, RoundNearest) # fallback
Base.round(a::DoubleF64, ::RoundingMode{:NearestTiesUp}) = round(a, RoundNearest) # fallback
Base.round(a::DoubleF64, ::RoundingMode{:FromZero}) =
    a >= zero(a) ? floor(a + DoubleF64(0.5)) : ceil(a - DoubleF64(0.5))

@inline function Base.round(a::DoubleF64, r::RoundingMode = RoundNearest)
    hi = round(a.hi, r)
    lo = 0.0

    if hi == a.hi
        # High word is an integer already. Round the low word.
        lo = round(a.lo, r)
        # Renormalize. This is needed if hi = some integer, lo = 1/2.
        hi, lo = quick_two_sum(hi, lo)
    else
        # High word is not an integer.
        if abs(hi - a.hi) == 0.5 && a.lo < 0.0
            # There is a tie in the high word; consult the low word to break it.
            hi -= 1.0
        end
    end

    return DoubleF64(hi, lo)
end

@inline function Base.floor(a::DoubleF64)
    hi = floor(a.hi)
    lo = 0.0

    if hi == a.hi
        lo = floor(a.lo)
        hi, lo = quick_two_sum(hi, lo)
    end

    return DoubleF64(hi, lo)
end

# We keep typed integer rounding helpers private and concrete. The broad
# `floor(::Type{I}, ::DoubleF64) where I <: Integer` shape has the same
# invalidation problem as broad `convert` methods, and the current code only
# needs a concrete integer target.
@inline function _floor_int64(a::DoubleF64)::Int64
    hi = floor(Int64, a.hi)
    lo = zero(Int64)
    if hi == a.hi
        lo = floor(Int64, a.lo)
    end
    return hi + lo
end

@inline function Base.ceil(a::DoubleF64)
    hi = ceil(a.hi)
    lo = 0.0

    if hi == a.hi
        lo = ceil(a.lo)
        hi, lo = quick_two_sum(hi, lo)
    end

    return DoubleF64(hi, lo)
end

@inline function _ceil_int64(a::DoubleF64)::Int64
    hi = ceil(Int64, a.hi)
    lo = zero(Int64)
    if hi == a.hi
        lo = ceil(Int64, a.lo)
    end
    return hi + lo
end

Base.trunc(a::DoubleF64) = a.hi >= 0.0 ? floor(a) : ceil(a)
_trunc_int64(a::DoubleF64)::Int64 = a.hi >= 0.0 ? _floor_int64(a) : _ceil_int64(a)
Base.isinteger(x::DoubleF64)::Bool = iszero(x - trunc(x))

# ---------------------------------------------------------------------------
# Decompose, ldexp, mul_pwr2
# ---------------------------------------------------------------------------

function Base.decompose(a::DoubleF64)::Tuple{Int128, Int, Int}
    hi, lo = a.hi, a.lo
    num1, pow1, den1 = Base.decompose(hi)
    num2, pow2, den2 = Base.decompose(lo)

    num = Int128(num1)

    pdiff = pow1 - pow2
    shift = min(pdiff, 52)
    signed_num = den1 * (Int128(num) << shift)  # den1 is +1/-1
    signed_num += den2 * (num2 >> (pdiff - shift))  # den2 is +1/-1

    num = abs(signed_num)
    den = signed_num >= 0 ? 1 : -1
    pow = pow1 - shift

    return num, pow, den
end

Base.ldexp(a::DoubleF64, exp::Int) = DoubleF64(ldexp(a.hi, exp), ldexp(a.lo, exp))

"""
    mul_pwr2(a::DoubleF64, b::Float64)

`a * b` where `b` is a power of 2. Exact when `b` is a power of 2.
"""
mul_pwr2(a::DoubleF64, b::Float64) = DoubleF64(a.hi * b, a.lo * b)

# ---------------------------------------------------------------------------
# Transcendental functions
# ---------------------------------------------------------------------------

# 1/n! for n = 1, …, 32, accurate to double-double precision.
const _INV_FACTORIAL = ntuple(
    n -> DoubleF64(one(BigFloat) / BigFloat(factorial(big(n)))), 32,
)
const double_log2 = DoubleF64(log(BigFloat(2)))
const _LOG2_F64 = 0.6931471805599453
# exp overflows above log(prevfloat(Inf)) and flushes to zero below log(nextfloat(0)).
const _EXP_MAX = 709.79
const _EXP_MIN = -745.2

"""
    exp(a::DoubleF64)

`m` is chosen so that `a - m·log 2` lies in `[-log2/2, log2/2]`; that remainder is
scaled down by 2⁹ so the Taylor series converges in a handful of terms, then the
result is squared back up nine times.
"""
function Base.exp(a::DoubleF64)::DoubleF64
    isnan(a) && return double_nan
    a.hi >= _EXP_MAX && return double_inf
    a.hi <= _EXP_MIN && return zero(DoubleF64)
    iszero(a) && return one(DoubleF64)

    m = round(a.hi / _LOG2_F64)
    r = mul_pwr2(a - double_log2 * m, 1 / 512)

    r2 = r * r
    s = r + mul_pwr2(r2, 0.5)
    p = r2 * r
    for n in 3:11
        s += p * _INV_FACTORIAL[n]
        p = p * r
    end
    # Undo the 1/512 scaling: exp(2r) - 1 = 2(exp(r) - 1) + (exp(r) - 1)².
    for _ in 1:9
        s = mul_pwr2(s, 2.0) + s * s
    end
    return ldexp(s + 1.0, Int(m))
end

# Taylor series on |x| <= π/4. The last retained term is below 2^-106 there.
function _sin_taylor(x::DoubleF64)::DoubleF64
    x2 = x * x
    s = x
    p = x
    for n in 3:2:31
        p = p * x2
        t = p * _INV_FACTORIAL[n]
        s = isodd((n - 1) ÷ 2) ? s - t : s + t
    end
    return s
end

function _cos_taylor(x::DoubleF64)::DoubleF64
    x2 = x * x
    s = one(DoubleF64)
    p = one(DoubleF64)
    for n in 2:2:30
        p = p * x2
        t = p * _INV_FACTORIAL[n]
        s = isodd(n ÷ 2) ? s - t : s + t
    end
    return s
end

function Base.sincos(a::DoubleF64)::Tuple{DoubleF64, DoubleF64}
    isfinite(a) || return (double_nan, double_nan)
    iszero(a) && return (zero(DoubleF64), one(DoubleF64))

    # `a - 2π·round(a/2π)` carries an absolute error of about |a|·2⁻¹⁰⁶, so past
    # |a| = 2⁵³ sum the angle from its two limbs, each of which Base reduces
    # exactly.
    if abs(a.hi) >= 0x1p53
        shi, chi = sincos(a.hi)
        slo, clo = sincos(a.lo)
        return (
            DoubleF64(shi * clo + chi * slo),
            DoubleF64(chi * clo - shi * slo),
        )
    end

    # Reduce to [-π, π], then to [-π/4, π/4] plus a quadrant index.
    t = a - double_2pi * round(a / double_2pi)
    q = round(t / double_pi2)
    t = t - double_pi2 * q
    s = _sin_taylor(t)
    c = _cos_taylor(t)

    j = mod(Int(_trunc_int64(q)), 4)
    j == 0 && return (s, c)
    j == 1 && return (c, -s)
    j == 2 && return (-s, -c)
    return (-c, s)
end

Base.sin(a::DoubleF64)::DoubleF64 = sincos(a)[1]
Base.cos(a::DoubleF64)::DoubleF64 = sincos(a)[2]

# Past this magnitude e^-|a| sits below the double-double ulp of e^|a|, so both
# functions equal e^|a|/2 and forming `e ± 1/e` would yield NaN.
const _SINH_LARGE = 40.0

function Base.cosh(a::DoubleF64)::DoubleF64
    isnan(a) && return double_nan
    isinf(a) && return double_inf
    iszero(a) && return one(DoubleF64)
    abs(a.hi) > _SINH_LARGE && return exp(abs(a) - double_log2)
    e = exp(a)
    return mul_pwr2(e + inv(e), 0.5)
end

function Base.sinh(a::DoubleF64)::DoubleF64
    isnan(a) && return double_nan
    isinf(a) && return a.hi > 0.0 ? double_inf : -double_inf
    iszero(a) && return zero(DoubleF64)
    if abs(a.hi) > _SINH_LARGE
        s = exp(abs(a) - double_log2)
        return a.hi > 0.0 ? s : -s
    end
    if abs(a.hi) > 0.05
        e = exp(a)
        return mul_pwr2(e - inv(e), 0.5)
    end
    # (e - 1/e)/2 cancels catastrophically near zero, so sum the series directly.
    x2 = a * a
    s = a
    p = a
    for n in 3:2:17
        p = p * x2
        s += p * _INV_FACTORIAL[n]
    end
    return s
end

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

function Base.show(io::IO, x::DoubleF64)
    return Printf.@printf io "%.32g" big(x)
end
