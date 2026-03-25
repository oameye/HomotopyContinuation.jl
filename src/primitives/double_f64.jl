# DoubleF64 — extended-precision arithmetic using error-free transformations.
# Ported from HomotopyContinuation.jl v2 (src/DoubleDouble.jl).
# Transcendental functions (exp, log, sin, cos, tan, asin, acos, atan) are omitted
# as they are not needed for polynomial system solving.

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

(::Type{T})(a::DoubleF64) where {T <: AbstractFloat} = convert(T, a)
Base.convert(::Type{T}, a::DoubleF64) where {T <: AbstractFloat} = convert(T, a.hi)
Base.convert(::Type{BigFloat}, a::DoubleF64) = big(a.hi) + big(a.lo)
Base.BigFloat(a::DoubleF64) = convert(BigFloat, a)
Base.convert(::Type{T}, a::DoubleF64) where {T <: Integer} = convert(T, a.hi)
Base.convert(::Type{Integer}, a::DoubleF64) = convert(Int64, a.hi)
Base.convert(::Type{BigInt}, a::DoubleF64) = convert(BigInt, big(a.hi) + big(a.lo))

Base.convert(::Type{DoubleF64}, x::DoubleF64) = x
Base.convert(::Type{DoubleF64}, x::AbstractFloat) = DoubleF64(x)
Base.convert(::Type{DoubleF64}, x::Irrational) = DoubleF64(x)
Base.convert(::Type{DoubleF64}, x::Integer) = DoubleF64(x)

Base.promote_rule(::Type{DoubleF64}, ::Type{<:Integer}) = DoubleF64
Base.promote_rule(::Type{DoubleF64}, ::Type{BigInt}) = BigFloat
Base.promote_rule(::Type{DoubleF64}, ::Type{BigFloat}) = BigFloat
Base.promote_rule(::Type{DoubleF64}, ::Type{Float64}) = DoubleF64
Base.promote_rule(::Type{DoubleF64}, ::Type{Float32}) = DoubleF64
Base.promote_rule(::Type{DoubleF64}, ::Type{Float16}) = DoubleF64

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

Compute `x * x` more efficiently than `x * x`.
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

@inline function Base.floor(::Type{I}, a::DoubleF64) where {I <: Integer}
    hi = floor(I, a.hi)
    lo = zero(I)

    if hi == a.hi
        lo = floor(I, a.lo)
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

@inline function Base.ceil(::Type{I}, a::DoubleF64) where {I <: Integer}
    hi = ceil(I, a.hi)
    lo = zero(I)

    if hi == a.hi
        lo = ceil(I, a.lo)
    end

    return hi + lo
end

Base.trunc(a::DoubleF64) = a.hi >= 0.0 ? floor(a) : ceil(a)
Base.trunc(::Type{I}, a::DoubleF64) where {I <: Integer} =
    a.hi >= 0.0 ? floor(I, a) : ceil(I, a)
Base.isinteger(x::DoubleF64) = iszero(x - trunc(x))

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
# Display
# ---------------------------------------------------------------------------

function Base.show(io::IO, x::DoubleF64)
    return Printf.@printf io "%.32g" big(x)
end
