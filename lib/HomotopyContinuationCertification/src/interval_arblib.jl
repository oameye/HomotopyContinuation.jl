# Bridge between `IComplexF64` intervals and Arb `Acb` balls.
#
# The Float64 Krawczyk path computes with `IComplex{Float64}`; certificates and
# the arbitrary-precision fallback store enclosures as Arb `Acb` matrices. These
# conversions move enclosures losslessly between the two representations.

"""
    IComplexF64(z::Arblib.AcbLike)

Read an Arb complex ball `z` into a Float64 rectangular interval, rounding the
real and imaginary endpoints outward so the enclosure is preserved.
"""
function IComplexF64(
        z::Arblib.AcbLike,
        a::Arblib.Arf = Arblib.Arf(; prec = 53),
        b::Arblib.Arf = Arblib.Arf(; prec = 53),
    )
    Arblib.get_interval!(a, b, Arblib.realref(z); prec = 53)
    re = Interval(Arblib.get_d(a, RoundDown), Arblib.get_d(b, RoundUp))
    Arblib.get_interval!(a, b, Arblib.imagref(z); prec = 53)
    im = Interval(Arblib.get_d(a, RoundDown), Arblib.get_d(b, RoundUp))
    return IComplexF64(re, im)
end

function Base.convert(
        ::Type{T},
        x::AbstractVector{IComplexF64},
    ) where {
        T <: Union{Arblib.AcbVector, Arblib.AcbRefVector, Arblib.AcbMatrix, Arblib.AcbRefMatrix},
    }
    y = T(mid.(x); prec = 53)
    m = Arblib.Mag()
    for (i, xᵢ) in enumerate(x)
        m[] = rad(xᵢ)
        Arblib.add_error!(y[i], m)
    end
    return y
end

function Base.setindex!(z::Union{Arblib.Acb, Arblib.AcbRef}, x::IComplexF64)
    rz = Arblib.realref(z)
    iz = Arblib.imagref(z)
    Arblib.midref(rz)[] = mid(real(x))
    Arblib.midref(iz)[] = mid(imag(x))
    Arblib.radref(rz)[] = rad(real(x))
    Arblib.radref(iz)[] = rad(imag(x))
    return z
end
function Base.setindex!(
        A::Union{Arblib.AcbMatrix, Arblib.AcbRefMatrix},
        x::IComplexF64,
        i::Integer,
        j::Integer,
    )
    Arblib.ref(A, i, j)[] = x
    return A
end

# Column `Acb` matrix to a `Vector{ComplexF64}` of midpoints. A named function
# rather than a `Vector{ComplexF64}` constructor overload, which would be type
# piracy (both types are foreign to this package).
function acb_complex_vector(A::Union{Arblib.AcbMatrix, Arblib.AcbRefMatrix})
    @assert size(A, 2) == 1
    return ComplexF64[ComplexF64(Arblib.ref(A, i, 1)) for i in 1:size(A, 1)]
end
