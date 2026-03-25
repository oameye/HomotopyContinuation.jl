fast_abs(z::Complex) = sqrt(abs2(z))
fast_abs(x::Real) = abs(x)

nanmin(a, b) = isnan(a) ? b : (isnan(b) ? a : min(a, b))
nanmax(a, b) = isnan(a) ? b : (isnan(b) ? a : max(a, b))

function nthroot(x::Real, N::Integer)
    return if N == 4
        sqrt(sqrt(x))
    elseif N == 2
        sqrt(x)
    elseif N == 3
        cbrt(x)
    elseif N == 1
        x
    elseif N == 0
        one(x)
    else
        x^(1 / N)
    end
end
