# Taylor coefficients 0..K at 0 of an analytic `g`, scalar- or vector-valued, from
# M samples on the circle of radius r (the discrete Cauchy integral).
function cauchy_coefficients(g, K::Int; M::Int = 64, r::Float64 = 0.1)
    samples = [g(r * cis(2π * j / M)) for j in 0:(M - 1)]
    return [
        sum(samples[j + 1] .* cis(-2π * j * k / M) for j in 0:(M - 1)) ./ (M * r^k)
            for k in 0:K
    ]
end
