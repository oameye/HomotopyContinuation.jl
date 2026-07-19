## Arbitrary-precision (Arb) Krawczyk fallback for `certify`.
##
## When the Float64 Krawczyk operator fails to certify a solution, `certify`
## retries at increasing bit precision using Arb `Acb` balls through the
## in-place `AcbInterpreter` (see model_kit/acb_interpreter.jl). This mirrors the
## Float64 `ε_inflation_krawczyk` but computes every step in place with a
## settable working precision, doubling the enclosure quality until either the
## Krawczyk test succeeds or `max_precision` is exceeded.

# The magnitude of an `Acb`/`AcbRef` as a `Float64` (writes into scratch `mag`).
magF64(x, mag::Mag) = Arblib.get(Arblib.get!(mag, x))

"""
    arb_ε_inflation_krawczyk(arb, p; prec)

Apply the ε-inflation Krawczyk operator in Arb arithmetic at precision `prec`
using the pre-allocated state in `arb::AcbCertCache`. The refined point lives in
`arb.x̃₀` and the approximate inverse in `arb.C` (both updated in place). Returns
`(certified, x₁::AcbMatrix, x₀::AcbMatrix, is_real)`.
"""
function arb_ε_inflation_krawczyk(
        arb::AcbCertCache,
        p::Union{Nothing, CertificationParameters};
        prec::Int,
    )
    C = arb.C
    r₀ = arb.r₀
    Δx₀ = arb.Δx₀
    x̃₀ = arb.x̃₀
    x₀ = arb.x₀
    x₁ = arb.x₁
    J_x₀ = arb.J_x₀
    M = arb.M
    δx = arb.δx
    m = arb.mag
    ip = arb_interval_params(p)
    # These buffers are free once their producer has run; reuse them as scratch.
    x̃₀′ = x̄₁ = Δx₀

    # Newton refinement with the fixed approximate inverse `C`, until the residual
    # magnitude stops improving.
    acc = Inf
    max_iters = 10
    for i in 1:max_iters
        acb_execute!(r₀, arb.eval_interpreter, x̃₀, ip)
        Arblib.mul!(Δx₀, C, r₀)
        acc_new = Arblib.get(Arblib.bound_inf_norm!(m, Δx₀))
        (acc_new > acc || i == max_iters) && break
        acc = acc_new
        Arblib.sub!(x̃₀, x̃₀, Δx₀)
        Arblib.get_mid!(x̃₀, x̃₀)
    end

    # ε-inflation around mid(x̃₀). The radius increase 2^(prec/4) is dynamic to
    # avoid hitting the precision limit in poorly-conditioned situations.
    n = size(x̃₀, 1)
    Arblib.get_mid!(x₀, x̃₀)
    incr_factor = exp2(div(prec, 4))
    mach_eps = exp2(-prec)
    for i in 1:n
        m[] = max(magF64(Δx₀[i], m), mach_eps) * incr_factor
        Arblib.add_error!(x₀[i], m)
    end

    acb_execute!(nothing, J_x₀, arb.jac_interpreter, x₀, ip)

    # M = C * J(x₀) - I
    Arblib.mul!(M, C, J_x₀)
    for i in 1:n
        Arblib.sub!(M[i, i], M[i, i], 1)
    end

    # Necessary condition ‖M‖ < 1/√2, lower-bounded by 0.7071.
    if Arblib.get(Arblib.bound_inf_norm!(m, M)) < 0.7071
        # x₁ = (x̃₀ - Δx₀) - (C * J(x₀) - I) * (x₀ - x̃₀)
        Arblib.sub!(δx, x₀, x̃₀)
        Arblib.mul!(x₁, M, δx)
        Arblib.sub!(x̃₀′, x̃₀, Δx₀)
        Arblib.sub!(x₁, x̃₀′, x₁)

        certified = Bool(Arblib.contains(x₀, x₁))
        Arblib.conjugate!(x̄₁, x₁)
        is_real = Bool(Arblib.contains(x₀, x̄₁))
    else
        certified = false
        is_real = false
    end

    if !certified
        # Refresh the approximate inverse for the next (higher-precision) attempt.
        Arblib.approx_inv!(C, J_x₀)
    end

    return certified, AcbMatrix(x₁), AcbMatrix(x₀), is_real
end

function extended_prec_certify_solution(
        F::System,
        solution_candidate::Vector{ComplexF64},
        x̃₀::AbstractVector,
        C::Matrix{ComplexF64},
        cert_params::Union{Nothing, CertificationParameters},
        cert_cache::CertificationCache,
        index::Int,
        is_real_system::Bool,
        ::Type{CertT};
        max_precision::Int = 256,
    ) where {CertT <: AbstractSolutionCertificate}
    arb = _arb(cert_cache)
    n = size(C, 1)

    prec = 128
    set_arb_precision!(arb, prec)

    # Seed the Arb inverse and refined point from the Float64 computation.
    @inbounds for j in 1:n, i in 1:n
        Arblib.set!(arb.C[i, j], C[i, j])
    end
    @inbounds for i in 1:n
        Arblib.set!(arb.x̃₀[i], x̃₀[i])
    end

    # Attempt at increasing precision. The precision is only raised when the
    # next step still fits within `max_precision`, so a failed certificate
    # reports the precision it was actually last attempted at (never overshoots
    # `max_precision`), and the cache is never raised past a precision that is
    # actually used.
    while true
        certified, x₁, x₀, is_real =
            arb_ε_inflation_krawczyk(arb, cert_params; prec = prec)
        is_real_system || (is_real = false)

        if certified
            is_complex_sol = any(
                i -> !Bool(Arblib.contains_zero(Arblib.imagref(Arblib.ref(x₁, i, 1)))),
                1:n,
            )
            is_complex_sol && (is_real = false)
            solution = ComplexF64[ComplexF64(Arblib.ref(x₁, i, 1)) for i in 1:n]
            return _arb_certificate(
                CertT, solution_candidate, is_real, is_complex_sol, index, prec,
                x₀, x₁, arb.x̃₀, arb.C, solution,
            )
        end

        prec + 64 > max_precision && break
        prec += 64
        set_arb_precision!(arb, prec)
    end

    # Certification failed at every attempted precision.
    return _uncertified_certificate(CertT, solution_candidate, index, prec)
end
