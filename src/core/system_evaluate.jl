## Allocating evaluation at the API boundary. The hot path is `evaluate!` /
## `evaluate_and_jacobian!` on preallocated buffers.

const EvaluableSystem = Union{SystemEvaluator, CloneableSystem, AbstractSystem}

# A system carrying a cached evaluator is evaluated through it. A bare `AbstractSystem`
# already implements the same interface, and wrapping one in a `SystemEvaluator` per call
# would cost more than the evaluation it is meant to serve.
_evaluation_target(S::SystemEvaluator)::SystemEvaluator = S
_evaluation_target(F::CloneableSystem)::SystemEvaluator = F.evaluator
_evaluation_target(F::AbstractSystem) = F

function _evaluation_arguments(
        S::EvaluableSystem, x::AbstractVector{<:Number}, p::AbstractVector{<:Number},
    )::Tuple{FSVec{ComplexF64}, FSVec{ComplexF64}}
    n = size(S)[2]
    length(x) == n || throw(
        ArgumentError(
            string("the system has ", n, " variables but got ", length(x), " values"),
        ),
    )
    np = nparameters(S)
    length(p) == np || throw(
        ArgumentError(
            string("the system has ", np, " parameters but got ", length(p), " values"),
        ),
    )
    xs = FSVec{ComplexF64}(undef, n)
    ps = FSVec{ComplexF64}(undef, np)
    copyto!(xs, x)
    copyto!(ps, p)
    return xs, ps
end

"""
    evaluate(F, x, p = ComplexF64[])

Value of the system `F` at the point `x` with parameters `p`. The result is real
when every entry is.

`F` may be a [`System`](@ref), a [`CompositionSystem`](@ref), a
[`FixedParameterSystem`](@ref) or an `AbstractSystem`. A `System` evaluates the
equations [`polynomials`](@ref) reports, which are normalized: an equation whose
coefficients are far from unit scale was divided by that scale, and
[`equation_scales`](@ref) records the factors.

One system evaluates one point at a time: concurrent calls on the same `F` share
its scratch space and would overwrite each other. `solve` and its executors are
the threaded entry points.
"""
function evaluate(
        F::EvaluableSystem, x::AbstractVector{<:Number},
        p::AbstractVector{<:Number} = ComplexF64[],
    )
    xs, ps = _evaluation_arguments(F, x, p)
    S = _evaluation_target(F)
    u = FSVec{ComplexF64}(undef, size(F)[1])
    evaluate!(u, S, xs, ps)
    return _narrow(collect(u))
end

"""
    jacobian(F, x, p = ComplexF64[])

Jacobian of the system `F` at the point `x` with parameters `p`. The result is
real when every entry is.
"""
function jacobian(
        F::EvaluableSystem, x::AbstractVector{<:Number},
        p::AbstractVector{<:Number} = ComplexF64[],
    )
    xs, ps = _evaluation_arguments(F, x, p)
    S = _evaluation_target(F)
    m, n = size(F)
    u = FSVec{ComplexF64}(undef, m)
    U = FSMat{ComplexF64}(undef, m, n)
    evaluate_and_jacobian!(u, U, S, xs, ps)
    return _narrow(collect(U))
end

(F::Union{CloneableSystem, AbstractSystem})(x::AbstractVector{<:Number}) = evaluate(F, x)

# `FixedParameterSystem` is left out: it has no parameters left to pass.
(F::Union{System, CompositionSystem, AbstractSystem})(
    x::AbstractVector{<:Number}, p::AbstractVector{<:Number},
) = evaluate(F, x, p)

"""
    is_real(F) -> Bool

Whether `F` maps real points to real values, i.e. whether its coefficients are real.

Exact for a [`System`](@ref), decided from the coefficients themselves. Any other
system is decided by evaluating at one random real point, and so is correct with
probability one.
"""
is_real(F::System)::Bool = _has_real_coefficients(polynomials(F))

function _has_real_coefficients(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
    )::Bool
    for p in polys, c in MP.coefficients(p)
        iszero(imag(ComplexF64(c))) || return false
    end
    return true
end

function _has_real_coefficients(exprs::AbstractVector{Expression})::Bool
    for e in exprs
        has_real_coefficients(e) || return false
    end
    return true
end

function is_real(
        F::Union{SystemEvaluator, CompositionSystem, FixedParameterSystem, AbstractSystem};
        rng::Random.AbstractRNG = Random.default_rng(),
    )::Bool
    S = _evaluation_target(F)
    m, n = size(F)
    u = FSVec{ComplexF64}(undef, m)
    x = FSVec{ComplexF64}(randn(rng, n))
    p = FSVec{ComplexF64}(randn(rng, nparameters(F)))
    evaluate!(u, S, x, p)
    return all(z -> iszero(imag(z)), u)
end
