## Solution certification via the Krawczyk method (interval arithmetic).
##
## `certify` proves that an approximate solution corresponds to a unique true
## solution contained in an explicit complex-interval box, following
## Breiding, Rose, Timme, "Certifying zeros of polynomial systems using interval
## arithmetic" (arXiv:2011.05000) and Moore's test.
##
## The Float64 path computes with `IComplex{Float64}` through the generic tape
## interpreter. When it fails, the arbitrary-precision Arb fallback retries at
## increasing precision (see certification_arb.jl).

# ─────────────────────────────────────────────────────────────────────────────
# Certificate types
# ─────────────────────────────────────────────────────────────────────────────

abstract type AbstractSolutionCertificate end

"""
    SolutionCertificate

Result of [`certify`](@ref) for a single solution. Holds the solution candidate
and, if certification succeeded, an `Arblib.AcbMatrix` of complex intervals that
contains a unique zero of the system.
"""
struct SolutionCertificate <: AbstractSolutionCertificate
    solution_candidate::Vector{ComplexF64}
    certified::Bool
    real::Bool
    complex::Bool
    index::Int
    prec::Int
    # Enclosure of the unique zero. A `0×0` sentinel marks an uncertified
    # certificate; the public accessors return `nothing` in that case. Keeping
    # the field concrete (rather than `Union{Nothing,AcbMatrix}`) preserves the
    # fully-typed struct layout.
    I::AcbMatrix
    # Double-precision midpoint of the certified interval, the best available
    # estimate of the true solution (empty when uncertified).
    solution::Vector{ComplexF64}
end

"""
    ExtendedSolutionCertificate

Like [`SolutionCertificate`](@ref) but also stores the Krawczyk operator data
(`I`, `I′` with `I′ ⊊ I`, the refined point `x̃`, and the approximate inverse `Y`).
"""
struct ExtendedSolutionCertificate <: AbstractSolutionCertificate
    solution_candidate::Vector{ComplexF64}
    certified::Bool
    real::Bool
    complex::Bool
    index::Int
    prec::Int
    # `0×0` sentinels mark an uncertified certificate (see `SolutionCertificate`).
    I::AcbMatrix
    I′::AcbMatrix
    x̃::AcbMatrix
    Y::AcbMatrix
    solution::Vector{ComplexF64}
end

# Sentinel enclosure marking an uncertified certificate.
_empty_acb() = AcbMatrix(0, 0)
_is_empty_acb(A::AcbMatrix) = size(A, 1) == 0

"""
    solution_candidate(certificate::AbstractSolutionCertificate)

Return the provided solution candidate.
"""
solution_candidate(C::AbstractSolutionCertificate) = C.solution_candidate

"""
    is_certified(certificate::AbstractSolutionCertificate)

Return `true` if `certificate` proves that its solution interval contains a
unique zero.
"""
is_certified(C::AbstractSolutionCertificate)::Bool = C.certified

"""
    is_real(certificate::AbstractSolutionCertificate)

Return `true` if `certificate` certifies that its solution interval contains a
true real zero. A `false` result does not prove the zero is non-real.
"""
is_real(C::AbstractSolutionCertificate)::Bool = C.real

"""
    is_complex(certificate::AbstractSolutionCertificate)

Return `true` if `certificate` certifies that its solution interval contains a
non-real complex zero.
"""
is_complex(C::AbstractSolutionCertificate)::Bool = C.complex

"""
    is_positive(certificate::AbstractSolutionCertificate)

Return `true` if the certified unique zero is real and positive in every coordinate.
"""
function is_positive(C::AbstractSolutionCertificate)::Bool
    (is_certified(C) && is_real(C)) || return false
    I = C.I
    return all(i -> Arblib.is_positive(real(Arblib.ref(I, i, 1))), 1:size(I, 1))
end

"""
    is_positive(certificate::AbstractSolutionCertificate, i::Integer)

Return `true` if the `i`-th coordinate of the certified unique zero is real and positive.
"""
function is_positive(C::AbstractSolutionCertificate, i::Integer)::Bool
    (is_certified(C) && is_real(C)) || return false
    return Arblib.is_positive(real(Arblib.ref(C.I, i, 1)))
end

"""
    certified_solution_interval(certificate::AbstractSolutionCertificate)

Return the `Arblib.AcbMatrix` of complex intervals containing a unique zero, or
`nothing` if `is_certified(certificate)` is `false`.
"""
certified_solution_interval(C::AbstractSolutionCertificate) = is_certified(C) ? C.I : nothing

"""
    certified_solution_interval_after_krawczyk(certificate::ExtendedSolutionCertificate)

Return the enclosure `I′` obtained by applying the Krawczyk operator to the
certified interval, or `nothing` if the certificate is not certified.
"""
certified_solution_interval_after_krawczyk(C::ExtendedSolutionCertificate) =
    is_certified(C) ? C.I′ : nothing

"""
    certificate_index(certificate::AbstractSolutionCertificate)

Return the index of the solution candidate this certificate was produced from.
"""
certificate_index(C::AbstractSolutionCertificate)::Int = C.index

"""
    precision(certificate::AbstractSolutionCertificate)

Return the maximal bit precision used to produce this certificate.
"""
Base.precision(C::AbstractSolutionCertificate)::Int = C.prec

"""
    solution_approximation(certificate::AbstractSolutionCertificate)

Return the midpoint of the certified interval as a `Vector{ComplexF64}`, or
`nothing` if the certificate is not certified.
"""
solution_approximation(C::AbstractSolutionCertificate) = is_certified(C) ? C.solution : nothing

"""
    krawczyk_operator_parameters(cert::ExtendedSolutionCertificate)

Return `(x = x̃, Y = Y)`, the Krawczyk operator parameters.
"""
krawczyk_operator_parameters(C::ExtendedSolutionCertificate) = (x = C.x̃, Y = C.Y)

function Base.show(
        io::IO,
        cert::AbstractSolutionCertificate;
        digits::Int = 16,
        more::Bool = false,
    )
    println(io, "SolutionCertificate:")
    println(io, "solution_candidate = [")
    for z in solution_candidate(cert)
        println(io, "  ", z, ",")
    end
    println(io, "]")
    print(io, "is_certified = ", is_certified(cert))
    if !isnothing(certified_solution_interval(cert))
        println(io)
        println(io, "certified_solution_interval = [")
        for z in certified_solution_interval(cert)
            println(io, "  ", string(z; digits = digits, more = more), ",")
        end
        println(io, "]")
        println(io, "precision = ", cert.prec)
        print(io, "is_real = ", is_real(cert))
    end
    if cert.index != 0
        println(io)
        print(io, "index = ", cert.index)
    end
    return
end

# ─────────────────────────────────────────────────────────────────────────────
# CertificationResult
# ─────────────────────────────────────────────────────────────────────────────

"""
    CertificationResult

The result of [`certify`](@ref) for multiple solutions. Holds the individual
[`SolutionCertificate`](@ref)s and groups of certificates that share a true solution.
"""
struct CertificationResult{C <: AbstractSolutionCertificate}
    certificates::Vector{C}
    duplicates::Vector{Vector{Int}}
    # The instruction sequence (straight-line program) of the certified system,
    # used by `show_straight_line_program`.
    slp::InstructionSequence
end

"""
    certificates(R::CertificationResult)

Return the stored [`SolutionCertificate`](@ref)s.
"""
certificates(R::CertificationResult) = R.certificates

"""
    distinct_certificates(R::CertificationResult)

Return one certificate per distinct certified solution interval.
"""
function distinct_certificates(R::CertificationResult)
    cs = certificates(R)
    isempty(R.duplicates) && return cs
    keep = trues(length(cs))
    for d in R.duplicates, k in 2:length(d)
        keep[d[k]] = false
    end
    return cs[keep]
end

"""
    distinct_solutions(R::CertificationResult)

Return the midpoint approximations of the distinct certified solution intervals.
"""
distinct_solutions(R::CertificationResult) = solution_approximation.(distinct_certificates(R))

"""
    ncertified(R::CertificationResult)

Return the number of certified solutions.
"""
ncertified(R::CertificationResult)::Int = count(is_certified, R.certificates)

"""
    nreal_certified(R::CertificationResult)

Return the number of certified real solutions.
"""
nreal_certified(R::CertificationResult)::Int =
    count(r -> is_certified(r) && is_real(r), R.certificates)

"""
    ncomplex_certified(R::CertificationResult)

Return the number of certified non-real complex solutions.
"""
ncomplex_certified(R::CertificationResult)::Int =
    count(r -> is_certified(r) && is_complex(r), R.certificates)

"""
    ndistinct_certified(R::CertificationResult)

Return the number of distinct certified solutions.
"""
function ndistinct_certified(R::CertificationResult)::Int
    ncert = ncertified(R)
    isempty(R.duplicates) && return ncert
    return ncert - sum(length, R.duplicates) + length(R.duplicates)
end

"""
    ndistinct_real_certified(R::CertificationResult)

Return the number of distinct certified real solutions.
"""
function ndistinct_real_certified(R::CertificationResult)::Int
    ncert = nreal_certified(R)
    isempty(R.duplicates) && return ncert
    return ncert - sum(R.duplicates) do dup
        is_real(R.certificates[dup[1]]) ? length(dup) - 1 : 0
    end
end

"""
    ndistinct_complex_certified(R::CertificationResult)

Return the number of distinct certified non-real complex solutions.
"""
function ndistinct_complex_certified(R::CertificationResult)::Int
    ncert = ncomplex_certified(R)
    isempty(R.duplicates) && return ncert
    return ncert - sum(R.duplicates) do dup
        is_complex(R.certificates[dup[1]]) ? length(dup) - 1 : 0
    end
end

"""
    show_straight_line_program([io::IO], R::CertificationResult)

Print the straight-line program (instruction sequence) of the certified system,
i.e. the interpreter tape that `certify` evaluated.
"""
show_straight_line_program(R::CertificationResult) = show_straight_line_program(stdout, R)
function show_straight_line_program(io::IO, R::CertificationResult)
    seq = R.slp
    println(io, "Straight-line program (", length(seq), " instructions):")
    for instr in seq.instructions
        op = instruction_op(instr)
        op == OpType.OP_STOP && continue
        args = String[]
        for k in 1:arity(op)
            v = instr.input[k]
            push!(args, should_use_index_not_reference(op, k) ? string(v) : string("#", v))
        end
        println(io, "  #", instruction_output(instr), " = ", op_call(op), "(", join(args, ", "), ")")
    end
    return
end

function Base.show(io::IO, R::CertificationResult)
    println(io, "CertificationResult")
    println(io, "===================")
    println(io, "• $(length(R.certificates)) solution candidates given")
    ncert = ncertified(R)
    print(io, "• $ncert certified solution intervals")
    nreal = nreal_certified(R)
    ncomplex = ncomplex_certified(R)
    print(io, " ($nreal real, $ncomplex complex")
    if nreal + ncomplex < ncert
        println(io, ", $(ncert - (nreal + ncomplex)) undecided)")
    else
        println(io, ")")
    end
    ndist = ndistinct_certified(R)
    print(io, "• $ndist distinct certified solution intervals")
    ndist_real = ndistinct_real_certified(R)
    ndist_complex = ndistinct_complex_certified(R)
    print(io, " ($ndist_real real, $ndist_complex complex")
    if ndist_real + ndist_complex < ndist
        print(io, ", $(ndist - (ndist_real + ndist_complex)) undecided)")
    else
        print(io, ")")
    end
    return
end

"""
    save(filename, R::CertificationResult)

Write a text representation of the certification result to disk.
"""
function save(filename, R::CertificationResult)
    open(filename, "w") do f
        println(f, "## Summary")
        show(f, R)
        println(f, "\n\n## Certificates")
        for cert in certificates(R)
            show(f, cert)
            println(f)
        end
    end
    return filename
end

# ─────────────────────────────────────────────────────────────────────────────
# Distinct-solution deduplication (interval tree keyed by squared distance)
# ─────────────────────────────────────────────────────────────────────────────

function squared_distance_interval(
        cert::AbstractSolutionCertificate,
        reference_point::Vector{ComplexF64},
    )
    a, b = Arblib.Arf(; prec = 53), Arblib.Arf(; prec = 53)
    I = cert.I::AcbMatrix
    n = length(reference_point)
    d = zero(Interval{Float64})
    for i in 1:n
        yᵢ = IComplexF64(Arblib.ref(I, i, 1), a, b)
        d +=
            sqr(real(yᵢ) - real(reference_point[i])) +
            sqr(imag(yᵢ) - imag(reference_point[i]))
    end
    return IntervalTrees.Interval(d.lo, d.hi)
end
function squared_distance_interval(
        candidate::AbstractVector{ComplexF64},
        reference_point::Vector{ComplexF64},
    )
    n = length(candidate)
    d = zero(Interval{Float64})
    for i in 1:n
        yᵢ = IComplexF64(candidate[i])
        d +=
            sqr(real(yᵢ) - real(reference_point[i])) +
            sqr(imag(yᵢ) - imag(reference_point[i]))
    end
    return IntervalTrees.Interval(d.lo, d.hi)
end

struct DistinctSolutionCertificates{S <: AbstractSolutionCertificate}
    reference_point::Vector{ComplexF64}
    distinct_tree::IntervalTrees.IntervalMap{Float64, S}
    acb_solution_candidate::AcbMatrix
end

function DistinctSolutionCertificates(
        reference_point::Vector{ComplexF64};
        extended_certificate::Bool = false,
    )
    S = extended_certificate ? ExtendedSolutionCertificate : SolutionCertificate
    return DistinctSolutionCertificates(
        reference_point,
        IntervalTrees.IntervalMap{Float64, S}(),
        AcbMatrix(length(reference_point), 1),
    )
end
DistinctSolutionCertificates(dim::Integer; kwargs...) =
    DistinctSolutionCertificates(randn(ComplexF64, dim); kwargs...)

Base.length(d::DistinctSolutionCertificates) = length(d.distinct_tree)
Base.show(io::IO, d::DistinctSolutionCertificates) =
    print(io, "DistinctSolutionCertificates with $(length(d)) certificates")

"""
    add_certificate!(distinct_sols, cert)

Insert `cert` into the interval tree unless an existing certificate's interval
overlaps it (a duplicate). Returns `(is_distinct, certificate)`.
"""
function add_certificate!(
        distinct_sols::DistinctSolutionCertificates,
        cert::AbstractSolutionCertificate,
    )
    d = squared_distance_interval(cert, distinct_sols.reference_point)
    for match in intersect(distinct_sols.distinct_tree, d)
        certᵢ = IntervalTrees.value(match)
        if Bool(Arblib.overlaps(cert.I, certᵢ.I))
            return (false, certᵢ)
        end
    end
    distinct_sols.distinct_tree[d] = cert
    return (true, cert)
end

"""
    is_solution_candidate_guaranteed_duplicate(distinct_sols, s)

Return `true` when the point `s` is provably contained in the interval of an
already-stored certificate, so it certifies a solution already accounted for.
Used to skip certifying candidates that are guaranteed duplicates.
"""
function is_solution_candidate_guaranteed_duplicate(
        distinct_sols::DistinctSolutionCertificates,
        s::AbstractVector{ComplexF64},
    )
    d = squared_distance_interval(s, distinct_sols.reference_point)
    assigned = false
    for match in intersect(distinct_sols.distinct_tree, d)
        certᵢ = IntervalTrees.value(match)
        if !assigned
            for (i, xᵢ) in enumerate(s)
                distinct_sols.acb_solution_candidate[i] = xᵢ
            end
            assigned = true
        end
        if Bool(Arblib.contains(certᵢ.I, distinct_sols.acb_solution_candidate))
            return true
        end
    end
    return false
end

# ─────────────────────────────────────────────────────────────────────────────
# Certification parameters
# ─────────────────────────────────────────────────────────────────────────────

struct CertificationParameters
    params::Vector{ComplexF64}
    interval_params::Vector{IComplexF64}
    arb_interval_params::AcbRefVector
end

function CertificationParameters(p::AbstractVector; prec::Int = 256)
    arb_ip = AcbRefVector(length(p); prec = prec)
    for (i, pᵢ) in enumerate(p)
        arb_ip[i][] = ComplexF64(pᵢ)
    end
    return CertificationParameters(
        convert(Vector{ComplexF64}, p),
        convert(Vector{IComplexF64}, p),
        arb_ip,
    )
end

certification_parameters(p::AbstractVector; prec::Int = 256) =
    CertificationParameters(p; prec = prec)
certification_parameters(::Nothing; prec::Int = 256) = nothing

complexF64_params(C::CertificationParameters) = C.params
complexF64_interval_params(C::CertificationParameters) = C.interval_params
arb_interval_params(C::CertificationParameters) = C.arb_interval_params
complexF64_params(::Nothing) = ComplexF64[]
complexF64_interval_params(::Nothing) = IComplexF64[]
arb_interval_params(::Nothing) = nothing

# ─────────────────────────────────────────────────────────────────────────────
# Arb (arbitrary-precision) certification cache
# ─────────────────────────────────────────────────────────────────────────────

"""
    AcbCertCache(F::System; prec = 128)

Pre-allocated Arb state for the extended-precision Krawczyk fallback. The
working precision grows during certification, so the buffers (and the tape views
inside the interpreters) are re-created at higher precision via
[`set_arb_precision!`](@ref); this is why the struct is mutable and the buffers
are not `const`.
"""
mutable struct AcbCertCache
    prec::Int
    const eval_interpreter::AcbInterpreter
    const jac_interpreter::AcbInterpreter
    C::AcbRefMatrix
    r₀::AcbRefMatrix
    Δx₀::AcbRefMatrix
    x̃₀::AcbRefMatrix
    x₀::AcbRefMatrix
    x₁::AcbRefMatrix
    J_x₀::AcbRefMatrix
    M::AcbRefMatrix
    δx::AcbRefMatrix
    const mag::Mag
end

function AcbCertCache(
        seq_eval::InstructionSequence, seq_jac::InstructionSequence, m::Int; prec::Int = 128,
    )
    eval_interpreter = AcbInterpreter(seq_eval; prec = prec)
    jac_interpreter = AcbInterpreter(seq_jac; prec = prec)
    return AcbCertCache(
        prec,
        eval_interpreter,
        jac_interpreter,
        AcbRefMatrix(m, m; prec = prec),
        AcbRefMatrix(m, 1; prec = prec),
        AcbRefMatrix(m, 1; prec = prec),
        AcbRefMatrix(m, 1; prec = prec),
        AcbRefMatrix(m, 1; prec = prec),
        AcbRefMatrix(m, 1; prec = prec),
        AcbRefMatrix(m, m; prec = prec),
        AcbRefMatrix(m, m; prec = prec),
        AcbRefMatrix(m, 1; prec = prec),
        Mag(),
    )
end
AcbCertCache(F::System; prec::Int = 128) =
    AcbCertCache(F._interp_f64.sequence, F._interp_jac.sequence, first(size(F)); prec = prec)

# Re-view an `AcbRefMatrix` at a new working precision, sharing the underlying data.
_setprec(A::AcbRefMatrix, p::Int) = Base.setprecision(A, p)

"""
    set_arb_precision!(cache::AcbCertCache, p::Int)

Raise the working precision of every Arb buffer and interpreter tape to `p`.
"""
function set_arb_precision!(cache::AcbCertCache, p::Int)
    cache.prec == p && return cache
    cache.prec = p
    cache.C = _setprec(cache.C, p)
    cache.r₀ = _setprec(cache.r₀, p)
    cache.Δx₀ = _setprec(cache.Δx₀, p)
    cache.x̃₀ = _setprec(cache.x̃₀, p)
    cache.x₀ = _setprec(cache.x₀, p)
    cache.x₁ = _setprec(cache.x₁, p)
    cache.J_x₀ = _setprec(cache.J_x₀, p)
    cache.M = _setprec(cache.M, p)
    cache.δx = _setprec(cache.δx, p)
    setprecision!(cache.eval_interpreter, p)
    setprecision!(cache.jac_interpreter, p)
    return cache
end

# ─────────────────────────────────────────────────────────────────────────────
# Certification cache
# ─────────────────────────────────────────────────────────────────────────────

"""
    CertificationCache(F::System)

Pre-allocated data structures for [`certify`](@ref). The Float64 Krawczyk path
uses interval interpreters built from `F`'s cached instruction sequences.

**Mutable justification:** the arbitrary-precision `arb` fallback state (~KBs of
Arb buffers) is only needed when the Float64 path fails, which is the exception,
not the rule. It is therefore left uninitialized and built lazily on first use
via [`_arb`](@ref); this is the one non-`const` field. Everything else is
`const`. The stored `seq_eval`/`seq_jac`/`m` let `_arb` build the fallback
without keeping a reference to `F`.
"""
mutable struct CertificationCache
    # Private evaluator clone so refinement newton never touches `F`'s shared
    # interpreter tapes; this makes the cache safe to use from its own task.
    const system_evaluator::SystemEvaluator
    const eval_interpreter_F64::Interpreter{Vector{IComplexF64}}
    const jac_interpreter_F64::Interpreter{Vector{IComplexF64}}
    const jac_interpreter_C64::Interpreter{Vector{ComplexF64}}
    const newton_cache::NewtonCache
    # Krawczyk Float64 buffers
    const J_C64::Matrix{ComplexF64}
    const C_C64::Matrix{ComplexF64}   # in-place approximate inverse of J_C64
    const u_C64::Vector{ComplexF64}
    const r₀::Vector{IComplexF64}
    const Δx₀::Vector{IComplexF64}
    const ix̃₀::Vector{IComplexF64}
    const u_scratch::Vector{IComplexF64}
    const Jx₀::Matrix{IComplexF64}
    const M::Matrix{IComplexF64}
    const δx::Vector{IComplexF64}
    const x₀::Vector{IComplexF64}
    const x₁::Vector{IComplexF64}
    # Instruction sequences + size, kept so `arb` can be built lazily.
    const seq_eval::InstructionSequence
    const seq_jac::InstructionSequence
    const m::Int
    # Arbitrary-precision fallback state, initialized lazily (see `_arb`).
    arb::AcbCertCache

    function CertificationCache(
            system_evaluator, eval_F64, jac_F64, jac_C64, newton_cache,
            J_C64, C_C64, u_C64, r₀, Δx₀, ix̃₀, u_scratch, Jx₀, M, δx, x₀, x₁,
            seq_eval, seq_jac, m,
        )
        # `arb` is intentionally left undefined here (lazy).
        return new(
            system_evaluator, eval_F64, jac_F64, jac_C64, newton_cache,
            J_C64, C_C64, u_C64, r₀, Δx₀, ix̃₀, u_scratch, Jx₀, M, δx, x₀, x₁,
            seq_eval, seq_jac, m,
        )
    end
end

function CertificationCache(F::System)
    m, n = size(F)
    m == n || throw(ArgumentError("We can only certify solutions to square systems."))
    seq_eval = F._interp_f64.sequence
    seq_jac = F._interp_jac.sequence
    return CertificationCache(
        _clone_system_evaluator(F),
        Interpreter(Vector{IComplexF64}, seq_eval),
        Interpreter(Vector{IComplexF64}, seq_jac),
        Interpreter(Vector{ComplexF64}, seq_jac),
        NewtonCache(F),
        zeros(ComplexF64, m, m),
        zeros(ComplexF64, m, m),
        zeros(ComplexF64, m),
        zeros(IComplexF64, m),
        zeros(IComplexF64, m),
        zeros(IComplexF64, m),
        zeros(IComplexF64, m),
        zeros(IComplexF64, m, m),
        zeros(IComplexF64, m, m),
        zeros(IComplexF64, m),
        zeros(IComplexF64, m),
        zeros(IComplexF64, m),
        seq_eval,
        seq_jac,
        m,
    )
end

"""
    _arb(cache::CertificationCache) -> AcbCertCache

Return the arbitrary-precision fallback state, building it on first use. Each
`CertificationCache` is used by a single task, so this lazy initialization is
not shared across threads.
"""
function _arb(cache::CertificationCache)::AcbCertCache
    isdefined(cache, :arb) || (cache.arb = AcbCertCache(cache.seq_eval, cache.seq_jac, cache.m))
    return cache.arb
end

# ─────────────────────────────────────────────────────────────────────────────
# Krawczyk operator (Float64)
# ─────────────────────────────────────────────────────────────────────────────

# Matrix multiply via muladd; faster than generic `*` for interval elements.
function sqr_mul!(C::AbstractVecOrMat{<:Number}, A::AbstractMatrix{<:Number}, B::AbstractVecOrMat{<:Number})
    n = size(A, 1)
    C .= zero(eltype(C))
    @inbounds for j in 1:size(B, 2)
        for k in 1:n
            bkj = B[k, j]
            for i in 1:n
                C[i, j] = muladd(A[i, k], bkj, C[i, j])
            end
        end
    end
    return C
end

function all2(f::F, a::AbstractVector, b::AbstractVector) where {F}
    @inbounds for i in eachindex(a, b)
        f(a[i], b[i]) || return false
    end
    return true
end

# Evaluate an interval interpreter with correct parameter arity.
Base.@propagate_inbounds function _exec_eval!(
        u, I::Interpreter, x, p::AbstractVector,
    )
    return isempty(I.sequence.parameters_range) ? execute!(u, I, x) : execute!(u, I, x, p)
end
Base.@propagate_inbounds function _exec_jac!(
        u, U, I::Interpreter, x, p::AbstractVector,
    )
    return isempty(I.sequence.parameters_range) ? execute!(u, U, I, x) :
        execute!(u, U, I, x, p)
end

"""
    ε_inflation_krawczyk(x̃₀, p, C, cert_cache)

Apply the ε-inflation Krawczyk operator around the refined point `x̃₀` with
approximate inverse `C`. Returns `(certified, x₁, x₀, is_real)`.
"""
function ε_inflation_krawczyk(
        x̃₀::AbstractVector,
        p::Union{Nothing, CertificationParameters},
        C::Matrix{ComplexF64},
        cert_cache::CertificationCache,
    )
    r₀ = cert_cache.r₀
    Δx₀ = cert_cache.Δx₀
    ix̃₀ = cert_cache.ix̃₀
    J_x₀ = cert_cache.Jx₀
    M = cert_cache.M
    δx = cert_cache.δx
    x₀ = cert_cache.x₀
    x₁ = cert_cache.x₁
    ip = complexF64_interval_params(p)

    ix̃₀ .= IComplexF64.(x̃₀)
    # r₀ = F(ix̃₀)
    _exec_eval!(r₀, cert_cache.eval_interpreter_F64, ix̃₀, ip)
    # Δx₀ = C * r₀
    sqr_mul!(Δx₀, C, r₀)

    # ε-inflation with a per-coordinate εᵢ (matches the weighted-norm strategy).
    @inbounds for i in eachindex(x̃₀)
        x̃₀_i = x̃₀[i]
        εᵢ = 512 * max(mag(Δx₀[i]), eps())
        x₀[i] = complex(
            Interval(real(x̃₀_i) - εᵢ, real(x̃₀_i) + εᵢ),
            Interval(imag(x̃₀_i) - εᵢ, imag(x̃₀_i) + εᵢ),
        )
    end

    _exec_jac!(cert_cache.u_scratch, J_x₀, cert_cache.jac_interpreter_F64, x₀, ip)

    # x₁ = (x̃₀ - Δx₀) - (C * J(x₀) - I) * (x₀ - x̃₀)
    sqr_mul!(M, C, J_x₀)
    @inbounds for i in 1:size(M, 1)
        M[i, i] -= 1
    end
    # Necessary condition ‖M‖ < 1/√2, lower-bounded by 0.7071.
    if inf_norm_bound(M) < 0.7071
        @inbounds for i in eachindex(x₀, x̃₀)
            δx[i] = x₀[i] - x̃₀[i]
        end
        sqr_mul!(x₁, M, δx)
        @inbounds for i in eachindex(x₁)
            x₁[i] = (x̃₀[i] - Δx₀[i]) - x₁[i]
        end
        certified = all2(isinterior, x₁, x₀)
        is_real =
            certified ? all2((a, b) -> isinterior(conj(a), b), x₁, x₀) : false
    else
        certified = false
        is_real = false
    end
    return certified, x₁, x₀, is_real
end

# ─────────────────────────────────────────────────────────────────────────────
# Certificate builders
# ─────────────────────────────────────────────────────────────────────────────
#
# The certificate type (`SolutionCertificate` vs `ExtendedSolutionCertificate`)
# is threaded through the whole pipeline as a type parameter so that every
# certification function returns a *concrete* type. Dispatching the builders on
# `::Type{CertT}` replaces the former runtime `extended_certificate::Bool`
# branch, which forced a `Union` return and broke inference.

# Build a certified Float64 certificate from Krawczyk data at 53-bit precision.
function _float64_certificate(
        ::Type{SolutionCertificate},
        candidate::Vector{ComplexF64}, is_real::Bool, is_complex_sol::Bool, index::Int,
        x₀::Vector{IComplexF64}, x₁::Vector{IComplexF64}, x̃₀::AbstractVector,
        C::Matrix{ComplexF64},
    )
    return SolutionCertificate(
        candidate, true, is_real, is_complex_sol, index, 53,
        AcbMatrix(x₁; prec = 53), collect(mid.(x₁)),
    )
end
function _float64_certificate(
        ::Type{ExtendedSolutionCertificate},
        candidate::Vector{ComplexF64}, is_real::Bool, is_complex_sol::Bool, index::Int,
        x₀::Vector{IComplexF64}, x₁::Vector{IComplexF64}, x̃₀::AbstractVector,
        C::Matrix{ComplexF64},
    )
    return ExtendedSolutionCertificate(
        candidate, true, is_real, is_complex_sol, index, 53,
        AcbMatrix(x₀; prec = 53), AcbMatrix(x₁; prec = 53),
        AcbMatrix(x̃₀; prec = 53), AcbMatrix(C; prec = 53), collect(mid.(x₁)),
    )
end

# Build a certified Arb certificate. `x̃₀_ref`/`C_ref` are only copied into an
# `AcbMatrix` for the extended variant, so the plain variant pays nothing.
function _arb_certificate(
        ::Type{SolutionCertificate},
        candidate::Vector{ComplexF64}, is_real::Bool, is_complex_sol::Bool, index::Int,
        prec::Int, x₀::AcbMatrix, x₁::AcbMatrix,
        x̃₀_ref::AcbRefMatrix, C_ref::AcbRefMatrix, sol::Vector{ComplexF64},
    )
    return SolutionCertificate(
        candidate, true, is_real, is_complex_sol, index, prec, x₁, sol,
    )
end
function _arb_certificate(
        ::Type{ExtendedSolutionCertificate},
        candidate::Vector{ComplexF64}, is_real::Bool, is_complex_sol::Bool, index::Int,
        prec::Int, x₀::AcbMatrix, x₁::AcbMatrix,
        x̃₀_ref::AcbRefMatrix, C_ref::AcbRefMatrix, sol::Vector{ComplexF64},
    )
    return ExtendedSolutionCertificate(
        candidate, true, is_real, is_complex_sol, index, prec,
        x₀, x₁, AcbMatrix(x̃₀_ref), AcbMatrix(C_ref), sol,
    )
end

# Build an uncertified (failed) certificate with sentinel enclosures.
_uncertified_certificate(::Type{SolutionCertificate}, candidate::Vector{ComplexF64}, index::Int, prec::Int) =
    SolutionCertificate(candidate, false, false, false, index, prec, _empty_acb(), ComplexF64[])
_uncertified_certificate(::Type{ExtendedSolutionCertificate}, candidate::Vector{ComplexF64}, index::Int, prec::Int) =
    ExtendedSolutionCertificate(
    candidate, false, false, false, index, prec,
    _empty_acb(), _empty_acb(), _empty_acb(), _empty_acb(), ComplexF64[],
)

# ─────────────────────────────────────────────────────────────────────────────
# Per-solution certification
# ─────────────────────────────────────────────────────────────────────────────

function certify_solution(
        F::System,
        solution_candidate::AbstractVector,
        cert_params::Union{Nothing, CertificationParameters},
        cert_cache::CertificationCache,
        index::Int,
        is_real_system::Bool,
        ::Type{CertT};
        max_precision::Int = 256,
        refine_solution::Bool = true,
    ) where {CertT <: AbstractSolutionCertificate}
    candidate = convert(Vector{ComplexF64}, solution_candidate)
    params_c64 = complexF64_params(cert_params)

    # Refine to machine precision. Use the cache's private evaluator clone (not
    # `F.evaluator`) so concurrent certification tasks never share tape state.
    if refine_solution
        res = _newton(
            cert_cache.system_evaluator,
            cert_cache.newton_cache,
            candidate,
            FSVec{ComplexF64}(params_c64),
            0.0,          # atol
            8 * eps(),    # rtol
            8,            # max_iters
            true,         # extended_precision
            1.0,          # contraction_factor
            typemax(Int), # min_contraction_iters
            Inf,          # max_abs_norm_first_update
            Inf,          # max_rel_norm_first_update
        )
        x̃₀ = solution(res)
    else
        x̃₀ = candidate
    end

    # C ≈ J(x̃₀)⁻¹, computed in place into the pre-allocated `C_C64` buffer
    # (`inv(J)` would allocate a fresh matrix per solution).
    _exec_jac!(cert_cache.u_C64, cert_cache.J_C64, cert_cache.jac_interpreter_C64, x̃₀, params_c64)
    C = cert_cache.C_C64
    copyto!(C, cert_cache.J_C64)
    LinearAlgebra.inv!(LinearAlgebra.lu!(C))

    certified, x₁, x₀, is_real = ε_inflation_krawczyk(x̃₀, cert_params, C, cert_cache)
    is_real_system || (is_real = false)

    if certified
        is_complex_sol = any(xi -> !(0.0 in imag(xi)), x₁)
        is_complex_sol && (is_real = false)
        return _float64_certificate(
            CertT, candidate, is_real, is_complex_sol, index, x₀, x₁, x̃₀, C,
        )
    end

    # Float64 certification failed: retry in extended precision.
    return extended_prec_certify_solution(
        F, candidate, x̃₀, C, cert_params, cert_cache, index, is_real_system, CertT;
        max_precision = max_precision,
    )
end

# Whether every coefficient of every polynomial of `F` is real.
function _is_real_system(F::System)::Bool
    for p in polynomials(F)
        for c in MP.coefficients(p)
            iszero(imag(ComplexF64(c))) || return false
        end
    end
    return true
end

# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

# Boundary: resolve the runtime `extended_certificate` flag to a concrete
# certificate type once, then dispatch to the type-stable implementation.
function _certify(
        F::System,
        solution_candidates::AbstractVector{<:AbstractVector{<:Number}},
        p::Union{Nothing, CertificationParameters},
        cache::CertificationCache;
        extended_certificate::Bool = false,
        show_progress::Bool = true,
        threading::Bool = true,
        max_precision::Int = 256,
        refine_solution::Bool = true,
    )
    CertT = extended_certificate ? ExtendedSolutionCertificate : SolutionCertificate
    return _certify_impl(
        F, solution_candidates, p, cache, CertT;
        show_progress = show_progress, threading = threading,
        max_precision = max_precision, refine_solution = refine_solution,
    )
end

function _certify_impl(
        F::System,
        solution_candidates::AbstractVector{<:AbstractVector{<:Number}},
        p::Union{Nothing, CertificationParameters},
        cache::CertificationCache,
        ::Type{CertT};
        show_progress::Bool = true,
        threading::Bool = true,
        max_precision::Int = 256,
        refine_solution::Bool = true,
    ) where {CertT <: AbstractSolutionCertificate}
    m, n = size(F)
    m == n || throw(ArgumentError("We can only certify solutions to square systems."))
    if isnothing(p) && nparameters(F) > 0
        throw(ArgumentError("The given system expects parameters but none are given."))
    end

    N = length(solution_candidates)
    is_real_system = _is_real_system(F)
    certs = Vector{CertT}(undef, N)

    progress = make_progress(N, show_progress; desc = "Certifying $N solutions... ")

    if threading && Threads.nthreads() > 1 && N > 1
        plock = ReentrantLock()
        # One cache per task (certify_solution mutates its buffers). Reuse the
        # caller-provided `cache` as one of them rather than discarding it, and
        # build only the remaining `nt - 1`.
        nt = min(Threads.nthreads(), N)
        pool = Channel{CertificationCache}(nt)
        put!(pool, cache)
        for _ in 2:nt
            put!(pool, CertificationCache(F))
        end
        @tasks for i in 1:N
            @set ntasks = nt
            @local task_cache = take!(pool)
            certs[i] = certify_solution(
                F, solution_candidates[i], p, task_cache, i, is_real_system, CertT;
                max_precision = max_precision,
                refine_solution = refine_solution,
            )
            progress === nothing || (@lock plock ProgressMeter.next!(progress))
        end
    else
        for i in 1:N
            certs[i] = certify_solution(
                F, solution_candidates[i], p, cache, i, is_real_system, CertT;
                max_precision = max_precision,
                refine_solution = refine_solution,
            )
            progress === nothing || ProgressMeter.next!(progress)
        end
    end

    # Group certificates that certify the same true solution.
    distinct_sols = DistinctSolutionCertificates(
        n; extended_certificate = (CertT === ExtendedSolutionCertificate),
    )
    duplicates_dict = Dict{Int, Vector{Int}}()
    for cert in certs
        is_certified(cert) || continue
        is_distinct, distinct_cert = add_certificate!(distinct_sols, cert)
        is_distinct && continue
        key = distinct_cert.index
        if haskey(duplicates_dict, key)
            push!(duplicates_dict[key], cert.index)
        else
            duplicates_dict[key] = [key, cert.index]
        end
    end
    duplicates = isempty(duplicates_dict) ? Vector{Int}[] : collect(values(duplicates_dict))

    return CertificationResult(certs, duplicates, F._interp_f64.sequence)
end

# ─────────────────────────────────────────────────────────────────────────────
# Public `certify` entry points
# ─────────────────────────────────────────────────────────────────────────────

"""
    certify(F, solutions, [p]; options...)
    certify(F, result, [p]; options...)

Attempt to certify that the approximate `solutions` correspond to true solutions
of the square polynomial system `F(x; p)` using interval arithmetic and the
Krawczyk method. Returns a [`CertificationResult`](@ref).

## Options
- `show_progress = true`: show a progress bar.
- `max_precision = 256`: maximum bit precision used in the Arb fallback.
- `extended_certificate = false`: also store the Krawczyk operator data.
"""
function certify end

# Canonical entry point. The other overloads normalize their input to a vector
# of solution vectors and delegate here. Keyword arguments are named and
# forwarded explicitly (never splatted) so inference is not blocked.
function certify(
        F::System,
        X::AbstractVector{<:AbstractVector{<:Number}},
        p::Union{Nothing, AbstractArray} = nothing,
        cache::CertificationCache = CertificationCache(F);
        target_parameters::Union{Nothing, AbstractArray} = nothing,
        show_progress::Bool = true,
        threading::Bool = true,
        max_precision::Int = 256,
        refine_solution::Bool = true,
        extended_certificate::Bool = false,
    )
    params = isnothing(p) ? target_parameters : p
    cert_params = certification_parameters(params; prec = max_precision)
    return _certify(
        F, X, cert_params, cache;
        show_progress, threading, max_precision, refine_solution, extended_certificate,
    )
end

function certify(
        F::System,
        x::AbstractVector{<:Number},
        p::Union{Nothing, AbstractArray} = nothing,
        cache::CertificationCache = CertificationCache(F);
        target_parameters::Union{Nothing, AbstractArray} = nothing,
        show_progress::Bool = true,
        threading::Bool = true,
        max_precision::Int = 256,
        refine_solution::Bool = true,
        extended_certificate::Bool = false,
    )
    return certify(
        F, [convert(Vector{ComplexF64}, x)], p, cache;
        target_parameters, show_progress, threading,
        max_precision, refine_solution, extended_certificate,
    )
end

function certify(
        F::System,
        X::Result,
        p::Union{Nothing, AbstractArray} = nothing,
        cache::CertificationCache = CertificationCache(F);
        target_parameters::Union{Nothing, AbstractArray} = nothing,
        show_progress::Bool = true,
        threading::Bool = true,
        max_precision::Int = 256,
        refine_solution::Bool = true,
        extended_certificate::Bool = false,
    )
    return certify(
        F, solutions(X), p, cache;
        target_parameters, show_progress, threading,
        max_precision, refine_solution, extended_certificate,
    )
end

function certify(
        F::System,
        r::PathResult,
        p::Union{Nothing, AbstractArray} = nothing,
        cache::CertificationCache = CertificationCache(F);
        target_parameters::Union{Nothing, AbstractArray} = nothing,
        show_progress::Bool = true,
        threading::Bool = true,
        max_precision::Int = 256,
        refine_solution::Bool = true,
        extended_certificate::Bool = false,
    )
    return certify(
        F, [solution(r)], p, cache;
        target_parameters, show_progress, threading,
        max_precision, refine_solution, extended_certificate,
    )
end

function certify(
        F::System,
        r::AbstractVector{<:PathResult},
        p::Union{Nothing, AbstractArray} = nothing,
        cache::CertificationCache = CertificationCache(F);
        target_parameters::Union{Nothing, AbstractArray} = nothing,
        show_progress::Bool = true,
        threading::Bool = true,
        max_precision::Int = 256,
        refine_solution::Bool = true,
        extended_certificate::Bool = false,
    )
    return certify(
        F, solution.(r), p, cache;
        target_parameters, show_progress, threading,
        max_precision, refine_solution, extended_certificate,
    )
end

function certify(
        F::System,
        X::MonodromyResult,
        cache::CertificationCache = CertificationCache(F);
        show_progress::Bool = true,
        threading::Bool = true,
        max_precision::Int = 256,
        refine_solution::Bool = true,
        extended_certificate::Bool = false,
    )
    return certify(
        F, solutions(X), parameters(X), cache;
        show_progress, threading, max_precision, refine_solution, extended_certificate,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# Streaming distinct-certified-solution accumulator
# ─────────────────────────────────────────────────────────────────────────────

"""
    DistinctCertifiedSolutions

Accumulate distinct certified solutions of a system on the fly. Unlike
[`certify`](@ref), which keeps a certificate per input solution, this keeps only
the distinct certified ones, which is useful when merging several large solution
sets into a single deduplicated set.

The struct is immutable; the interval tree of distinct certificates mutates in
place, guarded by `access_lock` so [`add_solution!`](@ref) is thread-safe.
"""
struct DistinctCertifiedSolutions{
        S <: System,
        P <: Union{Nothing, CertificationParameters},
        C <: AbstractSolutionCertificate,
    }
    system::S
    parameters::P
    cache::CertificationCache
    is_real_system::Bool
    access_lock::ReentrantLock
    # The certificate type `C` records whether this is an extended accumulator,
    # so no separate `extended_certificate` flag is stored.
    distinct::DistinctSolutionCertificates{C}
end

"""
    DistinctCertifiedSolutions(F::System, params; extended_certificate = false, max_precision = 256)

Create an empty accumulator for the (parametric) system `F` at parameters
`params` (`nothing` for a parameter-free system).
"""
function DistinctCertifiedSolutions(
        F::System,
        params;
        extended_certificate::Bool = false,
        max_precision::Int = 256,
    )
    m, n = size(F)
    m == n || throw(ArgumentError("We can only certify solutions to square systems."))
    if isnothing(params) && nparameters(F) > 0
        throw(ArgumentError("The given system expects parameters but none are given."))
    end
    cert_params = certification_parameters(params; prec = max_precision)
    distinct = DistinctSolutionCertificates(n; extended_certificate = extended_certificate)
    return DistinctCertifiedSolutions(
        F,
        cert_params,
        CertificationCache(F),
        _is_real_system(F),
        ReentrantLock(),
        distinct,
    )
end

Base.length(d::DistinctCertifiedSolutions) = length(d.distinct)
Base.show(io::IO, d::DistinctCertifiedSolutions) =
    print(io, "DistinctCertifiedSolutions with ", length(d), " distinct solutions")

"""
    add_solution!(d::DistinctCertifiedSolutions, sol, index = 0; max_precision = 256, refine_solution = true)

Certify `sol` and store it if it is a new distinct certified solution. Returns a
`(added::Bool, status::Symbol)` pair with `status` one of `:duplicate`,
`:certified_distinct`, or `:not_certified`.
"""
add_solution!(
    d::DistinctCertifiedSolutions,
    sol::AbstractVector{<:Number},
    index::Integer = 0;
    max_precision::Int = 256,
    refine_solution::Bool = true,
) = add_solution!(
    d, sol, index, d.cache;
    max_precision = max_precision, refine_solution = refine_solution,
)

function add_solution!(
        d::DistinctCertifiedSolutions{S, P, C},
        sol::AbstractVector{<:Number},
        index::Integer,
        cache::CertificationCache;
        max_precision::Int = 256,
        refine_solution::Bool = true,
    ) where {S, P, C}
    s = convert(Vector{ComplexF64}, sol)
    Base.@lock d.access_lock begin
        if is_solution_candidate_guaranteed_duplicate(d.distinct, s)
            return (false, :duplicate)
        end
    end
    # `C` is the concrete certificate type of this accumulator; pass it so
    # `certify_solution` returns a concrete type instead of a `Union`.
    cert = certify_solution(
        d.system, s, d.parameters, cache, Int(index), d.is_real_system, C;
        max_precision = max_precision,
        refine_solution = refine_solution,
    )
    is_certified(cert) || return (false, :not_certified)
    Base.@lock d.access_lock begin
        added, _ = add_certificate!(d.distinct, cert)
        return added ? (true, :certified_distinct) : (false, :duplicate)
    end
end

"""
    certificates(d::DistinctCertifiedSolutions)

Return the vector of stored distinct solution certificates.
"""
certificates(d::DistinctCertifiedSolutions) =
    collect(Base.values(d.distinct.distinct_tree))

"""
    solutions(d::DistinctCertifiedSolutions)

Return the midpoint approximations of the stored distinct certified solutions.
"""
solutions(d::DistinctCertifiedSolutions) = map(solution_approximation, certificates(d))

"""
    distinct_certified_solutions(F, S, p = nothing; threading = true, show_progress = true, max_precision = 256, refine_solution = true, extended_certificate = false)

Certify the solutions `S` of the (parametric) system `F` and return a
[`DistinctCertifiedSolutions`](@ref) holding only the distinct certified ones.
"""
function distinct_certified_solutions(
        F::System,
        S::AbstractVector{<:AbstractVector{<:Number}},
        p = nothing;
        threading::Bool = true,
        show_progress::Bool = true,
        max_precision::Int = 256,
        refine_solution::Bool = true,
        extended_certificate::Bool = false,
    )
    d = DistinctCertifiedSolutions(
        F, p;
        extended_certificate = extended_certificate, max_precision = max_precision,
    )
    return distinct_certified_solutions!(
        d, S;
        threading = threading, show_progress = show_progress,
        max_precision = max_precision, refine_solution = refine_solution,
    )
end

"""
    distinct_certified_solutions!(d::DistinctCertifiedSolutions, S; threading = true, show_progress = true, max_precision = 256, refine_solution = true)

Add every solution in `S` to `d`, keeping only distinct certified ones.
"""
function distinct_certified_solutions!(
        d::DistinctCertifiedSolutions,
        S::AbstractVector{<:AbstractVector{<:Number}};
        threading::Bool = true,
        show_progress::Bool = true,
        max_precision::Int = 256,
        refine_solution::Bool = true,
    )
    progress = make_progress(
        length(S), show_progress; desc = "Certifying $(length(S)) solutions... ",
    )
    if threading && Threads.nthreads() > 1
        plock = ReentrantLock()
        @tasks for i in eachindex(S)
            @local cache = CertificationCache(d.system)
            add_solution!(
                d, S[i], i, cache;
                max_precision = max_precision, refine_solution = refine_solution,
            )
            progress === nothing || (@lock plock ProgressMeter.next!(progress))
        end
    else
        for (i, sol) in enumerate(S)
            add_solution!(
                d, sol, i;
                max_precision = max_precision, refine_solution = refine_solution,
            )
            progress === nothing || ProgressMeter.next!(progress)
        end
    end
    return d
end
