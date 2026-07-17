## System — compiled polynomial system with cached interpreter pipeline.
#
# The System constructor runs the full pipeline:
#   MP polynomials → SExpr → CSE → InstructionSequence → Interpreters → FunctionWrappers

"""
    System

Compiled, ready-to-evaluate polynomial system. Caches the interpreter pipeline
so users pay the compilation cost once and reuse across multiple `solve` calls.

Construct from MultivariatePolynomials:

    F = System(polys; parameters=[], variables=...)

# Examples
```julia
@polyvar x y
F = System([x^2 + y - 1, x*y - 2])
solve(F)
solve(F, Polyhedral())
```
"""
struct System{P, V}
    polys::FSVec{P}
    parameters::FSVec{V}
    variables::FSVec{V}
    evaluator::SystemEvaluator
    degrees::Vector{Int}
    nvars::Int
    nparams::Int
    variable_groups::Vector{Vector{Int}}
    is_homogeneous::Bool
    support::Vector{Matrix{Int32}}
    coefficients::Vector{Vector{ComplexF64}}
    _interp_f64::Interpreter{Vector{ComplexF64}}
    _interp_df64::Interpreter{Vector{ComplexDF64}}
    _interp_jac::Interpreter{Vector{ComplexF64}}
    _interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}}
    _interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}}
    _interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}}
    compile_mode::CompileMode.T
end

## ── System constructor ──────────────────────────────────────────────────────

function System(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        parameters::AbstractVector,
        variables::AbstractVector,
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::System
    neqs = length(polys)
    nvars = length(variables)
    nparams = length(parameters)
    return _build_compiled_system(polys, variables, parameters, neqs, nvars, nparams, compile)
end

"""
    System(polys; parameters=[], variables=..., compile=CompileMode.INTERPRETED) -> System

Build a `System` from a vector of MultivariatePolynomials polynomials.
Compiles the full interpreter pipeline and caches everything for reuse.

`compile` controls the evaluation backend:
- `CompileMode.INTERPRETED` (default): tape-based interpreter
- `CompileMode.COMPILED`: RuntimeGeneratedFunctions compiled eval + Jacobian, interpreter Taylor
- `CompileMode.COMPILED_ALL`: additionally compiles the Taylor kernels

`INTERPRETED` is the default because it skips runtime code generation, so the
first `solve` starts tracking immediately. The compiled modes trade a
per-system compilation pause for modestly faster solves; worthwhile when
repeatedly solving the same system.
"""
function System(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters = nothing,
        variables = nothing,
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::System
    parameters === nothing && (parameters = _empty_vars(polys))
    variables === nothing && (variables = _effective_variables(polys, parameters))
    return System(polys, parameters, variables, compile)
end


## ── Accessors ────────────────────────────────────────────────────────────────

Base.size(F::System)::Tuple{Int, Int} = size(F.evaluator)
degrees(F::System)::Vector{Int} = F.degrees
nvariables(F::System)::Int = F.nvars
nparameters(F::System)::Int = F.nparams
polynomials(F::System) = F.polys
variables(F::System) = F.variables
parameters(F::System) = F.parameters
variable_groups(F::System) = F.variable_groups
is_homogeneous(F::System)::Bool = F.is_homogeneous
function support_coefficients(F::System)::Tuple{Vector{Matrix{Int32}}, Vector{Vector{ComplexF64}}}
    nparameters(F) == 0 ||
        throw(ArgumentError("support_coefficients(::System) is only defined for parameter-free systems"))
    return F.support, F.coefficients
end

@inline _to_fsvec(xs::AbstractVector{T}) where {T} = FSVec{T}(collect(xs))

function _is_homogeneous(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
    )::Bool
    Base.@nospecialize polys variables
    var_set = Set(variables)
    for poly in polys
        target_degree = nothing
        for term in MP.terms(poly)
            coeff = MP.coefficient(term)
            iszero(coeff) && continue
            mono = MP.monomial(term)
            degree = 0
            for (var, exp) in zip(MP.variables(mono), MP.exponents(mono))
                var in var_set || continue
                degree += exp
            end
            if isnothing(target_degree)
                target_degree = degree
            elseif degree != target_degree
                return false
            end
        end
    end
    return true
end

## ── Polynomial normalization ────────────────────────────────────────────────

function _normalize_polys(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
    )
    Base.@nospecialize polys
    # Only normalize when coefficients are large enough to cause numerical issues.
    # Threshold 1e8: systems with moderate coefficients (e.g. 100) are unchanged,
    # but O(10^19) integer coefficients get scaled to O(1).
    return map(polys) do p
        coeffs = MP.coefficients(p)
        nrm = maximum(c -> Float64(abs(c)), coeffs)
        nrm <= 1.0e8 && return p
        return p / nrm
    end
end

## ── FW-compatible wrapper functions ──────────────────────────────────────────


@noinline function _build_compiled_system(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
        parameters::AbstractVector,
        neqs::Int,
        nvars::Int,
        nparams::Int,
        compile::CompileMode.T,
    )::System
    Base.@nospecialize polys variables parameters
    polys = _normalize_polys(polys)

    supp, coeffs = if nparams == 0
        support_coefficients(polys, variables)
    else
        Vector{Matrix{Int32}}(), Vector{Vector{ComplexF64}}()
    end

    seq_eval = _build_instruction_sequence(polys, variables, parameters, false)
    seq_jac = _build_instruction_sequence(polys, variables, parameters, true)

    interp_f64 = Interpreter(Vector{ComplexF64}, seq_eval)
    interp_df64 = Interpreter(Vector{ComplexDF64}, seq_eval)
    interp_jac = Interpreter(Vector{ComplexF64}, seq_jac)
    interp_t1 = Interpreter(Vector{TruncatedTaylorSeries{2, ComplexF64}}, seq_eval)
    interp_t2 = Interpreter(Vector{TruncatedTaylorSeries{3, ComplexF64}}, seq_eval)
    interp_t3 = Interpreter(Vector{TruncatedTaylorSeries{4, ComplexF64}}, seq_eval)

    degs = Int[MP.maxdegree(p) for p in polys]
    is_homogeneous = _is_homogeneous(polys, variables)

    evaluator = if compile == CompileMode.INTERPRETED
        _build_system_evaluator(
            interp_f64, interp_df64, interp_jac,
            interp_t1, interp_t2, interp_t3,
            neqs, nvars, nparams,
        )
    elseif compile == CompileMode.COMPILED
        _build_compiled_evaluator(
            seq_eval, seq_jac, interp_df64,
            _build_taylor_fws(interp_t1, interp_t2, interp_t3),
            neqs, nvars, nparams,
        )
    else  # CompileMode.COMPILED_ALL
        _build_compiled_evaluator(
            seq_eval, seq_jac, interp_df64,
            _build_taylor_fws(seq_eval),
            neqs, nvars, nparams,
        )
    end

    return System(
        _to_fsvec(polys),
        _to_fsvec(parameters),
        _to_fsvec(variables),
        evaluator, degs, nvars, nparams,
        Vector{Int}[], is_homogeneous,
        supp, coeffs,
        interp_f64, interp_df64, interp_jac,
        interp_t1, interp_t2, interp_t3,
        compile,
    )
end

function _execute_eval_fw!(
        u::AbstractVector, interp::Interpreter, x::AbstractVector, p::AbstractVector,
    )::Nothing
    if isempty(interp.sequence.parameters_range)
        _load_inputs!(interp, x)
    else
        _load_inputs!(interp, x, p)
    end
    _execute_eval!(u, interp)
    return nothing
end

function _execute_jac_fw!(
        u::AbstractVector, U::AbstractMatrix, interp::Interpreter,
        x::AbstractVector, p::AbstractVector,
    )::Nothing
    if isempty(interp.sequence.parameters_range)
        _load_inputs!(interp, x)
    else
        _load_inputs!(interp, x, p)
    end
    _execute_jac!(u, U, interp)
    return nothing
end

function _build_system_evaluator(
        interp_f64::Interpreter{Vector{ComplexF64}},
        interp_df64::Interpreter{Vector{ComplexDF64}},
        interp_jac::Interpreter{Vector{ComplexF64}},
        interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}},
        interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}},
        interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}},
        neqs::Int,
        nvars::Int,
        nparams::Int,
    )::SystemEvaluator
    return SystemEvaluator(
        SysEvalFW((u, x, p) -> (_execute_eval_fw!(u, interp_f64, x, p); nothing)),
        SysEvalDF64FW((u, x, p) -> (_execute_eval_fw!(u, interp_df64, x, p); nothing)),
        SysEvalDF64OutFW((u, x, p) -> (_execute_eval_fw!(u, interp_df64, x, p); nothing)),
        SysEvalJacFW((u, U, x, p) -> (_execute_jac_fw!(u, U, interp_jac, x, p); nothing)),
        SysTaylor1FW((u, tx, p) -> (execute_taylor!(u, Val(1), interp_t1, tx, p); nothing)),
        SysTaylor2FW((u, tx, p) -> (execute_taylor!(u, Val(2), interp_t2, tx, p); nothing)),
        SysTaylor3FW((u, tx, p) -> (execute_taylor!(u, Val(3), interp_t3, tx, p); nothing)),
        SysTaylor1ParamFW((u, tx, tp) -> (execute_taylor!(u, Val(1), interp_t1, tx, tp); nothing)),
        SysTaylor2ParamFW((u, tx, tp) -> (execute_taylor!(u, Val(2), interp_t2, tx, tp); nothing)),
        SysTaylor3ParamFW((u, tx, tp) -> (execute_taylor!(u, Val(3), interp_t3, tx, tp); nothing)),
        (neqs, nvars),
        nparams,
    )
end
