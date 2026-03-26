## System — compiled polynomial system with cached interpreter pipeline.
#
# Merges the old PolynomialSystemInfo + SystemEvaluator into a single user-facing type.
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
struct System
    evaluator::SystemEvaluator
    degrees::Vector{Int}
    nvars::Int
    nparams::Int
    variable_groups::Vector{Vector{Int}}
    is_homogeneous::Bool
    support::Vector{Matrix{Int32}}
    coefficients::Vector{Vector{ComplexF64}}
    # GC roots — interpreters must stay alive for FunctionWrapper closures
    _seq_eval::InstructionSequence
    _seq_jac::InstructionSequence
    _interp_f64::Interpreter{Vector{ComplexF64}}
    _interp_df64::Interpreter{Vector{ComplexDF64}}
    _interp_jac::Interpreter{Vector{ComplexF64}}
    _interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}}
    _interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}}
    _interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}}
end

## ── Accessors ────────────────────────────────────────────────────────────────

Base.size(F::System)::Tuple{Int, Int} = size(F.evaluator)
degrees(F::System)::Vector{Int} = F.degrees
nvariables(F::System)::Int = F.nvars
nparameters(F::System)::Int = F.nparams

## ── FW-compatible wrapper functions ──────────────────────────────────────────

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

## ── System constructor ──────────────────────────────────────────────────────

"""
    System(polys; parameters=[], variables=...) -> System

Build a `System` from a vector of MultivariatePolynomials polynomials.
Compiles the full interpreter pipeline and caches everything for reuse.
"""
function System(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    )::System
    neqs = length(polys)
    nvars = length(variables)
    nparams = length(parameters)

    # ── Support and coefficients (non-parametric systems only) ────────────
    supp, coeffs = if nparams == 0
        support_coefficients(polys, variables)
    else
        Vector{Matrix{Int32}}(), Vector{Vector{ComplexF64}}()
    end

    # ── Build interpreters ────────────────────────────────────────────────
    interp_f64 = _build_interpreter(
        Vector{ComplexF64}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )
    interp_df64 = _build_interpreter(
        Vector{ComplexDF64}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )
    interp_jac = _build_interpreter(
        Vector{ComplexF64}, polys;
        parameters = parameters, variables = variables, include_jacobian = true,
    )
    interp_t1 = _build_interpreter(
        Vector{TruncatedTaylorSeries{2, ComplexF64}}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )
    interp_t2 = _build_interpreter(
        Vector{TruncatedTaylorSeries{3, ComplexF64}}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )
    interp_t3 = _build_interpreter(
        Vector{TruncatedTaylorSeries{4, ComplexF64}}, polys;
        parameters = parameters, variables = variables, include_jacobian = false,
    )

    # ── Metadata ──────────────────────────────────────────────────────────
    degs = Int[MP.maxdegree(p) for p in polys]

    is_homogeneous = try
        all(p -> MP.ishomogeneous(p), polys)
    catch
        false
    end

    # ── Wire FunctionWrappers ─────────────────────────────────────────────
    evaluator = SystemEvaluator(
        SysEvalFW((u, x, p) -> (_execute_eval_fw!(u, interp_f64, x, p); nothing)),
        SysEvalDF64FW((u, x, p) -> (_execute_eval_fw!(u, interp_df64, x, p); nothing)),
        SysEvalJacFW((u, U, x, p) -> (_execute_jac_fw!(u, U, interp_jac, x, p); nothing)),
        SysTaylor1FW((u, tx, p) -> (execute_taylor!(u, Val(1), interp_t1, tx, p); nothing)),
        SysTaylor2FW((u, tx, p) -> (execute_taylor!(u, Val(2), interp_t2, tx, p); nothing)),
        SysTaylor3FW((u, tx, p) -> (execute_taylor!(u, Val(3), interp_t3, tx, p); nothing)),
        (neqs, nvars),
        nparams,
    )

    return System(
        evaluator, degs, nvars, nparams,
        Vector{Int}[], is_homogeneous,
        supp, coeffs,
        interp_f64.sequence, interp_jac.sequence,
        interp_f64, interp_df64, interp_jac,
        interp_t1, interp_t2, interp_t3,
    )
end
