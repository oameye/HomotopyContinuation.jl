## system_eval — MP polynomial input → SystemEvaluator
#
# Converts a vector of MultivariatePolynomials into a fully-wired SystemEvaluator
# with all FunctionWrapper closures backed by tape-based interpreters.

"""
    PolynomialSystemInfo

Metadata and GC roots for a polynomial system compiled into a `SystemEvaluator`.

The `_interp_*` and `_seq_*` fields serve as GC roots — the FunctionWrapper closures
capture interpreters by reference, and without these fields the interpreters could be
garbage collected.
"""
struct PolynomialSystemInfo
    degrees::Vector{Int}
    nvars::Int
    nparams::Int
    variable_groups::Vector{Vector{Int}}
    is_homogeneous::Bool
    _seq_eval::InstructionSequence
    _seq_jac::InstructionSequence
    _interp_f64::Interpreter{Vector{ComplexF64}}
    _interp_df64::Interpreter{Vector{ComplexDF64}}
    _interp_jac::Interpreter{Vector{ComplexF64}}
    _interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}}
    _interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}}
    _interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}}
end

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

## ── system_eval ──────────────────────────────────────────────────────────────

"""
    system_eval(polys; parameters=[], variables=...) -> (PolynomialSystemInfo, SystemEvaluator)

Build a `SystemEvaluator` from a vector of MultivariatePolynomials polynomials.

Returns a tuple of `(info, evaluator)` where `info` holds metadata (degrees,
homogeneity, variable counts) and GC roots for the interpreter closures.
"""
function system_eval(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    )::Tuple{PolynomialSystemInfo, SystemEvaluator}
    neqs = length(polys)
    nvars = length(variables)
    nparams = length(parameters)

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
    degrees = Int[MP.maxdegree(p) for p in polys]

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

    info = PolynomialSystemInfo(
        degrees,
        nvars,
        nparams,
        Vector{Int}[],
        is_homogeneous,
        interp_f64.sequence,
        interp_jac.sequence,
        interp_f64,
        interp_df64,
        interp_jac,
        interp_t1,
        interp_t2,
        interp_t3,
    )

    return (info, evaluator)
end
