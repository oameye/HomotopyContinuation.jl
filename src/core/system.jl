## System — compiled polynomial system with cached interpreter pipeline.
#
# The System constructor runs the full pipeline:
#   MP polynomials → SExpr → CSE → InstructionSequence → Interpreters → FunctionWrappers
abstract type SystemShape end
struct UnderdeterminedShape <: SystemShape end
struct SquareShape <: SystemShape end
struct OverdeterminedShape <: SystemShape end

const SupportCoefficients = Tuple{Vector{Matrix{Int32}}, Vector{Vector{ComplexF64}}}

abstract type SystemCompileStrategy end
struct InterpretedCompile <: SystemCompileStrategy end
struct CompiledCompile <: SystemCompileStrategy end
struct CompiledAllCompile <: SystemCompileStrategy end

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
struct System{P, V, M, S <: SystemShape}
    polys::FSVec{P}
    parameters::FSVec{V}
    variables::FSVec{V}
    evaluator::SystemEvaluator
    degrees::Vector{Int}
    equation_scales::Vector{Float64}
    nvars::Int
    nparams::Int
    variable_groups::Vector{Vector{Int}}
    is_homogeneous::Bool
    _support_coefficients::LazyRef{SupportCoefficients}
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
        polys::AbstractVector,
        parameters::AbstractVector,
        variables::AbstractVector,
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::System
    neqs = length(polys)
    nvars = length(variables)
    nparams = length(parameters)
    shape = if neqs < nvars
        UnderdeterminedShape()
    elseif neqs == nvars
        SquareShape()
    else
        OverdeterminedShape()
    end
    builder = if compile == CompileMode.INTERPRETED
        _build_interpreted_system
    elseif compile == CompileMode.COMPILED
        _build_codegen_system
    else
        _build_codegen_all_system
    end
    # The compile mode and shape are construction-time policy. Hide their
    # closed unions from inference so the default path does not traverse all
    # three code-generation backends or all three shape instantiations.
    builder = Base.inferencebarrier(builder)
    shape = Base.inferencebarrier(shape)
    normalized, lowered = _lower_input(polys, variables, parameters)
    return _dispatch_system_build(
        builder, normalized, variables, parameters, lowered, neqs, nvars, nparams, shape,
    )
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

"""
    System(exprs::AbstractVector{Expression}; parameters=[], variables=..., compile=...) -> System

Build a `System` from symbolic [`Expression`](@ref)s. This is the front-end for
input that is not a polynomial: division, negative integer powers, `sqrt`, `sin`
and `cos`.

```julia
@var x y a b
System([sqrt(a + b) * x^2 - y, (x * y + a - sqrt(b))^2 - 3]; parameters = [a, b])
```
"""
function System(
        exprs::AbstractVector{Expression};
        parameters = nothing,
        variables = nothing,
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::System
    params = parameters === nothing ? Expression[] : _as_variables(parameters)
    vars = variables === nothing ? _default_variables(exprs, params) :
        _as_variables(variables)
    return System(exprs, params, vars, compile)
end

# `MP.RationalPoly` has no polynomial representation, so it routes through the
# expression front-end, keeping the MP variable order.
function System(
        polys::AbstractVector{<:MP.RationalPoly};
        parameters = nothing,
        variables = nothing,
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::System
    params = parameters === nothing ? Expression[] : _as_variables(parameters)
    vars = if variables === nothing
        _as_variables(
            _rational_variables(polys, parameters === nothing ? () : parameters),
        )
    else
        _as_variables(variables)
    end
    return System(_as_expressions(polys), params, vars, compile)
end

function _as_variables(vars::AbstractVector)::Vector{Expression}
    Base.@nospecialize vars
    out = Vector{Expression}(undef, length(vars))
    for (i, v) in enumerate(vars)
        out[i] = v isa Expression ? v : variable(Symbol(v))
    end
    return out
end

_as_variables(vars::Vector{Expression})::Vector{Expression} = vars

function _as_expressions(polys::AbstractVector)::Vector{Expression}
    Base.@nospecialize polys
    out = Vector{Expression}(undef, length(polys))
    for (i, p) in enumerate(polys)
        out[i] = convert(Expression, p)
    end
    return out
end


## ── Accessors ────────────────────────────────────────────────────────────────

Base.size(F::System)::Tuple{Int, Int} = size(F.evaluator)
@inline system_shape(::System{P, V, M, S}) where {P, V, M, S} = S()
degrees(F::System)::Vector{Int} = F.degrees
# Factor each input equation was divided by. It leaves `V(F)` alone but changes
# `F` as a map, which an inner composition stage has to undo.
equation_scales(F::System)::Vector{Float64} = F.equation_scales
nvariables(F::System)::Int = F.nvars
nparameters(F::System)::Int = F.nparams
polynomials(F::System) = F.polys
variables(F::System) = F.variables
parameters(F::System) = F.parameters
variable_groups(F::System) = F.variable_groups
is_homogeneous(F::System)::Bool = F.is_homogeneous
function support_coefficients(F::System)::SupportCoefficients
    nparameters(F) == 0 ||
        throw(ArgumentError("support_coefficients(::System) is only defined for parameter-free systems"))
    cache = F._support_coefficients
    if !is_installed(cache)
        install!(cache, support_coefficients(F.polys, F.variables))
    end
    return cache[]
end

@inline _to_fsvec(xs::AbstractVector{T}) where {T} = FSVec{T}(collect(xs))

# Linear scan: a `Set` costs more to compile than the scan costs to run, as long
# as callers keep the scan off the innermost loop.
function _contains_variable(variables::AbstractVector, var)::Bool
    Base.@nospecialize variables var
    for v in variables
        v == var && return true
    end
    return false
end

# Per-polynomial degree in `variables` only, and homogeneity in them, in one pass.
# `MP.maxdegree` would count parameters and inflate the Bezout number.
function _variable_degrees(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
    )::Tuple{Vector{Int}, Bool}
    Base.@nospecialize polys variables
    degs = Vector{Int}(undef, length(polys))
    homogeneous = true

    mask = Bool[]
    mask_variables = nothing
    for (i, poly) in enumerate(polys)
        maxdeg = 0
        mindeg = typemax(Int)
        for term in MP.terms(poly)
            iszero(MP.coefficient(term)) && continue
            mono = MP.monomial(term)
            mono_variables = MP.variables(mono)
            if mono_variables !== mask_variables
                resize!(mask, length(mono_variables))
                for (k, var) in enumerate(mono_variables)
                    mask[k] = _contains_variable(variables, var)
                end
                mask_variables = mono_variables
            end
            degree = 0
            for (keep, exp) in zip(mask, MP.exponents(mono))
                keep || continue
                degree += exp
            end
            degree > maxdeg && (maxdeg = degree)
            degree < mindeg && (mindeg = degree)
        end
        degs[i] = maxdeg
        # A polynomial with no nonzero term counts as homogeneous.
        homogeneous &= mindeg == typemax(Int) || mindeg == maxdeg
    end
    return degs, homogeneous
end

## ── Front-end lowering ──────────────────────────────────────────────────────

"""
Everything the builder needs from the input, independent of its representation.
"""
struct LoweredInput
    seq_eval::InstructionSequence
    seq_jac::InstructionSequence
    degrees::Vector{Int}
    is_homogeneous::Bool
    # factor each input equation was divided by
    scales::Vector{Float64}
end

# Called from a frame that still knows the concrete input type: dispatching behind
# the `@nospecialize` builder chain makes inference walk both front-ends on every
# build.
@noinline function _lower_input(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
        parameters::AbstractVector,
    )
    normalized, scales = _normalize_polys(polys)
    degs, is_homogeneous = _variable_degrees(normalized, variables)
    return normalized, LoweredInput(
            _build_instruction_sequence(normalized, variables, parameters, false),
            _build_instruction_sequence(normalized, variables, parameters, true),
            degs, is_homogeneous, scales,
        )
end

@noinline function _lower_input(
        exprs::AbstractVector{Expression},
        variables::AbstractVector,
        parameters::AbstractVector,
    )
    normalized, scales = _normalize_expressions(exprs)
    vars = _as_variables(variables)
    params = _as_variables(parameters)
    degs, is_homogeneous = _expression_degrees(normalized, vars)
    return normalized, LoweredInput(
            _build_instruction_sequence_from_expressions(normalized, vars, params, false),
            _build_instruction_sequence_from_expressions(normalized, vars, params, true),
            degs, is_homogeneous, scales,
        )
end

# Rational input reaching the positional constructor.
@noinline function _lower_input(
        polys::AbstractVector{<:MP.RationalPoly},
        variables::AbstractVector,
        parameters::AbstractVector,
    )
    return _lower_input(_as_expressions(polys), variables, parameters)
end

@noinline function _lower_input(
        polys::AbstractVector,
        variables::AbstractVector,
        parameters::AbstractVector,
    )
    Base.@nospecialize polys variables parameters
    throw(
        ArgumentError(
            "cannot build a `System` from input of type $(typeof(polys)): expected a " *
                "vector of MultivariatePolynomials polynomials or of `Expression`s.",
        ),
    )
end

## ── Polynomial normalization ────────────────────────────────────────────────

function _normalize_polys(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
    )
    Base.@nospecialize polys
    # Canonicalize coefficients to a floating polynomial type even when no
    # rescaling is required. Returning either the original integer polynomial
    # or a divided floating polynomial made the rest of construction infer an
    # abstract polynomial element type.
    scales = Vector{Float64}(undef, length(polys))
    normalized = map(enumerate(polys)) do (i, p)
        coeffs = MP.coefficients(p)
        nrm = maximum(c -> Float64(abs(c)), coeffs)
        scale = _normalization_scale(nrm)
        scales[i] = scale
        return p / scale
    end
    return normalized, scales
end

# Dividing an equation by a constant leaves `V(F)` unchanged, so an equation far above
# unit scale is brought back down. A scale of `0` (the zero equation) or a non-finite
# one carries no information and is left alone.
function _normalization_scale(nrm::Float64)::Float64
    (iszero(nrm) || !isfinite(nrm)) && return 1.0
    return nrm <= 1.0e8 ? 1.0 : nrm
end

# Scale measured by `expression_scale` rather than by the largest coefficient, which
# an expression tree does not carry.
function _normalize_expressions(
        exprs::AbstractVector{Expression},
    )::Tuple{Vector{Expression}, Vector{Float64}}
    scales = Vector{Float64}(undef, length(exprs))
    normalized = map(enumerate(exprs)) do (i, e)
        scale = _normalization_scale(expression_scale(e))
        scales[i] = scale
        return e / scale
    end
    return normalized, scales
end

## ── FW-compatible wrapper functions ──────────────────────────────────────────


@noinline function _dispatch_system_build(
        builder::Function,
        polys::AbstractVector,
        variables::AbstractVector,
        parameters::AbstractVector,
        lowered::LoweredInput,
        neqs::Int,
        nvars::Int,
        nparams::Int,
        shape::SystemShape,
    )::System
    Base.@nospecialize builder polys variables parameters shape
    return builder(polys, variables, parameters, lowered, neqs, nvars, nparams, shape)
end

@noinline function _build_interpreted_system(
        polys, variables, parameters, lowered::LoweredInput,
        neqs::Int, nvars::Int, nparams::Int, shape,
    )::System
    Base.@nospecialize polys variables parameters shape
    return _build_compiled_system(
        InterpretedCompile(), polys, variables, parameters, lowered,
        neqs, nvars, nparams, shape,
    )
end

@noinline function _build_codegen_system(
        polys, variables, parameters, lowered::LoweredInput,
        neqs::Int, nvars::Int, nparams::Int, shape,
    )::System
    Base.@nospecialize polys variables parameters shape
    return _build_compiled_system(
        CompiledCompile(), polys, variables, parameters, lowered,
        neqs, nvars, nparams, shape,
    )
end

@noinline function _build_codegen_all_system(
        polys, variables, parameters, lowered::LoweredInput,
        neqs::Int, nvars::Int, nparams::Int, shape,
    )::System
    Base.@nospecialize polys variables parameters shape
    return _build_compiled_system(
        CompiledAllCompile(), polys, variables, parameters, lowered,
        neqs, nvars, nparams, shape,
    )
end

@noinline function _build_compiled_system(
        strategy::C,
        polys::AbstractVector,
        variables::AbstractVector,
        parameters::AbstractVector,
        lowered::LoweredInput,
        neqs::Int,
        nvars::Int,
        nparams::Int,
        ::S,
    )::System where {C <: SystemCompileStrategy, S <: SystemShape}
    Base.@nospecialize polys variables parameters
    seq_eval = lowered.seq_eval
    seq_jac = lowered.seq_jac

    interp_f64 = Interpreter(Vector{ComplexF64}, seq_eval)
    interp_df64 = Interpreter(Vector{ComplexDF64}, seq_eval)
    interp_jac = Interpreter(Vector{ComplexF64}, seq_jac)
    interp_t1 = Interpreter(Vector{TruncatedTaylorSeries{2, ComplexF64}}, seq_eval)
    interp_t2 = Interpreter(Vector{TruncatedTaylorSeries{3, ComplexF64}}, seq_eval)
    interp_t3 = Interpreter(Vector{TruncatedTaylorSeries{4, ComplexF64}}, seq_eval)

    evaluator = _build_mode_evaluator(
        strategy, seq_eval, seq_jac,
        interp_f64, interp_df64, interp_jac,
        interp_t1, interp_t2, interp_t3,
        neqs, nvars, nparams,
    )
    M = _compile_mode(strategy)

    fs_polys = _to_fsvec(polys)
    fs_parameters = _to_fsvec(parameters)
    fs_variables = _to_fsvec(variables)
    return System{eltype(fs_polys), eltype(fs_variables), M, S}(
        fs_polys,
        fs_parameters,
        fs_variables,
        evaluator, lowered.degrees, lowered.scales, nvars, nparams,
        Vector{Int}[], lowered.is_homogeneous,
        LazyRef{SupportCoefficients}(),
        interp_f64, interp_df64, interp_jac,
        interp_t1, interp_t2, interp_t3,
        M,
    )
end

@inline _compile_mode(::InterpretedCompile) = CompileMode.INTERPRETED
@inline _compile_mode(::CompiledCompile) = CompileMode.COMPILED
@inline _compile_mode(::CompiledAllCompile) = CompileMode.COMPILED_ALL

function _build_mode_evaluator(
        ::InterpretedCompile, ::InstructionSequence, ::InstructionSequence,
        interp_f64, interp_df64, interp_jac, interp_t1, interp_t2, interp_t3,
        neqs::Int, nvars::Int, nparams::Int,
    )::SystemEvaluator
    return _build_system_evaluator(
        interp_f64, interp_df64, interp_jac,
        interp_t1, interp_t2, interp_t3,
        neqs, nvars, nparams,
    )
end

function _build_mode_evaluator(
        ::CompiledCompile, seq_eval::InstructionSequence, seq_jac::InstructionSequence,
        ::Interpreter{Vector{ComplexF64}}, interp_df64,
        ::Interpreter{Vector{ComplexF64}}, interp_t1, interp_t2, interp_t3,
        neqs::Int, nvars::Int, nparams::Int,
    )::SystemEvaluator
    return _build_compiled_evaluator(
        seq_eval, seq_jac, interp_df64,
        _build_taylor_fws(interp_t1, interp_t2, interp_t3),
        neqs, nvars, nparams,
    )
end

function _build_mode_evaluator(
        ::CompiledAllCompile, seq_eval::InstructionSequence, seq_jac::InstructionSequence,
        ::Interpreter{Vector{ComplexF64}}, interp_df64,
        ::Interpreter{Vector{ComplexF64}}, ::Interpreter, ::Interpreter, ::Interpreter,
        neqs::Int, nvars::Int, nparams::Int,
    )::SystemEvaluator
    return _build_compiled_evaluator(
        seq_eval, seq_jac, interp_df64, _build_taylor_fws(seq_eval),
        neqs, nvars, nparams,
    )
end

@noinline function _execute_eval_fw!(
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

@noinline function _execute_jac_fw!(
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

function _lazy_df64_interpreted(interp_df64::Interpreter{Vector{ComplexDF64}})
    return _lazy_df64(
        () -> (
            SysEvalDF64FW(
                (u, x, p) -> (_execute_eval_fw!(u, interp_df64, x, p); nothing),
            ),
            SysEvalDF64OutFW(
                (u, x, p) -> (_execute_eval_fw!(u, interp_df64, x, p); nothing),
            ),
        ),
    )
end

function _build_interpreted_evaluation_fws(
        interp_f64::Interpreter{Vector{ComplexF64}},
        interp_df64::Interpreter{Vector{ComplexDF64}},
        interp_jac::Interpreter{Vector{ComplexF64}},
    )
    return (
        SysEvalFW(
            (u, x, p) -> (_execute_eval_fw!(u, interp_f64, x, p); nothing),
        ),
        _lazy_df64_interpreted(interp_df64)...,
        SysEvalJacFW(
            (u, U, x, p) -> (_execute_jac_fw!(u, U, interp_jac, x, p); nothing),
        ),
    )
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
    taylor_1 = SysTaylor1FW(
        (u, tx, p) -> (execute_taylor!(u, Val(1), interp_t1, tx, p); nothing),
    )
    taylor_2 = SysTaylor2FW(
        (u, tx, p) -> (execute_taylor!(u, Val(2), interp_t2, tx, p); nothing),
    )
    taylor_3 = SysTaylor3FW(
        (u, tx, p) -> (execute_taylor!(u, Val(3), interp_t3, tx, p); nothing),
    )
    param_builder = if iszero(nparams)
        _build_zero_parameter_taylor_fws
    else
        _build_parameter_taylor_fws
    end
    param_builder = Base.inferencebarrier(param_builder)
    param_taylor = _dispatch_parameter_taylor_fws(
        param_builder, taylor_1, taylor_2, taylor_3, interp_t1, interp_t2, interp_t3,
    )
    evaluation_fws = _build_interpreted_evaluation_fws(
        interp_f64, interp_df64, interp_jac,
    )
    return SystemEvaluator(
        evaluation_fws...,
        taylor_1,
        taylor_2,
        taylor_3,
        param_taylor...,
        (neqs, nvars),
        nparams,
    )
end

@noinline function _dispatch_parameter_taylor_fws(
        builder::Function, taylor_1, taylor_2, taylor_3,
        interp_t1, interp_t2, interp_t3,
    )
    Base.@nospecialize builder taylor_1 taylor_2 taylor_3 interp_t1 interp_t2 interp_t3
    return builder(taylor_1, taylor_2, taylor_3, interp_t1, interp_t2, interp_t3)
end

function _build_zero_parameter_taylor_fws(
        taylor_1::SysTaylor1FW,
        taylor_2::SysTaylor2FW,
        taylor_3::SysTaylor3FW,
        ::Any, ::Any, ::Any,
    )
    # A TaylorVector with zero parameter columns is equivalent to the ordinary
    # empty-parameter call. Delegate to the already-built scalar wrappers so a
    # parameter-free System does not compile three unused convolution kernels.
    return (
        SysTaylor1ParamFW((u, tx, ::TaylorVector{2, ComplexF64}) -> (taylor_1(u, tx, _EMPTY_PARAMS); nothing)),
        SysTaylor2ParamFW((u, tx, ::TaylorVector{3, ComplexF64}) -> (taylor_2(u, tx, _EMPTY_PARAMS); nothing)),
        SysTaylor3ParamFW((u, tx, ::TaylorVector{4, ComplexF64}) -> (taylor_3(u, tx, _EMPTY_PARAMS); nothing)),
    )
end

function _build_parameter_taylor_fws(
        ::Any, ::Any, ::Any,
        interp_t1::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}},
        interp_t2::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}},
        interp_t3::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}},
    )
    return (
        SysTaylor1ParamFW((u, tx, tp) -> (execute_taylor!(u, Val(1), interp_t1, tx, tp); nothing)),
        SysTaylor2ParamFW((u, tx, tp) -> (execute_taylor!(u, Val(2), interp_t2, tx, tp); nothing)),
        SysTaylor3ParamFW((u, tx, tp) -> (execute_taylor!(u, Val(3), interp_t3, tx, tp); nothing)),
    )
end
