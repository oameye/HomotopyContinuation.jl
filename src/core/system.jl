## System — compiled polynomial system with cached interpreter pipeline.
#
# The System constructor runs the full pipeline:
#   Expression → SExpr → CSE → InstructionSequence → Interpreters → FunctionWrappers
# MultivariatePolynomials input is converted to `Expression` first, so one concrete
# `System` serves every front-end.
abstract type SystemShape end
struct UnderdeterminedShape <: SystemShape end
struct SquareShape <: SystemShape end
struct OverdeterminedShape <: SystemShape end

const SupportCoefficients = Tuple{Vector{Matrix{Int32}}, Vector{Vector{ComplexF64}}}

"""
    System

Compiled, ready-to-evaluate polynomial system. Caches the interpreter pipeline
so users pay the compilation cost once and reuse across multiple `solve` calls.

Construct from MultivariatePolynomials or from [`Expression`](@ref)s:

    F = System(polys; parameters=[], variables=...)

Either input is stored as `Expression`s, so `variables(F)`,
`parameters(F)` and the equations of `F` are `Expression`s.

# Examples
```julia
@polyvar x y
F = System([x^2 + y - 1, x*y - 2])
solve(F)
solve(F, Polyhedral())
```
"""
struct System
    polys::FSVec{Expression}
    parameters::FSVec{Expression}
    variables::FSVec{Expression}
    evaluator::SystemEvaluator
    degrees::Vector{Int}
    equation_scales::Vector{Float64}
    nvars::Int
    nparams::Int
    variable_groups::Vector{Vector{Int}}
    group_degrees::Matrix{Int}
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
        polys::Vector{Expression},
        parameters::Vector{Expression},
        variables::Vector{Expression},
        compile::CompileMode.T = CompileMode.INTERPRETED,
        variable_groups::Vector{Vector{Int}} = Vector{Int}[],
    )::System
    neqs = length(polys)
    nvars = length(variables)
    nparams = length(parameters)
    normalized, lowered = lower_input(polys, variables, parameters)
    isempty(variable_groups) ||
        (lowered = group_input(lowered, normalized, variables, variable_groups))
    return build_system(
        compile, normalized, variables, parameters, lowered, neqs, nvars, nparams,
    )
end

"""
    System(polys; parameters=[], variables=..., variable_groups=nothing,
           compile=CompileMode.INTERPRETED) -> System

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

`variable_groups` partitions the variables into groups, one degree per group
instead of one total degree. [`solve`](@ref) with [`TotalDegree`](@ref) then
tracks the multi-homogeneous Bezout number of paths, which is at most the total
Bezout number and usually far below it. The groups must cover every variable
exactly once, and when `variables` is omitted they also fix the variable order.

```julia
@polyvar x y v w
System([x * y - 2v * w, x^2 - 4v^2]; variable_groups = [[x, v], [y, w]])
```
"""
function System(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters = nothing,
        variables = nothing,
        variable_groups = nothing,
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::System
    parameters === nothing && (parameters = empty_vars(polys))
    if variables === nothing
        variables = variable_groups === nothing ?
            effective_variables_excluding(polys, parameters) : reduce(vcat, variable_groups)
    end
    return System(
        as_expressions(polys), as_variables(parameters), as_variables(variables),
        compile, variable_group_indices(variable_groups, variables),
    )
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
        variable_groups = nothing,
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::System
    params = parameters === nothing ? Expression[] : as_variables(parameters)
    vars = if variables !== nothing
        as_variables(variables)
    elseif variable_groups !== nothing
        as_variables(reduce(vcat, variable_groups))
    else
        default_variables(exprs, params)
    end
    return System(
        collect(Expression, exprs), params, vars, compile,
        variable_group_indices(variable_groups, vars),
    )
end

# `MP.RationalPoly` has no polynomial representation, so it routes through the
# expression front-end, keeping the MP variable order.
function System(
        polys::AbstractVector{<:MP.RationalPoly};
        parameters = nothing,
        variables = nothing,
        variable_groups = nothing,
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::System
    params = parameters === nothing ? Expression[] : as_variables(parameters)
    vars = if variables !== nothing
        as_variables(variables)
    elseif variable_groups !== nothing
        as_variables(reduce(vcat, variable_groups))
    else
        as_variables(
            rational_variables(polys, parameters === nothing ? () : parameters),
        )
    end
    return System(
        as_expressions(polys), params, vars, compile,
        variable_group_indices(variable_groups, vars),
    )
end

function as_variables(vars::AbstractVector)::Vector{Expression}
    Base.@nospecialize vars
    out = Vector{Expression}(undef, length(vars))
    for (i, v) in enumerate(vars)
        out[i] = v isa Expression ? v : polynomial_variable(v)
    end
    return out
end

as_variables(vars::Vector{Expression})::Vector{Expression} = vars

function as_expressions(polys::AbstractVector)::Vector{Expression}
    Base.@nospecialize polys
    out = Vector{Expression}(undef, length(polys))
    for (i, p) in enumerate(polys)
        out[i] = convert(Expression, p)
    end
    return out
end


## ── Accessors ────────────────────────────────────────────────────────────────

Base.size(F::System)::Tuple{Int, Int} = size(F.evaluator)
@inline function with_system_shape(f::F, S::System) where {F}
    m, n = size(S)
    return m < n ? f(UnderdeterminedShape()) :
        m == n ? f(SquareShape()) : f(OverdeterminedShape())
end
degrees(F::System)::Vector{Int} = F.degrees

"""
    equation_scales(F) -> Vector{Float64}

Factor each equation given to `F` was divided by at construction, one entry per
equation and `1.0` where nothing was divided. An equation whose coefficients sit
far from unit scale is normalized for tracker conditioning; that leaves the
solution set alone but changes the value of the map, so multiplying
[`evaluate`](@ref) entrywise by these factors recovers the equations as given.

Defined for a [`System`](@ref) and a [`Homotopy`](@ref).
"""
equation_scales(F::System)::Vector{Float64} = F.equation_scales
nvariables(F::System)::Int = F.nvars
nparameters(F::System)::Int = F.nparams
polynomials(F::System) = F.polys
variables(F::System) = F.variables
parameters(F::System) = F.parameters
variable_groups(F::System) = F.variable_groups

"""
    multi_degrees(F::System) -> Matrix{Int}

Degree of every equation of `F` in every variable group: one row per group of
`variable_groups(F)`, one column per equation. An ungrouped system has the
single row `degrees(F)`.
"""
multi_degrees(F::System)::Matrix{Int} =
    isempty(F.variable_groups) ? reshape(F.degrees, 1, :) : F.group_degrees
# For a grouped system: homogeneous in every group separately.
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

# Linear scan: a `Set` costs more to compile than the scan costs to run, as long
# as callers keep the scan off the innermost loop.
function contains_variable(variables::AbstractVector, var)::Bool
    Base.@nospecialize variables var
    for v in variables
        v == var && return true
    end
    return false
end

## ── Variable groups ─────────────────────────────────────────────────────────

variable_group_indices(::Nothing, ::AbstractVector)::Vector{Vector{Int}} = Vector{Int}[]

# Matched by name, so either front-end's variable type is accepted.
@noinline function variable_group_indices(
        groups, variables::AbstractVector,
    )::Vector{Vector{Int}}
    Base.@nospecialize groups variables
    index = Dict{Symbol, Int}(Symbol(v) => i for (i, v) in enumerate(variables))
    assigned = fill(0, length(variables))
    out = Vector{Vector{Int}}(undef, length(groups))
    for (j, group) in enumerate(groups)
        idx = Vector{Int}(undef, length(group))
        for (l, v) in enumerate(group)
            i = get(index, Symbol(v), 0)
            i == 0 && throw(
                ArgumentError("`$v` is in a variable group but not a variable of the system."),
            )
            assigned[i] == 0 || throw(
                ArgumentError("`$v` is in variable group $(assigned[i]) and in group $j."),
            )
            assigned[i] = j
            idx[l] = i
        end
        out[j] = idx
    end
    missing_var = findfirst(iszero, assigned)
    missing_var === nothing || throw(
        ArgumentError(
            "`$(variables[missing_var])` is a variable of the system but is in no " *
                "variable group; the groups must cover every variable.",
        ),
    )
    return out
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
    variable_groups::Vector{Vector{Int}}
    group_degrees::Matrix{Int}
end

LoweredInput(
    seq_eval::InstructionSequence, seq_jac::InstructionSequence, degrees::Vector{Int},
    is_homogeneous::Bool, scales::Vector{Float64},
) = LoweredInput(
    seq_eval, seq_jac, degrees, is_homogeneous, scales,
    Vector{Int}[], Matrix{Int}(undef, 0, 0),
)

lowered_input(sys::System)::LoweredInput = LoweredInput(
    sys._interp_f64.sequence, sys._interp_jac.sequence, sys.degrees,
    sys.is_homogeneous, sys.equation_scales, sys.variable_groups, sys.group_degrees,
)

@noinline function group_input(
        lowered::LoweredInput, polys::Vector{Expression}, variables::Vector{Expression},
        groups::Vector{Vector{Int}},
    )::LoweredInput
    D = Matrix{Int}(undef, length(groups), length(lowered.degrees))
    homogeneous = true
    for (j, group) in enumerate(groups)
        degs, hom = expression_degrees(polys, variables[group])
        D[j, :] .= degs
        homogeneous &= hom
    end
    return LoweredInput(
        lowered.seq_eval, lowered.seq_jac, lowered.degrees, homogeneous,
        lowered.scales, groups, D,
    )
end

function lower_input(
        exprs::Vector{Expression},
        vars::Vector{Expression},
        params::Vector{Expression},
    )::Tuple{Vector{Expression}, LoweredInput}
    normalized, scales = normalize_expressions(exprs)
    degs, is_homogeneous = expression_degrees(normalized, vars)
    return normalized, LoweredInput(
            build_instruction_sequence_from_expressions(normalized, vars, params, false),
            build_instruction_sequence_from_expressions(normalized, vars, params, true),
            degs, is_homogeneous, scales,
        )
end

## ── Polynomial normalization ────────────────────────────────────────────────

# Dividing an equation by a constant leaves `V(F)` unchanged, so one whose scale sits
# outside this band is brought back towards unity. A scale of `0` (the zero equation) or
# a non-finite one carries no information and is left alone.
const MIN_NORM_SCALE = 1.0e-6
const MAX_NORM_SCALE = 1.0e8

function normalization_scale(nrm::Float64)::Float64
    (iszero(nrm) || !isfinite(nrm)) && return 1.0
    MIN_NORM_SCALE <= nrm <= MAX_NORM_SCALE && return 1.0
    return nrm
end

# Scale measured by `expression_scale` rather than by the largest coefficient, which
# an expression tree does not carry.
function normalize_expressions(
        exprs::AbstractVector{Expression},
    )::Tuple{Vector{Expression}, Vector{Float64}}
    scales = Vector{Float64}(undef, length(exprs))
    normalized = map(enumerate(exprs)) do (i, e)
        scale = normalization_scale(expression_scale(e))
        scales[i] = scale
        return divide_coefficients(e, scale)
    end
    return normalized, scales
end

function divide_coefficients(e::Expression, scale::Float64)::Expression
    isone(scale) && return e
    terms = @match e begin
        SymExpr.EAdd(args) => args::Vector{Expression}
        _ => Expression[e]
    end
    return eadd(Expression[divide_coefficient(t, scale) for t in terms])
end

function divide_coefficient(term::Expression, scale::Float64)::Expression
    @match term begin
        SymExpr.ENum(val) => return SymExpr.ENum(val / scale)
        _ => nothing
    end
    base, c = split_coefficient(term)
    return emul(Expression[SymExpr.ENum(c / scale), base])
end

## ── System assembly ─────────────────────────────────────────────────────────

function build_system(
        compile::CompileMode.T, polys::Vector{Expression}, variables::Vector{Expression},
        parameters::Vector{Expression}, lowered::LoweredInput,
        neqs::Int, nvars::Int, nparams::Int,
    )::System
    strategy = compile == CompileMode.INTERPRETED ? InterpretedCompile() :
        compile == CompileMode.COMPILED ? CompiledCompile() : CompiledAllCompile()
    return build_compiled_system(
        strategy, polys, variables, parameters, lowered, neqs, nvars, nparams,
    )
end

@noinline function build_compiled_system(
        strategy::C,
        polys::Vector{Expression},
        variables::Vector{Expression},
        parameters::Vector{Expression},
        lowered::LoweredInput,
        neqs::Int,
        nvars::Int,
        nparams::Int,
    )::System where {C <: SystemCompileStrategy}
    seq_eval = lowered.seq_eval
    seq_jac = lowered.seq_jac

    interp_f64 = Interpreter(Vector{ComplexF64}, seq_eval)
    interp_df64 = Interpreter(Vector{ComplexDF64}, seq_eval)
    interp_jac = Interpreter(Vector{ComplexF64}, seq_jac)
    interp_t1 = Interpreter(Vector{TruncatedTaylorSeries{2, ComplexF64}}, seq_eval)
    interp_t2 = Interpreter(Vector{TruncatedTaylorSeries{3, ComplexF64}}, seq_eval)
    interp_t3 = Interpreter(Vector{TruncatedTaylorSeries{4, ComplexF64}}, seq_eval)

    evaluator = build_mode_evaluator(
        strategy, seq_eval, seq_jac,
        interp_f64, interp_df64, interp_jac,
        interp_t1, interp_t2, interp_t3,
        neqs, nvars, nparams,
    )
    M = compile_mode_of(strategy)

    return System(
        FSVec{Expression}(copy(polys)),
        FSVec{Expression}(copy(parameters)),
        FSVec{Expression}(copy(variables)),
        evaluator, lowered.degrees, lowered.scales, nvars, nparams,
        lowered.variable_groups, lowered.group_degrees, lowered.is_homogeneous,
        LazyRef{SupportCoefficients}(),
        interp_f64, interp_df64, interp_jac,
        interp_t1, interp_t2, interp_t3,
        M,
    )
end

@inline compile_mode_of(::InterpretedCompile) = CompileMode.INTERPRETED
@inline compile_mode_of(::CompiledCompile) = CompileMode.COMPILED
@inline compile_mode_of(::CompiledAllCompile) = CompileMode.COMPILED_ALL

function build_mode_evaluator(
        ::InterpretedCompile, ::InstructionSequence, ::InstructionSequence,
        interp_f64, interp_df64, interp_jac, interp_t1, interp_t2, interp_t3,
        neqs::Int, nvars::Int, nparams::Int,
    )::SystemEvaluator
    return build_system_evaluator(
        interp_f64, interp_df64, interp_jac,
        interp_t1, interp_t2, interp_t3,
        neqs, nvars, nparams,
    )
end

function build_mode_evaluator(
        mode::CompiledCompile, seq_eval::InstructionSequence,
        seq_jac::InstructionSequence,
        ::Interpreter{Vector{ComplexF64}}, interp_df64,
        ::Interpreter{Vector{ComplexF64}}, interp_t1, interp_t2, interp_t3,
        neqs::Int, nvars::Int, nparams::Int,
    )::SystemEvaluator
    return build_compiled_evaluator(
        mode, seq_eval, seq_jac, interp_df64,
        build_taylor_fws(interp_t1, interp_t2, interp_t3),
        neqs, nvars, nparams,
    )
end

function build_mode_evaluator(
        mode::CompiledAllCompile, seq_eval::InstructionSequence,
        seq_jac::InstructionSequence,
        ::Interpreter{Vector{ComplexF64}}, interp_df64,
        ::Interpreter{Vector{ComplexF64}}, ::Interpreter, ::Interpreter, ::Interpreter,
        neqs::Int, nvars::Int, nparams::Int,
    )::SystemEvaluator
    return build_compiled_evaluator(
        mode, seq_eval, seq_jac, interp_df64, build_taylor_fws(seq_eval),
        neqs, nvars, nparams,
    )
end

# One cloner per compile mode: the sequences are the whole recipe, the tapes
# around them are what must not be shared.

struct InterpretedCloner
    seq_eval::InstructionSequence
    seq_jac::InstructionSequence
    neqs::Int
    nvars::Int
    nparams::Int
end

function (c::InterpretedCloner)()::SystemEvaluator
    seq_eval, seq_jac = c.seq_eval, c.seq_jac
    return build_system_evaluator(
        Interpreter(Vector{ComplexF64}, seq_eval),
        Interpreter(Vector{ComplexDF64}, seq_eval),
        Interpreter(Vector{ComplexF64}, seq_jac),
        Interpreter(Vector{TruncatedTaylorSeries{2, ComplexF64}}, seq_eval),
        Interpreter(Vector{TruncatedTaylorSeries{3, ComplexF64}}, seq_eval),
        Interpreter(Vector{TruncatedTaylorSeries{4, ComplexF64}}, seq_eval),
        c.neqs, c.nvars, c.nparams,
    )
end

struct CompiledCloner{M <: Union{CompiledCompile, CompiledAllCompile}}
    mode::M
    seq_eval::InstructionSequence
    seq_jac::InstructionSequence
    neqs::Int
    nvars::Int
    nparams::Int
end

function (c::CompiledCloner)()::SystemEvaluator
    seq_eval = c.seq_eval
    return build_compiled_evaluator(
        c.mode, seq_eval, c.seq_jac,
        Interpreter(Vector{ComplexDF64}, seq_eval),
        clone_taylor_fws(c.mode, seq_eval),
        c.neqs, c.nvars, c.nparams,
    )
end

clone_taylor_fws(::CompiledCompile, seq_eval::InstructionSequence) =
    build_taylor_fws(
    Interpreter(Vector{TruncatedTaylorSeries{2, ComplexF64}}, seq_eval),
    Interpreter(Vector{TruncatedTaylorSeries{3, ComplexF64}}, seq_eval),
    Interpreter(Vector{TruncatedTaylorSeries{4, ComplexF64}}, seq_eval),
)

clone_taylor_fws(::CompiledAllCompile, seq_eval::InstructionSequence) =
    build_taylor_fws(seq_eval)

@noinline function execute_eval_fw!(
        u::AbstractVector, interp::Interpreter, x::AbstractVector, p::AbstractVector,
    )::Nothing
    if isempty(interp.sequence.parameters_range)
        load_inputs!(interp, x)
    else
        load_inputs!(interp, x, p)
    end
    execute_eval!(u, interp)
    return nothing
end

@noinline function execute_jac_fw!(
        u::AbstractVector, U::AbstractMatrix, interp::Interpreter,
        x::AbstractVector, p::AbstractVector,
    )::Nothing
    if isempty(interp.sequence.parameters_range)
        load_inputs!(interp, x)
    else
        load_inputs!(interp, x, p)
    end
    execute_jac!(u, U, interp)
    return nothing
end

function interpreted_df64_fws(interp_df64::Interpreter{Vector{ComplexDF64}})
    return (
        SysEvalDF64FW((u, x, p) -> (execute_eval_fw!(u, interp_df64, x, p); nothing)),
        SysEvalDF64OutFW((u, x, p) -> (execute_eval_fw!(u, interp_df64, x, p); nothing)),
    )
end

function build_interpreted_evaluation_fws(
        interp_f64::Interpreter{Vector{ComplexF64}},
        interp_df64::Interpreter{Vector{ComplexDF64}},
        interp_jac::Interpreter{Vector{ComplexF64}},
    )
    return (
        SysEvalFW(
            (u, x, p) -> (execute_eval_fw!(u, interp_f64, x, p); nothing),
        ),
        interpreted_df64_fws(interp_df64)...,
        SysEvalJacFW(
            (u, U, x, p) -> (execute_jac_fw!(u, U, interp_jac, x, p); nothing),
        ),
    )
end

function build_system_evaluator(
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
    param_taylor = iszero(nparams) ?
        build_zero_parameter_taylor_fws(
            taylor_1, taylor_2, taylor_3, interp_t1, interp_t2, interp_t3,
        ) :
        build_parameter_taylor_fws(
            taylor_1, taylor_2, taylor_3, interp_t1, interp_t2, interp_t3,
        )
    evaluation_fws = build_interpreted_evaluation_fws(
        interp_f64, interp_df64, interp_jac,
    )
    clone = SystemFactory(
        InterpretedCloner(
            interp_f64.sequence, interp_jac.sequence, neqs, nvars, nparams,
        ),
    )
    return SystemEvaluator(
        evaluation_fws...,
        taylor_1,
        taylor_2,
        taylor_3,
        param_taylor...,
        (neqs, nvars),
        nparams,
        clone,
    )
end

function build_zero_parameter_taylor_fws(
        taylor_1::SysTaylor1FW,
        taylor_2::SysTaylor2FW,
        taylor_3::SysTaylor3FW,
        ::Interpreter{Vector{TruncatedTaylorSeries{2, ComplexF64}}},
        ::Interpreter{Vector{TruncatedTaylorSeries{3, ComplexF64}}},
        ::Interpreter{Vector{TruncatedTaylorSeries{4, ComplexF64}}},
    )
    return (
        SysTaylor1ParamFW((u, tx, ::TaylorVector{2, ComplexF64}) -> (taylor_1(u, tx, EMPTY_PARAMS); nothing)),
        SysTaylor2ParamFW((u, tx, ::TaylorVector{3, ComplexF64}) -> (taylor_2(u, tx, EMPTY_PARAMS); nothing)),
        SysTaylor3ParamFW((u, tx, ::TaylorVector{4, ComplexF64}) -> (taylor_3(u, tx, EMPTY_PARAMS); nothing)),
    )
end

function build_parameter_taylor_fws(
        ::SysTaylor1FW, ::SysTaylor2FW, ::SysTaylor3FW,
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
