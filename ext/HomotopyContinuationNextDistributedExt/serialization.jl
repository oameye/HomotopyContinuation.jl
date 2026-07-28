## Wire formats for the system types a builder carries.
#
# A `SystemEvaluator` is a bundle of `FunctionWrapper`s closing over interpreter
# tapes, whose gensym'd types the generic serializer would have to match across
# processes. The instruction sequences are plain data and are the source of truth,
# so each of these ships them and rebuilds the evaluator on the far side.

const AbstractSer = Serialization.AbstractSerializer

function _compile_strategy(mode::HCN.CompileMode.T)
    mode === HCN.CompileMode.INTERPRETED && return HCN.InterpretedCompile()
    mode === HCN.CompileMode.COMPILED && return HCN.CompiledCompile()
    return HCN.CompiledAllCompile()
end

# ── System ──────────────────────────────────────────────────────────────────

function Serialization.serialize(s::AbstractSer, sys::HCN.System)
    Serialization.serialize_type(s, typeof(sys))
    Serialization.serialize(s, collect(sys.polys))
    Serialization.serialize(s, collect(sys.parameters))
    Serialization.serialize(s, collect(sys.variables))
    Serialization.serialize(s, sys.degrees)
    Serialization.serialize(s, sys.equation_scales)
    Serialization.serialize(s, sys.nvars)
    Serialization.serialize(s, sys.nparams)
    Serialization.serialize(s, sys.variable_groups)
    Serialization.serialize(s, sys.is_homogeneous)
    Serialization.serialize(s, sys._interp_f64.sequence)
    Serialization.serialize(s, sys._interp_jac.sequence)
    Serialization.serialize(s, size(sys.evaluator))
    Serialization.serialize(s, HCN.nparameters(sys.evaluator))
    return nothing
end

function Serialization.deserialize(
        s::AbstractSer, ::Type{HCN.System{P, V, M, S}},
    )::HCN.System{P, V, M, S} where {P, V, M, S}
    polys = Serialization.deserialize(s)::Vector{P}
    parameters = Serialization.deserialize(s)::Vector{V}
    variables = Serialization.deserialize(s)::Vector{V}
    degrees = Serialization.deserialize(s)::Vector{Int}
    equation_scales = Serialization.deserialize(s)::Vector{Float64}
    nvars = Serialization.deserialize(s)::Int
    nparams = Serialization.deserialize(s)::Int
    variable_groups = Serialization.deserialize(s)::Vector{Vector{Int}}
    is_homogeneous = Serialization.deserialize(s)::Bool
    seq_eval = Serialization.deserialize(s)::HCN.InstructionSequence
    seq_jac = Serialization.deserialize(s)::HCN.InstructionSequence
    neqs, nvariables = Serialization.deserialize(s)::Tuple{Int, Int}
    neval_params = Serialization.deserialize(s)::Int

    interp_f64 = HCN.Interpreter(Vector{ComplexF64}, seq_eval)
    interp_df64 = HCN.Interpreter(Vector{HCN.ComplexDF64}, seq_eval)
    interp_jac = HCN.Interpreter(Vector{ComplexF64}, seq_jac)
    interp_t1 = HCN.Interpreter(
        Vector{HCN.TruncatedTaylorSeries{2, ComplexF64}}, seq_eval,
    )
    interp_t2 = HCN.Interpreter(
        Vector{HCN.TruncatedTaylorSeries{3, ComplexF64}}, seq_eval,
    )
    interp_t3 = HCN.Interpreter(
        Vector{HCN.TruncatedTaylorSeries{4, ComplexF64}}, seq_eval,
    )
    evaluator = HCN._build_mode_evaluator(
        _compile_strategy(M), seq_eval, seq_jac,
        interp_f64, interp_df64, interp_jac, interp_t1, interp_t2, interp_t3,
        neqs, nvariables, neval_params,
    )

    return HCN.System{P, V, M, S}(
        HCN._to_fsvec(polys), HCN._to_fsvec(parameters), HCN._to_fsvec(variables),
        evaluator, degrees, equation_scales, nvars, nparams,
        variable_groups, is_homogeneous,
        HCN.LazyRef{HCN.SupportCoefficients}(),
        interp_f64, interp_df64, interp_jac, interp_t1, interp_t2, interp_t3,
        M,
    )
end

# ── _SupportSystem ──────────────────────────────────────────────────────────

function Serialization.serialize(s::AbstractSer, sys::HCN._SupportSystem)
    Serialization.serialize_type(s, typeof(sys))
    Serialization.serialize(s, sys.eval_sequence)
    Serialization.serialize(s, sys.jacobian_sequence)
    Serialization.serialize(s, size(sys.evaluator))
    Serialization.serialize(s, HCN.nparameters(sys.evaluator))
    return nothing
end

function Serialization.deserialize(
        s::AbstractSer, ::Type{HCN._SupportSystem},
    )::HCN._SupportSystem
    eval_sequence = Serialization.deserialize(s)::HCN.InstructionSequence
    jacobian_sequence = Serialization.deserialize(s)::HCN.InstructionSequence
    neqs, nvariables = Serialization.deserialize(s)::Tuple{Int, Int}
    nparams = Serialization.deserialize(s)::Int
    return HCN._SupportSystem(
        HCN._support_evaluator(
            eval_sequence, jacobian_sequence, neqs, nvariables, nparams,
        ),
        eval_sequence, jacobian_sequence,
    )
end

# ── CompositionSystem ───────────────────────────────────────────────────────

# A stage holds its system only inside its factory thunk, and every stage was
# built from one `System`, so the stage vector is rebuilt from those systems.
function Serialization.serialize(s::AbstractSer, C::HCN.CompositionSystem)
    Serialization.serialize_type(s, typeof(C))
    Serialization.serialize(s, [HCN._stage_system(stage) for stage in C.stages])
    Serialization.serialize(s, C.variables)
    Serialization.serialize(s, C.parameters)
    Serialization.serialize(s, C.degrees)
    Serialization.serialize(s, C.is_homogeneous)
    return nothing
end

function Serialization.deserialize(
        s::AbstractSer, ::Type{HCN.CompositionSystem},
    )::HCN.CompositionSystem
    stage_systems = Serialization.deserialize(s)::Vector
    variables = Serialization.deserialize(s)::Vector{HCN.Expression}
    parameters = Serialization.deserialize(s)::Vector{HCN.Expression}
    degrees = Serialization.deserialize(s)::Vector{Int}
    is_homogeneous = Serialization.deserialize(s)::Bool
    stages = HCN.CompositionStage[
        only(HCN._composition_stages(sys::HCN.System)) for sys in stage_systems
    ]
    return HCN.CompositionSystem(
        stages, HCN._fold_composition(stages), variables, parameters,
        degrees, is_homogeneous,
    )
end
