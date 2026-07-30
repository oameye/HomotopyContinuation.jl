## Wire formats for the system types a builder carries.
#
# A `SystemEvaluator` is a bundle of `FunctionWrapper`s closing over interpreter
# tapes, whose gensym'd types the generic serializer would have to match across
# processes. The instruction sequences are plain data and are the source of truth,
# so each of these ships them and rebuilds the evaluator on the far side.

const AbstractSer = Serialization.AbstractSerializer

# ── SystemEvaluator ─────────────────────────────────────────────────────────
#
# The kernel wrappers cannot cross a process; the rebuild thunk closes over plain
# data, so ship that and call it. This is how a homotopy reaches another process.
function Serialization.serialize(s::AbstractSer, ev::HCN.SystemEvaluator)
    Serialization.serialize_type(s, HCN.SystemEvaluator)
    Serialization.serialize(s, ev._clone.obj)
    return nothing
end

function Serialization.deserialize(
        s::AbstractSer, ::Type{HCN.SystemEvaluator},
    )::HCN.SystemEvaluator
    return _cloner(Serialization.deserialize(s))()::HCN.SystemEvaluator
end

# `FunctionWrapper` keeps an immutable callable behind a `Ref`, a mutable one bare.
_cloner(obj::Base.RefValue) = obj[]
_cloner(obj) = obj

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
    Serialization.serialize(s, HCN._lowered(sys))
    return nothing
end

function Serialization.deserialize(
        s::AbstractSer, ::Type{HCN.System{P, V, M, S}},
    )::HCN.System{P, V, M, S} where {P, V, M, S}
    polys = Serialization.deserialize(s)::Vector{P}
    parameters = Serialization.deserialize(s)::Vector{V}
    variables = Serialization.deserialize(s)::Vector{V}
    lowered = Serialization.deserialize(s)::HCN.LoweredInput

    return HCN._build_compiled_system(
        _compile_strategy(M), polys, variables, parameters, lowered,
        length(polys), length(variables), length(parameters), S(),
    )::HCN.System{P, V, M, S}
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

# ── FixedParameterSystem ────────────────────────────────────────────────────

# Only the source and the values ship; the binding is rebuilt on the far side.
function Serialization.serialize(s::AbstractSer, F::HCN.FixedParameterSystem)
    Serialization.serialize_type(s, typeof(F))
    Serialization.serialize(s, F.system)
    Serialization.serialize(s, F.parameters)
    return nothing
end

function Serialization.deserialize(
        s::AbstractSer, ::Type{HCN.FixedParameterSystem{S}},
    )::HCN.FixedParameterSystem{S} where {S}
    system = Serialization.deserialize(s)::S
    values = Serialization.deserialize(s)::Vector{ComplexF64}
    return HCN.FixedParameterSystem(system, values)
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
