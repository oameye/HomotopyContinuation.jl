# Precompile directives for the most common solve pipeline types.
# These use `precompile(f, types)` which only compiles (no execution),
# so they avoid the SymEngine reinitialization issues that block @compile_workload.

# Common matrix type for small systems (n <= 25)
const _MC = Matrix{ComplexF64}

# -- Parameter Homotopy path (InterpretedSystem) --
const _PH_I = ParameterHomotopy{InterpretedSystem}
const _ET_PH_I = EndgameTracker{_PH_I,_MC}
const _T_PH_I = Tracker{_PH_I,_MC}
const _S_PH_I = Solver{_ET_PH_I}

# -- StraightLineHomotopy path (InterpretedSystem, used by total_degree) --
const _SLH_I = StraightLineHomotopy{InterpretedSystem,InterpretedSystem}
const _ET_SLH_I = EndgameTracker{_SLH_I,_MC}
const _T_SLH_I = Tracker{_SLH_I,_MC}

# -- AffineChartHomotopy variants --
const _ACH_SLH_I = AffineChartHomotopy{_SLH_I}
const _ET_ACH_SLH_I = EndgameTracker{_ACH_SLH_I,_MC}
const _T_ACH_SLH_I = Tracker{_ACH_SLH_I,_MC}

# -- CoefficientHomotopy path (used by polyhedral) --
const _CH_I = CoefficientHomotopy{InterpretedSystem}
const _TH_I = ToricHomotopy{InterpretedSystem}
const _PT_I = PolyhedralTracker{_TH_I,_CH_I,_MC}
const _S_PT_I = Solver{_PT_I}

# Tracker core: init!, step!
for _T in (_T_PH_I, _T_SLH_I, _T_ACH_SLH_I)
    precompile(Tuple{typeof(init!),_T,Vector{ComplexF64},Float64,Float64})
    precompile(Tuple{typeof(step!),_T})
    precompile(Tuple{typeof(step!),_T,Bool})
end

# EndgameTracker: init!, step!, track
for _ET in (_ET_PH_I, _ET_SLH_I, _ET_ACH_SLH_I)
    precompile(Tuple{typeof(init!),_ET,Vector{ComplexF64},Float64})
    precompile(Tuple{typeof(step!),_ET})
    precompile(Tuple{typeof(track),_ET,Vector{ComplexF64},Float64})
end

# Solver: serial_solve, solve
for _S in (_S_PH_I,)
    precompile(Tuple{typeof(serial_solve),_S,Vector{Vector{ComplexF64}},Nothing,typeof(always_false)})
    precompile(Tuple{typeof(solve),_S,Vector{Vector{ComplexF64}}})
end

# PolyhedralTracker track
precompile(Tuple{
    typeof(track),
    _PT_I,
    Tuple{MixedSubdivisions.MixedCell,Vector{ComplexF64}},
})

# Result construction
precompile(Tuple{typeof(Result),Vector{PathResult}})

# MatrixWorkspace / Jacobian / linear algebra (for small systems)
precompile(Tuple{typeof(MatrixWorkspace),Matrix{ComplexF64}})
precompile(Tuple{typeof(Jacobian),Matrix{ComplexF64}})

# -- ModelKit: InterpretedSystem construction chain --
# These precompile the system construction pipeline that dominates TTFX.
const _IS = ModelKit.InstructionSequence
const _IR = ModelKit.IntermediateRepresentation
const _Interp = ModelKit.Interpreter

precompile(Tuple{Type{InterpretedSystem},System})
precompile(Tuple{typeof(ModelKit.instruction_sequence),System})
precompile(Tuple{typeof(ModelKit.instruction_sequence),_IR})
precompile(Tuple{typeof(ModelKit.interpreter),Type{Vector{ComplexF64}},_IS})
precompile(Tuple{typeof(ModelKit.interpreter),Type{ComplexF64},System})
precompile(Tuple{typeof(ModelKit.jacobian_interpreter),Type{ComplexF64},System})
precompile(Tuple{typeof(ModelKit.interpreter),Type{Vector{ComplexF64}},_Interp{Vector{ComplexF64}}})

# FixedParameterSystem construction
const _FPS_I = FixedParameterSystem{InterpretedSystem,Float64}
const _FPS_CF = FixedParameterSystem{InterpretedSystem,ComplexF64}
const _SLH_FPS = StraightLineHomotopy{_FPS_I,InterpretedSystem}
const _ET_SLH_FPS = EndgameTracker{_SLH_FPS,_MC}
const _T_SLH_FPS = Tracker{_SLH_FPS,_MC}

precompile(Tuple{typeof(init!),_T_SLH_FPS,Vector{ComplexF64},Float64,Float64})
precompile(Tuple{typeof(step!),_T_SLH_FPS})
precompile(Tuple{typeof(init!),_ET_SLH_FPS,Vector{ComplexF64},Float64})
precompile(Tuple{typeof(step!),_ET_SLH_FPS})
precompile(Tuple{typeof(track),_ET_SLH_FPS,Vector{ComplexF64},Float64})

# Polyhedral tracker internals
const _T_TH = Tracker{_TH_I,_MC}
const _ET_CH = EndgameTracker{_CH_I,_MC}
const _T_CH = Tracker{_CH_I,_MC}
precompile(Tuple{typeof(init!),_T_TH,Vector{ComplexF64},Float64,Float64})
precompile(Tuple{typeof(step!),_T_TH})
precompile(Tuple{typeof(init!),_T_CH,Vector{ComplexF64},Float64,Float64})
precompile(Tuple{typeof(step!),_T_CH})
precompile(Tuple{typeof(init!),_ET_CH,Vector{ComplexF64},Float64})
precompile(Tuple{typeof(track),_ET_CH,Vector{ComplexF64},Float64})

# -- Polyhedral full pipeline --
# These are the big TTFX costs: polyhedral(), collect(starts), solve(solver, starts)
const _PSI = PolyhedralStartSolutionsIterator{Vector{MixedSubdivisions.MixedCell}}
const _PT_EL = Tuple{MixedSubdivisions.MixedCell,Vector{ComplexF64}}

precompile(Tuple{typeof(polyhedral),System})
precompile(Tuple{typeof(collect),_PSI})
precompile(Tuple{typeof(solve),_S_PT_I,Vector{_PT_EL}})
precompile(Tuple{typeof(serial_solve),_S_PT_I,Vector{_PT_EL},Nothing,typeof(always_false)})
precompile(Tuple{typeof(track),_PT_I,_PT_EL})

# solver_startsolutions entry point
precompile(Tuple{typeof(solver_startsolutions),System})
precompile(Tuple{typeof(solver_startsolutions),System,Nothing})

# solve entry points — including kwcall forms for common keyword combinations
precompile(Tuple{typeof(solve),System})
precompile(Tuple{
    typeof(Core.kwcall),
    @NamedTuple{show_progress::Bool, compile::Val{false}},
    typeof(solve),
    System,
})
precompile(Tuple{
    typeof(Core.kwcall),
    @NamedTuple{show_progress::Bool, compile::Val{:none}},
    typeof(solve),
    System,
})
precompile(Tuple{
    typeof(Core.kwcall),
    @NamedTuple{show_progress::Bool},
    typeof(solve),
    System,
})
