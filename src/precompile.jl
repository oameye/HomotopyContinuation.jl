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
