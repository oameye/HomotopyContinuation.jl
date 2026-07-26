# Each executor is reached only through the `@cfunction` inside a
# `FunctionWrapper`, which leaves no backedge, so no workload caches these
# implicitly. A tape is data, not a type parameter, so the signatures name no
# user type and declaring them is enough. Each target must stay `@noinline`.

let
    V = FSVec{ComplexF64}
    M = FSMat{ComplexF64}
    VD = FSVec{ComplexDF64}
    interp_f64 = Interpreter{Vector{ComplexF64}}
    interp_df64 = Interpreter{Vector{ComplexDF64}}

    precompile(_execute_eval_fw!, (V, interp_f64, V, V))
    precompile(_execute_jac_fw!, (V, M, interp_f64, V, V))
    precompile(_execute_eval_fw!, (V, interp_df64, VD, V))
    precompile(_execute_eval_fw!, (VD, interp_df64, VD, V))

    for (K, N) in ((1, 2), (2, 3), (3, 4))
        interp_taylor = Interpreter{Vector{TruncatedTaylorSeries{N, ComplexF64}}}
        tv = TaylorVector{N, ComplexF64}
        precompile(execute_taylor!, (V, Val{K}, interp_taylor, tv, V))
        precompile(execute_taylor!, (V, Val{K}, interp_taylor, tv, tv))
    end
end
