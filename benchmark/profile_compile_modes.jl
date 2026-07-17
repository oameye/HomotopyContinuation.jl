# Flat profile of a full serial solve in INTERPRETED and COMPILED mode.
# Shows where solve time actually goes per mode. The mode-independent share
# (custom LU, ldiv, iterative refinement, norms, predictor logic, copies) is
# what caps the end-to-end gain from compiled kernels.
# Run standalone: julia --project=benchmark benchmark/profile_compile_modes.jl

using Profile
using DynamicPolynomials: @polyvar
using HomotopyContinuationNext: System, solve, CompileMode, TotalDegree, Serial

function _katsura(vars, n)
    lin = vars[1] + sum(2vars[i] for i in 2:(n + 1)) - 1
    eqs = [lin]
    for l in 0:(n - 1)
        eq = -vars[l + 1]
        for i in (-n):n
            j = l - i
            abs(i) <= n && abs(j) <= n && (eq += vars[abs(i) + 1] * vars[abs(j) + 1])
        end
        push!(eqs, eq)
    end
    return eqs
end

function flat_report(sys, label; reps = 300)
    solve(sys, TotalDegree(), Serial(); show_progress = false)
    Profile.clear()
    @profile for _ in 1:reps
        solve(sys, TotalDegree(), Serial(); show_progress = false)
    end
    io = IOBuffer()
    Profile.print(io; format = :flat, sortedby = :count, mincount = 20, C = false)
    txt = String(take!(io))
    println("\n════ flat profile: $label ════")
    for (i, line) in enumerate(split(txt, '\n'))
        i > 60 && break
        println(line)
    end
    return nothing
end

const N = 5
@polyvar kv[1:(N + 1)]
F = _katsura(kv, N)

flat_report(System(F; compile = CompileMode.INTERPRETED), "INTERPRETED")
flat_report(System(F; compile = CompileMode.COMPILED), "COMPILED")
