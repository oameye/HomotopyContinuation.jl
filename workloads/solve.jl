# Total degree and polyhedral start systems

const SOLVE = (
    :total_degree_interpreted_serial,
    :total_degree_compiled_serial,
    :total_degree_compiled_all_serial,
    :total_degree_interpreted_threaded,
    :polyhedral_interpreted_serial,
    :polyhedral_compiled_serial,
    :polyhedral_compiled_all_serial,
    :polyhedral_interpreted_threaded,
    :expression_frontend,
    :overdetermined_total_degree,
    :overdetermined_polyhedral,
    :singular_endgame,
    :progress_enabled,
    :large_symbolic_interpreted_build,
    :large_symbolic_compiled_all_build,
    :composition_solve,
    :fixed_parameters_solve,
)

run(::Val{:total_degree_interpreted_serial}) =
    solve_total_degree(CompileMode.INTERPRETED, Serial())
run(::Val{:total_degree_compiled_serial}) =
    solve_total_degree(CompileMode.COMPILED, Serial())
run(::Val{:total_degree_compiled_all_serial}) =
    solve_total_degree(CompileMode.COMPILED_ALL, Serial())
run(::Val{:total_degree_interpreted_threaded}) =
    solve_total_degree(CompileMode.INTERPRETED, Threaded(1))

run(::Val{:polyhedral_interpreted_serial}) =
    solve_polyhedral(CompileMode.INTERPRETED, Serial())
run(::Val{:polyhedral_compiled_serial}) =
    solve_polyhedral(CompileMode.COMPILED, Serial())
run(::Val{:polyhedral_compiled_all_serial}) =
    solve_polyhedral(CompileMode.COMPILED_ALL, Serial())
run(::Val{:polyhedral_interpreted_threaded}) =
    solve_polyhedral(CompileMode.INTERPRETED, Threaded(1))

function run(::Val{:expression_frontend})
    @var x y
    F = System([x^2 + y - 1, x * y - 0.5]; variables = [x, y])
    return solve(F, TotalDegree(; seed = SEED, show_progress = false), Serial())
end

function overdetermined_system()
    @polyvar x y
    return System([x^2 - 1, y^2 - 1, x * y - 1]; variables = [x, y])
end

run(::Val{:overdetermined_total_degree}) = solve(
    overdetermined_system(), TotalDegree(; seed = UInt32(0x42), show_progress = false),
    Serial(),
)
run(::Val{:overdetermined_polyhedral}) = solve(
    overdetermined_system(), Polyhedral(; seed = UInt32(0x42), show_progress = false),
    Serial(),
)

function run(::Val{:singular_endgame})
    @polyvar x y
    F = System([(x - 1)^2, y - 1]; variables = [x, y])
    return solve(F, TotalDegree(; seed = UInt32(3), show_progress = false), Serial())
end

run(::Val{:progress_enabled}) =
    solve_total_degree(CompileMode.INTERPRETED, Serial(); progress = true)

function katsura_system(n, mode)
    @polyvar x[1:(n + 1)]
    equations = [x[1] + sum(2x[i] for i in 2:(n + 1)) - 1]
    for l in 0:(n - 1)
        equation = -x[l + 1]
        for i in (-n):n
            j = l - i
            abs(j) <= n && (equation += x[abs(i) + 1] * x[abs(j) + 1])
        end
        push!(equations, equation)
    end
    return System(equations; variables = x, compile = mode)
end

run(::Val{:large_symbolic_interpreted_build}) =
    katsura_system(3, CompileMode.INTERPRETED)
run(::Val{:large_symbolic_compiled_all_build}) =
    katsura_system(3, CompileMode.COMPILED_ALL)

function run(::Val{:composition_solve})
    @var x y
    outer = System([x^2 + y^2 - 2, x - y]; variables = [x, y])
    inner = System([x + 0.5y, y - 0.25x]; variables = [x, y])
    return solve(
        compose(outer, inner), TotalDegree(; seed = SEED, show_progress = false), Serial(),
    )
end

function run(::Val{:fixed_parameters_solve})
    @polyvar x y a
    F = System([x^2 + y^2 - a, x - y]; variables = [x, y], parameters = [a])
    return solve(
        fix_parameters(F, [2.0]), TotalDegree(; seed = SEED, show_progress = false),
        Serial(),
    )
end
