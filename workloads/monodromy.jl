# Monodromy

const MONODROMY = (
    :monodromy_serial,
    :monodromy_threaded,
    :monodromy_group_action,
    :monodromy_subspace_trace,
    :monodromy_continuation,
    :verify_completeness,
)

function monodromy_system()
    @polyvar y[1:2] p[1:2]
    return System(
        [y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p,
    )
end

monodromy_algorithm(seed) =
    Monodromy(; target_solutions_count = 2, seed = UInt32(seed), show_progress = false)

run(::Val{:monodromy_serial}) = solve(monodromy_system(), monodromy_algorithm(7), Serial())
run(::Val{:monodromy_threaded}) = solve(monodromy_system(), monodromy_algorithm(7), Threaded())

function run(::Val{:monodromy_group_action})
    @polyvar x p
    F = System([x^2 - p]; variables = [x], parameters = [p])
    return solve(
        F, [[2.0 + 0.0im]], [4.0 + 0.0im],
        Monodromy(;
            group_action = solution -> ([-solution[1]],),
            seed = UInt32(11), show_progress = false,
        ),
        Serial(),
    )
end

function run(::Val{:monodromy_subspace_trace})
    @polyvar z[1:3]
    F = System([z[1]^2 + 2z[2]^2 + 3z[3]^2 + z[1] * z[2] - 1]; variables = z)
    return solve(F, Monodromy(; dim = 2, seed = UInt32(99), show_progress = false), Serial())
end

function run(::Val{:monodromy_continuation})
    F = monodromy_system()
    R = solve(F, monodromy_algorithm(7), Serial())
    return solve(
        F, R, ComplexF64[2.0, 1.0], Continuation(; seed = SEED, show_progress = false),
        Serial(),
    )
end

function run(::Val{:verify_completeness})
    F = monodromy_system()
    alg = monodromy_algorithm(21)
    return verify_solution_completeness(F, solve(F, alg, Serial()), alg, Serial())
end
