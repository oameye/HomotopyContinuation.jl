# Slices, witness sets and numerical algebraic geometry

const GEOMETRY = (
    :slice_solve,
    :slice_solve_projective,
    :result_iterator_lazy,
    :witness_set_build,
    :membership,
    :trace_test,
    :regeneration,
    :regeneration_unsorted,
    :intersection,
    :decomposition,
)

function run(::Val{:slice_solve})
    F, L = conic_and_line()
    return solve(F, L, TotalDegree(; seed = SEED, show_progress = false), Serial())
end

function run(::Val{:slice_solve_projective})
    @polyvar x y z
    F = System([x^2 + y^2 - z^2]; variables = [x, y, z])
    Random.seed!(6)
    L = rand_subspace(3; codim = 1, affine = false)
    return solve(F, L, TotalDegree(; seed = SEED, show_progress = false), Serial())
end

function run(::Val{:result_iterator_lazy})
    F, L = conic_and_line()
    ri = result_iterator(F, L, TotalDegree(; seed = SEED))
    return first(ri), Result(restrict(ri, selection(is_real, ri)))
end

function sphere_witness_set()
    @polyvar x y z
    F = System([x^2 + y^2 + z^2 - 1]; variables = [x, y, z])
    return solve(F, Witness(; dim = 2, seed = SEED, show_progress = false), Serial())
end

run(::Val{:witness_set_build}) = sphere_witness_set()

function run(::Val{:membership})
    W = sphere_witness_set()
    return membership(
        ComplexF64[0.6, 0.8, 0.0], W, Membership(; seed = SEED, show_progress = false),
        Serial(),
    )
end

function run(::Val{:trace_test})
    @polyvar z[1:3]
    Q = System([z[1]^4 + 2z[2]^4 + 3z[3]^4 + z[1] * z[2] - 1]; variables = z)
    W = solve(Q, Witness(; seed = UInt32(5), show_progress = false), Serial())
    return trace_test(W; seed = UInt32(1))
end

function sphere_and_plane()
    @polyvar x y z
    return System([(x^2 + y^2 + z^2 - 1) * (x - 0.5), (x^2 + y^2 + z^2 - 1) * (y - 0.5)]; variables = [x, y, z])
end

run(::Val{:regeneration}) =
    solve(sphere_and_plane(), Regeneration(; seed = SEED, show_progress = false), Serial())
run(::Val{:regeneration_unsorted}) = solve(
    sphere_and_plane(),
    Regeneration(; sorted = EquationSorting.UNSORTED, seed = SEED, show_progress = false),
    Serial(),
)

function run(::Val{:intersection})
    @polyvar x y z
    W = solve(
        System([x^2 + y^2 + z^2 - 1]; variables = [x, y, z]),
        Witness(; seed = SEED, show_progress = false), Serial(),
    )
    return intersect(W, x + y + z - 1, Intersection(; seed = SEED, show_progress = false), Serial())
end

run(::Val{:decomposition}) =
    solve(sphere_and_plane(), Decomposition(; seed = SEED, show_progress = false), Serial())
