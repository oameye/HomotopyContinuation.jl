# Parameter and homotopy continuation

const CONTINUATION = (
    :parameter_interpreted_serial,
    :parameter_compiled_all_threaded,
    :system_to_system_homotopy,
    :parameter_homotopy_solve,
    :parameter_sweep,
    :subspace_continuation_intrinsic,
    :subspace_continuation_extrinsic,
    :subspace_sweep_intrinsic,
    :subspace_sweep_extrinsic,
    :linear_subspace_homotopy,
    :find_start_pair,
    :path_info,
    :newton_standard,
    :newton_extended_underdetermined,
)

function run(::Val{:parameter_interpreted_serial})
    @polyvar x p
    F = System([x^2 - p]; variables = [x], parameters = [p])
    return solve(
        F, [[1.0 + 0.0im]], [1.0 + 0.0im], [9.0 + 0.0im],
        Continuation(; seed = UInt32(1), show_progress = false), Serial(),
    )
end

function run(::Val{:parameter_compiled_all_threaded})
    @polyvar x p
    F = System(
        [x^2 - p]; variables = [x], parameters = [p], compile = CompileMode.COMPILED_ALL,
    )
    return solve(
        F, [[1.0 + 0.0im]], [1.0 + 0.0im], [9.0 + 0.0im],
        Continuation(; seed = UInt32(1), show_progress = false), Threaded(1),
    )
end

function run(::Val{:system_to_system_homotopy})
    @polyvar x y
    G = System([x^2 - 1, y^2 - 1]; variables = [x, y])
    F = System([x^2 + y - 1, x * y - 0.5]; variables = [x, y])
    starts = [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
    return solve(G, F, starts, Continuation(; seed = SEED, show_progress = false), Serial())
end

function parameter_homotopy()
    @polyvar x y a b
    F = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
    return ParameterHomotopy(F, [1.0, 0.0], [2.0, 4.0])
end

run(::Val{:parameter_homotopy_solve}) = solve(
    parameter_homotopy(), [[1.0, 1.0]], Continuation(; show_progress = false), Serial(),
)

function run(::Val{:parameter_sweep})
    @polyvar x y a b c
    F = System(
        [x^2 + y^2 - 1, a * x + b * y + c];
        variables = [x, y], parameters = [a, b, c],
    )
    rng = MersenneTwister(17)
    p₀ = randn(rng, ComplexF64, 3)
    starts = [[1.0 + 0.0im, 0.0 + 0.0im], [-1.0 + 0.0im, 0.0 + 0.0im]]
    targets = [randn(rng, 3) for _ in 1:3]
    return solve(
        F, starts, p₀, targets, Sweep(; seed = SEED, show_progress = false), Serial(),
    )
end

function conic_line_starts()
    F, L₀ = conic_and_line()
    starts = solutions(solve(F, L₀, TotalDegree(; seed = SEED, show_progress = false), Serial()))
    return F, starts, L₀
end

function subspace_continuation(coords::SubspaceCoords.T)
    F, starts, L₀ = conic_line_starts()
    L₁ = rand_subspace(2; codim = 1)
    return solve(
        F, starts, L₀, L₁,
        Continuation(; coords = coords, seed = SEED, show_progress = false), Serial(),
    )
end

run(::Val{:subspace_continuation_intrinsic}) =
    subspace_continuation(SubspaceCoords.INTRINSIC)
run(::Val{:subspace_continuation_extrinsic}) =
    subspace_continuation(SubspaceCoords.EXTRINSIC)

function subspace_sweep(coords::SubspaceCoords.T)
    F, starts, L₀ = conic_line_starts()
    Random.seed!(23)
    targets = [rand_subspace(2; codim = 1) for _ in 1:3]
    return solve(
        F, starts, L₀, targets,
        Sweep(; coords = coords, seed = SEED, show_progress = false), Serial(),
    )
end

run(::Val{:subspace_sweep_intrinsic}) = subspace_sweep(SubspaceCoords.INTRINSIC)
run(::Val{:subspace_sweep_extrinsic}) = subspace_sweep(SubspaceCoords.EXTRINSIC)

function run(::Val{:linear_subspace_homotopy})
    Random.seed!(14)
    @polyvar w[1:3]
    F = System([w[1]^2 + w[2]^2 - w[3]^2]; variables = w)
    V = rand_subspace(3; dim = 2, affine = false)
    W = rand_subspace(3; dim = 2, affine = false)
    starts = solutions(solve(F, V, TotalDegree(; seed = SEED, show_progress = false), Serial()))
    return with_linear_subspace_homotopy(F, V, W) do H
        solve(H, starts, Continuation(; seed = SEED, show_progress = false), Serial())
    end
end

function run(::Val{:find_start_pair})
    @var x y q1 q2
    family = System(
        [x^2 + y^2 - q1, x + y - q2]; variables = [x, y], parameters = [q1, q2],
    )
    rng = MersenneTwister(0x5cc2)
    inner = System(randn(rng, ComplexF64, 2, 2) * [x, y] + randn(rng, ComplexF64, 2); variables = [x, y])
    return find_start_pair(family ∘ inner)
end

run(::Val{:path_info}) = path_info(parameter_homotopy(), [1.0, 1.0], 1.0, 0.0)

function run(::Val{:newton_standard})
    @polyvar x y
    F = System([x^2 + y^2 - 1, x - y]; variables = [x, y])
    root = inv(sqrt(2.0))
    return newton(F, [root + 0.05, root - 0.05])
end

function run(::Val{:newton_extended_underdetermined})
    @polyvar x y
    F = System([x^2 + y^2 - 1]; variables = [x, y])
    return newton(F, [1.1, 0.2]; extended_precision = true)
end
