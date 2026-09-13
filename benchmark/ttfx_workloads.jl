module TTFXWorkloads

using HomotopyContinuationNext
using HomotopyContinuationNext:
    AffineChartHomotopy,
    EndgameTracker,
    ExtrinsicSubspaceHomotopy,
    GroupActions,
    HomotopyEvaluator,
    IntrinsicSubspaceHomotopy,
    LinearSubspace,
    PathResult,
    PathResultCode,
    Result,
    Tracker,
    UniquePoints,
    DEFAULT_CLUSTER_ATOL,
    DEFAULT_CLUSTER_RTOL,
    _cluster_solutions,
    add!,
    ambient_coordinates!,
    intrinsic_coordinates!,
    linear_subspace_homotopy,
    on_chart!,
    rand_subspace,
    track!,
    verify_solution_completeness
using Random: MersenneTwister, Random, randn

const WORKLOADS = (
    :total_degree_interpreted_serial,
    :total_degree_compiled_serial,
    :total_degree_compiled_all_serial,
    :total_degree_interpreted_threaded,
    :polyhedral_interpreted_serial,
    :polyhedral_compiled_serial,
    :polyhedral_compiled_all_serial,
    :polyhedral_interpreted_threaded,
    :parameter_interpreted_serial,
    :parameter_compiled_all_threaded,
    :overdetermined_total_degree,
    :overdetermined_polyhedral,
    :singular_endgame,
    :large_symbolic_interpreted_build,
    :large_symbolic_compiled_all_build,
    :newton_standard,
    :newton_extended_underdetermined,
    :intrinsic_subspace_track,
    :extrinsic_subspace_track,
    :affine_chart,
    :slice_solve,
    :slice_solve_projective,
    :witness_set_build,
    :parameter_sweep,
    :subspace_sweep_intrinsic,
    :subspace_sweep_extrinsic,
    :result_iterator_lazy,
    :monodromy_serial,
    :monodromy_threaded,
    :monodromy_group_action,
    :monodromy_subspace_trace,
    :verify_completeness,
    :unique_points_group_action,
    :result_clustering,
    :progress_enabled,
)

workloads() = WORKLOADS

function _square_system(mode)
    @polyvar x y
    return System(
        [x^2 + y - 1, x * y - 0.5];
        variables = [x, y],
        compile = mode,
    )
end

function _solve_total_degree(mode, executor; progress = false)
    F = _square_system(mode)
    return solve(
        F,
        TotalDegree(; seed = UInt32(0x1234), show_progress = progress),
        executor,
    )
end

function _solve_polyhedral(mode, executor)
    F = _square_system(mode)
    return solve(
        F,
        Polyhedral(; seed = UInt32(0x1234), show_progress = false),
        executor,
    )
end

run(::Val{:total_degree_interpreted_serial}) =
    _solve_total_degree(CompileMode.INTERPRETED, Serial())
run(::Val{:total_degree_compiled_serial}) =
    _solve_total_degree(CompileMode.COMPILED, Serial())
run(::Val{:total_degree_compiled_all_serial}) =
    _solve_total_degree(CompileMode.COMPILED_ALL, Serial())
run(::Val{:total_degree_interpreted_threaded}) =
    _solve_total_degree(CompileMode.INTERPRETED, Threaded(1))

run(::Val{:polyhedral_interpreted_serial}) =
    _solve_polyhedral(CompileMode.INTERPRETED, Serial())
run(::Val{:polyhedral_compiled_serial}) =
    _solve_polyhedral(CompileMode.COMPILED, Serial())
run(::Val{:polyhedral_compiled_all_serial}) =
    _solve_polyhedral(CompileMode.COMPILED_ALL, Serial())
run(::Val{:polyhedral_interpreted_threaded}) =
    _solve_polyhedral(CompileMode.INTERPRETED, Threaded(1))

function run(::Val{:parameter_interpreted_serial})
    @polyvar x p
    F = System([x^2 - p]; variables = [x], parameters = [p])
    return solve(
        F,
        [[1.0 + 0.0im]],
        [1.0 + 0.0im],
        [9.0 + 0.0im],
        Continuation(; seed = UInt32(1), show_progress = false),
        Serial(),
    )
end

function run(::Val{:parameter_compiled_all_threaded})
    @polyvar x p
    F = System(
        [x^2 - p];
        variables = [x],
        parameters = [p],
        compile = CompileMode.COMPILED_ALL,
    )
    return solve(
        F,
        [[1.0 + 0.0im]],
        [1.0 + 0.0im],
        [9.0 + 0.0im],
        Continuation(; seed = UInt32(1), show_progress = false),
        Threaded(1),
    )
end

function _overdetermined_system()
    @polyvar x y
    return System([x^2 - 1, y^2 - 1, x * y - 1]; variables = [x, y])
end

function run(::Val{:overdetermined_total_degree})
    return solve(
        _overdetermined_system(),
        TotalDegree(; seed = UInt32(0x42), show_progress = false),
        Serial(),
    )
end

function run(::Val{:overdetermined_polyhedral})
    return solve(
        _overdetermined_system(),
        Polyhedral(; seed = UInt32(0x42), show_progress = false),
        Serial(),
    )
end

function run(::Val{:singular_endgame})
    @polyvar x y
    F = System([(x - 1)^2, y - 1]; variables = [x, y])
    return solve(
        F,
        TotalDegree(; seed = UInt32(3), show_progress = false),
        Serial(),
    )
end

function _katsura_system(n, mode)
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
    _katsura_system(3, CompileMode.INTERPRETED)
run(::Val{:large_symbolic_compiled_all_build}) =
    _katsura_system(3, CompileMode.COMPILED_ALL)

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

function _subspace_problem()
    @polyvar x y z
    p = (x * y - x^2) + 1 - z
    q = x^4 + x^2 - y - 1
    F = System(
        [
            p * q * (x - 3) * (x - 5),
            p * q * (y - 3) * (y - 5),
            p * (z - 3) * (z - 5),
        ];
        variables = [x, y, z],
    )
    L1 = LinearSubspace(reshape([1.0, 0.0, 0.0], 1, 3), [1.0])
    L2 = LinearSubspace(reshape([0.0, 1.0, 0.0], 1, 3), [1.0])
    start = ComplexF64[1.0, 1.0, 3.0]
    return F, L1, L2, start
end

function run(::Val{:extrinsic_subspace_track})
    F, L1, L2, start = _subspace_problem()
    H = ExtrinsicSubspaceHomotopy(F, L1, L2)
    tracker = EndgameTracker(Tracker(HomotopyEvaluator(H)))
    return track!(tracker, start)
end

function run(::Val{:intrinsic_subspace_track})
    F, L1, L2, start = _subspace_problem()
    H = IntrinsicSubspaceHomotopy(F, L1, L2)
    u = zeros(ComplexF64, size(H)[2])
    intrinsic_coordinates!(u, H, start, complex(1.0))
    tracker = EndgameTracker(Tracker(HomotopyEvaluator(H)))
    code = track!(tracker, u)
    ambient = zeros(ComplexF64, 3)
    ambient_coordinates!(ambient, H, Vector(tracker.tracker.state.x), complex(0.0))
    return code, ambient
end

function run(::Val{:affine_chart})
    Random.seed!(14)
    rng = MersenneTwister(14)
    @polyvar w[1:3]
    F = System([w[1]^2 + w[2]^2 - w[3]^2]; variables = w)
    V = rand_subspace(3; dim = 2, affine = false)
    W = rand_subspace(3; dim = 2, affine = false)
    H = linear_subspace_homotopy(F, V, W)
    x = randn(rng, ComplexF64, 3)
    H isa AffineChartHomotopy && on_chart!(x, H)
    return H, x
end

# ── Sliced, witness-set, sweep and lazy routes ────────────────────────────────

function _conic_and_line()
    Random.seed!(5)
    @polyvar x y
    F = System([x^2 + y^2 - 5]; variables = [x, y])
    return F, rand_subspace(2; codim = 1)
end

function run(::Val{:slice_solve})
    F, L = _conic_and_line()
    return solve(
        F, L, TotalDegree(; seed = UInt32(0x1234), show_progress = false), Serial(),
    )
end

function run(::Val{:slice_solve_projective})
    @polyvar x y z
    F = System([x^2 + y^2 - z^2]; variables = [x, y, z])
    Random.seed!(6)
    L = rand_subspace(3; codim = 1, affine = false)
    return solve(
        F, L, TotalDegree(; seed = UInt32(0x1234), show_progress = false), Serial(),
    )
end

function run(::Val{:witness_set_build})
    @polyvar x y z
    F = System([x^2 + y^2 + z^2 - 1]; variables = [x, y, z])
    return solve(
        F, Witness(; dim = 2, seed = UInt32(0x1234), show_progress = false), Serial(),
    )
end

function _parameter_sweep_problem()
    @polyvar x y a b c
    F = System(
        [x^2 + y^2 - 1, a * x + b * y + c];
        variables = [x, y], parameters = [a, b, c],
    )
    rng = MersenneTwister(17)
    p₀ = randn(rng, ComplexF64, 3)
    starts = [
        [1.0 + 0.0im, 0.0 + 0.0im],
        [-1.0 + 0.0im, 0.0 + 0.0im],
    ]
    return F, starts, p₀, [randn(rng, 3) for _ in 1:3]
end

function run(::Val{:parameter_sweep})
    F, starts, p₀, targets = _parameter_sweep_problem()
    return solve(
        F, starts, p₀, targets,
        Sweep(; seed = UInt32(0x1234), show_progress = false), Serial(),
    )
end

function _subspace_sweep_problem()
    F, L₀ = _conic_and_line()
    Random.seed!(23)
    targets = [rand_subspace(2; codim = 1) for _ in 1:3]
    starts = solutions(
        solve(F, L₀, TotalDegree(; show_progress = false), Serial()),
    )
    return F, starts, L₀, targets
end

function _subspace_sweep(intrinsic::Bool)
    F, starts, L₀, targets = _subspace_sweep_problem()
    return solve(
        F, starts, L₀, targets,
        Sweep(;
            intrinsic = intrinsic, seed = UInt32(0x1234), show_progress = false,
        ),
        Serial(),
    )
end

run(::Val{:subspace_sweep_intrinsic}) = _subspace_sweep(true)
run(::Val{:subspace_sweep_extrinsic}) = _subspace_sweep(false)

function run(::Val{:result_iterator_lazy})
    F, L = _conic_and_line()
    ri = result_iterator(F, L, TotalDegree(; seed = UInt32(0x1234)))
    return first(ri), Result(restrict(ri, selection(is_real, ri)))
end

function _monodromy_system()
    @polyvar y[1:2] p[1:2]
    return System(
        [y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]];
        variables = y,
        parameters = p,
    )
end

function run(::Val{:monodromy_serial})
    return solve(
        _monodromy_system(),
        Monodromy(;
            target_solutions_count = 2, seed = UInt32(7), show_progress = false,
        ),
        Serial(),
    )
end


function run(::Val{:monodromy_threaded})
    return solve(
        _monodromy_system(),
        Monodromy(;
            target_solutions_count = 2, seed = UInt32(7), show_progress = false,
        ),
        Threaded(),
    )
end

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
    F = System(
        [z[1]^2 + 2z[2]^2 + 3z[3]^2 + z[1] * z[2] - 1];
        variables = z,
    )
    return solve(
        F, Monodromy(; dim = 2, seed = UInt32(99), show_progress = false), Serial(),
    )
end

function run(::Val{:verify_completeness})
    F = _monodromy_system()
    alg = Monodromy(;
        target_solutions_count = 2, seed = UInt32(21), show_progress = false,
    )
    result = solve(F, alg, Serial())
    return verify_solution_completeness(F, result, alg, Serial())
end

function run(::Val{:unique_points_group_action})
    rng = MersenneTwister(43)
    base = [randn(rng, ComplexF64, 3) for _ in 1:100]
    cloud = vcat(
        base,
        [-point for point in base],
        [point .+ 1.0e-13 .* randn(rng, ComplexF64, 3) for point in base],
    )
    points = UniquePoints(3; group_actions = solution -> (-solution,))
    for (i, point) in enumerate(cloud)
        add!(points, point, i; atol = 1.0e-8)
    end
    return points
end

function _fake_path_result(solution::Vector{ComplexF64})
    return PathResult(
        PathResultCode.PATH_SUCCESS,
        solution,
        0.0,
        1.0e-12,
        1.0,
        1.0e-12,
        1.0e-12,
        1.0,
        1,
        false,
        10,
        0,
        0,
        false,
        copy(solution),
        0.0,
        0,
        ComplexF64[],
        Float64[],
        0,
    )
end

function run(::Val{:result_clustering})
    rng = MersenneTwister(99)
    path_results = PathResult[]
    for _ in 1:1_000
        solution = randn(rng, ComplexF64, 6)
        push!(path_results, _fake_path_result(solution))
        push!(
            path_results,
            _fake_path_result(
                solution .+ 1.0e-9 .* randn(rng, ComplexF64, 6),
            ),
        )
    end
    return _cluster_solutions(
        path_results, DEFAULT_CLUSTER_ATOL, DEFAULT_CLUSTER_RTOL, nothing,
    )
end

run(::Val{:progress_enabled}) =
    _solve_total_degree(CompileMode.INTERPRETED, Serial(); progress = true)

run(name::Symbol) = run(Val(name))

end
