# Results and utilities

const RESULTS = (
    :recluster,
    :unique_points_group_action,
    :solution_files,
)

function run(::Val{:recluster})
    @polyvar x y
    F = System([x^2 - y^2, x * y - 1]; variables = [x, y])
    r = solve(F, TotalDegree(; seed = SEED, show_progress = false), Serial())
    return recluster(r; group_action = s -> ([-s[1], -s[2]],))
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

function run(::Val{:solution_files})
    r = solve_total_degree(CompileMode.INTERPRETED, Serial())
    return mktempdir() do dir
        solutions_path = joinpath(dir, "solutions")
        parameters_path = joinpath(dir, "parameters")
        write_solutions(solutions_path, solutions(r))
        write_parameters(parameters_path, ComplexF64[1.0, 2.0])
        return read_solutions(solutions_path), read_parameters(parameters_path)
    end
end
