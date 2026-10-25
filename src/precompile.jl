@compile_workload begin
    let seed = UInt32(0x1234)
        default_solve = () -> begin
            @polyvar x y
            solve(System([x^2 + y - 1, x * y - 0.5]))
        end
        total_degree_serial = () -> begin
            @polyvar x y
            F = System([x^2 + y - 1, x * y - 0.5]; variables = [x, y])
            solve(F, TotalDegree(; seed, show_progress = false), Serial())
        end
        polyhedral_threaded = () -> begin
            @polyvar x y
            F = System([x^2 + y - 1, x * y - 0.5]; variables = [x, y])
            solve(F, Polyhedral(; seed, show_progress = false), Threaded())
        end
        parameter_homotopy_solve = () -> begin
            @polyvar x y a b
            F = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
            H = ParameterHomotopy(F, [1.0, 0.0], [2.0, 4.0])
            solve(H, [[1.0, 1.0]], Continuation(; show_progress = false), Serial())
        end
        monodromy_threaded = () -> begin
            @polyvar y[1:2] p[1:2]
            F = System(
                [y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]]; variables = y, parameters = p,
            )
            solve(
                F, Monodromy(; target_solutions_count = 2, seed, show_progress = false),
                Threaded(),
            )
        end
        for route in (
                default_solve, total_degree_serial, polyhedral_threaded,
                parameter_homotopy_solve, monodromy_threaded,
            )
            precompile(route, ()) || @warn "cannot precompile $route"
        end
    end
end
