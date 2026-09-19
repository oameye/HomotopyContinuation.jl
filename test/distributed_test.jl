using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar
using Distributed: Distributed, addprocs, rmprocs, workers, remotecall_eval
using Random: seed!

function same_path_semantics(a, b)
    pa, pb = path_results(a), path_results(b)
    length(pa) == length(pb) || return false
    ntracked(a) == ntracked(b) || return false
    nsolutions(a) == nsolutions(b) || return false
    nfailed(a) == nfailed(b) || return false
    nat_infinity(a) == nat_infinity(b) || return false
    seed(a) == seed(b) || return false

    for (u, v) in zip(pa, pb)
        path_number(u) == path_number(v) || return false
        is_success(u) == is_success(v) || return false
        is_finite(u) == is_finite(v) || return false
        is_at_infinity(u) == is_at_infinity(v) || return false
        is_excess_solution(u) == is_excess_solution(v) || return false
        steps(u) == steps(v) || return false
        accepted_steps(u) == accepted_steps(v) || return false
        rejected_steps(u) == rejected_steps(v) || return false
        length(solution(u)) == length(solution(v)) || return false
        length(start_solution(u)) == length(start_solution(v)) || return false
        maximum(abs.(solution(u) .- solution(v)); init = 0.0) < 1.0e-10 || return false
        maximum(abs.(start_solution(u) .- start_solution(v)); init = 0.0) < 1.0e-12 || return false
        if isfinite(residual(u)) && isfinite(residual(v))
            isapprox(residual(u), residual(v); rtol = 1.0e-9, atol = 1.0e-12) || return false
        else
            isequal(residual(u), residual(v)) || return false
        end
    end
    return true
end

_entry_result(entry::Result) = entry
_entry_result(entry::Tuple) = entry[1]

function same_sweep_semantics(a, b)
    length(a) == length(b) || return false
    return all(same_path_semantics(_entry_result(a[k]), _entry_result(b[k])) for k in eachindex(a))
end

same_points(a, b; tol = 1.0e-8) =
    length(a) == length(b) && all(u -> any(v -> maximum(abs.(u .- v)) < tol, b), a)

@testset "Distributed executor public behavior" begin
    @polyvar x y z a b

    F = System([x^3 + y^2 - 2x * y + 1, x * y^2 - 3x + 2y - 1])
    F_over = System([x^2 + y^2 - 1, x * y - 1 // 4, x - y + 1 // 10])
    F_param = System([x^2 + y^2 - a, x * y - b]; parameters = [a, b])
    F_curve = System([x^2 + y^2 + z^2 - 1, x + y + z])

    @testset "no worker process" begin
        @test workers() == [Distributed.myid()]
        err = try
            solve(F, TotalDegree(; seed = UInt32(99), show_progress = false), DistributedExecutor())
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("found no processes to track on", err.msg)
    end

    pids = addprocs(2; exeflags = ["--project=$(Base.active_project())", "-t2"])
    remotecall_eval(Main, pids, :(using HomotopyContinuation))

    try
        lockstep = DistributedExecutor(; pids = [pids[1]], tasks_per_process = 1)

        @testset "TotalDegree parity" begin
            alg = TotalDegree(; seed = UInt32(99), show_progress = false)
            serial = solve(F, alg, Serial())
            @test nsolutions(serial) > 0
            for exec in (
                    DistributedExecutor(),
                    DistributedExecutor(; batch_size = 1),
                    DistributedExecutor(; tasks_per_process = 1),
                    lockstep,
                )
                @test same_path_semantics(serial, solve(F, alg, exec))
            end
        end

        @testset "driver-side early stop" begin
            full = solve(F, TotalDegree(; seed = UInt32(99), show_progress = false), Serial())
            stopped = solve(
                F,
                TotalDegree(;
                    seed = UInt32(99), show_progress = false,
                    early_stop_callback = _ -> true,
                ),
                DistributedExecutor(; batch_size = 1),
            )
            @test 0 < stopped.tracked_paths < full.tracked_paths
            @test nfailed(stopped) == 0
            @test length(path_results(stopped)) == stopped.tracked_paths
        end

        @testset "overdetermined and composition" begin
            alg_over = TotalDegree(; seed = UInt32(31), show_progress = false)
            serial_over = solve(F_over, alg_over, Serial())
            @test same_path_semantics(serial_over, solve(F_over, alg_over, DistributedExecutor()))

            C = System([x + y, x - 2]) ∘ System([y^2 + 2x + 3, x - 1])
            alg_comp = TotalDegree(; seed = UInt32(17), show_progress = false)
            serial_comp = solve(C, alg_comp, Serial())
            @test same_path_semantics(serial_comp, solve(C, alg_comp, DistributedExecutor()))
        end

        @testset "variable groups" begin
            @polyvar u v s t
            alg = TotalDegree(; seed = UInt32(23), show_progress = false)
            for G in (
                    System([u * s - 2, u^2 - 4]; variable_groups = [[u], [s]]),
                    System(
                        [u * s - 2v * t, u^2 - 4 * v^2];
                        variable_groups = [[u, v], [s, t]],
                    ),
                )
                serial = solve(G, alg, Serial())
                @test nsolutions(serial) == 2
                @test same_path_semantics(serial, solve(G, alg, DistributedExecutor()))
            end
        end

        @testset "Polyhedral parity" begin
            alg = Polyhedral(; seed = UInt32(99), show_progress = false)
            serial = solve(F, alg, Serial())
            @test nsolutions(serial) > 0
            for exec in (DistributedExecutor(), DistributedExecutor(; batch_size = 1))
                @test same_path_semantics(serial, solve(F, alg, exec))
            end
        end

        p₀ = ComplexF64[3.0, 0.5]
        starts = solutions(
            solve(
                System([x^2 + y^2 - 3.0, x * y - 0.5]),
                TotalDegree(; seed = UInt32(7), show_progress = false),
                Serial(),
            ),
        )

        @testset "continuation routes" begin
            q = ComplexF64[2.3, 0.9]
            opts = (; seed = UInt32(55), show_progress = false)

            serial = solve(F_param, starts, p₀, q, Continuation(; opts...), Serial())
            @test length(starts) == 4
            for exec in (DistributedExecutor(), DistributedExecutor(; batch_size = 1))
                @test same_path_semantics(
                    serial,
                    solve(F_param, starts, p₀, q, Continuation(; opts...), exec),
                )
            end

            G = fix_parameters(F_param, p₀)
            H = fix_parameters(F_param, q)
            serial_target = solve(G, H, starts, Continuation(; opts...), Serial())
            @test nsolutions(serial_target) == 4
            @test same_path_semantics(
                serial_target,
                solve(G, H, starts, Continuation(; opts...), DistributedExecutor()),
            )

            hom = ParameterHomotopy(F_param, p₀, q)
            serial_hom = solve(hom, starts, Continuation(; opts...), Serial())
            @test same_path_semantics(
                serial_hom,
                solve(hom, starts, Continuation(; opts...), DistributedExecutor()),
            )
        end

        seed!(0x51ce)
        V = rand_subspace(3; codim = 1)
        W = rand_subspace(3; codim = 1)
        starts_V = solutions(
            solve(
                F_curve,
                V,
                TotalDegree(; seed = UInt32(3), show_progress = false),
                Serial(),
            ),
        )

        @testset "subspace continuation" begin
            @test length(starts_V) == 2
            for coords in (SubspaceCoords.EXTRINSIC, SubspaceCoords.INTRINSIC)
                opts = (; coords = coords, seed = UInt32(8), show_progress = false)
                serial = solve(F_curve, starts_V, V, W, Continuation(; opts...), Serial())
                @test same_path_semantics(
                    serial,
                    solve(F_curve, starts_V, V, W, Continuation(; opts...), DistributedExecutor()),
                )
            end
        end

        @testset "parameter and subspace sweeps" begin
            targets = [ComplexF64[2.0 + 0.1k, 0.4 + 0.05k] for k in 1:7]
            opts = (; seed = UInt32(55), show_progress = false)
            serial = solve(F_param, starts, p₀, targets, Sweep(; opts...), Serial())
            @test length(serial) == length(targets)
            for exec in (
                    DistributedExecutor(),
                    DistributedExecutor(; batch_size = 1),
                    DistributedExecutor(; batch_size = 3),
                    lockstep,
                )
                @test same_sweep_semantics(
                    serial,
                    solve(F_param, starts, p₀, targets, Sweep(; opts...), exec),
                )
            end

            subspace_targets = [rand_subspace(3; codim = 1) for _ in 1:5]
            subspace_opts = (; seed = UInt32(8), show_progress = false)
            serial_subspace =
                solve(F_curve, starts_V, V, subspace_targets, Sweep(; subspace_opts...), Serial())
            for exec in (DistributedExecutor(), DistributedExecutor(; batch_size = 3), lockstep)
                @test same_sweep_semantics(
                    serial_subspace,
                    solve(
                        F_curve, starts_V, V, subspace_targets,
                        Sweep(; subspace_opts...), exec,
                    ),
                )
            end
        end

        @testset "monodromy" begin
            G = System(
                [x^2 + y^2 - a^2, x * y - b^3];
                variables = [x, y], parameters = [a, b],
            )
            opts = (;
                seed = UInt32(123), show_progress = false,
                target_solutions_count = 4, max_loops_no_progress = 50,
            )
            serial = solve(G, Monodromy(; opts...), Serial())
            @test nsolutions(serial) == 4

            for exec in (DistributedExecutor(), DistributedExecutor(; batch_size = 1), lockstep)
                r = solve(G, Monodromy(; opts...), exec)
                @test nsolutions(r) == 4
                @test same_points(solutions(serial), solutions(r))
            end

            rperm = solve(
                G,
                Monodromy(; permutations = true, opts...),
                DistributedExecutor(),
            )
            perm = permutations(rperm)
            @test size(perm, 1) == 4
            for k in axes(perm, 2)
                @test sort(perm[:, k]) == [1, 2, 3, 4]
            end

            Q = System([x^2 + 2y^2 + 3z^2 + x * y - 1]; variables = [x, y, z])
            q_opts = (; dim = 2, seed = UInt32(99), show_progress = false)
            serial_q = solve(Q, Monodromy(; q_opts...), Serial())
            @test nsolutions(serial_q) == 2 && is_success(serial_q)
            for exec in (DistributedExecutor(), lockstep)
                r = solve(Q, Monodromy(; q_opts...), exec)
                @test nsolutions(r) == 2
                @test is_success(r)
                @test !isnan(trace(r)) && trace(r) < 1.0e-10
            end

            timed = solve(
                G,
                Monodromy(;
                    seed = UInt32(123), show_progress = false, timeout = 0.0,
                    target_solutions_count = 4, max_loops_no_progress = 50,
                ),
                DistributedExecutor(),
            )
            @test !is_success(timed)
        end

        @testset "public error reporting" begin
            alg = TotalDegree(; seed = UInt32(99), show_progress = false)
            bare = addprocs(1; exeflags = ["--project=$(Base.active_project())"])
            try
                err = try
                    solve(F, alg, DistributedExecutor(; pids = bare))
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("does not have HomotopyContinuation loaded", err.msg)
            finally
                rmprocs(bare)
            end

            err = try
                solve(
                    F_param,
                    starts,
                    p₀,
                    [ComplexF64[1.0]],
                    Sweep(; seed = UInt32(1), show_progress = false),
                    DistributedExecutor(),
                )
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin("length", err.msg)
        end
    finally
        rmprocs(pids)
    end
end
