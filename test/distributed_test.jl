using Test
using HomotopyContinuationNext
using HomotopyContinuationNext: Serial, Threaded, DistributedExecutor, Result,
    PathResult, TotalDegree, Polyhedral, System, CompileMode
using HomotopyContinuationNext: _SupportSystem, _stage_system, evaluate!, FSVec,
    nparameters, _distributed_solve!, _distributed_sweep_entries,
    _distributed_monodromy_solve!
using HomotopyContinuationNext: monodromy_solve, permutations, trace, is_success,
    MonodromyCode
using HomotopyContinuationNext: variable_groups, multi_degrees, is_homogeneous
using DynamicPolynomials: @polyvar
using Distributed: Distributed, addprocs, rmprocs, workers, remotecall_eval
using Serialization: serialize, deserialize
using CommonSolve: init
using Random: seed!

# Every field of every path is compared: a mismatch in the serialized system or
# in the global index mapping shows up there long before it changes a solution
# count.
function same_paths(a::Result, b::Result)
    pa, pb = path_results(a), path_results(b)
    length(pa) == length(pb) || return "path count $(length(pa)) vs $(length(pb))"
    for (k, (u, v)) in enumerate(zip(pa, pb))
        for field in fieldnames(PathResult)
            isequal(getfield(u, field), getfield(v, field)) && continue
            return "path $k field $field: $(getfield(u, field)) vs $(getfield(v, field))"
        end
    end
    return ""
end

# A sweep entry is `(result, target)` by default.
_entry_result(entry::Result) = entry
_entry_result(entry::Tuple) = entry[1]

function same_sweep(a::AbstractVector, b::AbstractVector)
    length(a) == length(b) || return "target count $(length(a)) vs $(length(b))"
    for k in eachindex(a)
        msg = same_paths(_entry_result(a[k]), _entry_result(b[k]))
        isempty(msg) || return "target $k: $msg"
    end
    return ""
end

roundtrip(x) = (io = IOBuffer(); serialize(io, x); seekstart(io); deserialize(io))

function evaluate_at(evaluator, x::Vector{ComplexF64}, p::Vector{ComplexF64})
    u = FSVec{ComplexF64}(zeros(ComplexF64, size(evaluator)[1]))
    xs = FSVec{ComplexF64}(copy(x))
    ps = FSVec{ComplexF64}(copy(p))
    evaluate!(u, evaluator, xs, ps)
    return collect(u)
end

@testset "Distributed executor" begin

    @polyvar x y z a b

    F = System([x^3 + y^2 - 2x * y + 1, x * y^2 - 3x + 2y - 1])
    F_over = System([x^2 + y^2 - 1, x * y - 1 // 4, x - y + 1 // 10])
    F_param = System([x^2 + y^2 - a, x * y - b]; parameters = [a, b])
    F_curve = System([x^2 + y^2 + z^2 - 1, x + y + z])

    @testset "system serialization" begin
        @testset "System, compile mode $mode" for mode in (
                CompileMode.INTERPRETED, CompileMode.COMPILED,
                CompileMode.COMPILED_ALL,
            )
            G = System([x^3 + y^2 - 2x * y + 1, x * y^2 - 3x + 2y - 1]; compile = mode)
            H = roundtrip(G)
            @test typeof(H) === typeof(G)
            @test H.degrees == G.degrees
            @test H.compile_mode === G.compile_mode
            @test size(H.evaluator) == size(G.evaluator)
            pt = ComplexF64[0.7, -1.3]
            @test evaluate_at(H.evaluator, pt, ComplexF64[]) ==
                evaluate_at(G.evaluator, pt, ComplexF64[])
        end

        @testset "parametric System" begin
            H = roundtrip(F_param)
            @test nparameters(H.evaluator) == 2
            pt, ps = ComplexF64[0.7, -1.3], ComplexF64[2.0, 5.0]
            @test evaluate_at(H.evaluator, pt, ps) ==
                evaluate_at(F_param.evaluator, pt, ps)
        end

        @testset "grouped System" begin
            @polyvar u v s t
            G = System(
                [u * s - 2v * t, u^2 - 4 * v^2]; variable_groups = [[u, v], [s, t]],
            )
            H = roundtrip(G)
            @test variable_groups(H) == variable_groups(G)
            @test multi_degrees(H) == multi_degrees(G)
            @test is_homogeneous(H) == is_homogeneous(G)
        end

        @testset "_SupportSystem" begin
            support_system = init(F, Polyhedral(), Serial()).builder.support_system
            @test support_system isa _SupportSystem
            other = roundtrip(support_system)
            pt = ComplexF64[0.7, -1.3]
            ps = ComplexF64[0.3 + 0.1im * k for k in 1:nparameters(support_system.evaluator)]
            @test evaluate_at(other.evaluator, pt, ps) ==
                evaluate_at(support_system.evaluator, pt, ps)
        end

        @testset "CompositionSystem" begin
            f = System([y^2 + 2x + 3, x - 1])
            g = System([x + y * a, x - b]; parameters = [a, b])
            C = g ∘ f
            D = roundtrip(C)
            @test typeof(D) === typeof(C)
            @test length(D.stages) == length(C.stages)
            @test D.degrees == C.degrees
            @test D.is_homogeneous == C.is_homogeneous
            @test [_stage_system(s).degrees for s in D.stages] ==
                [_stage_system(s).degrees for s in C.stages]
            pt, ps = ComplexF64[0.7, -1.3], ComplexF64[2.0, 5.0]
            @test evaluate_at(D.evaluator, pt, ps) == evaluate_at(C.evaluator, pt, ps)
        end
    end

    # `ProgressMeter` pulls in `Distributed`, so the extension is always active
    # and the fallbacks are reachable only through the hooks.
    @testset "extension hooks" begin
        @test Base.get_extension(
            HomotopyContinuationNext, :HomotopyContinuationNextDistributedExt,
        ) !== nothing
        @test_throws ArgumentError _distributed_solve!(nothing)
        @test_throws ArgumentError _distributed_sweep_entries(
            nothing, [], Int[], nothing, identity, identity, nothing,
        )
        @test_throws ArgumentError _distributed_monodromy_solve!(
            nothing, nothing, nothing, nothing, nothing,
        )
    end

    # Before `addprocs`, so `workers()` reports nothing but this process.
    @testset "no worker process" begin
        @test workers() == [Distributed.myid()]
        err = try
            solve(F, TotalDegree(; seed = UInt32(99)), DistributedExecutor(); show_progress = false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("found no processes to track on", err.msg)
    end

    pids = addprocs(
        2; exeflags = ["--project=$(Base.active_project())", "-t2"],
    )
    remotecall_eval(Main, pids, :(using HomotopyContinuationNext))

    try
        # One process running one task sees the paths in the same order a serial
        # solve does, so every route must agree with `Serial()` to the last bit.
        lockstep = DistributedExecutor(; pids = [pids[1]], tasks_per_process = 1)

        @testset "total degree" begin
            alg = TotalDegree(; seed = UInt32(99))
            serial = solve(F, alg, Serial(); show_progress = false)
            @test nsolutions(serial) > 0
            for exec in (
                    DistributedExecutor(),
                    DistributedExecutor(; batch_size = 1),
                    DistributedExecutor(; tasks_per_process = 1),
                    lockstep,
                )
                @test same_paths(
                    serial, solve(F, alg, exec; show_progress = false),
                ) == ""
            end
        end

        @testset "total degree, overdetermined" begin
            alg = TotalDegree(; seed = UInt32(31))
            serial = solve(F_over, alg, Serial(); show_progress = false)
            @test same_paths(
                serial, solve(F_over, alg, DistributedExecutor(); show_progress = false),
            ) == ""
        end

        @testset "total degree, composition" begin
            C = System([x + y, x - 2]) ∘ System([y^2 + 2x + 3, x - 1])
            alg = TotalDegree(; seed = UInt32(17))
            serial = solve(C, alg, Serial(); show_progress = false)
            @test same_paths(
                serial, solve(C, alg, DistributedExecutor(); show_progress = false),
            ) == ""
        end

        @testset "total degree, variable groups" begin
            @polyvar u v s t
            alg = TotalDegree(; seed = UInt32(23))
            for G in (
                    System([u * s - 2, u^2 - 4]; variable_groups = [[u], [s]]),
                    System(
                        [u * s - 2v * t, u^2 - 4 * v^2];
                        variable_groups = [[u, v], [s, t]],
                    ),
                    System(
                        [(u^2 - 4 * v^2) * (u * s - v * t), u * s - v * t, u^2 - v^2];
                        variable_groups = [[u, v], [s, t]],
                    ),
                )
                serial = solve(G, alg, Serial(); show_progress = false)
                @test nsolutions(serial) == 2
                @test same_paths(
                    serial, solve(G, alg, DistributedExecutor(); show_progress = false),
                ) == ""
            end
        end

        @testset "polyhedral" begin
            alg = Polyhedral(; seed = UInt32(99))
            serial = solve(F, alg, Serial(); show_progress = false)
            @test nsolutions(serial) > 0
            for exec in (DistributedExecutor(), DistributedExecutor(; batch_size = 1))
                @test same_paths(
                    serial, solve(F, alg, exec; show_progress = false),
                ) == ""
            end
        end

        p₀ = ComplexF64[3.0, 0.5]
        starts = solutions(
            solve(
                System([x^2 + y^2 - 3.0, x * y - 0.5]),
                TotalDegree(; seed = UInt32(7)), Serial(); show_progress = false,
            ),
        )

        @testset "parameter homotopy" begin
            @test length(starts) == 4
            q = ComplexF64[2.3, 0.9]
            opts = (; seed = UInt32(55), show_progress = false)
            serial = solve(F_param, starts, p₀, q, Serial(); opts...)
            for exec in (DistributedExecutor(), DistributedExecutor(; batch_size = 1))
                @test same_paths(serial, solve(F_param, starts, p₀, q, exec; opts...)) == ""
            end
        end

        seed!(0x51ce)
        V = rand_subspace(3; codim = 1)
        W = rand_subspace(3; codim = 1)
        starts_V = solutions(
            solve(
                F_curve, V, TotalDegree(; seed = UInt32(3)), Serial();
                show_progress = false,
            ),
        )

        @testset "subspace to subspace, intrinsic = $intr" for intr in (false, true)
            @test length(starts_V) == 2
            opts = (; intrinsic = intr, seed = UInt32(8), show_progress = false)
            serial = solve(F_curve, starts_V, V, W, Serial(); opts...)
            @test same_paths(
                serial, solve(F_curve, starts_V, V, W, DistributedExecutor(); opts...),
            ) == ""
        end

        @testset "parameter sweep" begin
            targets = [ComplexF64[2.0 + 0.1k, 0.4 + 0.05k] for k in 1:7]
            opts = (; seed = UInt32(55), show_progress = false)
            serial = solve_targets(F_param, starts, p₀, targets, Serial(); opts...)
            @test length(serial) == length(targets)
            # `batch_size = 3` puts a batch boundary inside a target, which is
            # what exercises the retarget bookkeeping.
            for exec in (
                    DistributedExecutor(),
                    DistributedExecutor(; batch_size = 1),
                    DistributedExecutor(; batch_size = 3),
                    lockstep,
                )
                @test same_sweep(
                    serial, solve_targets(F_param, starts, p₀, targets, exec; opts...),
                ) == ""
            end
        end

        @testset "subspace sweep" begin
            targets = [rand_subspace(3; codim = 1) for _ in 1:5]
            opts = (; seed = UInt32(8), show_progress = false)
            serial = solve_targets(F_curve, starts_V, V, targets, Serial(); opts...)
            @test length(serial) == length(targets)
            for exec in (
                    DistributedExecutor(),
                    DistributedExecutor(; batch_size = 1),
                    DistributedExecutor(; batch_size = 3),
                    lockstep,
                )
                @test same_sweep(
                    serial, solve_targets(F_curve, starts_V, V, targets, exec; opts...),
                ) == ""
            end
        end

        # Only the tracking is distributed, so this compares solution sets rather
        # than paths. `target_solutions_count` keeps the heuristic stop from cutting
        # one of the runs short at a different point.
        @testset "monodromy" begin
            G = System(
                [x^2 + y^2 - a^2, x * y - b^3]; variables = [x, y],
                parameters = [a, b],
            )
            opts = (;
                seed = UInt32(123), show_progress = false,
                target_solutions_count = 4, max_loops_no_progress = 50,
            )
            serial = monodromy_solve(G, Serial(); opts...)
            @test nsolutions(serial) == 4

            for exec in (
                    DistributedExecutor(),
                    DistributedExecutor(; batch_size = 1),
                    lockstep,
                )
                r = monodromy_solve(G, exec; opts...)
                @test r.returncode == serial.returncode
                @test nsolutions(r) == 4
                for s in solutions(serial)
                    @test minimum(maximum(abs.(s .- t)) for t in solutions(r)) < 1.0e-8
                end
            end

            @testset "permutations" begin
                r = monodromy_solve(
                    G, DistributedExecutor(); permutations = true, opts...,
                )
                perm = permutations(r)
                @test size(perm, 1) == 4
                for k in axes(perm, 2)
                    @test sort(perm[:, k]) == [1, 2, 3, 4]
                end
            end

            # Trace columns are collected remotely and folded into the driver's
            # trace matrix, which is what stops this run.
            @testset "subspace with trace test" begin
                Q = System([x^2 + 2y^2 + 3z^2 + x * y - 1]; variables = [x, y, z])
                q_opts = (; dim = 2, seed = UInt32(99), show_progress = false)
                serial_q = monodromy_solve(Q, Serial(); q_opts...)
                @test nsolutions(serial_q) == 2 && is_success(serial_q)
                for exec in (DistributedExecutor(), lockstep)
                    r = monodromy_solve(Q, exec; q_opts...)
                    @test nsolutions(r) == 2
                    @test is_success(r)
                    @test trace(r) !== nothing && trace(r) < 1.0e-10
                end
            end

            @testset "timeout" begin
                r = monodromy_solve(
                    G, DistributedExecutor(); seed = UInt32(123),
                    show_progress = false, timeout = 0.0,
                    target_solutions_count = 4, max_loops_no_progress = 50,
                )
                @test r.returncode == MonodromyCode.TIMEOUT
            end
        end

        @testset "error reporting" begin
            alg = TotalDegree(; seed = UInt32(99))

            # A worker that never loaded the package must say so, rather than
            # failing somewhere inside deserialization.
            bare = addprocs(1; exeflags = ["--project=$(Base.active_project())"])
            try
                err = try
                    solve(
                        F, alg, DistributedExecutor(; pids = bare);
                        show_progress = false,
                    )
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                @test occursin("does not have HomotopyContinuationNext loaded", err.msg)
            finally
                rmprocs(bare)
            end

            # An error thrown while tracking arrives as the error the system
            # raised, not as a `RemoteException` wrapping it.
            err = try
                solve_targets(
                    F_param, starts, p₀, [ComplexF64[1.0]], DistributedExecutor();
                    seed = UInt32(1), show_progress = false,
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
