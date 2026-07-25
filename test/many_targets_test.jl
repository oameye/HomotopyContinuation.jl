using Test
using HomotopyContinuationNext
using HomotopyContinuationNext: Serial, Threaded, Result, TotalDegree, ParameterHomotopy,
    TrackerOptions, AmbientWorkerState, IntrinsicWorkerState, ExtrinsicSubspaceHomotopy
using DynamicPolynomials: @polyvar
using LinearAlgebra: norm
using Random: seed!

@testset "Many targets" begin

    @polyvar x y a b c
    F = System(
        [x^2 + y^2 - 1, a * x + b * y + c];
        variables = [x, y], parameters = [a, b, c],
    )
    seed!(0x2718)
    p₀ = randn(ComplexF64, 3)
    S₀ = solutions(
        solve(
            System([x^2 + y^2 - 1, p₀[1] * x + p₀[2] * y + p₀[3]]; variables = [x, y]),
            Serial(); show_progress = false,
        ),
    )
    params = [rand(3) for _ in 1:20]

    @testset "default return shape" begin
        for exec in (Serial(), Threaded())
            res = solve(
                F, S₀, params, exec; start_parameters = p₀, show_progress = false,
            )
            @test res isa Vector{Tuple{Result, Vector{Float64}}}
            @test length(res) == length(params)
            @test all(t -> nsolutions(first(t)) == 2, res)
            @test [last(t) for t in res] == params
        end
    end

    # Threading runs over the (target, path) product, so results must land in the
    # right (target, path) slot for every ratio of targets to tasks, in
    # particular for fewer targets than tasks, where a target's paths are split
    # across several tasks and each task retargets its own homotopy.
    @testset "threaded matches serial for $n target(s)" for n in (1, 2, 3, 7, 20)
        tg = params[1:n]
        rs = solve(F, S₀, tg, Serial(); start_parameters = p₀, show_progress = false)
        rt = solve(F, S₀, tg, Threaded(); start_parameters = p₀, show_progress = false)
        @test length(rs) == length(rt) == n
        @test [last(t) for t in rt] == tg
        for k in 1:n
            by = z -> (round(real(z[1]); digits = 9), round(imag(z[1]); digits = 9))
            a = sort(solutions(first(rs[k])); by = by)
            b = sort(solutions(first(rt[k])); by = by)
            @test length(a) == length(b) == 2
            @test maximum(norm.(a .- b, Inf)) < 1.0e-8
        end
    end

    @testset "transform_result" begin
        res = solve(
            F, S₀, params, Threaded();
            start_parameters = p₀, transform_result = (r, p) -> real_solutions(r),
            show_progress = false,
        )
        @test res isa Vector{Vector{Vector{Float64}}}
        @test length(res) == length(params)
        @test !isempty(res)
    end

    @testset "flatten" begin
        res = solve(
            F, S₀, params, Serial();
            start_parameters = p₀, transform_result = (r, p) -> real_solutions(r),
            flatten = true, show_progress = false,
        )
        @test res isa Vector{Vector{Float64}}
        nested = solve(
            F, S₀, params, Serial();
            start_parameters = p₀, transform_result = (r, p) -> real_solutions(r),
            show_progress = false,
        )
        @test length(res) == sum(length, nested)
    end

    @testset "flatten rejects non-array entries" begin
        @test_throws ArgumentError solve(
            F, S₀, params, Serial();
            start_parameters = p₀, flatten = true, show_progress = false,
        )
    end

    @testset "transform_parameters" begin
        res = solve(
            F, S₀, 1:20, Serial();
            start_parameters = p₀,
            transform_result = (r, p) -> (real_solutions(r), p),
            transform_parameters = _ -> rand(3),
            show_progress = false,
        )
        @test res isa Vector{Tuple{Vector{Vector{Float64}}, Int}}
        @test [last(t) for t in res] == collect(1:20)
    end

    # A second call on the first target would draw twice from a randomized closure.
    @testset "transform_parameters is called once per target ($exec)" for exec in
        (Serial(), Threaded())
        for n in (1, 3, 20)
            calls = Ref(0)
            counted = q -> (calls[] += 1; q)
            solve(
                F, S₀, params[1:n], exec;
                start_parameters = p₀, transform_parameters = counted,
                show_progress = false,
            )
            @test calls[] == n
        end
    end

    @testset "solutions match a single-target solve" begin
        q = params[1]
        sweep = first(
            solve(
                F, S₀, [q], Serial(); start_parameters = p₀, show_progress = false,
            ),
        )
        single = solve(
            F, S₀, Serial();
            start_parameters = p₀, target_parameters = q, show_progress = false,
        )
        a1 = sort(solutions(first(sweep)); by = real ∘ first)
        a2 = sort(solutions(single); by = real ∘ first)
        @test length(a1) == length(a2) == 2
        @test maximum(norm.(a1 .- a2, Inf)) < 1.0e-8
    end

    @testset "retargets instead of rebuilding" begin
        # The homotopy handle in the worker state is the one the tracker uses, so a
        # retarget is visible through the FunctionWrapper firewall.
        cache = HomotopyContinuationNext._init_parameter_sweep(
            F, S₀, params[1], Serial(), p₀, UInt32(1),
            TrackerOptions(), EndgameOptions(), false,
        )
        H = cache.worker.homotopy
        @test H isa ParameterHomotopy
        HomotopyContinuationNext._retarget!(cache.worker, ComplexF64.(params[2]))
        @test Vector(H.target_p) ≈ ComplexF64.(params[2])
        @test cache.worker.homotopy === H
    end

    @testset "targets of the wrong length are rejected" begin
        # `target_parameters!` copies into a fixed-length buffer, so a short target
        # must not silently leave stale parameter values behind.
        ragged = [params[1], [1.0, 2.0]]
        @test_throws ArgumentError solve(
            F, S₀, ragged, Serial(); start_parameters = p₀, show_progress = false,
        )
    end

    @testset "empty targets" begin
        @test_throws ArgumentError solve(
            F, S₀, Vector{Float64}[], Serial();
            start_parameters = p₀, show_progress = false,
        )
    end

    # ── Subspace sweep ─────────────────────────────────────────────────────

    @polyvar u v
    f = System([u^2 + v^2 - 1]; variables = [u, v])
    L₀ = rand_subspace(2; dim = 1)
    S = solutions(solve(f, L₀; show_progress = false))
    subspaces = [rand_subspace(2; dim = 1) for _ in 1:30]

    @testset "target subspaces, $label" for (label, intrinsic) in
        (("auto", nothing), ("intrinsic", true), ("extrinsic", false))
        for exec in (Serial(), Threaded())
            res = solve(
                f, S, L₀, subspaces, exec;
                intrinsic = intrinsic, show_progress = false,
            )
            @test res isa Vector{Tuple{Result, LinearSubspace{ComplexF64}}}
            @test all(t -> nsolutions(first(t)) == 2, res)
            for (r, L) in res
                E = extrinsic(L)
                @test maximum(s -> maximum(abs, E.A * s - E.b), solutions(r)) < 1.0e-10
            end
        end
    end

    # Same (target, path) threading as the parameter sweep, over both regimes.
    @testset "target subspaces, threaded matches serial for $n target(s)" for n in
        (1, 2, 30)
        for intrinsic in (true, false)
            tg = subspaces[1:n]
            rs = solve(
                f, S, L₀, tg, Serial(); intrinsic = intrinsic, show_progress = false,
            )
            rt = solve(
                f, S, L₀, tg, Threaded(); intrinsic = intrinsic, show_progress = false,
            )
            @test length(rs) == length(rt) == n
            for k in 1:n
                @test last(rt[k]) === tg[k]
                by = z -> (round(real(z[1]); digits = 9), round(imag(z[1]); digits = 9))
                a = sort(solutions(first(rs[k])); by = by)
                b = sort(solutions(first(rt[k])); by = by)
                @test length(a) == length(b) == 2
                @test maximum(norm.(a .- b, Inf)) < 1.0e-8
            end
        end
    end

    @testset "target subspaces: transform_parameters is called once per target" begin
        for exec in (Serial(), Threaded()), n in (1, 30)
            calls = Ref(0)
            counted = L -> (calls[] += 1; L)
            solve(
                f, S, L₀, subspaces[1:n], exec;
                transform_parameters = counted, show_progress = false,
            )
            @test calls[] == n
        end
    end

    @testset "target subspaces of the wrong dimension are rejected" begin
        full = HomotopyContinuationNext._full_subspace(2)
        for intrinsic in (true, false)
            @test_throws ArgumentError solve(
                f, S, L₀, [subspaces[1], full], Serial();
                intrinsic = intrinsic, show_progress = false,
            )
        end
    end

    @testset "target subspaces, flatten" begin
        res = solve(
            f, S, L₀, subspaces, Serial();
            transform_result = (r, L) -> solutions(r), flatten = true,
            show_progress = false,
        )
        @test res isa Vector{Vector{ComplexF64}}
        @test length(res) == 2 * length(subspaces)
    end

    @testset "subspace sweep agrees with single moves" begin
        pair = subspaces[1:2]
        sweep = solve(f, S, L₀, pair, Serial(); intrinsic = false, show_progress = false)
        for (k, L) in enumerate(pair)
            single = solve(
                f, S, L₀, L, Serial(); intrinsic = false, show_progress = false,
            )
            a1 = sort(solutions(first(sweep[k])); by = real ∘ first)
            a2 = sort(solutions(single); by = real ∘ first)
            @test maximum(norm.(a1 .- a2, Inf)) < 1.0e-8
        end
    end
end
