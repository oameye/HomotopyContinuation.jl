using Test
using HomotopyContinuation
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
            TotalDegree(; show_progress = false),
            Serial(),
        ),
    )
    params = [rand(3) for _ in 1:20]

    @testset "default return shape" begin
        for exec in (Serial(), Threaded())
            res = solve(F, S₀, p₀, params, Sweep(; show_progress = false), exec)
            @test res isa Vector{Tuple{Result, Vector{Float64}}}
            @test length(res) == length(params)
            @test all(t -> nsolutions(first(t)) == 2, res)
            @test [last(t) for t in res] == params
        end
    end

    # Threading runs over the (target, path) product, so every public result must
    # be independent of how the work is partitioned across tasks.
    @testset "threaded matches serial for $n target(s)" for n in (1, 2, 3, 7, 20)
        tg = params[1:n]
        opts = (; seed = UInt32(0x5EED), show_progress = false)
        rs = solve(F, S₀, p₀, tg, Sweep(; opts...), Serial())
        rt = solve(F, S₀, p₀, tg, Sweep(; opts...), Threaded())
        @test length(rs) == length(rt) == n
        @test [last(t) for t in rt] == tg
        for k in 1:n
            by = z -> (round(real(z[1]); digits = 9), round(imag(z[1]); digits = 9))
            a = sort(solutions(first(rs[k])); by = by)
            b = sort(solutions(first(rt[k])); by = by)
            @test length(a) == length(b) == 2
            @test a == b
        end
    end

    @testset "transform_result" begin
        res = solve(
            F,
            S₀,
            p₀,
            params,
            Sweep(; transform_result = (r, p) -> real_solutions(r), show_progress = false),
            Threaded(),
        )
        @test res isa Vector{Vector{Vector{Float64}}}
        @test length(res) == length(params)
        @test !isempty(res)
    end

    @testset "flatten_results" begin
        nested = solve(
            F,
            S₀,
            p₀,
            params,
            Sweep(; transform_result = (r, p) -> real_solutions(r), show_progress = false),
            Serial(),
        )
        res = flatten_results(nested)
        @test res isa Vector{Vector{Float64}}
        @test length(res) == sum(length, nested)
    end

    @testset "flatten_results rejects non-array entries" begin
        nested = solve(F, S₀, p₀, params, Sweep(; show_progress = false), Serial())
        @test_throws ArgumentError flatten_results(nested)
    end

    @testset "transform_parameters" begin
        res = solve(
            F,
            S₀,
            p₀,
            1:20,
            Sweep(;
                transform_result = (r, p) -> (real_solutions(r), p),
                transform_parameters = _ -> rand(3), show_progress = false,
            ),
            Serial(),
        )
        @test res isa Vector{Tuple{Vector{Vector{Float64}}, Int}}
        @test [last(t) for t in res] == collect(1:20)
    end

    @testset "transform_parameters is called once per target ($exec)" for exec in
        (Serial(), Threaded())
        for n in (1, 3, 20)
            calls = Ref(0)
            counted = q -> (calls[] += 1; q)
            solve(
                F,
                S₀,
                p₀,
                params[1:n],
                Sweep(; transform_parameters = counted, show_progress = false),
                exec,
            )
            @test calls[] == n
        end
    end

    @testset "solutions match a single-target solve" begin
        q = params[1]
        sweep = first(
            solve(F, S₀, p₀, [q], Sweep(; show_progress = false), Serial()),
        )
        single = solve(F, S₀, p₀, q, Continuation(; show_progress = false), Serial())
        a1 = sort(solutions(first(sweep)); by = real ∘ first)
        a2 = sort(solutions(single); by = real ∘ first)
        @test length(a1) == length(a2) == 2
        @test maximum(norm.(a1 .- a2, Inf)) < 1.0e-8
    end

    @testset "targets of the wrong length are rejected" begin
        ragged = [params[1], [1.0, 2.0]]
        @test_throws ArgumentError solve(
            F, S₀, p₀, ragged, Sweep(; show_progress = false), Serial(),
        )
    end

    @testset "empty targets" begin
        @test_throws ArgumentError solve(
            F, S₀, p₀, Vector{Float64}[], Sweep(; show_progress = false), Serial(),
        )
    end

    # ── Subspace sweep ─────────────────────────────────────────────────────

    @polyvar u v
    f = System([u^2 + v^2 - 1]; variables = [u, v])
    L₀ = rand_subspace(2; dim = 1)
    S = solutions(solve(f, L₀, TotalDegree(; show_progress = false)))
    subspaces = [rand_subspace(2; dim = 1) for _ in 1:30]

    @testset "target subspaces, $label" for (label, kw) in
        (
            ("auto", NamedTuple()),
            ("intrinsic", (; coords = SubspaceCoords.INTRINSIC)),
            ("extrinsic", (; coords = SubspaceCoords.EXTRINSIC)),
        )
        for exec in (Serial(), Threaded())
            res = solve(f, S, L₀, subspaces, Sweep(; show_progress = false, kw...), exec)
            @test res isa Vector{Tuple{Result, LinearSubspace{ComplexF64}}}
            @test all(t -> nsolutions(first(t)) == 2, res)
            for (r, L) in res
                E = extrinsic(L)
                @test maximum(s -> maximum(abs, E.A * s - E.b), solutions(r)) < 1.0e-10
            end
        end
    end

    TASK_COUNTS = unique(min.((1, 2, 3, 8), Threads.nthreads()))

    @testset "target subspaces, threaded matches serial for $n target(s)" for n in
        (1, 2, 30)
        for coords in (SubspaceCoords.INTRINSIC, SubspaceCoords.EXTRINSIC),
                nt in TASK_COUNTS
            tg = subspaces[1:n]
            opts = (;
                coords = coords, seed = UInt32(0x5EED), show_progress = false,
            )
            rs = solve(f, S, L₀, tg, Sweep(; opts...), Serial())
            rt = solve(f, S, L₀, tg, Sweep(; opts...), Threaded(nt))
            @test length(rs) == length(rt) == n
            for k in 1:n
                @test last(rt[k]) === tg[k]
                by = z -> (round(real(z[1]); digits = 9), round(imag(z[1]); digits = 9))
                a = sort(solutions(first(rs[k])); by = by)
                b = sort(solutions(first(rt[k])); by = by)
                @test length(a) == length(b) == 2
                @test a == b
            end
        end
    end

    @testset "target subspaces, order independent" begin
        for coords in (SubspaceCoords.INTRINSIC, SubspaceCoords.EXTRINSIC)
            tg = subspaces[1:6]
            opts = (;
                coords = coords, seed = UInt32(0x5EED), show_progress = false,
            )
            fwd = solve(f, S, L₀, tg, Sweep(; opts...), Serial())
            rev = reverse(solve(f, S, L₀, reverse(tg), Sweep(; opts...), Serial()))
            for k in eachindex(tg)
                by = z -> (round(real(z[1]); digits = 9), round(imag(z[1]); digits = 9))
                a = sort(solutions(first(fwd[k])); by = by)
                b = sort(solutions(first(rev[k])); by = by)
                @test length(a) == length(b) == 2
                @test a == b
            end
        end
    end

    @testset "target subspaces: transform_parameters is called once per target" begin
        for exec in (Serial(), Threaded()), n in (1, 30)
            calls = Ref(0)
            counted = L -> (calls[] += 1; L)
            solve(
                f,
                S,
                L₀,
                subspaces[1:n],
                Sweep(; transform_parameters = counted, show_progress = false),
                exec,
            )
            @test calls[] == n
        end
    end

    @testset "target subspaces of the wrong dimension are rejected" begin
        full = rand_subspace(2; dim = 2)
        for coords in (SubspaceCoords.INTRINSIC, SubspaceCoords.EXTRINSIC)
            @test_throws ArgumentError solve(
                f,
                S,
                L₀,
                [subspaces[1], full],
                Sweep(; coords = coords, show_progress = false),
                Serial(),
            )
        end
    end

    @testset "target subspaces, flatten_results" begin
        res = flatten_results(
            solve(
                f,
                S,
                L₀,
                subspaces,
                Sweep(;
                    transform_result = (r, L) -> solutions(r), show_progress = false,
                ),
                Serial(),
            ),
        )
        @test res isa Vector{Vector{ComplexF64}}
        @test length(res) == 2 * length(subspaces)
    end

    @testset "subspace sweep agrees with single moves" begin
        pair = subspaces[1:2]
        sweep = solve(
            f, S, L₀, pair,
            Sweep(; coords = SubspaceCoords.EXTRINSIC, show_progress = false),
            Serial(),
        )
        for (k, L) in enumerate(pair)
            single = solve(
                f,
                S,
                L₀,
                L,
                Continuation(; coords = SubspaceCoords.EXTRINSIC, show_progress = false),
                Serial(),
            )
            a1 = sort(solutions(first(sweep[k])); by = real ∘ first)
            a2 = sort(solutions(single); by = real ∘ first)
            @test maximum(norm.(a1 .- a2, Inf)) < 1.0e-8
        end
    end
end
