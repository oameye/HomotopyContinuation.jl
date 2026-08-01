using Test
using DynamicPolynomials: @polyvar
using HomotopyContinuationNext: Homotopy, Expression, System, ParameterHomotopy,
    @var, solve, solutions, nsolutions, fix_parameters, variables, parameters,
    nvariables, nparameters, expressions, evaluate, jacobian, Serial, Threaded,
    CompileMode

@testset "Homotopy: symbolic surface" begin
    @var x y z t
    h = [x^2 + y + z + 2t, 4 * x^2 * z^2 * y + 4z - 6x * y * z^2]
    H = Homotopy(h, [x, y, z], t)

    @test nvariables(H) == 3
    @test nparameters(H) == 0
    @test variables(H) == [x, y, z]
    @test parameters(H) == Expression[]
    @test expressions(H) == h
    @test size(H) == (2, 3)
    @test size(H, 1) == 2
    @test size(H, 2) == 3
    @test length(H) == 2

    out = sprint(show, H)
    @test startswith(out, "Homotopy in t of length 2\n 3 variables: x, y, z\n\n")
    @test count(==('\n'), out) == 4
    @test sprint(show, H; context = :compact => true) ==
        string("[", h[1], ", ", h[2], "]")

    @test Homotopy(convert(Vector{Any}, h), [x, y, z], t) == H
    @test Homotopy(h, [x, y, z], t; compile = CompileMode.COMPILED) == H

    @var a b
    P = Homotopy([x^2 - a, y - b * t], [x, y], t; parameters = [a, b])
    @test nparameters(P) == 2
    @test parameters(P) == [a, b]
    pout = sprint(show, P)
    @test occursin(" 2 variables: x, y\n 2 parameters: a, b\n", pout)
    @test P != H

    @testset "evaluation" begin
        @test H([1, 2, 3], 0.5) == [7.0, -24.0]
        @test evaluate(H, [1, 2, 3], 0.5) == H([1, 2, 3], 0.5)
        @test P([2, 3], 0.5, [4, 6]) == [0.0, 0.0]
        @test H([1, 2, 3], 0.5im) isa Vector{ComplexF64}
        # Jacobian in the variables only, so one column per variable.
        @test jacobian(H, [1, 2, 3], 0.5) == [2.0 1.0 1.0; 36.0 -18.0 -20.0]
        @test jacobian(P, [2, 3], 0.5, [4, 6]) == [4.0 0.0; 0.0 1.0]
    end

    @testset "fix_parameters" begin
        Q = fix_parameters(P, [4, 6])
        @test Q isa Homotopy
        @test nparameters(Q) == 0
        @test parameters(Q) == Expression[]
        @test variables(Q) == [x, y]
        # The values are substituted into the equations, so no `a`, `b` is left.
        @test Set(variables(expressions(Q))) == Set([t, x, y])
        @test Q([2, 3], 0.5) == P([2, 3], 0.5, [4, 6])
        @test Q([1, 2], 0.25) == P([1, 2], 0.25, [4, 6])
        @test jacobian(Q, [2, 3], 0.5) == jacobian(P, [2, 3], 0.5, [4, 6])
    end

    @testset "MultivariatePolynomials input" begin
        @polyvar u v s
        M = Homotopy([u^2 + v + 2s, u * v - s], [u, v], s)
        @test nvariables(M) == 2
        @test nparameters(M) == 0
        @test M([2, 3], 0.5) == [8.0, 5.5]
    end

    @testset "rejected input" begin
        # The path parameter must be a variable, and must not double as one.
        @test_throws ArgumentError Homotopy(h, [x, y, z], t + 1)
        @test_throws ArgumentError Homotopy(h, [x, y, z, t], t)
        @test_throws ArgumentError Homotopy(h, [x, y, z], t; parameters = [t])
        # An undeclared symbol is an error rather than a silent free variable.
        @test_throws ArgumentError Homotopy(h, [x, y], t)
        @test_throws ArgumentError fix_parameters(P, [1.0])
        @test_throws ArgumentError fix_parameters(H, [1.0])
        @test_throws ArgumentError P([2, 3], 0.5, [4])
        @test_throws ArgumentError H([1, 2], 0.5)
    end
end

@testset "Homotopy: tracking" begin
    @var x y t
    G = [x^2 - 1, y^2 - 1]
    F = [x^2 + 2 * y^2 - 3, x * y - 1]
    starts = [[1.0 + 0im, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]

    H = Homotopy(t .* G .+ (1 - t) .* F, [x, y], t)
    r = solve(H, starts)
    @test nsolutions(r) == 4
    # (±1, ±1) and (±√2, ±1/√2)
    expected = [
        [1.0, 1.0], [-1.0, -1.0], [sqrt(2), 1 / sqrt(2)], [-sqrt(2), -1 / sqrt(2)],
    ]
    for e in expected
        @test any(s -> maximum(abs.(s .- e)) < 1.0e-8, solutions(r))
    end

    @testset "matches the equivalent ParameterHomotopy" begin
        @var a
        S = System(a .* G .+ (1 - a) .* F; variables = [x, y], parameters = [a])
        ref = solve(ParameterHomotopy(S, [1.0], [0.0]), starts)
        @test sort(round.(abs.(reduce(vcat, solutions(r))); digits = 8)) ==
            sort(round.(abs.(reduce(vcat, solutions(ref))); digits = 8))
    end

    @testset "executors agree" begin
        rs = solve(H, starts, Serial())
        rt = solve(H, starts, Threaded())
        @test nsolutions(rs) == nsolutions(rt) == 4
        @test sort(round.(abs.(reduce(vcat, solutions(rs))); digits = 8)) ==
            sort(round.(abs.(reduce(vcat, solutions(rt))); digits = 8))
    end

    @testset "parameters must be bound before tracking" begin
        @var c
        P = Homotopy(
            t .* G .+ (1 - t) .* [x^2 + 2 * y^2 - c, x * y - 1], [x, y], t;
            parameters = [c]
        )
        @test_throws ArgumentError solve(P, starts)
        @test nsolutions(solve(fix_parameters(P, [3.0]), starts)) == 4
    end
end
