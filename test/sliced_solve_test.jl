using Test
using HomotopyContinuationNext
using HomotopyContinuationNext: CompileMode, TotalDegree, Polyhedral, Result,
    Serial, Threaded, nparameters, nvariables, is_homogeneous, polynomials,
    variables, parameters, _sliced_solve_system, _sliced_solve_setup,
    _init_sliced_total_degree, _rebuild_sliced, _sliced_evaluator, SlicedSystem,
    LinearSubspace, FSVec, FSMat, TaylorVector, vectors,
    evaluate!, evaluate_and_jacobian!, taylor!, path_results, steps,
    fix_parameters
using DynamicPolynomials: @polyvar
using CommonSolve: CommonSolve
using LinearAlgebra: norm

# Largest residual of the linear equations of `L` at `x`.
function subspace_residual(L, x)
    E = extrinsic(L)
    return isempty(E.b) ? 0.0 : maximum(abs, E.A * x - E.b)
end

@testset "Sliced solve" begin

    @testset "slice: shape and equations" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L = rand_subspace(2; codim = 1)
        G = slice(F, L)
        @test size(G) == (2, 2)
        @test length(polynomials(G)) == 2
        @test degrees(G) == [2, 1]
        @test collect(variables(G)) == collect(variables(F))
    end

    @testset "slice: preserves parameters and compile mode" begin
        @polyvar x y a
        F = System(
            [x^2 + y^2 - a]; variables = [x, y], parameters = [a],
            compile = CompileMode.COMPILED
        )
        G = slice(F, rand_subspace(2; codim = 1))
        @test nparameters(G) == 1
        @test collect(parameters(G)) == collect(parameters(F))
        @test G.compile_mode == CompileMode.COMPILED
    end

    @testset "slice: chart row" begin
        @polyvar x y z
        F = System([x^2 + y^2 - z^2]; variables = [x, y, z])
        L = rand_subspace(3; codim = 1, affine = false)
        @test size(slice(F, L)) == (2, 3)
        @test size(slice(F, L; chart = randn(ComplexF64, 3))) == (3, 3)
    end

    @testset "slice: ambient dimension mismatch" begin
        @polyvar x y z
        F = System([x^2 + y^2 - z^2]; variables = [x, y, z])
        @test_throws ArgumentError slice(F, rand_subspace(2; codim = 1))
    end

    @testset "solve(F, L): affine line through a conic" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L = rand_subspace(2; codim = 1)
        res = solve(F, L; show_progress = false)
        @test nsolutions(res) == 2
        for s in solutions(res)
            @test length(s) == 2
            @test abs(s[1]^2 + s[2]^2 - 5) < 1.0e-10
            @test subspace_residual(L, s) < 1.0e-10
        end
    end

    @testset "solve(F, L): two equations in three variables" begin
        @polyvar x y z
        F = System([x^2 + y^2 - 5, x * y + 1]; variables = [x, y, z])
        L = rand_subspace(3; codim = 1)
        res = solve(F, L; show_progress = false)
        @test nsolutions(res) == 4
        for s in solutions(res)
            @test subspace_residual(L, s) < 1.0e-10
        end
    end

    @testset "solve(F, L): projective" begin
        @polyvar x y z
        F = System([x^2 + y^2 - z^2]; variables = [x, y, z])
        @test is_homogeneous(F)
        L = rand_subspace(3; codim = 1, affine = false)
        res = solve(F, L; show_progress = false)
        @test nsolutions(res) == 2
        for s in solutions(res)
            @test abs(s[1]^2 + s[2]^2 - s[3]^2) < 1.0e-10 * norm(s, Inf)^2
            @test subspace_residual(L, s) < 1.0e-10 * norm(s, Inf)
        end
    end

    @testset "solve(F, L): parametric" begin
        @polyvar x y a
        F = System([x^2 + y^2 - a]; variables = [x, y], parameters = [a])
        L = rand_subspace(2; codim = 1)
        res = solve(fix_parameters(F, [5.0]), L; show_progress = false)
        @test nsolutions(res) == 2
        for s in solutions(res)
            @test abs(s[1]^2 + s[2]^2 - 5) < 1.0e-10
        end
        @test_throws ArgumentError solve(F, L; show_progress = false)
    end

    @testset "solve(F, L): fixing parameters of a parameter-free system" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        @test_throws ArgumentError fix_parameters(F, [1.0])
    end

    @testset "solve(F, L): overdetermined slice keeps the excess checker" begin
        @polyvar x y
        # The four points of V(F) do not lie on a generic line, so every
        # endpoint of the (overdetermined) sliced system is an excess solution.
        F = System([x^2 + y^2 - 5, x * y - 1]; variables = [x, y])
        res = solve(F, rand_subspace(2; codim = 1); show_progress = false)
        @test nsolutions(res) == 0
        @test nexcess_solutions(res) > 0
    end

    @testset "solve(F, L): Polyhedral" begin
        @polyvar x y z
        F = System([x^2 + y^2 - 5, x * y + 1]; variables = [x, y, z])
        L = rand_subspace(3; codim = 1)
        res = solve(F, L, Polyhedral(); show_progress = false)
        @test nsolutions(res) == 4
        for s in solutions(res)
            @test subspace_residual(L, s) < 1.0e-10
        end
    end

    @testset "solve(F, L): executors agree" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L = rand_subspace(2; codim = 1)
        alg = TotalDegree(; seed = UInt32(0x1234))
        serial = solve(F, L, alg, Serial(); show_progress = false)
        threaded = solve(F, L, alg, Threaded(); show_progress = false)
        @test nsolutions(serial) == nsolutions(threaded) == 2
        @test seed(serial) == seed(threaded) == UInt32(0x1234)
        @test nsolutions(solve(F, L, Serial(); show_progress = false)) == 2
    end

    @testset "solve(F, L): reproducible chart in the projective case" begin
        @polyvar x y z
        F = System([x^2 + y^2 - z^2]; variables = [x, y, z])
        L = rand_subspace(3; codim = 1, affine = false)
        seed32 = UInt32(0x2024)
        G1 = _sliced_solve_system(F, L, seed32)
        G2 = _sliced_solve_system(F, L, seed32)
        @test collect(polynomials(G1)) == collect(polynomials(G2))
    end

    @testset "SlicedSystem matches the rebuilt polynomial system" begin
        @polyvar x y z
        F = System([x^3 + y^2 * z - 2 * z^3, x * y - z^2]; variables = [x, y, z])
        L = rand_subspace(3; codim = 1)
        alg = TotalDegree(; seed = UInt32(11))
        G, chart = _sliced_solve_setup(F, L, alg.seed)
        wrapped = _sliced_evaluator(
            G.evaluator, convert(LinearSubspace{ComplexF64}, L), chart,
        )
        rebuilt = _rebuild_sliced(G, L, chart).evaluator
        @test size(wrapped) == size(rebuilt) == (3, 3)

        p = FSVec{ComplexF64}(ComplexF64[])
        x0 = FSVec{ComplexF64}(randn(ComplexF64, 3))
        u1 = FSVec{ComplexF64}(zeros(ComplexF64, 3))
        u2 = FSVec{ComplexF64}(zeros(ComplexF64, 3))
        U1 = FSMat{ComplexF64}(zeros(ComplexF64, 3, 3))
        U2 = FSMat{ComplexF64}(zeros(ComplexF64, 3, 3))

        evaluate!(u1, wrapped, x0, p)
        evaluate!(u2, rebuilt, x0, p)
        @test maximum(abs, Vector(u1) - Vector(u2)) < 1.0e-12

        evaluate_and_jacobian!(u1, U1, wrapped, x0, p)
        evaluate_and_jacobian!(u2, U2, rebuilt, x0, p)
        @test maximum(abs, Vector(u1) - Vector(u2)) < 1.0e-12
        @test maximum(abs, Matrix(U1) - Matrix(U2)) < 1.0e-12

        # Taylor coefficients must agree with a nonzero highest-order row too:
        # a straight-line homotopy asks for order K-1 with that row filled in.
        rows = [randn(ComplexF64, 3) for _ in 1:4]
        for K in (1, 2, 3)
            tx = TaylorVector{K + 1, ComplexF64}(3)
            for (i, vv) in enumerate(vectors(tx))
                vv .= rows[i]
            end
            a1 = FSVec{ComplexF64}(zeros(ComplexF64, 3))
            a2 = FSVec{ComplexF64}(zeros(ComplexF64, 3))
            taylor!(a1, Val(K), wrapped, tx, p)
            taylor!(a2, Val(K), rebuilt, tx, p)
            @test maximum(abs, Vector(a1) - Vector(a2)) < 1.0e-12
        end
    end

    @testset "SlicedSystem route tracks the same paths as the rebuild route" begin
        @polyvar x y z
        F = System([x^3 + y^2 * z - 2 * z^3, x * y - z^2]; variables = [x, y, z])
        L = rand_subspace(3; codim = 1)
        alg = TotalDegree(; seed = UInt32(11))
        G, chart = _sliced_solve_setup(F, L, alg.seed)
        wrapped = CommonSolve.solve!(
            _init_sliced_total_degree(G, L, chart, alg, Serial(), false),
        )
        rebuilt = CommonSolve.solve!(
            CommonSolve.init(
                _rebuild_sliced(G, L, chart), alg, Serial(); show_progress = false,
            ),
        )
        @test nsolutions(wrapped) == nsolutions(rebuilt)
        @test [steps(pr) for pr in path_results(wrapped)] ==
            [steps(pr) for pr in path_results(rebuilt)]
        a1 = sort(solutions(wrapped); by = real ∘ first)
        a2 = sort(solutions(rebuilt); by = real ∘ first)
        @test maximum(norm.(a1 .- a2, Inf)) < 1.0e-8
    end

    @testset "SlicedSystem linear rows and validation" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L = rand_subspace(2; codim = 1)
        S = SlicedSystem(F.evaluator, L)
        @test size(S) == (2, 2)
        p = FSVec{ComplexF64}(ComplexF64[])
        x0 = FSVec{ComplexF64}(randn(ComplexF64, 2))
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u, S, x0, p)
        E = extrinsic(L)
        @test abs(u[2] - (E.A * Vector(x0) - E.b)[1]) < 1.0e-12
        @test_throws ArgumentError SlicedSystem(F.evaluator, rand_subspace(3; codim = 1))
        @test_throws ArgumentError SlicedSystem(F.evaluator, L, randn(ComplexF64, 3))
    end

    @testset "solve(F, L): CommonSolve init/solve! interface" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        L = rand_subspace(2; codim = 1)
        cache = CommonSolve.init(F, L, TotalDegree(), Serial(); show_progress = false)
        res = CommonSolve.solve!(cache)
        @test res isa Result
        @test nsolutions(res) == 2
    end
end
