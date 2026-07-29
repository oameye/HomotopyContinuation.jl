using Test, Random
using HomotopyContinuationNext: System, FixedParameterSystem, fix_parameters, CompileMode,
    TotalDegree, Polyhedral, Serial, Threaded, compose,
    solve, solutions, nsolutions, nexcess_solutions, path_results, steps,
    evaluate!, evaluate_and_jacobian!, taylor!,
    TaylorVector, ComplexDF64, FSVec, FSMat, nparameters, SystemEvaluator
import HomotopyContinuationNext as Next
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: MultivariatePolynomials as MP
using CommonSolve: CommonSolve

# Sort key that is stable under path ordering, for comparing two solve routes.
_key(R) = sort(
    solutions(R);
    by = z -> (round(real(z[1]); digits = 8), round(imag(z[1]); digits = 8)),
)

@testset "fix_parameters" begin
    Random.seed!(0x1f3a55c1)
    @polyvar x y a b
    polys = [x^2 + a * y^2 - b, x * y^3 - a * b + 2]
    pvals = ComplexF64[1.7 - 0.4im, 2.3]
    F = System(polys; variables = [x, y], parameters = [a, b])

    # Oracle: the values substituted by hand.
    G = System(
        [MP.polynomial(MP.subs(f, [a, b] => pvals)) for f in polys]; variables = [x, y],
    )
    empty_p = FSVec{ComplexF64}(ComplexF64[])
    xvals = randn(ComplexF64, 2)
    xf = FSVec{ComplexF64}(xvals)
    truth = ComplexF64[f([x, y] => xvals, [a, b] => pvals) for f in polys]

    @testset "a System substitutes into the equations" begin
        H = fix_parameters(F, pvals)
        @test H isa System
        @test nparameters(H) == 0
        @test Next.degrees(H) == Next.degrees(F)
        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u, H.evaluator, xf, empty_p)
        @test u ≈ truth rtol = 1.0e-12
    end

    # A composition has no equations to substitute into, so the values are bound
    # at the evaluator level.
    @testset "a composition binds" begin
        C = compose(System([x + y, x - y]; variables = [x, y]), F)
        H = fix_parameters(C, pvals)
        @test H isa FixedParameterSystem
        @test nparameters(H) == 0
        @test size(H) == size(C)
        @test Next.degrees(H) == Next.degrees(C)
        @test Next.system_shape(H) === Next.system_shape(C)
    end

    @testset "rejects a mismatched parameter count" begin
        @test_throws ArgumentError fix_parameters(F, ComplexF64[1.0])
        @test_throws ArgumentError fix_parameters(F, ComplexF64[1.0, 2.0, 3.0])
        @test_throws ArgumentError fix_parameters(G, ComplexF64[1.0])
    end

    # The bound evaluator must agree with the substituted system on every
    # interface method, since routes reach it through the same wrappers.
    @testset "the bound evaluator matches the substituted system" begin
        bound = Next._bound_evaluator(F.evaluator, pvals)
        @test nparameters(bound) == 0
        @test size(bound) == (2, 2)

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u, bound, xf, empty_p)
        @test u ≈ truth rtol = 1.0e-12

        # DF64 input, Float64 output: the extended-precision residual path.
        u2 = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u2, bound, FSVec{ComplexDF64}(ComplexDF64.(xvals)), empty_p)
        @test u2 ≈ truth rtol = 1.0e-12

        u3 = FSVec{ComplexDF64}(zeros(ComplexDF64, 2))
        evaluate!(u3, bound, FSVec{ComplexDF64}(ComplexDF64.(xvals)), empty_p)
        @test ComplexF64.(Vector(u3)) ≈ truth rtol = 1.0e-12

        U = FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        evaluate_and_jacobian!(u, U, bound, xf, empty_p)
        @test u ≈ truth rtol = 1.0e-12
        for i in 1:2, j in 1:2
            dp = MP.differentiate(polys[i], [x, y][j])
            @test U[i, j] ≈ dp([x, y] => xvals, [a, b] => pvals) rtol = 1.0e-12
        end
    end

    # A frozen `p` has the constant series, so both parameter forms of `taylor!`
    # must agree with the substituted system.
    @testset "taylor! order $K, $(P === nothing ? "scalar" : "TaylorVector") parameters" for
        K in 1:3, P in (nothing, TaylorVector)
        bound = Next._bound_evaluator(F.evaluator, pvals)
        tx = TaylorVector{K + 1, ComplexF64}(2)
        tx.data .= randn(ComplexF64, K + 1, 2)

        u = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        if P === nothing
            taylor!(u, Val(K), bound, tx, empty_p)
        else
            taylor!(u, Val(K), bound, tx, TaylorVector{K + 1, ComplexF64}(0))
        end

        expected = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        taylor!(expected, Val(K), G.evaluator, tx, empty_p)
        @test u ≈ expected rtol = 1.0e-10
    end

    # Threading clones per worker; each clone must be independent and carry the
    # same bound values.
    @testset "a bound composition clones per worker" begin
        C = compose(System([x + y, x - y]; variables = [x, y]), F)
        H = fix_parameters(C, pvals)
        e1 = Next._clone_system_evaluator(H)
        e2 = Next._clone_system_evaluator(H)
        @test e1 !== e2
        @test nparameters(e1) == nparameters(e2) == 0
        u1 = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        u2 = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u1, e1, xf, empty_p)
        evaluate!(u2, e2, xf, empty_p)
        @test u1 ≈ u2 rtol = 1.0e-14
        u0 = FSVec{ComplexF64}(zeros(ComplexF64, 2))
        evaluate!(u0, H.evaluator, xf, empty_p)
        @test u1 ≈ u0 rtol = 1.0e-14
    end
end

@testset "solving a fixed-parameter system" begin
    @polyvar x y a b
    F = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
    # The same system with the parameters substituted by hand.
    G = System([x^2 - 2, x * y + 2]; variables = [x, y])

    @testset "total degree matches the substituted system path for path" begin
        seed = UInt32(99)
        r = solve(
            fix_parameters(F, [2, 4]), TotalDegree(; seed = seed), Serial();
            show_progress = false,
        )
        ref = solve(G, TotalDegree(; seed = seed), Serial(); show_progress = false)
        @test nsolutions(r) == nsolutions(ref) == 2
        @test _key(r) ≈ _key(ref)
        @test sort(steps.(path_results(r))) == sort(steps.(path_results(ref)))
    end

    @testset "polyhedral matches the substituted system" begin
        seed = UInt32(99)
        r = solve(
            fix_parameters(F, [2, 4]), Polyhedral(; seed = seed), Serial();
            show_progress = false,
        )
        ref = solve(G, Polyhedral(; seed = seed), Serial(); show_progress = false)
        @test nsolutions(r) == nsolutions(ref) == 2
        @test _key(r) ≈ _key(ref)
    end

    @testset "solutions solve the system at those parameter values" begin
        r = solve(
            fix_parameters(F, [2, 4]), TotalDegree(; seed = UInt32(5)), Serial();
            show_progress = false,
        )
        for s in solutions(r)
            @test abs(s[1]^2 - 2) < 1.0e-10
            @test abs(s[1] * s[2] - 2 + 4) < 1.0e-10
        end
    end

    @testset "$(nameof(typeof(exec))) executor" for exec in (Serial(), Threaded())
        r = solve(
            fix_parameters(F, [2, 4]), TotalDegree(; seed = UInt32(17)), exec;
            show_progress = false,
        )
        @test nsolutions(r) == 2
        @test _key(r) ≈ _key(
            solve(
                fix_parameters(F, [2, 4]), TotalDegree(; seed = UInt32(17)), Serial();
                show_progress = false,
            ),
        )
    end

    # Fixing a `System`'s parameters adds no cache specialization of its own,
    # since the route sees an ordinary parameter-free `System`.
    @testset "the substituted route reuses the parameter-free cache type" begin
        cache = CommonSolve.init(
            fix_parameters(F, [2, 4]), TotalDegree(; seed = UInt32(41)), Serial();
            show_progress = false,
        )
        @test cache.builder.target_system isa System
        @test typeof(cache) === typeof(
            CommonSolve.init(
                Next.fix_parameters(F, ComplexF64[2, 4]),
                TotalDegree(; seed = UInt32(41)), Serial(); show_progress = false,
            ),
        )
    end

    @testset "compile mode $mode" for mode in
        (CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL)
        Fm = System(
            [x^2 - a, x * y - a + b];
            variables = [x, y], parameters = [a, b], compile = mode,
        )
        r = solve(
            fix_parameters(Fm, [2, 4]), TotalDegree(; seed = UInt32(23)), Serial();
            show_progress = false,
        )
        @test nsolutions(r) == 2
        @test all(s -> abs(s[1]^2 - 2) < 1.0e-10, solutions(r))
    end

    @testset "overdetermined, both routes" begin
        @polyvar u v p1 p2
        Fo = System(
            [u^2 + v^2 - p1, u * v - p2, (u^2 + v^2 - p1) * (u - v)];
            variables = [u, v], parameters = [p1, p2],
        )
        Go = System(
            [u^2 + v^2 - 5, u * v - 2, (u^2 + v^2 - 5) * (u - v)]; variables = [u, v],
        )
        for alg in (TotalDegree(; seed = UInt32(7)), Polyhedral(; seed = UInt32(7)))
            r = solve(fix_parameters(Fo, [5, 2]), alg, Serial(); show_progress = false)
            ref = solve(Go, alg, Serial(); show_progress = false)
            @test nsolutions(r) == nsolutions(ref)
            @test nexcess_solutions(r) == nexcess_solutions(ref)
            @test _key(r) ≈ _key(ref)
        end
    end

    # The bound route: a composition tracks through the wrapped evaluator.
    @testset "composition" begin
        C = compose(System([x + y, x - y]; variables = [x, y]), F)
        Cref = compose(System([x + y, x - y]; variables = [x, y]), G)
        r = solve(
            fix_parameters(C, [2, 4]), TotalDegree(; seed = UInt32(3)), Serial();
            show_progress = false,
        )
        ref = solve(Cref, TotalDegree(; seed = UInt32(3)), Serial(); show_progress = false)
        @test nsolutions(r) == nsolutions(ref)
        @test _key(r) ≈ _key(ref)
    end

    @testset "a parametric system is rejected by name" begin
        for alg in (TotalDegree(), Polyhedral())
            @test_throws ArgumentError solve(F, alg, Serial(); show_progress = false)
        end
    end
end
