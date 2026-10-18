using Test, Random
using HomotopyContinuation
using HomotopyContinuation: TaylorVector, ComplexDF64
using DynamicPolynomials: @polyvar
using MultivariatePolynomials: MultivariatePolynomials as MP
using FixedSizeArrays: FixedSizeVectorDefault

@static if VERSION < v"1.11"
    buffer(v::AbstractVector{T}) where {T} = Vector{T}(v)
else
    buffer(v::AbstractVector{T}) where {T} = FixedSizeVectorDefault{T}(v)
end

value_at(polys, vars, x) = ComplexF64[f(vars => x) for f in polys]
jacobian_at(polys, vars, x) =
    ComplexF64[MP.differentiate(f, v)(vars => x) for f in polys, v in vars]

_key(R) = sort(
    solutions(R);
    by = z -> (round(real(z[1]); digits = 8), round(imag(z[1]); digits = 8)),
)

@testset "fix_parameters public behavior" begin
    Random.seed!(0x1f3a55c1)
    @polyvar x y a b
    polys = [x^2 + a * y^2 - b, x * y^3 - a * b + 2]
    pvals = ComplexF64[1.7 - 0.4im, 2.3]
    F = System(polys; variables = [x, y], parameters = [a, b])
    xvals = randn(ComplexF64, 2)

    @testset "a System substitutes its parameters" begin
        H = fix_parameters(F, pvals)
        @test H isa System
        @test nparameters(H) == 0
        @test isempty(parameters(H))
        @test degrees(H) == degrees(F)
        @test evaluate(H, xvals) ≈ evaluate(F, xvals, pvals) rtol = 1.0e-12
        @test jacobian(H, xvals) ≈ jacobian(F, xvals, pvals) rtol = 1.0e-12
    end

    @testset "a composition binds through the same public behavior" begin
        C = compose(System([x + y, x - y]; variables = [x, y]), F)
        H = fix_parameters(C, pvals)
        @test H isa FixedParameterSystem
        @test nparameters(H) == 0
        @test size(H) == size(C)
        @test degrees(H) == degrees(C)
        @test evaluate(H, xvals) ≈ evaluate(C, xvals, pvals) rtol = 1.0e-12
        @test jacobian(H, xvals) ≈ jacobian(C, xvals, pvals) rtol = 1.0e-12
    end

    # The independent oracle substitutes the parameter values into the
    # polynomials symbolically, then evaluates and differentiates them.
    @testset "agrees with symbolic substitution" begin
        @polyvar λ
        substituted = [MP.subs(f, [a, b] => pvals) for f in polys]
        H = fix_parameters(F, pvals)
        scales = equation_scales(H)
        @test evaluate(H, xvals) .* scales ≈ value_at(substituted, [x, y], xvals) rtol = 1.0e-12
        @test jacobian(H, xvals) .* scales ≈ jacobian_at(substituted, [x, y], xvals) rtol = 1.0e-12

        outer = [x + y, x - y]
        C = fix_parameters(compose(System(outer; variables = [x, y]), F), pvals)
        composed = [MP.subs(g, [x, y] => substituted) for g in outer]
        @test evaluate(C, xvals) ≈ value_at(composed, [x, y], xvals) rtol = 1.0e-12
        @test jacobian(C, xvals) ≈ jacobian_at(composed, [x, y], xvals) rtol = 1.0e-12

        # Mutating evaluation, in double-double precision and to Taylor order 3.
        P = ParameterHomotopy(H, ComplexF64[], ComplexF64[])
        u = buffer(zeros(ComplexF64, 2))
        evaluate!(u, P, buffer(ComplexDF64.(xvals)), complex(0.0))
        @test u .* scales ≈ value_at(substituted, [x, y], xvals) rtol = 1.0e-12
        for K in 2:3
            X = randn(K + 1, 2) .+ im .* randn(K + 1, 2)
            tx = TaylorVector{K + 1, ComplexF64}(2)
            tx[1] = Tuple(X[:, 1])
            tx[2] = Tuple(X[:, 2])
            taylor!(u, Val(K), P, tx, complex(0.0))
            series = [sum(X[k + 1, j] * λ^k for k in 0:K) for j in 1:2]
            expected = [MP.coefficient(f([x, y] => series), λ^K) for f in substituted]
            @test u .* scales ≈ expected rtol = 1.0e-10
        end
    end

    @testset "rejects a mismatched parameter count" begin
        G = System([x^2 + y, x * y - 1]; variables = [x, y])
        @test_throws ArgumentError fix_parameters(F, ComplexF64[1.0])
        @test_throws ArgumentError fix_parameters(F, ComplexF64[1.0, 2.0, 3.0])
        @test_throws ArgumentError fix_parameters(G, ComplexF64[1.0])
    end
end

@testset "solving a fixed-parameter system" begin
    @polyvar x y a b
    F = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
    G = System([x^2 - 2, x * y + 2]; variables = [x, y])

    @testset "total degree matches the substituted system path for path" begin
        seed = UInt32(99)
        r = solve(
            fix_parameters(F, [2, 4]),
            TotalDegree(; seed = seed, show_progress = false),
            Serial(),
        )
        ref = solve(G, TotalDegree(; seed = seed, show_progress = false), Serial())
        @test nsolutions(r) == nsolutions(ref) == 2
        @test _key(r) ≈ _key(ref)
        @test sort(steps.(path_results(r))) == sort(steps.(path_results(ref)))
    end

    @testset "polyhedral matches the substituted system" begin
        seed = UInt32(99)
        r = solve(
            fix_parameters(F, [2, 4]),
            Polyhedral(; seed = seed, show_progress = false),
            Serial(),
        )
        ref = solve(G, Polyhedral(; seed = seed, show_progress = false), Serial())
        @test nsolutions(r) == nsolutions(ref) == 2
        @test _key(r) ≈ _key(ref)
    end

    @testset "solutions solve the system at those parameter values" begin
        r = solve(
            fix_parameters(F, [2, 4]),
            TotalDegree(; seed = UInt32(5), show_progress = false),
            Serial(),
        )
        for s in solutions(r)
            @test maximum(abs, evaluate(F, s, [2, 4])) < 1.0e-10
        end
    end

    @testset "$(nameof(typeof(exec))) executor" for exec in (Serial(), Threaded())
        r = solve(
            fix_parameters(F, [2, 4]),
            TotalDegree(; seed = UInt32(17), show_progress = false),
            exec,
        )
        ref = solve(
            fix_parameters(F, [2, 4]),
            TotalDegree(; seed = UInt32(17), show_progress = false),
            Serial(),
        )
        @test nsolutions(r) == 2
        @test _key(r) ≈ _key(ref)
    end

    @testset "compile mode $mode" for mode in
        (CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL)
        Fm = System(
            [x^2 - a, x * y - a + b];
            variables = [x, y], parameters = [a, b], compile = mode,
        )
        r = solve(
            fix_parameters(Fm, [2, 4]),
            TotalDegree(; seed = UInt32(23), show_progress = false),
            Serial(),
        )
        @test nsolutions(r) == 2
        @test all(s -> maximum(abs, evaluate(Fm, s, [2, 4])) < 1.0e-10, solutions(r))
    end

    @testset "overdetermined, both routes" begin
        @polyvar u v p1 p2
        Fo = System(
            [u^2 + v^2 - p1, u * v - p2, (u^2 + v^2 - p1) * (u - v)];
            variables = [u, v], parameters = [p1, p2],
        )
        Go = System(
            [u^2 + v^2 - 5, u * v - 2, (u^2 + v^2 - 5) * (u - v)];
            variables = [u, v],
        )
        for alg in (
                TotalDegree(; seed = UInt32(7), show_progress = false),
                Polyhedral(; seed = UInt32(7), show_progress = false),
            )
            r = solve(fix_parameters(Fo, [5, 2]), alg, Serial())
            ref = solve(Go, alg, Serial())
            @test nsolutions(r) == nsolutions(ref)
            @test nexcess_solutions(r) == nexcess_solutions(ref)
            @test _key(r) ≈ _key(ref)
        end
    end

    @testset "composition" begin
        C = compose(System([x + y, x - y]; variables = [x, y]), F)
        Cref = compose(System([x + y, x - y]; variables = [x, y]), G)
        r = solve(
            fix_parameters(C, [2, 4]),
            TotalDegree(; seed = UInt32(3), show_progress = false),
            Serial(),
        )
        ref = solve(Cref, TotalDegree(; seed = UInt32(3), show_progress = false), Serial())
        @test nsolutions(r) == nsolutions(ref)
        @test _key(r) ≈ _key(ref)
    end

    @testset "a parametric system is rejected by name" begin
        for alg in (TotalDegree(), Polyhedral())
            @test_throws ArgumentError solve(F, alg, Serial())
        end
    end
end
