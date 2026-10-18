using Test
using HomotopyContinuation
using LinearAlgebra: det, dot

@testset "Expression frontend" begin
    @testset "variables and unique variables" begin
        @var p q r[1:3] M[1:2, 1:2]
        @test Symbol(p) === :p
        @test Symbol(q) === :q
        @test Symbol.(r) == [:r₁, :r₂, :r₃]
        @test size(M) == (2, 2)
        @test Symbol(M[1, 2]) === :M₁₋₂

        @var s
        @unique_var s
        @test Symbol(s) !== :s
        @test startswith(string(Symbol(s)), "s")
    end

    @testset "algebraic canonicalization" begin
        @var x y
        @test iszero(x - x)
        @test isone(x * inv(x))
        @test x + x == 2x
        @test x * x == x^2
        @test x^0 == one(Expression)
        @test x^1 == x
        @test (2x) * (3y) == 6x * y
        @test (x * y)^2 == x^2 * y^2
        @test sqrt(Expression(4)) == Expression(2)
        @test sin(Expression(0)) == zero(Expression)
        @test cos(Expression(0)) == one(Expression)
        @test x / x == one(Expression)
        @test x^2 / x == x
        @test x + y == y + x
        @test x * y == y * x
        @test x + 1 == 1 + x
        @test 2x - x == x
        @test x^-2 == inv(x^2)
        @test_throws ArgumentError x^(x + 1)

        minus_one_plus = Expression(ComplexF64(-1.0, 0.0))
        minus_one_minus = Expression(ComplexF64(-1.0, -0.0))
        @test minus_one_plus == minus_one_minus
        @test hash(minus_one_plus) == hash(minus_one_minus)
        @test expand((x + y) * (x - y)) == x^2 - y^2
    end

    @testset "variable discovery is deterministic" begin
        @var z[1:12] u
        f = sum(z) + u
        @test Symbol.(variables(f)) == [:u; Symbol.(z)]
        @test variables(f; parameters = [u]) == collect(z)
    end

    @testset "symbolic differentiation matches analysis and finite differences" begin
        @var x y a b
        @test differentiate(x^3, x) == 3x^2
        @test differentiate(x * y, [x, y]) == Expression[y, x]
        @test differentiate(1 / x, x) == -inv(x^2)
        @test differentiate(sin(x), x) == cos(x)
        @test differentiate(cos(x), x) == -sin(x)
        @test differentiate(sqrt(x), x) == 1 / (2sqrt(x))
        @test differentiate(sqrt(a + b), x) == zero(Expression)
        @test differentiate(Expression[x * y, x + y], [x, y]) == Expression[y x; 1 1]

        f = sqrt(a + x) / y + sin(x * y) - cos(y)^2
        F = System(
            [differentiate(f, x), differentiate(f, y)];
            variables = [x, y], parameters = [a],
        )
        x0, y0, a0 = 0.7, 1.3, 2.1
        h = 1.0e-6
        g(xv, yv) = sqrt(a0 + xv) / yv + sin(xv * yv) - cos(yv)^2
        numeric = [
            (g(x0 + h, y0) - g(x0 - h, y0)) / (2h),
            (g(x0, y0 + h) - g(x0, y0 - h)) / (2h),
        ]
        @test real.(evaluate(F, [x0, y0], [a0])) ≈ numeric atol = 1.0e-7
    end

    @testset "substitution preserves algebra" begin
        @var x y z w u
        @test subs(x^2 + y, x => y + 1) == (y + 1)^2 + y
        @test subs(x / y, [x, y] => [1, 2]) == Expression(0.5)
        @test subs(Expression[x + y, x * y], [x, y] => [z, 2]) == Expression[z + 2, 2z]
        @test subs(sqrt(x), x => 9) == Expression(3)

        f = x^2 * (x + y * w)
        @test subs(f, x => z) == z^2 * (z + w * y)
        @test subs(f, [x, y] => [z^2, z + 2]) == z^4 * (w * (2 + z) + z^2)
        @test subs(f, [x, y] => [z^2, z + 2], w => u) == z^4 * (u * (2 + z) + z^2)
        @test subs(f, Dict(x => z^2, y => Expression(3), w => u)) == z^4 * (3u + z^2)
        @test_throws ArgumentError subs(f, [x, y] => [z])

        @test to_number(subs(f, [x, y, w] => [2, 3, -5])) == -52
        @test to_number.(subs([f, 2f], [x, y, w] => [2, 3, -5])) == [-52, -104]
    end

    @testset "determinant and conjugation obey their mathematical identities" begin
        @var x[1:2, 1:2]
        @test det(x[1:1, 1:1]) == x[1, 1]
        @test det(x) == -x[2, 1] * x[1, 2] + x[2, 2] * x[1, 1]
        @test det(x') == det(x)
        @test det(transpose(x)) == det(x)

        @var y[1:3, 1:3]
        cofactor = sum(
            (isodd(j) ? 1 : -1) * y[1, j] * det(y[2:3, [k for k in 1:3 if k != j]])
                for j in 1:3
        )
        @test det(y) == cofactor
        @test det(Expression[2 0; 0 3]) == Expression(6)
        @test_throws DimensionMismatch det(y[1:2, 1:3])

        @var u v
        @test conj(Expression(1 + 2im)) == Expression(1 - 2im)
        @test conj(u) == u
        @test conj((1 + 2im) * u) == (1 - 2im) * u
        @test conj((1 + 2im) * u^2 + 3im * v - 4im) == (1 - 2im) * u^2 - 3im * v + 4im
        @test conj(conj((1 + 2im) * u)) == (1 + 2im) * u
        A = Expression[(1 + 2im) * u 3im; v 1]
        @test A' == Expression[(1 - 2im) * u v; -3im 1]
    end

    @testset "degree and rational normalization" begin
        @var x y z a
        @test degree(x^2 * y + 1, [x, y]) == 3
        @test degree(a^5 * x, [x]) == 1
        @test degree(1 / x, [x]) == -1
        @test degree(sqrt(x), [x]) == -1
        @test degree(sqrt(a) * x^2, [x]) == 2
        @test degree((x^2 + y^2 - z) / a, [x, y, z]) == 2

        @test num_den(x^2 + y) == (x^2 + y, Expression(1))
        @test num_den(1 / x) == (Expression(1), x)
        @test num_den(x / (y - 1) + y + z) ==
            (x + (y - 1) * y + (y - 1) * z, y - 1)
        @test num_den(x / (2y)) == (0.5x, y)
        p, q = num_den(1 / (x - 1) + 1 / (x - 1)^2 + 1 / y)
        @test q == (x - 1)^2 * y
        @test p == (x - 1) * y + y + (x - 1)^2
        @test num_den((1 + 1 / x)^-2) == (x^2, (x + 1)^2)

        for f in (x / (y - 1) + y, (1 + 1 / x)^-2 * y, a / (x * y) + 1 / (x + y))
            numerator, denominator = num_den(f)
            values = Dict(x => 0.3 + 0.7im, y => -1.1 + 0.2im, a => 2.0 - 0.5im)
            @test to_number(subs(f, values)) ≈
                to_number(subs(numerator, values)) / to_number(subs(denominator, values))
        end
    end

    @testset "polynomial and Expression frontends are numerically equivalent" begin
        @polyvar px py pu[1:2]
        @var x y

        e = Expression(px^2 * py - 3)
        @test Symbol.(variables(e)) == [:px, :py]

        rational = pu[1] / px^2 + pu[2]
        F = System([rational, px + py - 1]; parameters = pu, variables = [px, py])
        @test nparameters(F) == 2
        @test nvariables(F) == 2
        @test evaluate(F, [2.0, -1.0], [8.0, 3.0]) ≈ ComplexF64[5, 0]

        polys = [px^2 * py - 3px + 1, px * py^2 + 2py]
        F_poly = System(polys)
        F_expr = System(Expression.(polys); variables = [px, py])
        point = ComplexF64[0.4 + 0.2im, -1.1 + 0.7im]
        @test evaluate(F_poly, point) ≈ evaluate(F_expr, point)
        @test jacobian(F_poly, point) ≈ jacobian(F_expr, point)
    end

    @testset "modeling scenarios compose public symbolic operations" begin
        @testset "bottleneck" begin
            @var x y z
            f = [
                (0.3x^2 + 0.5z + 0.3x + 1.2y^2 - 1.1)^2 +
                    (0.7(y - 0.5x)^2 + y + 1.2z^2 - 1)^2 - 0.3,
            ]
            vars = [x, y, z]
            @unique_var q[1:3] v[1:1] w[1:1]
            J = differentiate(f, vars)
            f′ = subs(f, vars => q)
            J′ = subs(J, vars => q)
            Nx = (vars - q) - J' * v
            Ny = (vars - q) - J′' * w
            F = System([f; f′; Nx; Ny]; variables = [vars; q; v; w])
            @test size(F) == (8, 8)
        end

        @testset "Steiner tangency system" begin
            @var x[1:2] a[1:5] c[1:6] y[1:2, 1:5] v[1:6, 1:5]
            quadric(z) = [z[i] * z[j] for i in 1:3 for j in i:3]
            f = sum([a; 1] .* quadric([x; 1]))
            ∇ = differentiate(f, x)
            g = sum(c .* quadric([x; 1]))
            ∇g = differentiate(g, x)

            function incidence(i)
                fi = subs(f, x => y[:, i])
                ∇i = subs(∇, x => y[:, i])
                gi = subs(g, x => y[:, i], c => v[:, i])
                ∇gi = subs(∇g, x => y[:, i], c => v[:, i])
                return [fi; gi; det([∇i ∇gi])]
            end

            F = System(
                vcat(map(incidence, 1:5)...);
                variables = [a; vec(y)], parameters = vec(v),
            )
            @test size(F) == (15, 15)
            @test nparameters(F) == 30
        end

        @testset "reach of a plane curve" begin
            @var x y
            f = (x^3 - x * y^2 + y + 1)^2 * (x^2 + y^2 - 1) + y^2 - 5
            ∇ = differentiate(f, [x, y])
            H = differentiate(∇, [x, y])
            g = dot(∇, ∇)
            v = [-∇[2]; ∇[1]]
            h = v' * H * v
            ∇σ = g .* differentiate(h, [x, y]) - ((3 / 2) * h) .* differentiate(g, [x, y])
            F = System([dot(v, ∇σ); f]; variables = [x, y])
            @test size(F) == (2, 2)
        end
    end

    @testset "display reflects symbolic structure" begin
        @var x y
        @test occursin("sqrt(", sprint(show, sqrt(x)))
        @test occursin("sin(", sprint(show, sin(x)))
        @test occursin("cos(", sprint(show, cos(x)))
        @test sprint(show, x) == "x"
        @test occursin("^-1", sprint(show, inv(x + y)))
    end
end
