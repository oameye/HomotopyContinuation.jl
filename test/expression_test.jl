using Test
import HomotopyContinuationNext as Next
using HomotopyContinuationNext: Expression, System, @var, @unique_var, @polyvar,
    differentiate, subs, num_den, variable, unique_variable, expression_to_sexpr,
    SExprT
using DynamicPolynomials: DynamicPolynomials
using LinearAlgebra: det, dot

@testset "Expression frontend" begin
    @testset "@var declares variables and arrays" begin
        @var p q r[1:3] M[1:2, 1:2]
        @test Symbol(p) === :p
        @test Symbol(q) === :q
        @test r isa Array{Expression}
        @test Symbol.(r) == [:r₁, :r₂, :r₃]
        @test size(M) == (2, 2)
        @test Symbol(M[1, 2]) === :M₁₋₂
        @test Next.is_variable(p)
        @test !Next.is_variable(p + q)
        @test Next.is_number(Expression(2))
        @test !Next.is_number(p)
        @test Next.expr_number(Expression(2)) == ComplexF64(2)
        @test Next.expr_number(p) === nothing
    end

    @testset "@unique_var avoids collisions" begin
        @var s
        @unique_var s
        # The binding is still `s`, but the generated name differs.
        @test Symbol(s) !== :s
        @test startswith(string(Symbol(s)), "s")
    end

    @testset "unique_variable renames until free" begin
        @var t
        w = unique_variable(:t, Expression[t], Expression[])
        @test w != t
        @test unique_variable(:z, Expression[t], Expression[]) == variable(:z)
    end

    @testset "canonicalization and constant folding" begin
        @var x y
        @test iszero(x - x)
        @test isone(x * inv(x))
        @test x + x == 2 * x
        @test x * x == x^2
        @test x^0 == one(Expression)
        @test x^1 == x
        @test (2 * x) * (3 * y) == 6 * (x * y)
        @test (x * y)^2 == x^2 * y^2
        @test sqrt(Expression(4)) == Expression(2)
        @test sin(Expression(0)) == zero(Expression)
        @test cos(Expression(0)) == one(Expression)
        @test x / x == one(Expression)
        @test x^2 / x == x
        @test x + y == y + x
        @test x * y == y * x
    end

    @testset "equal expressions hash alike" begin
        @var x y
        # A coefficient picks up whichever sign of zero the arithmetic that built
        # it produced: `-(y^2)` carries `-1.0 + 0.0im`, while expanding the same
        # term multiplies through to `-1.0 - 0.0im`. The two compare equal, so
        # they have to hash equal as well, or `Dict{Expression}` and the
        # hash-ordered term sort disagree about which terms are the same.
        minus_one_plus = Expression(ComplexF64(-1.0, 0.0))
        minus_one_minus = Expression(ComplexF64(-1.0, -0.0))
        @test minus_one_plus == minus_one_minus
        @test hash(minus_one_plus) == hash(minus_one_minus)
        @test hash(minus_one_plus * y^2) == hash(minus_one_minus * y^2)

        # The payoff: expansion is canonical, so it does not depend on how the
        # expression it is handed was built.
        @test Next.expand((x + y) * (x - y)) == x^2 - y^2
        @test Next.expand(x^2 - y^2) == x^2 - y^2
    end

    @testset "arithmetic promotes numbers" begin
        @var x
        @test x + 1 == 1 + x
        @test 2x - x == x
        @test zero(Expression) + x == x
        @test x^-2 == inv(x^2)
        @test_throws ArgumentError x^(x + 1)
    end

    @testset "variable discovery is sorted by index" begin
        @var z[1:12] u
        f = sum(z) + u
        # Base names sort lexicographically, indices numerically (z₂ before z₁₀).
        @test Symbol.(Next.variables(f)) == [:u; Symbol.(z)]
        @test Next.variables(f; parameters = [u]) == collect(z)
    end

    @testset "differentiate" begin
        @var x y a b

        @test differentiate(x^3, x) == 3 * x^2
        @test differentiate(x * y, [x, y]) == Expression[y, x]
        @test differentiate(1 / x, x) == -inv(x^2)
        @test differentiate(sin(x), x) == cos(x)
        @test differentiate(cos(x), x) == -sin(x)
        @test differentiate(sqrt(x), x) == 1 / (2 * sqrt(x))
        @test differentiate(sqrt(a + b), x) == zero(Expression)

        J = differentiate(Expression[x * y, x + y], [x, y])
        @test J == Expression[y x; 1 1]
    end

    @testset "differentiate matches finite differences" begin
        @var x y a
        f = sqrt(a + x) / y + sin(x * y) - cos(y)^2
        dfdx = differentiate(f, x)
        dfdy = differentiate(f, y)
        g(xv, yv, av) = sqrt(av + xv) / yv + sin(xv * yv) - cos(yv)^2

        x0, y0, a0 = 0.7, 1.3, 2.1
        h = 1.0e-6
        num_x = (g(x0 + h, y0, a0) - g(x0 - h, y0, a0)) / (2h)
        num_y = (g(x0, y0 + h, a0) - g(x0, y0 - h, a0)) / (2h)

        F = System([dfdx, dfdy]; variables = [x, y], parameters = [a])
        u = Next.FSVec{ComplexF64}(zeros(ComplexF64, 2))
        Next.evaluate!(
            u, F.evaluator,
            Next.FSVec{ComplexF64}(ComplexF64[x0, y0]),
            Next.FSVec{ComplexF64}(ComplexF64[a0]),
        )
        @test real(u[1]) ≈ num_x atol = 1.0e-7
        @test real(u[2]) ≈ num_y atol = 1.0e-7
    end

    @testset "subs" begin
        @var x y z w u
        @test subs(x^2 + y, x => y + 1) == (y + 1)^2 + y
        @test subs(x / y, [x, y] => [1, 2]) == Expression(0.5)
        @test subs(Expression[x + y, x * y], [x, y] => [z, 2]) ==
            Expression[z + 2, 2 * z]
        @test subs(sqrt(x), x => 9) == Expression(3)

        f = x^2 * (x + y * w)
        @test subs(f, x => z) == z^2 * (z + w * y)
        @test subs([f], x => z) == [z^2 * (z + w * y)]
        @test subs(f, [x, y] => [z^2, z + 2]) == z^4 * (w * (2 + z) + z^2)
        @test subs(f, [x, y] => [z^2, z + 2], w => u) == z^4 * (u * (2 + z) + z^2)
        @test subs(f, x => z^2, y => 3, w => u) == z^4 * (3 * u + z^2)
        @test subs(f, Dict(x => z^2, y => Expression(3), w => u)) ==
            z^4 * (3 * u + z^2)
        @test subs([f, 2f], Dict(x => z^2, y => Expression(3), w => u)) ==
            [z^4 * (3 * u + z^2), 2 * z^4 * (3 * u + z^2)]
        @test_throws ArgumentError subs(f, [x, y] => [z])
    end

    @testset "substituting every variable folds to a number" begin
        @var x y w
        f = x^2 * (x + y * w)
        @test subs(f, [x, y, w] => [2, 3, -5]) == Expression(-52)
        @test subs([f, 2f], [x, y, w] => [2, 3, -5]) ==
            Expression[-52, -104]
        @test Next.expr_number(subs(f, Dict(x => 2, y => 3, w => -5))) == -52

        # Float64 coefficients stay real after folding.
        @var a b
        constraints = [
            -0.2 * (-4.2467 * (0.1 + 1.0 * a^2) + 1.0 * b^2),
            -0.222 * (-4.49 * (0.5 + 1.0 * a^2) + 1.0 * b^2),
        ]
        vals = subs(constraints, Dict(a => 0.25, b => 0.75))
        @test all(v -> imag(Next.expr_number(v)) == 0, vals)
    end

    @testset "determinant" begin
        @var x[1:2, 1:2]
        @test det(x[1:1, 1:1]) == x[1, 1]
        @test det(x) == -x[2, 1] * x[1, 2] + x[2, 2] * x[1, 1]
        @test det(x') == -x[2, 1] * x[1, 2] + x[2, 2] * x[1, 1]
        @test det(transpose(x)) == -x[2, 1] * x[1, 2] + x[2, 2] * x[1, 1]

        @var y[1:3, 1:3]
        cofactor = sum(
            (isodd(j) ? 1 : -1) * y[1, j] *
                det(y[2:3, [k for k in 1:3 if k != j]]) for j in 1:3
        )
        @test det(y) == cofactor
        @test det(Expression[2 0; 0 3]) == Expression(6)
        @test det(Expression[x[1, 1] 0; 0 0]) == zero(Expression)
        @test_throws DimensionMismatch det(y[1:2, 1:3])
    end

    @testset "conjugation" begin
        @var x y
        # Variables stand for real symbols: the tape has no conjugation instruction.
        @test conj(Expression(1 + 2im)) == Expression(1 - 2im)
        @test conj(x) == x
        @test conj((1 + 2im) * x) == (1 - 2im) * x
        @test adjoint((1 + 2im) * x) == (1 - 2im) * x
        @test transpose((1 + 2im) * x) == (1 + 2im) * x
        @test conj((1 + 2im) * x^2 + 3im * y - 4im) == (1 - 2im) * x^2 - 3im * y + 4im
        @test conj(sqrt(2im * x)) == sqrt(-2im * x)
        @test conj(x^2 + 2 * x * y) == x^2 + 2 * x * y
        @test conj(conj((1 + 2im) * x)) == (1 + 2im) * x

        @polyvar u
        @test adjoint((1 + 2im) * u) == (1 - 2im) * u

        # `'` on a matrix transposes and conjugates entrywise.
        A = Expression[(1 + 2im) * x 3im; y 1]
        @test A' == Expression[(1 - 2im) * x y; -3im 1]
    end

    @testset "degree and is_polynomial" begin
        @var x y a
        @test Next.degree(x^2 * y + 1, [x, y]) == 3
        @test Next.degree(a^5 * x, [x]) == 1
        @test Next.degree(1 / x, [x]) == -1
        @test Next.degree(sqrt(x), [x]) == -1
        @test Next.degree(sqrt(a) * x^2, [x]) == 2
        @test Next.is_polynomial(x^2 + y, [x, y])
        @test !Next.is_polynomial(x / y, [x, y])
        @test Next.is_polynomial(x / a, [x])
    end

    @testset "rational functions" begin
        @var x y z
        f = x^2 + y^2 - z
        g = x / (y - 1) + y + z - 1
        h = f * g
        @test Next.is_polynomial(f, [x, y, z])
        @test !Next.is_polynomial(g, [x, y, z])
        @test !Next.is_polynomial(h, [x, y, z])
        @test Next.degree(h, [x, y, z]) == -1

        # Denominators that carry no variable leave the degree intact.
        @var a
        @test Next.is_polynomial(f / a, [x, y, z])
        @test Next.degree(f / a, [x, y, z]) == 2
    end

    @testset "num_den" begin
        @var x y z a
        @test num_den(x^2 + y) == (x^2 + y, Expression(1))
        @test num_den(Expression(3)) == (Expression(3), Expression(1))
        @test num_den(1 / x) == (Expression(1), x)
        @test num_den(x / (y - 1) + y + z) ==
            (x + (y - 1) * y + (y - 1) * z, y - 1)
        # The numeric part of a denominator moves into the numerator.
        @test num_den(x / (2 * y)) == (0.5 * x, y)
        # Each base enters the common denominator with its highest power.
        p, q = num_den(1 / (x - 1) + 1 / (x - 1)^2 + 1 / y)
        @test q == (x - 1)^2 * y
        @test p == (x - 1) * y + y + (x - 1)^2
        # Negative powers of a rational base invert it.
        @test num_den((1 + 1 / x)^-2) == (x^2, (x + 1)^2)
        # No factoring, so structurally different bases do not cancel.
        @test num_den((x - 1)^2 / (x^2 - 2 * x + 1)) ==
            ((x - 1)^2, x^2 - 2 * x + 1)
        # `sqrt`, `sin` and `cos` have no rational normal form.
        @test num_den(sqrt(x) / y + 1) == (sqrt(x) + y, y)
        @test num_den(sqrt(1 / x)) == (sqrt(1 / x), Expression(1))
        # f == num / den on random values.
        for f in (x / (y - 1) + y, (1 + 1 / x)^-2 * y, a / (x * y) + 1 / (x + y))
            p, q = num_den(f)
            v = Dict(x => 0.3 + 0.7im, y => -1.1 + 0.2im, a => 2.0 - 0.5im)
            @test Next.expr_number(Next.subs(f, v)) ≈
                Next.expr_number(Next.subs(p, v)) / Next.expr_number(Next.subs(q, v))
        end

        # A rational factor in a product form. Nothing is expanded, so the
        # numerator agrees with the factored form by value, not structurally.
        f = x^2 + y^2 - z
        g = x / (y - 1) + y + z - 1
        p, q = num_den(f * g)
        @test q == y - 1
        v = Dict(x => 0.4 + 0.1im, y => 1.3 - 0.2im, z => -0.7 + 0.9im)
        @test Next.expr_number(Next.subs(p, v)) ≈
            Next.expr_number(Next.subs(f * (x + (y - 1) * (y + z - 1)), v))
    end

    @testset "has_real_coefficients" begin
        @var x
        @test Next.has_real_coefficients(2 * x^2 - 1)
        @test !Next.has_real_coefficients((1 + 2im) * x)
        @test !Next.has_real_coefficients(sqrt(im * x))
    end

    @testset "MultivariatePolynomials conversion" begin
        @polyvar px py pu[1:2]

        e = Expression(px^2 * py - 3)
        @var x y
        # Names must round-trip so `@polyvar` and `@var` variables agree.
        @test Next.variables(e) == [variable(:px), variable(:py)]

        r = pu[1] / px^2 + pu[2]
        er = Expression(r)
        @test !Next.is_polynomial(er, [variable(:px)])

        F = System([r, px + py - 1]; parameters = pu, variables = [px, py])
        @test Next.nparameters(F) == 2
        @test Next.nvariables(F) == 2

        u = Next.FSVec{ComplexF64}(zeros(ComplexF64, 2))
        Next.evaluate!(
            u, F.evaluator,
            Next.FSVec{ComplexF64}(ComplexF64[2.0, -1.0]),
            Next.FSVec{ComplexF64}(ComplexF64[8.0, 3.0]),
        )
        @test u[1] ≈ 8.0 / 4.0 + 3.0
        @test u[2] ≈ 0.0
    end

    @testset "lowering agrees with the polynomial front-end" begin
        @polyvar px py
        polys = [px^2 * py - 3 * px + 1, px * py^2 + 2 * py]
        F_poly = System(polys)
        F_expr = System(Expression.(polys); variables = [px, py])

        x = Next.FSVec{ComplexF64}(ComplexF64[0.4 + 0.2im, -1.1 + 0.7im])
        p = Next.FSVec{ComplexF64}(ComplexF64[])
        u1 = Next.FSVec{ComplexF64}(zeros(ComplexF64, 2))
        u2 = Next.FSVec{ComplexF64}(zeros(ComplexF64, 2))
        U1 = Next.FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        U2 = Next.FSMat{ComplexF64}(zeros(ComplexF64, 2, 2))
        Next.evaluate_and_jacobian!(u1, U1, F_poly.evaluator, x, p)
        Next.evaluate_and_jacobian!(u2, U2, F_expr.evaluator, x, p)
        @test collect(u1) ≈ collect(u2)
        @test collect(U1) ≈ collect(U2)
        @test Next.degrees(F_poly) == Next.degrees(F_expr)
    end

    @testset "expression_to_sexpr rejects unknown symbols" begin
        @var x y
        var_map = Dict(:x => 1)
        param_map = Dict{Symbol, Int}()
        @test expression_to_sexpr(x^2, var_map, param_map) isa SExprT
        @test_throws ArgumentError expression_to_sexpr(x + y, var_map, param_map)
    end

    @testset "modeling" begin
        @testset "bottleneck" begin
            @var x y z
            f = [
                (0.3 * x^2 + 0.5z + 0.3x + 1.2 * y^2 - 1.1)^2 +
                    (0.7 * (y - 0.5x)^2 + y + 1.2 * z^2 - 1)^2 - 0.3,
            ]
            vars = [x, y, z]
            n, m = length(vars), length(f)
            @unique_var q[1:n] v[1:m] w[1:m]
            J = differentiate(f, vars)
            f′ = subs(f, vars => q)
            J′ = subs(J, vars => q)
            Nx = (vars - q) - J' * v
            Ny = (vars - q) - J′' * w
            F = System([f; f′; Nx; Ny]; variables = [vars; q; v; w])
            @test size(F) == (8, 8)
        end

        @testset "steiner" begin
            @var x[1:2] a[1:5] c[1:6] y[1:2, 1:5] v[1:6, 1:5]
            # The six degree-2 monomials in (x₁, x₂, 1).
            quadric(z) = [z[i] * z[j] for i in 1:3 for j in i:3]

            f = sum([a; 1] .* quadric([x; 1]))
            ∇ = differentiate(f, x)
            g = sum(c .* quadric([x; 1]))
            ∇_2 = differentiate(g, x)

            # The conic f is tangent to the conic gᵢ at the point y[:, i].
            function incidence(i)
                fᵢ = subs(f, x => y[:, i])
                ∇ᵢ = subs(∇, x => y[:, i])
                Cᵢ = subs(g, x => y[:, i], c => v[:, i])
                ∇_Cᵢ = subs(∇_2, x => y[:, i], c => v[:, i])
                return [fᵢ; Cᵢ; det([∇ᵢ ∇_Cᵢ])]
            end

            F = System(
                vcat(map(incidence, 1:5)...);
                variables = [a; vec(y)], parameters = vec(v),
            )
            @test size(F) == (15, 15)
            @test Next.nparameters(F) == 30
        end

        @testset "reach of a plane curve" begin
            @var x y
            f = (x^3 - x * y^2 + y + 1)^2 * (x^2 + y^2 - 1) + y^2 - 5
            ∇ = differentiate(f, [x, y])
            H = differentiate(∇, [x, y])

            g = dot(∇, ∇)
            v = [-∇[2]; ∇[1]]
            h = v' * H * v
            dg = differentiate(g, [x, y])
            dh = differentiate(h, [x, y])
            ∇σ = g .* dh - ((3 / 2) * h) .* dg

            F = System([dot(v, ∇σ); f]; variables = [x, y])
            @test size(F) == (2, 2)
        end
    end

    @testset "show" begin
        @var x y
        @test occursin("sqrt(", sprint(show, sqrt(x)))
        @test occursin("sin(", sprint(show, sin(x)))
        @test occursin("cos(", sprint(show, cos(x)))
        @test sprint(show, x) == "x"
        @test occursin("^-1", sprint(show, inv(x + y)))
    end
end
