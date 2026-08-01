using Test
using HomotopyContinuationNext: Expression, System, @var, subs, variables, degree,
    expand, to_dict, horner, monomials, dense_poly, rand_poly, coefficients,
    coeffs_as_dense_poly, exponents_coefficients, poly_from_exponents_coefficients,
    to_number, evaluate, jacobian, is_real, compose
using Random: MersenneTwister

@testset "expand" begin
    @var x y
    @test expand((x + y)^2) == 2 * x * y + x^2 + y^2
    @test expand((x + y)^3) == x^3 + 3 * x^2 * y + 3 * x * y^2 + y^3
    @test expand((x + y) * (x - y)) == x^2 - y^2
    # A sum under a negative power stays put; the rest of the product expands.
    @test expand((x + y)^-1 * (x + y)) == Expression(1)
    @test expand(x * (x + y)^-2) == x * (x + y)^-2
    # The sum multiplying a `sqrt` distributes; the one under it does not.
    @test expand(sqrt(x + y) * (x + y)) == x * sqrt(x + y) + y * sqrt(x + y)
    @test expand(Expression(3)) == Expression(3)
    @test expand(x) == x
    @test expand([(x + y)^2, x]) == [2 * x * y + x^2 + y^2, x]
end

@testset "to_dict" begin
    @var x y a
    @test to_dict(x^2 + y * x * a + y * x, [x, y]) ==
        Dict([2, 0] => Expression(1), [1, 1] => a + 1)
    @test degree(x * y^5 + y^2, [x, y]) == 6

    # Non-`vars` variables land in the coefficient, expanded.
    @test to_dict((a + 1) * (a + 2) * x, [x]) == Dict([1] => a^2 + 3 * a + 2)
    # A cancellation inside a coefficient leaves no term behind.
    @test to_dict((a - a) * x + y, [x]) == Dict([0] => y)
    @test_throws ArgumentError to_dict(x / y, [x, y])
    @test_throws ArgumentError to_dict(sqrt(x), [x])
    # Non-polynomial in `vars` only: `sqrt(a)` is a coefficient.
    @test to_dict(sqrt(a) * x, [x]) == Dict([1] => sqrt(a))
end

@testset "monomials" begin
    @var x y
    @test monomials([x, y], 2; affine = false) == [x^2, x * y, y^2]
    @test monomials([x, y], 2; homogeneous = true) == [x^2, x * y, y^2]
    @test monomials([x, y], 2) == [x^2, x * y, y^2, x, y, Expression(1)]
    @test monomials([x, y], [1, 2]) == [x^2, x * y, y^2, x, y]
    @test monomials([x, y], 0) == [Expression(1)]
    @test length(monomials([x, y], 3)) == 10
end

@testset "dense_poly / rand_poly / coefficients" begin
    @var x y
    f, c = dense_poly([x, y], 3)
    @test length(c) == 10
    g = rand_poly(MersenneTwister(7), Float64, [x, y], 3)
    @test subs(f, c => coefficients(g, [x, y])) == g
    _, coeffs = exponents_coefficients(g, [x, y])
    @test subs(f, c => coeffs) == g

    h = rand_poly(MersenneTwister(7), [x, y], 2)
    @test length(coefficients(h, [x, y])) == 6
    @test rand_poly(MersenneTwister(7), Float64, [x, y], 3; homogeneous = true) ==
        rand_poly(MersenneTwister(7), Float64, [x, y], 3; homogeneous = true)

    @var z[1:3]
    fz, cz = dense_poly(z, 3; coeff_name = :c)
    gz = z[1]^3 + z[2]^3 + z[3]^3 - 1
    @test subs(fz, cz => coeffs_as_dense_poly(gz, z, 3)) == gz

    @var a
    @test_throws ArgumentError coefficients(a * x, [x])
    # A term the dense polynomial cannot carry is an error, not a silent drop.
    @test_throws ArgumentError coeffs_as_dense_poly(x^4, [x, y], 3)
end

@testset "exponents_coefficients round trip" begin
    @var x y
    f = x^2 + x * y - 1
    vars = variables(f)
    M, c = exponents_coefficients(f, vars)
    @test M == Int32[2 1 0; 0 1 0]
    g = poly_from_exponents_coefficients(M, c, vars)
    Mg, cg = exponents_coefficients(g, vars)
    @test M == Mg
    @test c == cg

    fc = x^2 + x * y - randn(MersenneTwister(3), ComplexF64)
    Mc, cc = exponents_coefficients(fc, variables(fc))
    @test poly_from_exponents_coefficients(Mc, cc, variables(fc)) isa Expression

    @test exponents_coefficients(x^3 - 2 * x, x) == (Int32[3 1;], ComplexF64[1, -2])
    @test coefficients(x^3 - 2 * x, x) == ComplexF64[1, -2]
    @test_throws ArgumentError poly_from_exponents_coefficients(M, c[1:2], vars)
    @test_throws ArgumentError poly_from_exponents_coefficients(M, c, vars[1:1])
end

@testset "horner" begin
    @var u v c[1:3]
    f = c[1] + c[2] * v + c[3] * u^2 * v^2 + c[3] * u^3 * v
    @test expand(horner(f)) == expand(f)

    # Every coefficient of a generic quintic in 4 variables restricted to a line.
    n = 4
    @var x[1:n]
    ν = monomials(x, 5; affine = false)
    @var q[1:(length(ν) - 1)]
    F = sum(q[i] * ν[i] for i in 1:(length(ν) - 1)) + 1
    @var a[1:(n - 1)] b[1:(n - 1)] t
    G = subs(F, x => [a; 1] .* t + [b; 0])
    FcapL = collect(values(to_dict(G, [t])))
    @test length(FcapL) == 6
    @test all(f -> expand(horner(f)) == expand(f), FcapL)
    @test all(f -> expand(horner(f, [a; b])) == expand(f), FcapL)

    @var y
    @test horner(x[1]^3 - 2 * x[1], x[1]) == (x[1]^2 - 2) * x[1]
    # Not polynomial in the given variables: returned unchanged.
    @test horner(x[1] / y + x[1]) == x[1] / y + x[1]
    @test horner(x[1] - x[1]) == Expression(0)
end

@testset "to_number and convert" begin
    @var x
    s = x + 1 - x
    @test to_number(s) == ComplexF64(1)
    @test convert(Int, s) == 1
    @test convert(Int32, s) == Int32(1)
    @test convert(Int64, s) == Int64(1)
    @test convert(BigInt, s) == BigInt(1)
    @test convert(BigFloat, s) == BigFloat(1)
    @test convert(Float64, s) == Float64(1)
    @test convert(ComplexF64, s) == ComplexF64(1)

    r = x + 1.0 - x
    @test convert(Float64, r) == Float64(1)
    @test convert(ComplexF64, r) == ComplexF64(1)
    @test convert(BigFloat, x + BigFloat(1.0) - x) == BigFloat(1.0)
    @test convert(ComplexF64, x + (1.0 + 0.0im) - x) == ComplexF64(1)

    @test_throws ArgumentError to_number(x)
    @test_throws InexactError convert(Int, x + 0.5 - x)
end

@testset "evaluate on expressions" begin
    @var x y
    @test evaluate([x^2, x * y], [x, y] => [2, 3]) == [4.0, 6.0]
    @test evaluate([x^2, x * y], [x, y] => [2, 3]) isa Vector{Float64}
    @test evaluate(x^2 + y, Dict(x => 2, y => 3)) == 7.0
    @test evaluate(x^2, x => 1 + 2im) isa ComplexF64
    @test (x^2)(x => 3) == 9.0
    @test evaluate([x y; x^2 y^2], [x, y] => [2, 3]) == [2.0 3.0; 4.0 9.0]
    @test_throws ArgumentError evaluate(x + y, x => 1)

    # A real-coefficient system evaluates to real numbers.
    constraints = [
        -0.2 * (-4.2467 * (0.1 + 1.0 * x^2) + 1.0 * y^2),
        -0.222 * (-4.49 * (0.5 + 1.0 * x^2) + 1.0 * y^2),
    ]
    @test evaluate(constraints, Dict(x => 0.25, y => 0.75)) isa Vector{Float64}
end

@testset "evaluate and jacobian on systems" begin
    @var x y a
    F = System([x^2 + y, x * y]; variables = [x, y])
    @test F([2, 3]) == [7.0, 6.0]
    @test F([2, 3]) isa Vector{Float64}
    @test jacobian(F, [2, 3]) == [4.0 1.0; 3.0 2.0]
    @test evaluate(F, [2, 3]) == F([2, 3])
    @test F([2im, 3]) isa Vector{ComplexF64}

    P = System([x^2 + a, x * y]; variables = [x, y], parameters = [a])
    @test P([2, 3], [5]) == [9.0, 6.0]
    @test size(jacobian(P, [2, 3], [5])) == (2, 2)
    @test_throws ArgumentError P([2, 3])
    @test_throws ArgumentError P([2, 3, 4], [5])

    @test_throws ArgumentError F([1, 2, 3])

    G = System([x + y]; variables = [x, y])
    C = compose(G, F)
    @test C([2, 3]) == [13.0]

    @testset "issue #511: a float system evaluates real" begin
        @var z[1:1]
        R = System(1.0 * z .^ 2; variables = z)
        @test R([2]) isa Vector{Float64}
        @test R([2]) == [4.0]
    end
end

@testset "is_real" begin
    @var x
    @test is_real(System([x - 1]))
    @test !is_real(System([x - 1 + 1.0e-16 * im]))
    @test is_real(System([x^2 - 2 * x + 1]))
    @test !is_real(System([im * x]))

    @var y a
    P = System([x^2 + a * y]; variables = [x, y], parameters = [a])
    @test is_real(P)
    @test !is_real(System([im * a * x]; variables = [x], parameters = [a]))
end
