using Test, Random
using HomotopyContinuation
using HomotopyContinuation: TrackerOptions, MonodromyOptions, linear_subspace,
    is_linear, dim, EquationSorting
using DynamicPolynomials: @polyvar
import MultivariatePolynomials as MP

function rand_poly(T, vars, d; homogeneous = false)
    monos = homogeneous ? MP.monomials(vars, d) : MP.monomials(vars, 0:d)
    return sum(randn(T) * m for m in monos)
end
rand_poly(vars, d; homogeneous = false) = rand_poly(Float64, vars, d; homogeneous = homogeneous)


@testset "Numerical Irreducible Decomposition" begin
    @testset "Bricard6R" begin
        z1x = 1
        z1y = 0
        z1z = 0
        z6x = 0
        z6y = 0
        z6z = 1
        a1 = 1
        a2 = 1
        a3 = 1
        a4 = 1
        a5 = 1
        d2 = 0
        d3 = 0
        d4 = 0
        d5 = 0
        c1 = 0
        c2 = 0
        c3 = 0
        c4 = 0
        c5 = 0
        px = 0
        py = 1
        pz = 0

        @polyvar z2x z2y z2z z3x z3y z3z z4x z4y z4z z5x z5y z5z

        unit2 = z2x^2 + z2y^2 + z2z^2 - 1
        unit3 = z3x^2 + z3y^2 + z3z^2 - 1
        unit4 = z4x^2 + z4y^2 + z4z^2 - 1
        unit5 = z5x^2 + z5y^2 + z5z^2 - 1
        twist1 = z1x * z2x + z1y * z2y + z1z * z2z - c1
        twist2 = z2x * z3x + z2y * z3y + z2z * z3z - c2
        twist3 = z3x * z4x + z3y * z4y + z3z * z4z - c3
        twist4 = z4x * z5x + z4y * z5y + z4z * z5z - c4
        twist5 = z5x * z6x + z5y * z6y + z5z * z6z - c5
        x1x = z1y * z2z - z1z * z2y
        x2x = z2y * z3z - z2z * z3y
        x3x = z3y * z4z - z3z * z4y
        x4x = z4y * z5z - z4z * z5y
        x5x = z5y * z6z - z5z * z6y
        x1y = z1z * z2x - z1x * z2z
        x2y = z2z * z3x - z2x * z3z
        x3y = z3z * z4x - z3x * z4z
        x4y = z4z * z5x - z4x * z5z
        x5y = z5z * z6x - z5x * z6z
        x1z = z1x * z2y - z1y * z2x
        x2z = z2x * z3y - z2y * z3x
        x3z = z3x * z4y - z3y * z4x
        x4z = z4x * z5y - z4y * z5x
        x5z = z5x * z6y - z5y * z6x
        X =
            a1 * x1x + d2 * z2x + a2 * x2x + d3 * z3x + a3 * x3x +
            d4 * z4x + a4 * x4x + d5 * z5x + a5 * x5x - px
        Y =
            a1 * x1y + d2 * z2y + a2 * x2y + d3 * z3y + a3 * x3y +
            d4 * z4y + a4 * x4y + d5 * z5y + a5 * x5y - py
        Z =
            a1 * x1z + d2 * z2z + a2 * x2z + d3 * z3z + a3 * x3z +
            d4 * z4z + a4 * x4z + d5 * z5z + a5 * x5z - pz

        Bricard6R = System(
            [unit2, unit3, unit4, unit5, twist1, twist2, twist3, twist4, twist5, X, Y, Z];
            variables = [z2x, z2y, z2z, z3x, z3y, z3z, z4x, z4y, z4z, z5x, z5y, z5z],
        )

        N_Bricard6R = solve(Bricard6R, Decomposition(; show_progress = false))
        @test degrees(N_Bricard6R) == Dict(1 => [8])
        @test ncomponents(N_Bricard6R) == 1

        N_Bricard6R_c4 = solve(
            Bricard6R, Decomposition(; max_codim = 4, show_progress = false),
        )
        @test ncomponents(N_Bricard6R_c4) == 0
    end

    @testset "ACR" begin
        @polyvar xx_Di xx_Da xx_Ya xx_Yi
        @polyvar xx_CXY xx_CXYp xx_G xx_CNA xx_X
        @polyvar xx_A xx_N xx_CXA xx_T xx_CXT xx_Xp

        F_ACR = [
            -(5 / 3) * xx_Di + (2 / 3) * xx_Da,
            (5 / 3) * xx_Di - (2 / 3) * xx_Da,
            -(1 / 3) * xx_Ya * xx_X - (2 / 3) * xx_Ya +
                (5 / 6) * xx_Yi + (6 / 5) * xx_CXY + 4 * xx_CXYp,
            -8 * xx_Da * xx_Yi + (5 / 7) * xx_G * xx_CNA + (2 / 3) * xx_Ya -
                (49 / 30) * xx_Yi,
            8 * xx_Da * xx_Yi - (5 / 7) * xx_G * xx_CNA + (4 / 5) * xx_Yi,
            8 * xx_Da * xx_Yi - (5 / 7) * xx_G * xx_CNA +
                (2 / 3) * xx_A * xx_N + (4 / 5) * xx_Yi - (8 / 3) * xx_CNA,
            -(2 / 3) * xx_A * xx_N - (7 / 8) * xx_A * xx_X +
                (8 / 3) * xx_CNA + (5 / 7) * xx_CXA,
            -(2 / 3) * xx_A * xx_N + (8 / 3) * xx_CNA - (5 / 2) * xx_N + 1,
            -(1 / 3) * xx_Ya * xx_X - (7 / 8) * xx_A * xx_X - xx_X * xx_T - (4 / 3) * xx_X +
                (6 / 5) * xx_CXY + 3 * xx_CXT + (5 / 7) * xx_CXA + 1,
            (1 / 3) * xx_Ya * xx_X - (26 / 5) * xx_CXY,
            4 * xx_CXY - 4 * xx_CXYp,
            4 * xx_CXYp - 5 * xx_Xp,
            -xx_X * xx_T + 3 * xx_CXT,
            xx_X * xx_T - 3 * xx_CXT,
            (7 / 8) * xx_A * xx_X - (5 / 7) * xx_CXA,
        ]

        # Monodromy-based decomposition is probabilistic: unseeded, ACR gives
        # the correct Dict(4 => [7]) only ~90% of the time, so a bare `==`
        # assertion on an unseeded run is flaky. Seed for a deterministic
        # check; reproducibility relies on regeneration/decompose drawing
        # their randomness from the (seeded) global RNG.
        N_ACR = solve(F_ACR, Decomposition(; seed = UInt32(0x1234), show_progress = false))
        @test degrees(N_ACR) == Dict(4 => [7])
    end

    @testset "Union of a sphere, a line, and a point" begin
        @polyvar x y z

        S = [x^2 + y^2 + z^2 - 1]
        L = [2 * x - z, 2 * y - z]
        P = [x + y + 2 * z - 4, y - z, x - z]

        F = System([s * l * p for s in S for l in L for p in P])

        NID = solve(F, Decomposition(; seed = 0x7a4845b9, show_progress = false))

        @test ncomponents(NID, 0) == 1
        @test degrees(NID)[0] == [1]
    end

    # The seed alone determines the result: the same seed from two different
    # ambient RNG states must agree, and a route must not advance the ambient
    # stream on the way.
    @testset "seeds determine the result" begin
        @polyvar x y z
        F = [x * y, x * z]
        s = UInt32(0xBEEF)

        function from_two_states(f)
            Random.seed!(1)
            a = f()
            Random.seed!(2)
            b = f()
            return a, b
        end

        function advances_ambient(f)
            Random.seed!(7)
            a = rand()
            Random.seed!(7)
            f()
            return a != rand()
        end

        reg = () -> solve(F, Regeneration(; seed = s, show_progress = false), Serial())
        R1, R2 = from_two_states(reg)
        @test degree.(R1) == degree.(R2)
        @test !advances_ambient(reg)

        dec = () -> solve(F, Decomposition(; seed = s, show_progress = false), Serial())
        N1, N2 = from_two_states(dec)
        @test degrees(N1) == degrees(N2)
        @test !advances_ambient(dec)

        W = solve(System([x^2 + y^2 - 5]), Witness(; seed = s, show_progress = false))
        cut = () -> only(intersect(W, x - y, Intersection(; seed = s)))
        I1, I2 = from_two_states(cut)
        @test degree(I1) == degree(I2)
        @test !advances_ambient(cut)

        @polyvar a b c
        Fp = System(
            [x^2 + y^2 - 1, a * x + b * y + c];
            variables = [x, y], parameters = [a, b, c],
        )
        mono = () -> solve(Fp, Monodromy(; seed = s, show_progress = false), Serial())
        M1, M2 = from_two_states(mono)
        by = t -> (round(real(t[1]); digits = 8), round(imag(t[1]); digits = 8))
        @test sort(solutions(M1); by = by) ≈ sort(solutions(M2); by = by)
        @test !advances_ambient(mono)
    end
end
