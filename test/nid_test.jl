using Test, Random
using HomotopyContinuationNext
using HomotopyContinuationNext: TrackerOptions
using DynamicPolynomials: @polyvar
import MultivariatePolynomials as MP

function rand_poly(T, vars, d; homogeneous = false)
    monos = homogeneous ? MP.monomials(vars, d) : MP.monomials(vars, 0:d)
    return sum(randn(T) * m for m in monos)
end
rand_poly(vars, d; homogeneous = false) = rand_poly(Float64, vars, d; homogeneous = homogeneous)

@testset "Numerical Irreducible Decomposition" begin
    @testset "Union of 1 2-dim, 2 1-dim and 8 points" begin
        @polyvar x y z
        p = (x * y - x^2) + 1 - z
        q = x^4 + x^2 - y - 1
        F = [
            p * q * (x - 3) * (x - 5),
            p * q * (y - 3) * (y - 5),
            p * (z - 3) * (z - 5),
        ]

        W = regeneration(F)
        @test sort(degree.(W); rev = true) == [8, 8, 2]
        @test isconcretetype(eltype(W))

        dec = decompose(W)
        @test all(W -> is_irreducible(W) == Irreducibility.IRREDUCIBLE, dec)
        @test eltype(dec) === eltype(W)

        # sorting
        W = regeneration(F; sorted = false, show_progress = false)
        @test sort(degree.(W); rev = true) == [8, 8, 2]

        # limited codimension
        W = regeneration(F; max_codim = 2, show_progress = false)
        @test sort(degree.(W); rev = true) == [8, 2]

        # no threading
        N = nid(F; threading = false, show_progress = false)
        @test isa(N, NumericalIrreducibleDecomposition)

        # seed
        s = 0x42c9d504
        N = nid(F; seed = s, show_progress = false)
        @test seed(N) == s
        @test isconcretetype(typeof(N))

        # Without an explicit seed a random one is drawn and recorded, so the
        # result always carries a seed that reproduces it.
        N = nid(F; show_progress = false)
        @test seed(N) isa UInt32

        # seed roundtrip / stability
        N = nid(F; seed = 0xc770fa47, show_progress = false)
        degs = degrees(N)
        @test degs[2] == [2]
        @test sort(degs[1]) == [4, 4]

        N = nid(F; show_monodromy_progress = true, show_progress = false)
        @test isa(N, NumericalIrreducibleDecomposition)

        N = nid(F; warning = false, show_progress = false)
        @test isa(N, NumericalIrreducibleDecomposition)

        # options
        N_fails = nid(
            F;
            endgame_options = EndgameOptions(; max_endgame_steps = 0),
            show_progress = false,
        )
        @test isempty(witness_sets(N_fails))

        N2 = nid(
            F;
            tracker_options = TrackerOptions(; extended_precision = false),
            show_progress = false,
        )
        @test isa(N2, NumericalIrreducibleDecomposition)

        N3 = nid(
            F;
            monodromy_options = MonodromyOptions(; trace_test_tol = 1.0e-5),
            show_progress = false,
        )
        @test isa(N3, NumericalIrreducibleDecomposition)

        metric = (x, y) -> sum(abs2, x .- y)
        identity_options = MonodromyOptions(;
            distance = metric, triangle_inequality = false,
            unique_points_atol = 2.0e-12, unique_points_rtol = 3.0e-9,
        )
        copied_options = HomotopyContinuationNext._decompose_monodromy_options(
            identity_options,
        )
        @test copied_options.distance === metric
        @test copied_options.triangle_inequality === false
        @test copied_options.unique_points_atol == 2.0e-12
        @test copied_options.unique_points_rtol == 3.0e-9

        zero_metric = (x, y) -> 0.0
        identity_points = UniquePoints(
            2; distance = zero_metric, triangle_inequality = false,
        )
        identity = HomotopyContinuationNext.DecompositionPointIdentity(
            identity_points, Vector{Vector{ComplexF64}}(), Int[], BitVector(),
        )
        @test HomotopyContinuationNext._point_identity!(
            identity, ComplexF64[0, 0], 1.0e-14, 1.0e-8,
        ) == 1
        @test HomotopyContinuationNext._point_identity!(
            identity, ComplexF64[10, 10], 1.0e-14, 1.0e-8,
        ) == 1
        @test length(identity.master) == 1

        # number of components
        @test ncomponents(N3) == 11
        @test ncomponents(N3; dims = [1, 2]) == 3
        @test ncomponents(N3, 1) == 2
        @test n_components(N3) == 11
        @test n_components(N3; dims = [1, 2]) == 3
        @test n_components(N3, 1) == 2

        # max_codim = 1
        N4 = nid(F; max_codim = 1, show_progress = false)
        @test isa(N4, NumericalIrreducibleDecomposition)
    end

    @testset "rational systems" begin
        @var x y z
        g = System([x^2 + y^2 - z, x / (y - 1) + y + z - 1]; variables = [x, y, z])
        N = nid(g; show_progress = false)
        @test ncomponents(N) == 1
        W = first(witness_sets(N)[1])
        @test degree(W) == 4
        @test !membership(randn(3), W)
        @test membership(solutions(W)[1], W)
        U = intersect(W, x / (y - 1) + y + z - 1)
        @test degree(U) == 4
        V = intersect(W, x * y^3 - z^4 + x^3 - 8)
        @test degree(V) == 16

        # A numerator and a denominator sharing a structural factor: the zero of
        # the numerator at `r = 1` is a pole of the equation, not a point of its
        # variety, so it must not enter the witness superset.
        @var r s
        H = System([(r^2 - 1) / (r - 1) + s, r * s - 2]; variables = [r, s])
        R = regeneration(H; show_progress = false)
        @test degree.(R) == [2]
        @test all(pt -> abs(pt[1] * pt[2] - 2) < 1.0e-8, solutions(first(R)))
        @test all(pt -> abs(pt[1] + 1 + pt[2]) < 1.0e-8, solutions(first(R)))

        # A zero sitting close to a pole is still a zero: `(r - ε)/r` vanishes at
        # `r = ε`, a full `ε` away from the denominator variety.
        for ε in (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
            Hε = System([(r - ε) / r]; variables = [r, s])
            @test degree.(regeneration(Hε; show_progress = false)) == [1]
        end
        # A shared zero leaves nothing behind: only `r = -1` survives here.
        @test degree.(
            regeneration(
                System([(r^2 - 1) / (r - 1)]; variables = [r, s]);
                show_progress = false
            )
        ) == [1]

        # Which numerator zeros are poles is a property of the variety, so
        # rescaling an equation or its denominator may not change the answer.
        for k in (1.0e-4, 1.0e4, 1.0e8)
            Hk = System(
                [k * ((r^2 - 1) / (r - 1) + s), r * s - 2]; variables = [r, s]
            )
            @test degree.(regeneration(Hk; show_progress = false)) == [2]
            Hd = System(
                [(r^2 - 1) / (k * (r - 1)) + s, r * s - 2]; variables = [r, s]
            )
            @test degree.(regeneration(Hd; show_progress = false)) == [2]
        end

        # Mixing the two input front-ends in one `intersect` is rejected.
        @polyvar a b
        Wp = witness_set(System([a^2 + b^2 - 1]); show_progress = false)
        @test_throws ArgumentError intersect(Wp, x + y)
    end

    @testset "equation scaling" begin
        # Multiplying an equation by a constant leaves its variety, and so the witness
        # superset, alone in either front-end.
        @polyvar u v
        @var p q
        for k in (1.0e-8, 1.0e-4, 1.0, 1.0e4, 1.0e12)
            @test degree.(
                regeneration(
                    System([k * (u + 1 + v), u * v - 2]); show_progress = false,
                )
            ) == [2]
            @test degree.(
                regeneration(
                    System([k * (p + 1 + q), p * q - 2]; variables = [p, q]);
                    show_progress = false,
                )
            ) == [2]
        end
    end

    @testset "intersect with a hypersurface" begin
        # The hypersurface lives in the ambient space of the witness set and may leave
        # variables out. A homogeneous one is sliced affinely like any other.
        @polyvar x1 x2 x3 x4
        Wp = witness_set(System([x1^2 + x2^2 + x3^2 - 4]); show_progress = false)
        @test degree(intersect(Wp, x1^2 - x2^2 - 1; show_progress = false)) == 4
        @test degree(intersect(Wp, x1^2 - x2^2; show_progress = false)) == 4
        @test_throws ArgumentError intersect(Wp, x1^2 - x4^2; show_progress = false)

        @var y1 y2 y3
        We = witness_set(
            System([y1^2 + y2^2 + y3^2 - 4]; variables = [y1, y2, y3]);
            show_progress = false,
        )
        @test degree(intersect(We, y1^2 - y2^2 - 1; show_progress = false)) == 4
        @test degree(intersect(We, y1^2 - y2^2; show_progress = false)) == 4
        @test degree(
            intersect(We, (y1^2 - y2^2) / (y1 - y2); show_progress = false)
        ) == 2
    end

    @testset "Hypersurface of degree 5" begin
        @polyvar x[1:4]
        f = rand_poly(ComplexF64, x, 5)
        Hyp = System([f]; variables = x)

        N_Hyp = numerical_irreducible_decomposition(Hyp; show_progress = false)
        @test degrees(N_Hyp) == Dict(3 => [5])
        @test ncomponents(N_Hyp) == 1
    end

    @testset "Curve of degree 6" begin
        @polyvar x[1:3]
        f = rand_poly(ComplexF64, x, 2)
        g = rand_poly(ComplexF64, x, 3)
        Curve = System([f, g]; variables = x)

        N_Curve = nid(Curve; show_progress = false)
        @test degrees(N_Curve) == Dict(1 => [6])
        @test ncomponents(N_Curve) == 1
    end

    @testset "Overdetermined Test" begin
        @polyvar x y z
        TwistedCubicSphere =
            [x * z - y^2, y - z^2, x - y * z, rand_poly(ComplexF64, [x, y, z], 1)]

        N_TwistedCubicSphere = nid(TwistedCubicSphere)
        @test degrees(N_TwistedCubicSphere) == Dict(0 => [1, 1, 1])
        @test ncomponents(N_TwistedCubicSphere) == 3
    end

    @testset "Three Lines" begin
        @polyvar x y z
        f = x * z + y
        g = y * z + x
        ThreeLines = System([f, g]; variables = [x, y, z])

        N_ThreeLines = nid(ThreeLines; show_progress = false)
        @test degrees(N_ThreeLines) == Dict(1 => [1, 1, 1])
        @test ncomponents(N_ThreeLines) == 3
    end

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

        N_Bricard6R = nid(Bricard6R; show_progress = false)
        @test degrees(N_Bricard6R) == Dict(1 => [8])
        @test ncomponents(N_Bricard6R) == 1

        N_Bricard6R_c4 = nid(Bricard6R; max_codim = 4, show_progress = false)
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
        N_ACR = nid(F_ACR; seed = UInt32(0x1234), show_progress = false)
        @test degrees(N_ACR) == Dict(4 => [7])
    end

    @testset "Union of a sphere, a line, and a point" begin
        @polyvar x y z

        S = [x^2 + y^2 + z^2 - 1]
        L = [2 * x - z, 2 * y - z]
        P = [x + y + 2 * z - 4, y - z, x - z]

        F = System([s * l * p for s in S for l in L for p in P])

        NID = numerical_irreducible_decomposition(F; seed = 0x7a4845b9, show_progress = false)

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

        reg = () -> regeneration(
            F; seed = s, show_progress = false, threading = false,
        )
        R1, R2 = from_two_states(reg)
        @test degree.(R1) == degree.(R2)
        @test !advances_ambient(reg)

        dec = () -> nid(F; seed = s, show_progress = false, threading = false)
        N1, N2 = from_two_states(dec)
        @test degrees(N1) == degrees(N2)
        @test !advances_ambient(dec)

        W = witness_set(System([x^2 + y^2 - 5]); seed = s, show_progress = false)
        cut = () -> intersect(W, x - y; seed = s)
        I1, I2 = from_two_states(cut)
        @test degree(I1) == degree(I2)
        @test !advances_ambient(cut)

        @polyvar a b c
        Fp = System(
            [x^2 + y^2 - 1, a * x + b * y + c];
            variables = [x, y], parameters = [a, b, c],
        )
        mono = () -> monodromy_solve(
            Fp; seed = s, show_progress = false, threading = false,
        )
        M1, M2 = from_two_states(mono)
        by = t -> (round(real(t[1]); digits = 8), round(imag(t[1]); digits = 8))
        @test sort(solutions(M1); by = by) ≈ sort(solutions(M2); by = by)
        @test !advances_ambient(mono)
    end
end
