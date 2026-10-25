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
    @testset "Union of 1 2-dim, 2 1-dim and 8 points" begin
        @polyvar x y z
        p = (x * y - x^2) + 1 - z
        q = x^4 + x^2 - y - 1
        F = [
            p * q * (x - 3) * (x - 5),
            p * q * (y - 3) * (y - 5),
            p * (z - 3) * (z - 5),
        ]

        W = solve(F, Regeneration(; seed = 0x8a1f2c3d))
        @test sort(degree.(W); rev = true) == [8, 8, 2]

        dec = solve(W, Decomposition(; seed = 0x8a1f2c3d))
        @test all(W -> is_irreducible(W) == Irreducibility.IRREDUCIBLE, dec)

        # sorting
        W = solve(
            F,
            Regeneration(;
                sorted = EquationSorting.UNSORTED,
                seed = 0x8a1f2c3d,
                show_progress = false,
            ),
        )
        @test sort(degree.(W); rev = true) == [8, 8, 2]

        # limited codimension
        W = solve(
            F,
            Regeneration(; max_codim = 2, seed = 0x8a1f2c3d, show_progress = false),
        )
        @test sort(degree.(W); rev = true) == [8, 2]

        # no threading
        N = solve(F, Decomposition(; show_progress = false), Serial())
        @test isa(N, NumericalIrreducibleDecomposition)

        # seed
        s = 0x42c9d504
        N = solve(F, Decomposition(; seed = s, show_progress = false))
        @test seed(N) == s

        # Without an explicit seed a random one is drawn and recorded, so the
        # result always carries a seed that reproduces it.
        N = solve(F, Decomposition(; show_progress = false))
        @test seed(N) isa UInt32

        # seed roundtrip / stability
        N = solve(F, Decomposition(; seed = 0xc770fa47, show_progress = false))
        degs = degrees(N)
        @test degs[2] == [2]
        @test sort(degs[1]) == [4, 4]

        N = solve(F, Decomposition(; show_progress = false))
        @test isa(N, NumericalIrreducibleDecomposition)

        N = solve(F, Decomposition(; warning = false, show_progress = false))
        @test isa(N, NumericalIrreducibleDecomposition)

        # options
        N_fails = solve(
            F,
            Decomposition(;
                endgame_options = EndgameOptions(; max_endgame_steps = 0),
                show_progress = false,
            ),
        )
        @test isempty(witness_sets(N_fails))

        N2 = solve(
            F,
            Decomposition(;
                tracker_options = TrackerOptions(; extended_precision = false),
                show_progress = false,
            ),
        )
        @test isa(N2, NumericalIrreducibleDecomposition)

        # Seeded: unseeded, the monodromy decomposition occasionally finds only one
        # of the two 1-dimensional components counted below.
        N3 = solve(
            F,
            Decomposition(;
                monodromy = MonodromyOptions(; trace_test_tol = 1.0e-5),
                seed = UInt32(0x1234), show_progress = false,
            ),
        )
        @test isa(N3, NumericalIrreducibleDecomposition)

        # number of components
        @test ncomponents(N3) == 11
        @test ncomponents(N3; dims = [1, 2]) == 3
        @test ncomponents(N3, 1) == 2
        @test n_components(N3) == 11
        @test n_components(N3; dims = [1, 2]) == 3
        @test n_components(N3, 1) == 2

        # max_codim = 1
        N4 = solve(
            F, Decomposition(; max_codim = 1, show_progress = false),
        )
        @test isa(N4, NumericalIrreducibleDecomposition)
    end

    @testset "rational systems" begin
        @var x y z
        g = System([x^2 + y^2 - z, x / (y - 1) + y + z - 1]; variables = [x, y, z])
        N = solve(g, Decomposition(; show_progress = false))
        @test ncomponents(N) == 1
        W = first(witness_sets(N)[1])
        @test degree(W) == 4
        @test !membership(randn(3), W)
        @test membership(solutions(W)[1], W)
        U = intersect(W, x / (y - 1) + y + z - 1)
        @test degree(only(U)) == 4
        V = intersect(W, x * y^3 - z^4 + x^3 - 8)
        @test degree(only(V)) == 16

        # A numerator and a denominator sharing a structural factor: the zero of
        # the numerator at `r = 1` is a pole of the equation, not a point of its
        # variety, so it must not enter the witness superset.
        @var r s
        H = System([(r^2 - 1) / (r - 1) + s, r * s - 2]; variables = [r, s])
        R = solve(H, Regeneration(; show_progress = false))
        @test degree.(R) == [2]
        @test all(pt -> abs(pt[1] * pt[2] - 2) < 1.0e-8, solutions(first(R)))
        @test all(pt -> abs(pt[1] + 1 + pt[2]) < 1.0e-8, solutions(first(R)))

        # A zero sitting close to a pole is still a zero: `(r - ε)/r` vanishes at
        # `r = ε`, a full `ε` away from the denominator variety.
        for ε in (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8)
            Hε = System([(r - ε) / r]; variables = [r, s])
            @test degree.(solve(Hε, Regeneration(; show_progress = false))) == [1]
        end
        # A shared zero leaves nothing behind: only `r = -1` survives here.
        @test degree.(
            solve(
                System([(r^2 - 1) / (r - 1)]; variables = [r, s]),
                Regeneration(; show_progress = false),
            )
        ) == [1]

        # Which numerator zeros are poles is a property of the variety, so
        # rescaling an equation or its denominator may not change the answer.
        for k in (1.0e-4, 1.0e4, 1.0e8)
            Hk = System(
                [k * ((r^2 - 1) / (r - 1) + s), r * s - 2]; variables = [r, s]
            )
            @test degree.(solve(Hk, Regeneration(; show_progress = false))) == [2]
            Hd = System(
                [(r^2 - 1) / (k * (r - 1)) + s, r * s - 2]; variables = [r, s]
            )
            @test degree.(solve(Hd, Regeneration(; show_progress = false))) == [2]
        end

        # A hypersurface in variables the witness set does not have is rejected.
        @polyvar a b
        Wp = solve(System([a^2 + b^2 - 1]), Witness(; show_progress = false))
        @test_throws ArgumentError intersect(Wp, x + y)
    end

    @testset "equation scaling" begin
        # Multiplying an equation by a constant leaves its variety, and so the witness
        # superset, alone in either front-end.
        @polyvar u v
        @var p q
        for k in (1.0e-8, 1.0e-4, 1.0, 1.0e4, 1.0e12)
            @test degree.(
                solve(
                    System([k * (u + 1 + v), u * v - 2]),
                    Regeneration(; show_progress = false),
                )
            ) == [2]
            @test degree.(
                solve(
                    System([k * (p + 1 + q), p * q - 2]; variables = [p, q]),
                    Regeneration(; show_progress = false),
                )
            ) == [2]
        end
    end

    @testset "intersect with a hypersurface" begin
        # The hypersurface lives in the ambient space of the witness set and may leave
        # variables out. A homogeneous one is sliced affinely like any other.
        @polyvar x1 x2 x3 x4
        Wp = solve(System([x1^2 + x2^2 + x3^2 - 4]), Witness(; show_progress = false))
        @test degree(only(intersect(Wp, x1^2 - x2^2 - 1, Intersection(; show_progress = false)))) == 4
        @test degree(only(intersect(Wp, x1^2 - x2^2, Intersection(; show_progress = false)))) == 4
        @test_throws ArgumentError intersect(Wp, x1^2 - x4^2, Intersection(; show_progress = false))

        @var y1 y2 y3
        We = solve(
            System([y1^2 + y2^2 + y3^2 - 4]; variables = [y1, y2, y3]),
            Witness(; show_progress = false),
        )
        @test degree(only(intersect(We, y1^2 - y2^2 - 1, Intersection(; show_progress = false)))) == 4
        @test degree(only(intersect(We, y1^2 - y2^2, Intersection(; show_progress = false)))) == 4
        @test degree(
            only(
                intersect(
                    We, (y1^2 - y2^2) / (y1 - y2),
                    Intersection(; show_progress = false),
                ),
            ),
        ) == 2
    end

    @testset "Hypersurface of degree 5" begin
        @polyvar x[1:4]
        f = rand_poly(ComplexF64, x, 5)
        Hyp = System([f]; variables = x)

        N_Hyp = solve(Hyp, Decomposition(; show_progress = false))
        @test degrees(N_Hyp) == Dict(3 => [5])
        @test ncomponents(N_Hyp) == 1
    end

    @testset "Curve of degree 6" begin
        @polyvar x[1:3]
        f = rand_poly(ComplexF64, x, 2)
        g = rand_poly(ComplexF64, x, 3)
        Curve = System([f, g]; variables = x)

        N_Curve = solve(Curve, Decomposition(; show_progress = false))
        @test degrees(N_Curve) == Dict(1 => [6])
        @test ncomponents(N_Curve) == 1
    end

    @testset "Homogeneous systems" begin
        @polyvar x[1:4]
        a = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
        b = x[1]^3 + x[2]^3 + 2x[3]^3 + 3x[4]^3
        c = x[1]^4 + 2x[2]^4 + 4x[3]^4 - x[4]^4
        # V(a*c, b*c) = V(c) ∪ V(a, b): a quartic surface and a degree-6 curve
        # in P³. Homogeneous input is regenerated projectively, so the witness
        # sets are cut by linear (not affine) subspaces.
        G = System([a * c, b * c]; variables = x)
        N = solve(G, Decomposition(; seed = 0x00000078, show_progress = false))
        @test degrees(N) == Dict(2 => [4], 1 => [6])
        @test ncomponents(N) == 2

        Ws = witness_sets(N)
        W1, W2 = first(Ws[2]), first(Ws[1])
        @test all(W -> is_linear(linear_subspace(W)), [W1, W2])
        # Every rescaling of a point of a projective component is a member. The
        # slice through a rescaled witness point of the curve shares a plane
        # with the witness slice, so the membership homotopy follows a geodesic
        # with vanishing principal angles.
        scales = (2.0 + 1.0im, 1.0, 1.0e-8, 1.0e8, 1.0e-8im, -3.7e5 + 2.0e5im)
        for seed in (0x00000002, 0x00000005), W in (W1, W2), q in solutions(W)
            alg = Membership(; seed, show_progress = false)
            @test all(membership([s .* q for s in scales], W, alg))
            @test all(s -> membership(s .* q, W, alg), scales)
        end
        # A generic point of the quartic surface is not on the sextic curve,
        # at any scale.
        p = first(solutions(W1))
        alg = Membership(; seed = 0x00000002, show_progress = false)
        @test !any(membership([s .* p for s in scales], W2, alg))

        # `EquationSorting.RANDOMIZED` combines equations of different degrees,
        # which does not stay homogeneous.
        @test_throws ArgumentError solve(
            G,
            Regeneration(;
                sorted = EquationSorting.RANDOMIZED, show_progress = false,
            ),
        )
    end

    @testset "randomization" begin
        @polyvar x y z
        # A degree-2 and a degree-3 surface meeting in a degree-6 curve. The
        # random upper-triangular combination leaves V(f, g) unchanged.
        f = rand_poly(ComplexF64, [x, y, z], 2)
        g = rand_poly(ComplexF64, [x, y, z], 3)
        N = solve(
            System([f, g]; variables = [x, y, z]),
            Decomposition(;
                sorted = EquationSorting.RANDOMIZED, show_progress = false,
            ),
        )
        @test degrees(N) == Dict(1 => [6])
        @test ncomponents(N) == 1
    end

    @testset "Overdetermined Test" begin
        @polyvar x y z
        TwistedCubicSphere =
            [x * z - y^2, y - z^2, x - y * z, rand_poly(ComplexF64, [x, y, z], 1)]

        N_TwistedCubicSphere = solve(TwistedCubicSphere, Decomposition())
        @test degrees(N_TwistedCubicSphere) == Dict(0 => [1, 1, 1])
        @test ncomponents(N_TwistedCubicSphere) == 3
    end

    @testset "Three Lines" begin
        @polyvar x y z
        f = x * z + y
        g = y * z + x
        ThreeLines = System([f, g]; variables = [x, y, z])

        N_ThreeLines = solve(ThreeLines, Decomposition(; show_progress = false))
        @test degrees(N_ThreeLines) == Dict(1 => [1, 1, 1])
        @test ncomponents(N_ThreeLines) == 3
    end
end
