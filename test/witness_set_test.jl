using Test, Random
using LinearAlgebra: norm
using HomotopyContinuation
using DynamicPolynomials: @polyvar
import MultivariatePolynomials as MP

function witness_rand_poly(T, vars, d; homogeneous = false)
    monos = homogeneous ? MP.monomials(vars, d) : MP.monomials(vars, 0:d)
    return sum(randn(T) * m for m in monos)
end
witness_rand_poly(vars, d; homogeneous = false) =
    witness_rand_poly(Float64, vars, d; homogeneous)

@testset "Witness Sets" begin
    @testset "affine witness sets and slice moves" begin
        @polyvar x y
        F = System([x^2 + y^2 - 5]; variables = [x, y])
        W = solve(F, Witness(; show_progress = false))

        @test dim(W) == 1
        @test codim(W) == 1
        @test degree(W) == 2
        @test solutions(W) isa Vector{Vector{ComplexF64}}
        @test trace_test(W) < 1.0e-8

        L = LinearSubspace([1 1], [-1])
        W_L = solve(W, L, Witness())
        @test degree(W_L) == 2
        @test sort(real.(solutions(W_L))) ≈ [[-2, 1], [1, -2]]
        @test linear_subspace(W_L) == convert(typeof(linear_subspace(W_L)), L)
        @test all(pt -> norm(L(pt)) < 1.0e-10, solutions(W_L))

        seeded1 = solve(F, Witness(; seed = UInt32(0x1234), show_progress = false), Serial())
        seeded2 = solve(F, Witness(; seed = UInt32(0x1234), show_progress = false), Serial())
        @test linear_subspace(seeded1) == linear_subspace(seeded2)
        @test points(seeded1) == points(seeded2)

        @test_throws MethodError trace_test(W; shwo_progress = false)
    end

    @testset "polynomial input forms" begin
        @polyvar x y
        f = x^2 + y^2 - 5
        @test degree(solve(f, Witness(; show_progress = false))) == 2
        @test degree(solve([f], Witness(; show_progress = false))) == 2

        L = LinearSubspace([1 1], [-1])
        for input in (f, [f])
            W = solve(input, L, Witness(; show_progress = false))
            @test sort(real.(solutions(W))) ≈ [[-2, 1], [1, -2]]
        end

        @polyvar p
        Fpar = System([x^2 + y^2 - p]; variables = [x, y], parameters = [p])
        @test_throws ArgumentError WitnessSet(Fpar, L, Vector{Vector{ComplexF64}}())
    end

    @testset "projective witness sets and homogeneous slices" begin
        @polyvar x y z
        F = System([x^2 + y^2 - 5z^2]; variables = [x, y, z])
        W = solve(F, Witness(; show_progress = false))

        @test dim(W) == 1
        @test codim(W) == 2
        @test degree(W) == 2
        @test trace_test(W) < 1.0e-8

        for L in (
                LinearSubspace([1 1 1]),
                rand_subspace(3; codim = 1, affine = false),
                rand_subspace([x, y, z]; codim = 1, affine = false),
            )
            moved = solve(W, L, Witness())
            @test degree(moved) == 2
            @test all(pt -> norm(L(pt)) < 1.0e-8 * max(1, norm(pt)), solutions(moved))
        end

        @test_throws ErrorException solve(
            W,
            rand_subspace(3; codim = 1, affine = true),
            Witness(),
        )
    end

    @testset "projective membership is scale invariant" begin
        @polyvar x y z
        W = solve(
            System([x^2 + y^2 - 5z^2]; variables = [x, y, z]),
            Witness(; show_progress = false),
        )

        pt = ComplexF64[3, 4, sqrt(5)]
        scaled = pt .* (2.0 + 1.0im)
        off = randn(ComplexF64, 3)
        witness = solutions(W)[1]
        alg = Membership(; show_progress = false)

        @test membership(pt, W, alg, Serial())
        @test membership(scaled, W, alg, Serial())
        @test !membership(off, W, alg, Serial())
        @test membership([scaled, off, witness], W, alg, Serial()) == [true, false, true]
        @test membership([scaled, off, witness], W, alg, Threaded()) == [true, false, true]
    end

    @testset "zero-dimensional witness sets" begin
        @polyvar s t
        projective = solve(
            System([s^2 - t^2]; variables = [s, t]),
            Witness(; show_progress = false),
        )
        @test degree(projective) == 2
        @test dim(projective) == 0
        @test trace_test(projective) < 1.0e-8

        @polyvar a b
        F = System([a^2 - 1, b^2 - 4]; variables = [a, b])
        affine = solve(F, Witness(; show_progress = false))
        @test degree(affine) == 4
        @test dim(affine) == 0
        @test membership(
            [ComplexF64[1, 2], ComplexF64[1, 3]],
            affine,
            Membership(; show_progress = false),
            Serial(),
        ) == [true, false]
        @test degree(solve(F, Witness(; dim = 0, show_progress = false))) == 4
    end

    @testset "fixed parameters" begin
        @polyvar u v p q
        F = System([u^2 + v^2 - p * q]; variables = [u, v], parameters = [p, q])
        fixed = fix_parameters(F, [2.0, 2.5])
        W = solve(fixed, Witness(; show_progress = false))

        @test degree(W) == 2
        @test dim(W) == 1
        @test nparameters(system(W)) == 0
        @test trace_test(W) < 1.0e-8
        @test degree(solve(W, LinearSubspace([1 1], [-1]), Witness())) == 2
        @test degree(
            solve(
                fixed,
                rand_subspace(2; codim = 1),
                Witness(; show_progress = false),
            ),
        ) == 2

        @test_throws ArgumentError solve(F, Witness())
        @test_throws ArgumentError solve(fix_parameters(F, [1.0]), Witness())
        G = System([u^2 + v^2 - 5]; variables = [u, v])
        @test_throws ArgumentError solve(fix_parameters(G, [1.0]), Witness())
        @test_throws ArgumentError solve(F, Regeneration())
    end

    @testset "dimension and codimension selection" begin
        Random.seed!(0x77)
        @polyvar x[1:6]

        homogeneous = System(
            [
                witness_rand_poly(x, 2; homogeneous = true),
                witness_rand_poly(x, 2; homogeneous = true),
                witness_rand_poly(x, 2; homogeneous = true),
                witness_rand_poly(x, 2; homogeneous = true),
            ]
        )
        @test degree(solve(homogeneous, Witness(; dim = 1, show_progress = false))) == 16
        @test degree(solve(homogeneous, Witness(; codim = 4, show_progress = false))) == 16
        @test degree(solve(homogeneous, Witness(; show_progress = false))) == 16

        affine = System(
            [
                witness_rand_poly(x, 2),
                witness_rand_poly(x, 2),
                witness_rand_poly(x, 2),
                witness_rand_poly(x, 2),
            ]
        )
        @test degree(solve(affine, Witness(; dim = 2, show_progress = false))) == 16
        @test degree(solve(affine, Witness(; codim = 4, show_progress = false))) == 16
        @test degree(solve(affine, Witness(; show_progress = false))) == 16
    end

    @polyvar x y z
    p = (x * y - x^2) + 1 - z
    q = x^4 + x^2 - y - 1
    reducible = [
        p * q * (x - 3) * (x - 5),
        p * q * (y - 3) * (y - 5),
        p * (z - 3) * (z - 5),
    ]

    @testset "membership" begin
        seed = UInt32(0x5eed)
        W = solve(
            System(reducible),
            Witness(; codim = 2, seed, show_progress = false),
        )
        off = randn(3)
        on = solutions(W)[1]
        alg = Membership(; seed, show_progress = false)

        @test !membership(off, W, alg)
        @test membership(on, W, alg)
        @test membership([off, on], W, alg) == [false, true]
        @test membership(
            [off, on],
            W,
            Membership(; seed, show_progress = true),
        ) == [false, true]
    end

    @testset "affine intersections" begin
        hypersurfaces = [
            solve(System([f]), Witness(; show_progress = false)) for f in reducible
        ]
        pairwise = intersect(hypersurfaces[1], hypersurfaces[2])
        components = vcat(
            [
                intersect(W, hypersurfaces[3], Intersection(; show_progress = false))
                    for W in pairwise
            ]...
        )
        @test sort(degree.(components)) == [2, 8, 8]

        threaded_pairwise = intersect(
            hypersurfaces[1],
            hypersurfaces[2],
            Intersection(),
            Threaded(),
        )
        threaded_components = vcat(
            [
                intersect(W, hypersurfaces[3], Intersection(; show_progress = false), Threaded())
                    for W in threaded_pairwise
            ]...
        )
        @test sort(degree.(threaded_components)) == [2, 8, 8]
    end

    @testset "projective intersections" begin
        @polyvar w[1:4]
        a = w[1]^2 + w[2]^2 + w[3]^2 + w[4]^2
        b = w[1]^3 + w[2]^3 + 2w[3]^3 + 3w[4]^3
        c = w[1]^4 + 2w[2]^4 + 4w[3]^4 - w[4]^4

        H = [
            solve(System([g]; variables = w), Witness(; show_progress = false))
                for g in (a * c, b * c)
        ]
        components = intersect(H[1], H[2], Intersection(; show_progress = false))
        @test sort(degree.(components)) == [4, 6]
        @test sort(dim.(components)) == [1, 2]
        @test all(W -> is_linear(linear_subspace(W)), components)

        direct = intersect(H[1], b * c, Intersection(; show_progress = false))
        @test sort(degree.(direct)) == [4, 6]

        affine = solve(
            System([a * c - 1]; variables = w),
            Witness(; show_progress = false),
        )
        @test_throws ArgumentError intersect(
            H[1],
            affine,
            Intersection(; show_progress = false),
        )
    end

    @testset "public sliced solves and witness-set moves" begin
        Random.seed!(0x5eed)
        @polyvar a1 a2
        circle = System([a1^2 + a2^2 - 5]; variables = [a1, a2])
        L1 = rand_subspace(2; codim = 1)
        L2 = rand_subspace(2; codim = 1)

        W1 = solve(circle, L1, Witness(; show_progress = false))
        @test degree(W1) == 2
        W2 = solve(W1, L2, Witness())
        @test degree(W2) == 2
        @test all(pt -> norm(L2(pt)) < 1.0e-10, solutions(W2))

        @polyvar a3
        sphere = a1^2 + a2^2 + a3^2 - 1
        parabola = a2 - a1^2
        L = rand_subspace(3; codim = 1)
        four_points = solve([sphere, parabola], L, Witness(; show_progress = false))
        @test degree(four_points) == 4
        for pt in solutions(four_points)
            @test abs(sphere([a1, a2, a3] => pt)) < 1.0e-8
            @test abs(parabola([a1, a2, a3] => pt)) < 1.0e-8
            @test norm(L(pt)) < 1.0e-8
        end
    end

    @testset "seeded routes are reproducible and do not advance ambient RNG" begin
        @polyvar x y z
        Faff = System([x^2 + y^2 - 5]; variables = [x, y])
        Fhom = System([x^2 + y^2 - z^2]; variables = [x, y, z])
        Laff = rand_subspace(2; codim = 1)
        Lhom = rand_subspace(3; codim = 1, affine = false)

        points_agree(a, b) =
            length(a) == length(b) && all(u ≈ v for (u, v) in zip(a, b))
        function from_two_states(f)
            Random.seed!(1)
            a = f()
            Random.seed!(2)
            b = f()
            return a, b
        end
        function advances_ambient(f)
            Random.seed!(7)
            expected = rand()
            Random.seed!(7)
            f()
            return expected != rand()
        end

        seed = UInt32(0xBEEF)
        routes = (
            () -> solve(Faff, Witness(; seed, show_progress = false)),
            () -> solve(Faff, Laff, Witness(; seed, show_progress = false)),
            () -> solve(Fhom, Witness(; seed, show_progress = false)),
            () -> solve(Fhom, Lhom, Witness(; seed, show_progress = false)),
        )
        for route in routes
            a, b = from_two_states(route)
            @test points_agree(solutions(a), solutions(b))
            @test !advances_ambient(route)
        end

        W = solve(Faff, Witness(; seed = UInt32(0x55), show_progress = false))
        L2 = rand_subspace(2; codim = 1)
        move = () -> solve(W, L2, Witness(; seed))
        a, b = from_two_states(move)
        @test points_agree(solutions(a), solutions(b))
        @test !advances_ambient(move)

        trace_route = () -> trace_test(W; seed)
        t1, t2 = from_two_states(trace_route)
        @test t1 ≈ t2
        @test !advances_ambient(trace_route)

        on = solutions(W)[1]
        membership_route = () -> membership(
            on,
            W,
            Membership(; seed, show_progress = false),
        )
        m1, m2 = from_two_states(membership_route)
        @test m1 == m2 == true
        @test !advances_ambient(membership_route)
    end
end

@testset "Witness-set ambient coordinates" begin
    @polyvar x y z
    F = System([x]; variables = [x, y])
    H = System([z]; variables = [x, z])
    R = System([x]; variables = [y, x])
    L = LinearSubspace(zeros(ComplexF64, 0, 2), ComplexF64[])
    empty_points = Vector{Vector{ComplexF64}}()

    W = WitnessSet(F, L, copy(empty_points))
    WH = WitnessSet(H, L, copy(empty_points))
    WR = WitnessSet(R, L, copy(empty_points))

    @test_throws ArgumentError intersect(
        W,
        WH,
        Intersection(; show_progress = false),
        Serial(),
    )
    @test_throws ArgumentError intersect(
        W,
        WR,
        Intersection(; show_progress = false),
        Serial(),
    )
end
