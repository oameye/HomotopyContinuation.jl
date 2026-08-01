using Test, Random
using LinearAlgebra
using HomotopyContinuationNext
using HomotopyContinuationNext: corank, extrinsic, TrackerOptions,
    IntrinsicSubspaceHomotopy, ExtrinsicSubspaceHomotopy,
    EndgameTracker, Tracker, HomotopyEvaluator, PathResult,
    intrinsic_coordinates!, ambient_coordinates!, track!, is_success, FSVec,
    fix_parameters, is_linear, linear_subspace, dim
using DynamicPolynomials: @polyvar
import MultivariatePolynomials as MP

# Test helper: dense random polynomial in `vars` of degree `d`.
# `homogeneous` restricts to monomials of degree exactly d.
function rand_poly(T, vars, d; homogeneous = false)
    monos = homogeneous ? MP.monomials(vars, d) : MP.monomials(vars, 0:d)
    return sum(randn(T) * m for m in monos)
end
rand_poly(vars, d; homogeneous = false) = rand_poly(Float64, vars, d; homogeneous = homogeneous)

@testset "Witness Sets" begin

    @testset "affine" begin
        @polyvar x y

        F = System([x^2 + y^2 - 5]; variables = [x, y])

        W = solve(F, Witness(; show_progress = false))

        @test dim(W) == 1
        @test codim(W) == 1
        @test degree(W) == 2
        @test solutions(W) isa Vector{Vector{ComplexF64}}

        L = LinearSubspace([1 1], [-1])

        W_L = solve(W, L, Witness())
        @test degree(W_L) == 2
        @test sort(real.(solutions(W_L))) ≈ [[-2, 1], [1, -2]]
        @test linear_subspace(W_L) == convert(typeof(linear_subspace(W_L)), L)

        @test trace_test(W) < 1.0e-8
        @test_throws MethodError trace_test(W; shwo_progress = false)

        W_seed₁ = solve(F, Witness(; seed = UInt32(0x1234), show_progress = false), Serial())
        W_seed₂ = solve(F, Witness(; seed = UInt32(0x1234), show_progress = false), Serial())
        @test linear_subspace(W_seed₁) == linear_subspace(W_seed₂)
        @test points(W_seed₁) == points(W_seed₂)
    end

    @testset "polynomial input forms" begin
        @polyvar x y

        # single polynomial and vector of polynomials
        @test degree(solve(x^2 + y^2 - 5, Witness(; show_progress = false))) == 2
        @test degree(solve([x^2 + y^2 - 5], Witness(; show_progress = false))) == 2

        L = LinearSubspace([1 1], [-1])
        W_L = solve(x^2 + y^2 - 5, L, Witness(; show_progress = false))
        @test sort(real.(solutions(W_L))) ≈ [[-2, 1], [1, -2]]
        W_L2 = solve([x^2 + y^2 - 5], L, Witness(; show_progress = false))
        @test sort(real.(solutions(W_L2))) ≈ [[-2, 1], [1, -2]]

        # a WitnessSet requires a parameter-free system
        @polyvar p
        Fpar = System([x^2 + y^2 - p]; variables = [x, y], parameters = [p])
        @test_throws ArgumentError WitnessSet(Fpar, L, Vector{Vector{ComplexF64}}())
    end

    @testset "projective" begin
        @polyvar x y z

        F = System([x^2 + y^2 - 5z^2]; variables = [x, y, z])

        W = solve(F, Witness(; show_progress = false))

        @test dim(W) == 1
        @test codim(W) == 2
        @test degree(W) == 2
        @test solutions(W) isa Vector{Vector{ComplexF64}}
        @test trace_test(W) < 1.0e-8

        L = LinearSubspace([1 1 1])
        W_L = solve(W, L, Witness())
        @test degree(W_L) == 2

        L = rand_subspace(3; codim = 1, affine = false)
        W_L = solve(W, L, Witness())
        @test degree(W_L) == 2

        L = rand_subspace([x, y, z]; codim = 1, affine = false)
        W_L = solve(W, L, Witness())
        @test degree(W_L) == 2

        L = rand_subspace(3; codim = 1, affine = true)
        @test_throws ErrorException solve(W, L, Witness())
    end

    @testset "projective membership" begin
        @polyvar x y z

        F = System([x^2 + y^2 - 5z^2]; variables = [x, y, z])
        W = solve(F, Witness(; show_progress = false))
        @test W.projective

        # any projective representative works (arbitrary complex scaling)
        pt_on = ComplexF64[3, 4, sqrt(5)] .* (2.0 + 1.0im)
        pt_off = randn(ComplexF64, 3)
        q0 = solutions(W)[1]

        @test membership(pt_on, W, Membership(; show_progress = false), Serial())
        @test !membership(pt_off, W, Membership(; show_progress = false), Serial())
        @test membership([pt_on, pt_off, q0], W, Membership(; show_progress = false), Serial()) ==
            [true, false, true]
        # threaded driver (with one thread this still exercises the task path)
        @test membership([pt_on, pt_off, q0], W, Membership(; show_progress = false), Threaded()) ==
            [true, false, true]
    end

    @testset "zero-dimensional" begin
        # projective: two points in P^1
        @polyvar s t
        W0p = solve(System([s^2 - t^2]; variables = [s, t]), Witness(; show_progress = false))
        @test degree(W0p) == 2
        @test dim(W0p) == 0
        @test trace_test(W0p) < 1.0e-8

        # affine: four isolated points
        @polyvar a b
        F0 = System([a^2 - 1, b^2 - 4]; variables = [a, b])
        W0a = solve(F0, Witness(; show_progress = false))
        @test degree(W0a) == 4
        @test dim(W0a) == 0
        @test membership(
            [ComplexF64[1, 2], ComplexF64[1, 3]],
            W0a,
            Membership(; show_progress = false),
            Serial(),
        ) ==
            [true, false]

        # explicit dim = 0 kwarg
        @test degree(solve(F0, Witness(; dim = 0, show_progress = false))) == 4
    end

    @testset "parametric (fix_parameters)" begin
        @polyvar u v p q
        F = System([u^2 + v^2 - p * q]; variables = [u, v], parameters = [p, q])

        W = solve(fix_parameters(F, [2.0, 2.5]), Witness(; show_progress = false))
        @test degree(W) == 2
        @test dim(W) == 1
        # the stored system is parameter-free (parameters substituted)
        @test HomotopyContinuationNext.nparameters(system(W)) == 0
        @test trace_test(W) < 1.0e-8

        # the fixed witness set moves like any other
        W_L = solve(W, LinearSubspace([1 1], [-1]), Witness())
        @test degree(W_L) == 2

        # explicit subspace entry point takes a fixed-parameter system too
        L = rand_subspace(2; codim = 1)
        W2 = solve(fix_parameters(F, [2.0, 2.5]), L, Witness(; show_progress = false))
        @test degree(W2) == 2

        # errors: missing, spurious, and wrong-length parameter values
        @test_throws ArgumentError solve(F, Witness())
        @test_throws ArgumentError solve(fix_parameters(F, [1.0]), Witness())
        G = System([u^2 + v^2 - 5]; variables = [u, v])
        @test_throws ArgumentError solve(fix_parameters(G, [1.0]), Witness())
        # regeneration rejects parametric systems loudly
        @test_throws ArgumentError solve(F, Regeneration())
    end

    @testset "dim / codim" begin
        @polyvar x[1:6]
        homogeneous = true

        f = System(
            [
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
            ]
        )

        @test degree(solve(f, Witness(; dim = 1, show_progress = false))) == 16
        @test degree(solve(f, Witness(; codim = 4, show_progress = false))) == 16
        @test degree(solve(f, Witness(; show_progress = false))) == 16

        homogeneous = false
        f = System(
            [
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
            ]
        )
        @test degree(solve(f, Witness(; dim = 2, show_progress = false))) == 16
        @test degree(solve(f, Witness(; codim = 4, show_progress = false))) == 16
        @test degree(solve(f, Witness(; show_progress = false))) == 16
    end

    @polyvar x y z
    p = (x * y - x^2) + 1 - z
    q = x^4 + x^2 - y - 1
    F = [
        p * q * (x - 3) * (x - 5),
        p * q * (y - 3) * (y - 5),
        p * (z - 3) * (z - 5),
    ]

    @testset "membership" begin
        W = solve(System(F), Witness(; codim = 2, show_progress = false))

        pt = randn(3)
        q0 = solutions(W)[1]

        @test !membership(pt, W, Membership(; show_progress = false))
        @test membership(q0, W, Membership(; show_progress = false))
        a = membership([pt, q0], W, Membership(; show_progress = false))
        @test a == [false, true]
        @test membership([pt, q0], W, Membership(; show_progress = true)) == [false, true]
    end

    @testset "intersect" begin
        H = [solve(System([f]), Witness(; show_progress = false)) for f in F]
        B = intersect(H[1], H[2])
        C = vcat([intersect(Hi, H[3], Intersection(; show_progress = false)) for Hi in B]...)
        @test sort(degree.(C)) == [2, 8, 8]
    end

    @testset "intersect projective" begin
        @polyvar w[1:4]
        a = w[1]^2 + w[2]^2 + w[3]^2 + w[4]^2
        b = w[1]^3 + w[2]^3 + 2w[3]^3 + 3w[4]^3
        c = w[1]^4 + 2w[2]^4 + 4w[3]^4 - w[4]^4
        # Two homogeneous hypersurfaces in P³ sharing the quartic V(c): the
        # intersection is that surface plus the degree-6 curve V(a, b).
        Hp = [
            solve(System([g]; variables = w), Witness(; show_progress = false))
                for g in [a * c, b * c]
        ]
        @test all(W -> W.projective, Hp)
        B = intersect(Hp[1], Hp[2], Intersection(; show_progress = false))
        @test sort(degree.(B)) == [4, 6]
        @test sort(dim.(B)) == [1, 2]
        @test all(W -> W.projective && is_linear(linear_subspace(W)), B)

        # A projective witness set and a homogeneous hypersurface given directly.
        Bf = intersect(Hp[1], b * c, Intersection(; show_progress = false))
        @test sort(degree.(Bf)) == [4, 6]

        # Mixing a projective and an affine witness set slices in the wrong
        # ambient space, so it is rejected rather than answered wrongly.
        Waff = solve(
            System([a * c - 1]; variables = w), Witness(; show_progress = false),
        )
        @test !Waff.projective
        @test_throws ArgumentError intersect(
            Hp[1], Waff, Intersection(; show_progress = false),
        )
    end

    @testset "sliced solve and subspace moves" begin
        Random.seed!(0x5eed)
        @polyvar a1 a2
        Fcirc = System([a1^2 + a2^2 - 5]; variables = [a1, a2])
        l1 = rand_subspace(2; codim = 1)
        l2 = rand_subspace(2; codim = 1)

        # slicing the circle with a codim-1 subspace gives 2 points
        W1 = solve(Fcirc, l1, Witness(; show_progress = false))
        @test degree(W1) == 2

        # extrinsic move: the moved points lie on the target subspace
        W2 = solve(W1, l2, Witness())
        @test degree(W2) == 2
        for pt in solutions(W2)
            @test norm(l2(pt)) < 1.0e-10
        end

        # intrinsic move between the same subspaces reaches all points too
        Hom = linear_subspace_homotopy(Fcirc, l1, l2; intrinsic = true)
        @test Hom isa IntrinsicSubspaceHomotopy
        eg = EndgameTracker(
            Tracker(HomotopyEvaluator(Hom); options = TrackerOptions()),
            EndgameOptions(),
        )
        u = FSVec{ComplexF64}(zeros(ComplexF64, 1))
        amb = zeros(ComplexF64, 2)
        moved = 0
        for s in solutions(W1)
            intrinsic_coordinates!(u, Hom, ComplexF64.(s), complex(1.0))
            track!(eg, u)
            pr = PathResult(eg; path_number = 0, start_solution = ComplexF64.(s))
            if is_success(pr)
                ambient_coordinates!(amb, Hom, HomotopyContinuationNext.solution(pr), complex(0.0))
                norm(l2(amb)) < 1.0e-10 && (moved += 1)
            end
        end
        @test moved == 2

        # extrinsic regime at init: sphere ∩ parabola sliced by a codim-1
        # subspace (dim(L) = 2 > codim(L) = 1) gives 4 points; the input is
        # a polynomial vector
        @polyvar a3
        S = a1^2 + a2^2 + a3^2 - 1
        P = a2 - a1^2
        L = rand_subspace(3; codim = 1)
        W4 = solve([S, P], L, Witness(; show_progress = false))
        @test degree(W4) == 4

        # the sliced system is [F; A x − b]
        Ls = rand_subspace(3; dim = 1)
        G = HomotopyContinuationNext._sliced_system([S, P], [a1, a2, a3], Ls)
        @test size(G) == (4, 3)
        xr = randn(ComplexF64, 3)
        u_out = FSVec{ComplexF64}(zeros(ComplexF64, 4))
        HomotopyContinuationNext.evaluate!(
            u_out, G.evaluator, FSVec{ComplexF64}(xr), FSVec{ComplexF64}(ComplexF64[]),
        )
        E = extrinsic(Ls)
        expected = [
            S([a1, a2, a3] => xr), P([a1, a2, a3] => xr), (E.A * xr - E.b)...,
        ]
        @test Vector(u_out) ≈ expected atol = 1.0e-12
    end

    @testset "intersect threaded" begin
        # exercise the threaded u-homotopy root tracking explicitly
        H = [solve(System([f]), Witness(; show_progress = false)) for f in F]
        B = intersect(H[1], H[2], Intersection(), Threaded())
        C = vcat(
            [intersect(Hi, H[3], Intersection(; show_progress = false), Threaded()) for Hi in B]...
        )
        @test sort(degree.(C)) == [2, 8, 8]
    end

    # The seed alone determines the result: the same seed from two different
    # ambient RNG states must agree, and a route must not advance the ambient
    # stream on the way.
    @testset "seeds determine the result" begin
        @polyvar x y z

        Faff = System([x^2 + y^2 - 5]; variables = [x, y])
        Fhom = System([x^2 + y^2 - z^2]; variables = [x, y, z])
        Laff = rand_subspace(2; codim = 1)
        Lhom = rand_subspace(3; codim = 1, affine = false)

        pts_agree(a, b) =
            length(a) == length(b) && all(u ≈ v for (u, v) in zip(a, b))

        # Two calls with one seed, from deliberately different ambient states.
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

        s = UInt32(0xBEEF)
        routes = (
            ("solve(F, Witness())", () -> solve(Faff, Witness(; seed = s, show_progress = false))),
            (
                "solve(F, L, Witness())",
                () -> solve(Faff, Laff, Witness(; seed = s, show_progress = false)),
            ),
            (
                "solve(F, Witness()) projective",
                () -> solve(Fhom, Witness(; seed = s, show_progress = false)),
            ),
            (
                "solve(F, L, Witness()) projective",
                () -> solve(Fhom, Lhom, Witness(; seed = s, show_progress = false)),
            ),
        )
        @testset "$name" for (name, route) in routes
            W1, W2 = from_two_states(route)
            @test pts_agree(solutions(W1), solutions(W2))
            @test !advances_ambient(route)
        end

        W = solve(Faff, Witness(; seed = UInt32(0x55), show_progress = false))
        L2 = rand_subspace(2; codim = 1)

        move = () -> solve(W, L2, Witness(; seed = s))
        M1, M2 = from_two_states(move)
        @test pts_agree(solutions(M1), solutions(M2))
        @test !advances_ambient(move)

        trace = () -> trace_test(W; seed = s)
        t1, t2 = from_two_states(trace)
        @test t1 ≈ t2
        @test !advances_ambient(trace)

        q = solutions(W)[1]
        mem = () -> membership(q, W, Membership(; seed = s, show_progress = false))
        m1, m2 = from_two_states(mem)
        @test m1 == m2 == true
        @test !advances_ambient(mem)
    end

end
