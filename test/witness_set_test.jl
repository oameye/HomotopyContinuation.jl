using Test, Random
using LinearAlgebra
using HomotopyContinuationNext
using HomotopyContinuationNext: corank, extrinsic, TrackerOptions,
    IntrinsicSubspaceHomotopy, ExtrinsicSubspaceHomotopy,
    EndgameTracker, Tracker, HomotopyEvaluator, PathResult,
    intrinsic_coordinates!, ambient_coordinates!, track!, is_success, FSVec
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

        W = witness_set(F; show_progress = false)

        @test dim(W) == 1
        @test codim(W) == 1
        @test degree(W) == 2
        @test solutions(W) isa Vector{Vector{ComplexF64}}

        L = LinearSubspace([1 1], [-1])

        W_L = witness_set(W, L)
        @test degree(W_L) == 2
        @test sort(real.(solutions(W_L))) ≈ [[-2, 1], [1, -2]]
        @test linear_subspace(W_L) == convert(typeof(linear_subspace(W_L)), L)

        @test trace_test(W) < 1.0e-8
        @test_throws MethodError trace_test(W; shwo_progress = false)

        W_seed₁ = witness_set(F; seed = 0x1234, threading = false, show_progress = false)
        W_seed₂ = witness_set(F; seed = 0x1234, threading = false, show_progress = false)
        @test linear_subspace(W_seed₁) == linear_subspace(W_seed₂)
        @test points(W_seed₁) == points(W_seed₂)
    end

    @testset "polynomial input forms" begin
        @polyvar x y

        # single polynomial and vector of polynomials
        @test degree(witness_set(x^2 + y^2 - 5; show_progress = false)) == 2
        @test degree(witness_set([x^2 + y^2 - 5]; show_progress = false)) == 2

        L = LinearSubspace([1 1], [-1])
        W_L = witness_set(x^2 + y^2 - 5, L; show_progress = false)
        @test sort(real.(solutions(W_L))) ≈ [[-2, 1], [1, -2]]
        W_L2 = witness_set([x^2 + y^2 - 5], L; show_progress = false)
        @test sort(real.(solutions(W_L2))) ≈ [[-2, 1], [1, -2]]

        # a WitnessSet requires a parameter-free system
        @polyvar p
        Fpar = System([x^2 + y^2 - p]; variables = [x, y], parameters = [p])
        @test_throws ArgumentError WitnessSet(Fpar, L, Vector{Vector{ComplexF64}}())
    end

    @testset "projective" begin
        @polyvar x y z

        F = System([x^2 + y^2 - 5z^2]; variables = [x, y, z])

        W = witness_set(F; show_progress = false)

        @test dim(W) == 1
        @test codim(W) == 2
        @test degree(W) == 2
        @test solutions(W) isa Vector{Vector{ComplexF64}}
        @test trace_test(W) < 1.0e-8

        L = LinearSubspace([1 1 1])
        W_L = witness_set(W, L)
        @test degree(W_L) == 2

        L = rand_subspace(3; codim = 1, affine = false)
        W_L = witness_set(W, L)
        @test degree(W_L) == 2

        L = rand_subspace([x, y, z]; codim = 1, affine = false)
        W_L = witness_set(W, L)
        @test degree(W_L) == 2

        L = rand_subspace(3; codim = 1, affine = true)
        @test_throws ErrorException witness_set(W, L)
    end

    @testset "projective membership" begin
        @polyvar x y z

        F = System([x^2 + y^2 - 5z^2]; variables = [x, y, z])
        W = witness_set(F; show_progress = false)
        @test W.projective

        # any projective representative works (arbitrary complex scaling)
        pt_on = ComplexF64[3, 4, sqrt(5)] .* (2.0 + 1.0im)
        pt_off = randn(ComplexF64, 3)
        q0 = solutions(W)[1]

        @test membership(pt_on, W; threading = false, show_progress = false)
        @test !membership(pt_off, W; threading = false, show_progress = false)
        @test membership([pt_on, pt_off, q0], W; threading = false, show_progress = false) ==
            [true, false, true]
        # threaded driver (with one thread this still exercises the task path)
        @test membership([pt_on, pt_off, q0], W; threading = true, show_progress = false) ==
            [true, false, true]
    end

    @testset "zero-dimensional" begin
        # projective: two points in P^1
        @polyvar s t
        W0p = witness_set(System([s^2 - t^2]; variables = [s, t]); show_progress = false)
        @test degree(W0p) == 2
        @test dim(W0p) == 0
        @test trace_test(W0p) < 1.0e-8

        # affine: four isolated points
        @polyvar a b
        F0 = System([a^2 - 1, b^2 - 4]; variables = [a, b])
        W0a = witness_set(F0; show_progress = false)
        @test degree(W0a) == 4
        @test dim(W0a) == 0
        @test membership([ComplexF64[1, 2], ComplexF64[1, 3]], W0a; threading = false, show_progress = false) ==
            [true, false]

        # explicit dim = 0 kwarg
        @test degree(witness_set(F0; dim = 0, show_progress = false)) == 4
    end

    @testset "parametric (target_parameters)" begin
        @polyvar u v p q
        F = System([u^2 + v^2 - p * q]; variables = [u, v], parameters = [p, q])

        W = witness_set(F; target_parameters = [2.0, 2.5], show_progress = false)
        @test degree(W) == 2
        @test dim(W) == 1
        # the stored system is parameter-free (parameters substituted)
        @test HomotopyContinuationNext.nparameters(system(W)) == 0
        @test trace_test(W) < 1.0e-8

        # the fixed witness set moves like any other
        W_L = witness_set(W, LinearSubspace([1 1], [-1]))
        @test degree(W_L) == 2

        # explicit subspace entry point takes target_parameters too
        L = rand_subspace(2; codim = 1)
        W2 = witness_set(F, L; target_parameters = [2.0, 2.5], show_progress = false)
        @test degree(W2) == 2

        # errors: missing, spurious, and wrong-length parameter values
        @test_throws ArgumentError witness_set(F)
        @test_throws ArgumentError witness_set(F; target_parameters = [1.0])
        G = System([u^2 + v^2 - 5]; variables = [u, v])
        @test_throws ArgumentError witness_set(G; target_parameters = [1.0])
        # regeneration rejects parametric systems loudly
        @test_throws ArgumentError regeneration(F)
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

        @test degree(witness_set(f; dim = 1, show_progress = false)) == 16
        @test degree(witness_set(f; codim = 4, show_progress = false)) == 16
        @test degree(witness_set(f; show_progress = false)) == 16

        homogeneous = false
        f = System(
            [
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
                rand_poly(x, 2; homogeneous = homogeneous),
            ]
        )
        @test degree(witness_set(f; dim = 2, show_progress = false)) == 16
        @test degree(witness_set(f; codim = 4, show_progress = false)) == 16
        @test degree(witness_set(f; show_progress = false)) == 16
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
        W = witness_set(System(F); codim = 2, show_progress = false)

        pt = randn(3)
        q0 = solutions(W)[1]

        @test !membership(pt, W; show_progress = false)
        @test membership(q0, W; show_progress = false)
        a = membership([pt, q0], W; show_progress = false)
        @test a == [false, true]
        @test membership([pt, q0], W; show_progress = true) == [false, true]
    end

    @testset "intersect" begin
        H = [witness_set(System([f]); show_progress = false) for f in F]
        B = intersect(H[1], H[2])
        C = vcat([intersect(Hi, H[3]; show_progress = false) for Hi in B]...)
        @test sort(degree.(C)) == [2, 8, 8]
    end

    @testset "sliced solve and subspace moves" begin
        Random.seed!(0x5eed)
        @polyvar a1 a2
        Fcirc = System([a1^2 + a2^2 - 5]; variables = [a1, a2])
        l1 = rand_subspace(2; codim = 1)
        l2 = rand_subspace(2; codim = 1)

        # slicing the circle with a codim-1 subspace gives 2 points
        W1 = witness_set(Fcirc, l1; show_progress = false)
        @test degree(W1) == 2

        # extrinsic move: the moved points lie on the target subspace
        W2 = witness_set(W1, l2)
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
        W4 = witness_set([S, P], L; show_progress = false)
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
        H = [witness_set(System([f]); show_progress = false) for f in F]
        B = intersect(H[1], H[2]; threading = true)
        C = vcat(
            [intersect(Hi, H[3]; show_progress = false, threading = true) for Hi in B]...
        )
        @test sort(degree.(C)) == [2, 8, 8]
    end

end
