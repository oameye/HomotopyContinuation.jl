using Test, Random
using LinearAlgebra: norm
using HomotopyContinuationNext
using HomotopyContinuationNext: FSVec, FSMat, TaylorVector, ComplexDF64, DoubleF64,
    evaluate!, evaluate_and_jacobian!, taylor!, equation_scales, nparameters,
    nvariables, variables, parameters, subs, newton, NewtonReturnCode,
    find_start_pair, degrees, is_homogeneous,
    SystemEvaluator, _StartPairSystem, _clone_system_evaluator

fsvec(v) = FSVec{ComplexF64}(collect(ComplexF64, v))

function evaluate_system(F, x, p = ComplexF64[])
    u = fsvec(zeros(ComplexF64, size(F)[1]))
    evaluate!(u, F.evaluator, fsvec(x), fsvec(p))
    return collect(u)
end

function jacobian_system(F, x, p = ComplexF64[])
    m, n = size(F)
    u = fsvec(zeros(ComplexF64, m))
    U = FSMat{ComplexF64}(zeros(ComplexF64, m, n))
    evaluate_and_jacobian!(u, U, F.evaluator, fsvec(x), fsvec(p))
    return collect(U)
end

relerr(a, b) = maximum(abs, a .- b) / max(maximum(abs, b), 1.0)

@testset "compose: shape, variables, parameters" begin
    @var x y a b
    f = System([y^2 + 2x + 3, x - 1]; variables = [x, y])
    g = System([x + y * a, x - b]; variables = [x, y], parameters = [a, b])

    @test size(g ∘ f) == (2, 2)
    @test nvariables(g ∘ f) == 2
    @test variables(g ∘ f) == [x, y]
    @test parameters(g ∘ f) == [a, b]
    @test nparameters(g ∘ f) == 2

    @test parameters(g ∘ g) == [a, b]
    @test nparameters(g ∘ g) == 2
    @test parameters(f ∘ g) == [a, b]
    @test parameters(f ∘ f) == Expression[]
    @test nparameters(f ∘ f) == 0

    # three stages flatten into one composition
    @test parameters(f ∘ g ∘ f) == [a, b]
    @test nparameters(f ∘ g ∘ f) == 2
    @test length((f ∘ g ∘ f).stages) == 3
    @test compose(g, f) isa CompositionSystem

    # a composition is accepted as a stage on either side
    @test size((g ∘ f) ∘ f) == (2, 2)
    @test size(g ∘ (f ∘ f)) == (2, 2)

    # sprint(show, ...) must not throw
    @test occursin("CompositionSystem", sprint(show, g ∘ f))
end

@testset "compose: rejected input" begin
    @var x y z a c
    f = System([y^2 + 2x + 3, x - 1]; variables = [x, y])
    wide = System([x + y + z]; variables = [x, y, z])
    ga = System([x + y * a, x - a]; variables = [x, y], parameters = [a])
    gc = System([x + y * c, x - c]; variables = [x, y], parameters = [c])

    # `wide` has three variables, `f` has two equations
    @test_throws ArgumentError wide ∘ f
    @test_throws ArgumentError ga ∘ gc
end

@testset "composition evaluates the substituted system" begin
    Random.seed!(0x5cbb)
    @var x y a b
    fexprs = [y^2 + 2x + 3, x - 1]
    gexprs = [(x^2 + y * a)^2 - 3y, x - b^2]
    f = System(fexprs; variables = [x, y])
    g = System(gexprs; variables = [x, y], parameters = [a, b])
    C = g ∘ f
    # the reference is normalized as one equation, the composition stage by stage
    ref = System(subs(gexprs, [x, y] => fexprs); variables = [x, y], parameters = [a, b])
    corr = equation_scales(ref) ./ equation_scales(g)

    for _ in 1:3
        x₀ = randn(ComplexF64, 2)
        p₀ = randn(ComplexF64, 2)
        @test relerr(evaluate_system(C, x₀, p₀) ./ corr, evaluate_system(ref, x₀, p₀)) < 1.0e-12
        @test relerr(jacobian_system(C, x₀, p₀) ./ corr, jacobian_system(ref, x₀, p₀)) < 1.0e-12

        # extended-precision residual
        xd = FSVec{ComplexDF64}(ComplexDF64.(x₀))
        u1 = FSVec{ComplexDF64}(zeros(ComplexDF64, 2))
        u2 = FSVec{ComplexDF64}(zeros(ComplexDF64, 2))
        evaluate!(u1, C.evaluator, xd, fsvec(p₀))
        evaluate!(u2, ref.evaluator, xd, fsvec(p₀))
        @test relerr(ComplexF64.(collect(u1)) ./ corr, ComplexF64.(collect(u2))) < 1.0e-12

        for K in 1:3
            tx = TaylorVector{K + 1, ComplexF64}(2)
            tx.data .= randn(ComplexF64, K + 1, 2)
            tp = TaylorVector{K + 1, ComplexF64}(2)
            tp.data .= randn(ComplexF64, K + 1, 2)
            o1 = fsvec(zeros(ComplexF64, 2))
            o2 = fsvec(zeros(ComplexF64, 2))

            taylor!(o1, Val(K), C.evaluator, tx, fsvec(p₀))
            taylor!(o2, Val(K), ref.evaluator, tx, fsvec(p₀))
            @test relerr(collect(o1) ./ corr, collect(o2)) < 1.0e-10

            taylor!(o1, Val(K), C.evaluator, tx, tp)
            taylor!(o2, Val(K), ref.evaluator, tx, tp)
            @test relerr(collect(o1) ./ corr, collect(o2)) < 1.0e-10
        end
    end
end

@testset "composition evaluates with a parameter-free outer stage" begin
    Random.seed!(0x5cc0)
    @var x y a b
    fexprs = [y^2 + 2x + 3, x - 1]
    gexprs = [x + y * a, x - b^2]
    f = System(fexprs; variables = [x, y])
    g = System(gexprs; variables = [x, y], parameters = [a, b])
    C = f ∘ g
    ref = System(subs(fexprs, [x, y] => gexprs); variables = [x, y], parameters = [a, b])
    corr = equation_scales(ref) ./ equation_scales(f)

    x₀ = randn(ComplexF64, 2)
    p₀ = randn(ComplexF64, 2)
    @test relerr(evaluate_system(C, x₀, p₀) ./ corr, evaluate_system(ref, x₀, p₀)) < 1.0e-12
    @test relerr(jacobian_system(C, x₀, p₀) ./ corr, jacobian_system(ref, x₀, p₀)) < 1.0e-12
end

@testset "composition degrees and homogeneity" begin
    @var x y z

    # every stage homogeneous with one degree per stage
    f = System([x^2 + y * z, y^2 - x * z, z^2 + x * y]; variables = [x, y, z])
    g = System([x * y - z^2, x^2 + y * z, x * z]; variables = [x, y, z])
    C = g ∘ f
    @test is_homogeneous(C)
    @test degrees(C) == [4, 4, 4]
    @test degrees(C) == degrees(System(C))
    @test is_homogeneous(System(C))

    # a linear change of coordinates keeps the outer degrees
    L = System([2x - y + z, x + 3z, y - z]; variables = [x, y, z])
    @test is_homogeneous(g ∘ L)
    @test degrees(g ∘ L) == [2, 2, 2]

    # the inner degrees weight the outer equations: `x * y ↦ x^2 * y`, `x^2 ↦ x^4`
    fm = System([x^2, y]; variables = [x, y])
    gm = System([x * y, x^2]; variables = [x, y])
    @test is_homogeneous(gm ∘ fm)
    @test degrees(gm ∘ fm) == [3, 4]
    @test degrees(gm ∘ fm) == degrees(System(gm ∘ fm))
    @test is_homogeneous(System(gm ∘ fm))

    # an outer equation that is not homogeneous in those weights breaks it
    gw = System([x + y, x^2]; variables = [x, y])
    @test !is_homogeneous(gw ∘ fm)
    @test degrees(gw ∘ fm) == [2, 4]
    @test degrees(gw ∘ fm) == degrees(System(gw ∘ fm))

    # an inhomogeneous stage makes the composition inhomogeneous
    fi = System([x^2 + 1, y^2]; variables = [x, y])
    @test !is_homogeneous(gm ∘ fi)
    @test degrees(gm ∘ fi) == [4, 4]

    # a non-polynomial stage equation gives `-1` only where it is used
    fnp = System([x^2, 1 / y]; variables = [x, y])
    @test degrees(gm ∘ fnp) == [-1, 4]
    @test degrees(gm ∘ fnp) == degrees(System(gm ∘ fnp))
    @test !is_homogeneous(gm ∘ fnp)
end

@testset "System(::CompositionSystem)" begin
    Random.seed!(0x5cbf)
    @var x y a b
    fexprs = [y^2 + 2x + 3, x - 1]
    gexprs = [(x^2 + y * a)^2 - 3y, x - b^2]
    f = System(fexprs; variables = [x, y])
    g = System(gexprs; variables = [x, y], parameters = [a, b])

    for C in (g ∘ f, f ∘ g ∘ f)
        S = System(C)
        @test size(S) == size(C)
        @test variables(S) == variables(C)
        @test parameters(S) == parameters(C)
        # `System` renormalizes, so the two agree up to a factor per equation
        outer = C.stages[end]
        corr = equation_scales(S) ./ outer.scales
        for _ in 1:3
            x₀ = randn(ComplexF64, 2)
            p₀ = randn(ComplexF64, 2)
            @test relerr(
                evaluate_system(C, x₀, p₀) ./ corr, evaluate_system(S, x₀, p₀),
            ) < 1.0e-12
            @test relerr(
                jacobian_system(C, x₀, p₀) ./ corr, jacobian_system(S, x₀, p₀),
            ) < 1.0e-12
        end
    end

    # the inner stage's normalization is undone symbolically too
    @var u v
    fb = System([1.0e9 * v + u, u - 1]; variables = [u, v])
    gb = System([u + v, u * v]; variables = [u, v])
    Cb = gb ∘ fb
    Sb = System(Cb)
    x₀ = randn(ComplexF64, 2)
    @test relerr(
        evaluate_system(Cb, x₀) ./ equation_scales(Sb), evaluate_system(Sb, x₀),
    ) < 1.0e-12
end

@testset "start pair system: joint (x, p) evaluation" begin
    Random.seed!(0x5cc1)
    @var x y q1 q2
    F = System(
        [x^2 * q1 + y^2 - q2, x * y * q2 + q1^2 - 3];
        variables = [x, y], parameters = [q1, q2],
    )
    # the same equations with the parameters promoted to variables
    joint = System(
        [x^2 * q1 + y^2 - q2, x * y * q2 + q1^2 - 3]; variables = [x, y, q1, q2],
    )
    SP = SystemEvaluator(_StartPairSystem(F.evaluator))
    @test size(SP) == (2, 4)
    @test nparameters(SP) == 0

    for _ in 1:3
        xp = randn(ComplexF64, 4)
        u1 = fsvec(zeros(ComplexF64, 2))
        u2 = fsvec(zeros(ComplexF64, 2))
        U1 = FSMat{ComplexF64}(zeros(ComplexF64, 2, 4))
        U2 = FSMat{ComplexF64}(zeros(ComplexF64, 2, 4))
        evaluate_and_jacobian!(u1, U1, SP, fsvec(xp), fsvec(ComplexF64[]))
        evaluate_and_jacobian!(u2, U2, joint.evaluator, fsvec(xp), fsvec(ComplexF64[]))
        @test relerr(collect(u1), collect(u2)) < 1.0e-12
        @test relerr(collect(U1), collect(U2)) < 1.0e-12

        for K in 1:3
            tx = TaylorVector{K + 1, ComplexF64}(4)
            tx.data .= randn(ComplexF64, K + 1, 4)
            o1 = fsvec(zeros(ComplexF64, 2))
            o2 = fsvec(zeros(ComplexF64, 2))
            taylor!(o1, Val(K), SP, tx, fsvec(ComplexF64[]))
            taylor!(o2, Val(K), joint.evaluator, tx, fsvec(ComplexF64[]))
            @test relerr(collect(o1), collect(o2)) < 1.0e-10
        end
    end
end

@testset "composition undoes the inner equation scaling" begin
    Random.seed!(0x5cbc)
    @var x y
    fexprs = [1.0e9 * y + x, x - 1]
    gexprs = [x + y, x * y]
    f = System(fexprs; variables = [x, y])
    g = System(gexprs; variables = [x, y])
    @test !all(isone, equation_scales(f))

    C = g ∘ f
    ref = System(subs(gexprs, [x, y] => fexprs); variables = [x, y])
    x₀ = randn(ComplexF64, 2)
    @test relerr(
        evaluate_system(C, x₀) ./ equation_scales(ref), evaluate_system(ref, x₀),
    ) < 1.0e-12
end

@testset "solving a composition from a start system" begin
    @var a b c x y z u v
    e = System([u + 1, v - 2]; variables = [u, v])
    f = System([a * b - 2, a * c - 1]; variables = [a, b, c])
    g = System([x + y, y + 3, x + 2]; variables = [x, y])
    C = e ∘ f ∘ g
    @test size(C) == (2, 2)
    # total degree folds the stage degrees, so the equations are never rebuilt
    @test degrees(C) == [2, 2]

    r = solve(C, TotalDegree(; show_progress = false))
    @test nsolutions(r) == 2
    for s in solutions(r)
        @test norm(evaluate_system(C, s), Inf) < 1.0e-10
    end

    rp = solve(C, Polyhedral(; show_progress = false))
    @test nsolutions(rp) == 2

    # a non-polynomial stage names the composition, not the outermost equations
    @var s t
    nonpolynomial = System([s^2, 1 / t]; variables = [s, t]) ∘
        System([s + t, s - t]; variables = [s, t])
    @test_throws ArgumentError solve(
        System([s * t, s^2]; variables = [s, t]) ∘ nonpolynomial,
        TotalDegree(; show_progress = false),
    )
end

@testset "cloning a composition" begin
    @var x y a b
    f = System([y^2 + 2x + 3, x - 1]; variables = [x, y])
    g = System([x + y * a, x - b]; variables = [x, y], parameters = [a, b])
    C = g ∘ f
    clone = _clone_system_evaluator(C)
    @test size(clone) == size(C)
    @test nparameters(clone) == nparameters(C)

    x₀ = randn(ComplexF64, 2)
    p₀ = randn(ComplexF64, 2)
    u1 = fsvec(zeros(ComplexF64, 2))
    u2 = fsvec(zeros(ComplexF64, 2))
    evaluate!(u1, C.evaluator, fsvec(x₀), fsvec(p₀))
    evaluate!(u2, clone, fsvec(x₀), fsvec(p₀))
    @test collect(u1) == collect(u2)
end

@testset "newton and parameter homotopy through a composition" begin
    Random.seed!(0x5cbd)
    @var x y q1 q2
    F = System([x^2 + y^2 - q1, x + y - q2]; variables = [x, y], parameters = [q1, q2])
    A = randn(ComplexF64, 2, 2)
    c = randn(ComplexF64, 2)
    L = System(A * [x, y] + c; variables = [x, y])
    C = F ∘ L

    p₀ = ComplexF64[3.0, 1.0]
    target = System([x^2 + y^2 - 3.0, x + y - 1.0]; variables = [x, y])
    truth = solutions(solve(target, TotalDegree(; show_progress = false)))
    @test length(truth) == 2
    starts = [A \ (s - c) for s in truth]

    res = newton(C, starts[1] .+ 1.0e-6; p = p₀)
    @test res.return_code == NewtonReturnCode.NEWTON_SUCCESS
    @test norm(A * res.x + c - truth[1], Inf) < 1.0e-8

    q = ComplexF64[5.0, 2.0]
    r = solve(C, starts, p₀, q, Continuation(; show_progress = false))
    @test nsolutions(r) == 2
    for s in solutions(r)
        z = A * s + c
        @test abs(z[1]^2 + z[2]^2 - q[1]) < 1.0e-8
        @test abs(z[1] + z[2] - q[2]) < 1.0e-8
    end
end

@testset "monodromy through a composition" begin
    Random.seed!(0x5cbe)
    @var x y q1 q2
    F = System([x^2 + y^2 - q1, x + y - q2]; variables = [x, y], parameters = [q1, q2])
    A = randn(ComplexF64, 2, 2)
    c = randn(ComplexF64, 2)
    C = F ∘ System(A * [x, y] + c; variables = [x, y])

    p₀ = ComplexF64[3.0, 1.0]
    target = System([x^2 + y^2 - 3.0, x + y - 1.0]; variables = [x, y])
    truth = solutions(solve(target, TotalDegree(; show_progress = false)))
    start = A \ (truth[1] - c)

    r = solve(
        C,
        [start],
        p₀,
        Monodromy(;
            target_solutions_count = 2, show_progress = false, seed = UInt32(0x5cbe),
        ),
        Serial(),
    )
    @test nsolutions(r) == 2
    found = sort([A * s + c for s in solutions(r)]; by = z -> real(z[1]))
    expected = sort(truth; by = z -> real(z[1]))
    @test norm(found[1] - expected[1], Inf) < 1.0e-8
    @test norm(found[2] - expected[2], Inf) < 1.0e-8

end

@testset "start pair and monodromy without an explicit start" begin
    Random.seed!(0x5cc2)
    @var x y q1 q2
    F = System([x^2 + y^2 - q1, x + y - q2]; variables = [x, y], parameters = [q1, q2])
    A = randn(ComplexF64, 2, 2)
    c = randn(ComplexF64, 2)
    C = F ∘ System(A * [x, y] + c; variables = [x, y])

    pair = find_start_pair(C)
    @test pair !== nothing
    start, p = pair
    @test p !== nothing
    @test length(start) == 2
    @test length(p) == 2
    @test norm(evaluate_system(C, start, p), Inf) < 1.0e-10

    r = solve(
        C,
        Monodromy(;
            target_solutions_count = 2, show_progress = false, seed = UInt32(0x5cc2),
        ),
        Serial(),
    )
    @test nsolutions(r) == 2

    # a parameter-free composition returns a zero and no parameters
    Cf = System([x^2 + y^2 - 3.0, x + y - 1.0]; variables = [x, y]) ∘
        System(A * [x, y] + c; variables = [x, y])
    free_pair = find_start_pair(Cf)
    @test free_pair !== nothing
    @test free_pair[2] === nothing
    @test norm(evaluate_system(Cf, free_pair[1]), Inf) < 1.0e-10
end
