using Test
using Random: Random, MersenneTwister, randn, rand
import HomotopyContinuationNext as Next
using HomotopyContinuationNext: Expression, System, CompileMode, @var, @polyvar,
    differentiate, solve, solutions, nsolutions,
    verify_solution_completeness, Serial, TotalDegree, Polyhedral,
    Continuation, Monodromy, Witness, Regeneration, Decomposition, Intersection,
    TaylorVector, ComplexDF64, HomotopyEvaluator, StraightLineHomotopy,
    Interpreter, execute!,
    fix_parameters
using FixedSizeArrays: FixedSizeArray

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

fsv(v) = FSVec{ComplexF64}(collect(ComplexF64, v))
fsm(m) = FSMat{ComplexF64}(collect(ComplexF64, m))

function eval_system(F, x, p)
    u = fsv(zeros(size(F)[1]))
    Next.evaluate!(u, F.evaluator, fsv(x), fsv(p))
    return collect(u)
end

function eval_jacobian(F, x, p)
    m, n = size(F)
    u = fsv(zeros(m))
    U = fsm(zeros(m, n))
    Next.evaluate_and_jacobian!(u, U, F.evaluator, fsv(x), fsv(p))
    return collect(u), collect(U)
end

# Coefficients of λ ↦ g(λ) at 0 from samples on a circle of radius r.
function cauchy_coefficients(g, K::Int; M::Int = 128, r::Float64 = 0.1)
    out = Vector{ComplexF64}[]
    for j in 0:(M - 1)
        v = g(r * cis(2π * j / M))
        isempty(out) && append!(out, [zeros(ComplexF64, length(v)) for _ in 0:K])
        for k in 0:K
            out[k + 1] .+= v .* cis(-2π * j * k / M)
        end
    end
    return [out[k + 1] ./ (M * r^k) for k in 0:K]
end

# Coefficients of λ ↦ f(x(λ); p) at 0, with each argument given by its series.
function cauchy_system_coefficients(f, coeffs; K::Int, M::Int = 128, r::Float64 = 0.1)
    n = length(coeffs)
    series(λ) = [
        sum(coeffs[i][k + 1] * λ^k for k in 0:(length(coeffs[i]) - 1)) for i in 1:n
    ]
    return cauchy_coefficients(λ -> f(series(λ)), K; M = M, r = r)
end

include("test_systems.jl")

const MODES = (CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL)

@testset "Non-polynomial system sweep: $name" for (name, exprs, vars, params, ref) in
    NONPOLYNOMIAL_SYSTEM_COLLECTION

    rng = MersenneTwister(0x00e8b1a5 + length(name))
    m, n, r = length(exprs), length(vars), length(params)
    # Real, order-one points: away from the poles and off the branch cut.
    xv = ComplexF64.(0.7 .+ rand(rng, n))
    pv = ComplexF64.(0.7 .+ rand(rng, r))
    coeffs = [
        [i == 1 ? xv[j] : 0.15 * (randn(rng, ComplexF64)) for i in 1:4] for j in 1:n
    ]
    pcoeffs = [
        [i == 1 ? pv[j] : 0.15 * (randn(rng, ComplexF64)) for i in 1:4] for j in 1:r
    ]

    truth_u = ref(xv, pv)
    h = 1.0e-6
    truth_J = zeros(ComplexF64, m, n)
    for j in 1:n
        e = zeros(ComplexF64, n)
        e[j] = h
        truth_J[:, j] .= (ref(xv .+ e, pv) .- ref(xv .- e, pv)) ./ (2h)
    end

    truth_taylor = cauchy_system_coefficients(
        w -> ref(w[1:n], w[(n + 1):(n + r)]), [coeffs; pcoeffs]; K = 3, r = 0.05,
    )

    @testset "symbolic tape roundtrip" begin
        F = System(exprs; variables = vars, parameters = params)
        I = Interpreter(Vector{Expression}, F._interp_f64.sequence)
        out = Vector{Expression}(undef, m)
        execute!(out, I, collect(vars), collect(params))
        @test out == collect(exprs)
    end

    @testset "$mode" for mode in MODES
        F = System(exprs; variables = vars, parameters = params, compile = mode)
        @test size(F) == (m, n)

        u_got, J_got = eval_jacobian(F, xv, pv)
        @test u_got ≈ truth_u rtol = 1.0e-10
        @test J_got ≈ truth_J rtol = 1.0e-5

        u_df64 = FSVec{ComplexDF64}(zeros(ComplexDF64, m))
        Next.evaluate!(
            u_df64, F.evaluator,
            FSVec{ComplexDF64}(ComplexDF64.(xv)), fsv(pv),
        )
        @test ComplexF64.(collect(u_df64)) ≈ truth_u rtol = 1.0e-10

        @testset "taylor! K=$K" for K in 1:3
            xdata = FSMat{ComplexF64}(zeros(ComplexF64, K + 1, n))
            pdata = FSMat{ComplexF64}(zeros(ComplexF64, K + 1, max(r, 1)))
            for j in 1:n, k in 0:K
                xdata[k + 1, j] = coeffs[j][k + 1]
            end
            for j in 1:r, k in 0:K
                pdata[k + 1, j] = pcoeffs[j][k + 1]
            end
            u = fsv(zeros(m))
            Next.taylor!(
                u, Val(K), F.evaluator,
                TaylorVector{K + 1, ComplexF64}(xdata),
                r == 0 ? fsv(ComplexF64[]) :
                    TaylorVector{K + 1, ComplexF64}(pdata[:, 1:r]),
            )
            @test collect(u) ≈ truth_taylor[K + 1] atol = 1.0e-7
        end
    end
end

# H(x,t) = γ·t·G(x) + (1-t)·F(x) with F non-polynomial and its parameters already
# substituted in.
@testset "StraightLineHomotopy sweep: $name" for (name, exprs, vars, params, ref) in
    NONPOLYNOMIAL_SYSTEM_COLLECTION

    rng = MersenneTwister(0x00c0ffee + length(name))
    m, n, r = length(exprs), length(vars), length(params)
    pv = ComplexF64.(0.7 .+ rand(rng, r))
    target = r == 0 ? collect(exprs) : Next.subs(exprs, params => pv)
    start = [sum(vars) - i for i in 1:m]
    γ = ComplexF64(cis(2π * 0.3))
    t = 0.37 + 0.21im
    X = ComplexF64.(0.7 .+ rand(rng, 4, n))
    X[2:4, :] .= 0.15 .* randn(rng, ComplexF64, 3, n)
    x0 = X[1, :]

    fref(z) = ref(z, pv)
    gref(z) = ComplexF64[sum(z) - i for i in 1:m]
    href(z, s) = γ * s .* gref(z) .+ (1 - s) .* fref(z)

    @testset "$mode" for mode in MODES
        G = System(start; variables = vars, compile = mode)
        F = System(target; variables = vars, compile = mode)
        He = HomotopyEvaluator(
            StraightLineHomotopy(G.evaluator, F.evaluator; γ = γ),
        )

        u = fsv(zeros(m))
        Next.evaluate!(u, He, fsv(x0), ComplexF64(t))
        @test collect(u) ≈ href(x0, t) rtol = 1.0e-10

        U = fsm(zeros(m, n))
        Next.evaluate_and_jacobian!(u, U, He, fsv(x0), ComplexF64(t))
        @test collect(u) ≈ href(x0, t) rtol = 1.0e-10
        h = 1.0e-6
        for j in 1:n
            e = zeros(ComplexF64, n)
            e[j] = h
            @test collect(U)[:, j] ≈ (href(x0 .+ e, t) .- href(x0 .- e, t)) ./ (2h) rtol =
                1.0e-5
        end

        # Val(1) takes a plain point: x is constant, so only the ∂/∂t term remains.
        fill!(u, 0)
        Next.taylor!(u, Val(1), He, fsv(x0), ComplexF64(t))
        @test collect(u) ≈ γ .* gref(x0) .- fref(x0) rtol = 1.0e-10

        @testset "taylor! K=$K" for K in 2:3
            xλ(λ) = [sum(X[k + 1, i] * λ^k for k in 0:K) for i in 1:n]
            truth = cauchy_coefficients(λ -> href(xλ(λ), t + λ), K; r = 0.05)
            data = FSMat{ComplexF64}(ComplexF64.(X[1:(K + 1), :]))
            fill!(u, 0)
            Next.taylor!(
                u, Val(K), He, TaylorVector{K + 1, ComplexF64}(data), ComplexF64(t),
            )
            @test collect(u) ≈ truth[K + 1] atol = 1.0e-7
        end
    end
end

@testset "Non-polynomial input" begin
    @testset "rational system evaluation and Jacobian" begin
        @var x y u[1:4]
        F = System(
            [u[1] / x^2 + u[2], u[3] / y^2 + u[4]];
            variables = [x, y], parameters = u,
        )
        @test size(F) == (2, 2)
        @test Next.degrees(F) == [-1, -1]

        xv = [1.4 + 0.3im, -0.8 + 0.5im]
        pv = [2.0, -1.0, 3.0, 0.5]
        ref(z) = [pv[1] / z[1]^2 + pv[2], pv[3] / z[2]^2 + pv[4]]

        u_got, J_got = eval_jacobian(F, xv, pv)
        @test u_got ≈ ref(xv)
        J_exact = ComplexF64[
            -2pv[1] / xv[1]^3 0
            0 -2pv[3] / xv[2]^3
        ]
        @test J_got ≈ J_exact
    end

    @testset "sqrt parameters: compile-mode parity and derivatives" begin
        @var x y a b
        exprs = [sqrt(a + b) * x^2 - y, (x * y + a - sqrt(b))^2 - 3]
        xv = [1.3 + 0.2im, -0.7 + 0.4im]
        pv = [2.0, 3.0]
        ref(z) = [
            sqrt(pv[1] + pv[2]) * z[1]^2 - z[2],
            (z[1] * z[2] + pv[1] - sqrt(pv[2]))^2 - 3,
        ]

        systems = [
            System(exprs; parameters = [a, b], compile = mode) for
                mode in (
                    CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL,
                )
        ]

        u_ref, J_ref = eval_jacobian(systems[1], xv, pv)
        @test u_ref ≈ ref(xv)
        for F in systems[2:end]
            u_got, J_got = eval_jacobian(F, xv, pv)
            @test u_got ≈ u_ref
            @test J_got ≈ J_ref
        end

        # Jacobian against central differences.
        h = 1.0e-6
        for j in 1:2
            e = zeros(ComplexF64, 2)
            e[j] = h
            numeric = (ref(xv .+ e) .- ref(xv .- e)) ./ (2h)
            @test J_ref[:, j] ≈ numeric atol = 1.0e-6
        end
    end

    @testset "extended-precision residual on a transcendental system" begin
        @var x y a b
        F = System([sin(a) * x + cos(y) - b, x^2 + y - 1]; parameters = [a, b])
        xv = ComplexDF64[ComplexDF64(1.3), ComplexDF64(-0.7)]
        pv = [2.0, 3.0]
        u = fsv(zeros(2))
        Next.evaluate!(u, F.evaluator, FSVec{ComplexDF64}(xv), fsv(pv))
        @test collect(u) ≈ [
            sin(2.0) * 1.3 + cos(-0.7) - 3.0,
            1.3^2 - 0.7 - 1.0,
        ]
    end

    @testset "Taylor orders 1-3 against a Cauchy-integral oracle" begin
        @var x y a b
        exprs = [sqrt(a + b) * x^2 - y / x, sin(x) + (y + a)^-2]
        pv = [2.0, 3.0]
        ref(z) = [
            sqrt(pv[1] + pv[2]) * z[1]^2 - z[2] / z[1],
            sin(z[1]) + (z[2] + pv[1])^-2,
        ]

        coeffs = [
            ComplexF64[1.3 + 0.2im, 0.3 - 0.1im, -0.15 + 0.2im, 0.05 + 0.1im],
            ComplexF64[-0.7 + 0.4im, 0.2 + 0.15im, 0.1 - 0.05im, -0.2 + 0.1im],
        ]
        truth = cauchy_system_coefficients(ref, coeffs; K = 3)

        for mode in (CompileMode.INTERPRETED, CompileMode.COMPILED_ALL)
            F = System(exprs; parameters = [a, b], compile = mode)
            for K in 1:3
                data = FSMat{ComplexF64}(zeros(ComplexF64, K + 1, 2))
                for i in 1:2, k in 0:K
                    data[k + 1, i] = coeffs[i][k + 1]
                end
                tx = TaylorVector{K + 1, ComplexF64}(data)
                u = fsv(zeros(2))
                Next.taylor!(u, Val(K), F.evaluator, tx, fsv(pv))
                @test collect(u) ≈ truth[K + 1] atol = 1.0e-8
            end
        end
    end

    @testset "Taylor-valued parameters against a Cauchy-integral oracle" begin
        @var x y a
        exprs = [a / x - y, sqrt(a) * x + sin(y)]
        ref(z, p) = [p[1] / z[1] - z[2], sqrt(p[1]) * z[1] + sin(z[2])]

        xc = [
            ComplexF64[1.3 + 0.2im, 0.3 - 0.1im, -0.15 + 0.2im, 0.05 + 0.1im],
            ComplexF64[-0.7 + 0.4im, 0.2 + 0.15im, 0.1 - 0.05im, -0.2 + 0.1im],
        ]
        pc = [ComplexF64[2.0 + 0.3im, 0.4 - 0.2im, 0.1 + 0.05im, -0.15 + 0.1im]]

        # Coefficients of λ ↦ F(x(λ); p(λ)), so the parameter series convolves too.
        combined(w) = ref(w[1:2], w[3:3])
        truth = cauchy_system_coefficients(combined, [xc; pc]; K = 3)

        F = System(exprs; variables = [x, y], parameters = [a])
        for K in 1:3
            xdata = FSMat{ComplexF64}(zeros(ComplexF64, K + 1, 2))
            pdata = FSMat{ComplexF64}(zeros(ComplexF64, K + 1, 1))
            for k in 0:K
                xdata[k + 1, 1] = xc[1][k + 1]
                xdata[k + 1, 2] = xc[2][k + 1]
                pdata[k + 1, 1] = pc[1][k + 1]
            end
            u = fsv(zeros(2))
            Next.taylor!(
                u, Val(K), F.evaluator,
                TaylorVector{K + 1, ComplexF64}(xdata),
                TaylorVector{K + 1, ComplexF64}(pdata),
            )
            @test collect(u) ≈ truth[K + 1] atol = 1.0e-8
        end
    end

    @testset "parameter homotopy tracking through a sqrt parameter" begin
        # sqrt(a) x² + x - 1 = 0, y = 1 - x. Track from a = 4 to a = 9.
        @var x y a
        F = System([sqrt(a) * x^2 + x - 1, x + y - 1]; parameters = [a])

        roots(s) = [(-1 + sign * sqrt(1 + 4s)) / (2s) for sign in (1, -1)]
        starts = [ComplexF64[xi, 1 - xi] for xi in roots(2.0)]   # sqrt(4) = 2
        res = solve(
            F,
            starts,
            ComplexF64[4.0],
            ComplexF64[9.0],
            Continuation(; show_progress = false),
            Serial(),
        )
        @test nsolutions(res) == 2
        got = sort(real.(first.(solutions(res))))
        @test got ≈ sort(roots(3.0)) atol = 1.0e-8   # sqrt(9) = 3
    end

    @testset "monodromy on rational functions" begin
        @var x y u[1:4]
        counts = map((CompileMode.INTERPRETED, CompileMode.COMPILED_ALL)) do mode
            F = System(
                [u[1] / x^2 + u[2], u[3] / y^2 + u[4]];
                variables = [x, y], parameters = u, compile = mode,
            )
            res = solve(
                F,
                Monodromy(;
                    target_solutions_count = 4, max_loops_no_progress = 100,
                    show_progress = false,
                ),
                Serial(),
            )
            return nsolutions(res)
        end
        @test all(==(4), counts)
    end

    @testset "monodromy on a rational triangulation system" begin
        @var A1[1:3, 1:4] A2[1:3, 1:4]
        @var x[1:3] u1[1:2] u2[1:2]
        y1 = A1 * [x; 1]
        y2 = A2 * [x; 1]
        f = sum((u1 - y1[1:2] ./ y1[3]) .^ 2) + sum((u2 - y2[1:2] ./ y2[3]) .^ 2)
        F = System(
            differentiate(f, x);
            variables = x, parameters = [u1; u2; vec(A1); vec(A2)],
        )
        res = solve(
            F,
            Monodromy(;
                max_loops_no_progress = 100, target_solutions_count = 6,
                show_progress = false,
            ),
            Serial(),
        )
        @test nsolutions(res) == 6
    end

    @testset "verify_solution_completeness on an expression system" begin
        @var x y a b c
        F = System([x^2 + y^2 - 1, a * x + b * y + c]; parameters = [a, b, c])
        q = ComplexF64[1, 2, 3]
        sols = [
            ComplexF64[-0.6 - 0.8im, -1.2 + 0.4im],
            ComplexF64[-0.6 + 0.8im, -1.2 - 0.4im],
        ]
        @test verify_solution_completeness(F, sols, q, Monodromy(; show_progress = false)) == true
        @test verify_solution_completeness(F, sols[1:1], q, Monodromy(; show_progress = false)) == false
    end

    @testset "MP rational input builds the same system" begin
        @polyvar px py pu[1:4]
        F_mp = System(
            [pu[1] / px^2 + pu[2], pu[3] / py^2 + pu[4]];
            variables = [px, py], parameters = pu,
        )
        @var x y u[1:4]
        F_expr = System(
            [u[1] / x^2 + u[2], u[3] / y^2 + u[4]];
            variables = [x, y], parameters = u,
        )
        xv = [1.4 + 0.3im, -0.8 + 0.5im]
        pv = [2.0, -1.0, 3.0, 0.5]
        @test eval_system(F_mp, xv, pv) ≈ eval_system(F_expr, xv, pv)
    end

    @testset "start systems reject non-polynomial input" begin
        @var x y
        F = System([x / y - 2, x^2 + y^2 - 5])
        @test_throws ArgumentError solve(F, TotalDegree(; show_progress = false), Serial())
        @test_throws ArgumentError solve(F, Polyhedral(; show_progress = false), Serial())
        L = Next.LinearSubspace(ComplexF64[1.0 1.0], ComplexF64[1.0])
        @test_throws ArgumentError solve(F, L, TotalDegree(; show_progress = false), Serial())
        @test_throws ArgumentError Next.solve(
            System([x / y - 1]; variables = [x, y]),
            Next.Witness(; show_progress = false),
        )
    end

    @testset "polyhedral start systems on expression input" begin
        @var x y
        @polyvar u v
        @testset "support matches the polynomial front-end" begin
            expr = Next.support_coefficients(
                System([x^2 * y + 3y - 1, x + (x + y)^2 - 1]; variables = [x, y]),
            )
            poly = Next.support_coefficients(System([u^2 * v + 3v - 1, u + (u + v)^2 - 1]))
            @test expr == poly
        end

        sols(r) = sort(
            [
                (round(real(s[1]); digits = 6), round(imag(s[1]); digits = 6))
                    for s in solutions(r)
            ]
        )

        @testset "dense system" begin
            F = System([x^2 + y - 1, x + y^2 - 1]; variables = [x, y])
            G = System([u^2 + v - 1, u + v^2 - 1])
            @test sols(solve(F, Polyhedral(; show_progress = false), Serial())) ==
                sols(solve(G, Polyhedral(; show_progress = false), Serial()))
        end

        # Sparse enough that the BKK bound is below the Bezout number.
        @testset "sparse system" begin
            F = System([x^3 * y^2 - 3, x^2 * y^3 - 5]; variables = [x, y])
            r = solve(F, Polyhedral(; show_progress = false), Serial())
            @test nsolutions(r) == 5
            @test sols(r) == sols(solve(F, TotalDegree(; show_progress = false), Serial()))
        end

        @testset "overdetermined system is squared up" begin
            F = System([x^2 + y^2 - 1, x - y, x^3 - y^3]; variables = [x, y])
            @test sols(solve(F, Polyhedral(; show_progress = false), Serial())) ==
                [(-0.707107, -0.0), (0.707107, -0.0)]
        end

        @testset "a parameter is not a variable" begin
            @var a
            F = System([x^2 + a * y - 1, x + y^2 - 1]; variables = [x, y], parameters = [a])
            @test_throws ArgumentError Next.support_coefficients(F)
        end
    end

    @testset "sliced solve substitutes parameters through the expression front-end" begin
        @var x y a
        @polyvar u v b
        L = Next.LinearSubspace(ComplexF64[1.0 1.0], ComplexF64[1.0])
        sols(r) = sort(
            [
                (round(real(s[1]); digits = 6), round(imag(s[1]); digits = 6))
                    for s in solutions(r)
            ]
        )
        F = System([a * x^2 + y^2 - 1]; variables = [x, y], parameters = [a])
        G = System([b * u^2 + v^2 - 1]; variables = [u, v], parameters = [b])
        @test sols(solve(fix_parameters(F, [2.0]), L, TotalDegree(; show_progress = false))) ==
            sols(solve(fix_parameters(G, [2.0]), L, TotalDegree(; show_progress = false)))

        # Substitution has to reach a non-polynomial equation too.
        H = System([a / x + y - 2]; variables = [x, y], parameters = [a])
        @test Next.polynomials(fix_parameters(H, ComplexF64[3.0])) ==
            [Next.subs(a / x + y - 2, a => 3.0)]
    end

    @testset "regeneration takes polynomial and rational expression input" begin
        @var x y z
        # Polynomial, but built through the expression front-end.
        F = System([x^2 + y^2 - z, x + y + z - 1]; variables = [x, y, z])
        @test Next.degree.(Next.solve(F, Next.Regeneration(; show_progress = false))) == [2]
        @test Next.ncomponents(Next.solve(F, Next.Decomposition(; show_progress = false))) == 1
        W = Next.solve(F, Next.Witness(; show_progress = false))
        @test Next.degree(W) == 2

        # Rational: the u-homotopy carries the denominator, and the hypersurface
        # witness sets come from the numerators.
        G = System([x^2 + y^2 - z, x / (y - 1) + y + z - 1]; variables = [x, y, z])
        WG = Next.solve(G, Next.Regeneration(; show_progress = false))
        @test Next.degree.(WG) == [4]

        # Rebuilding equations cannot take `sqrt` of a variable, in a denominator just
        # as in a numerator.
        for f in (
                sqrt(x) + y - 1, 1 / sqrt(x) + y - 1, x / (1 + sqrt(x)) + y - 1,
                1 / sin(x) + y - 1,
            )
            H = System([f, x * y - z]; variables = [x, y, z])
            @test_throws ArgumentError Next.solve(H, Next.Regeneration(; show_progress = false))
            @test_throws ArgumentError Next.solve(H, Next.Decomposition(; show_progress = false))
        end

        # `intersect` gates its hypersurface argument the same way.
        WF = first(Next.solve(F, Next.Regeneration(; show_progress = false)))
        @test_throws ArgumentError intersect(WF, 1 / sqrt(x), Intersection(; show_progress = false))
        @test_throws ArgumentError intersect(WF, x / (1 + sqrt(y)), Intersection(; show_progress = false))
    end
end
