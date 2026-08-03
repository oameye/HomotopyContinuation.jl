using Test
using HomotopyContinuationNext: @polyvar, @var, System, Expression
using HomotopyContinuationNextCertification: AcbInterpreter, acb_execute!, setprecision!,
    acb_op_exp!, acb_op_sinh!, acb_op_cosh!, acb_op_tan!, acb_op_tanh!, acb_op_asin!,
    acb_op_acos!, acb_op_pow!, acb_op_sqrt!
import DynamicPolynomials as DP
import Arblib

# Systems chosen so the compiled tape covers the fused instructions (mul3/mul4,
# muladd/mulsub/submul, sqr/cb/pow_int) as well as plain add and mul.
function acb_test_systems()
    @polyvar x y z a b
    return [
        (
            "cubic pair",
            [x^3 * y - x * y^2 + x^2 - 3, x^2 * y + y^3 - x + 2],
            [x, y], typeof(x)[],
        ),
        (
            "parametric",
            [x^3 * y - x * y^2 + a * x^2 - 3 * b, x^2 * y + y^3 - a * x + 2 * b],
            [x, y], [a, b],
        ),
        (
            "fused products",
            [
                x * y * z + x^2 * y^2 * z^2 - a,
                (x + y + z + a)^2 - x^5,
                x * y + z * a - b * x^3 * y,
            ],
            [x, y, z], [a, b],
        ),
    ]
end

mp_at(f, vars, x, params, p) = isempty(params) ? f(vars => x) : f(vars => x, params => p)

# Tapes carrying inv/invsqr from division and the unary sqrt/sin/cos, with value
# and Jacobian written out by hand.
function nonpolynomial_acb_systems()
    @var x y a b
    return [
        (
            "rational",
            [a / x^2 + b * y, x * y - a], [x, y], [a, b],
            (z, p) -> [p[1] / z[1]^2 + p[2] * z[2], z[1] * z[2] - p[1]],
            (z, p) -> [
                -2 * p[1] / z[1]^3 p[2]
                z[2] z[1]
            ],
        ),
        (
            "transcendental",
            [sqrt(a) * x + sin(y), cos(x * y) - b], [x, y], [a, b],
            (z, p) -> [sqrt(p[1]) * z[1] + sin(z[2]), cos(z[1] * z[2]) - p[2]],
            (z, p) -> [
                sqrt(p[1]) cos(z[2])
                -z[2] * sin(z[1] * z[2]) -z[1] * sin(z[1] * z[2])
            ],
        ),
        (
            "exp, tan and hyperbolics",
            [exp(x) + tan(y) - a, sinh(x) * cosh(y) + tanh(b * x)], [x, y], [a, b],
            (z, p) -> [
                exp(z[1]) + tan(z[2]) - p[1],
                sinh(z[1]) * cosh(z[2]) + tanh(p[2] * z[1]),
            ],
            (z, p) -> [
                exp(z[1]) 1 + tan(z[2])^2
                cosh(z[1]) * cosh(z[2]) + p[2] * (1 - tanh(p[2] * z[1])^2) sinh(z[1]) *
                    sinh(z[2])
            ],
        ),
        (
            "inverse trig and a fractional power",
            [asin(x / 4) + (y + 4)^(3 // 2), acos(y / 4) + (x + 4)^(-4 // 3)],
            [x, y], Expression[],
            (z, p) -> [
                asin(z[1] / 4) + (z[2] + 4)^(3 / 2),
                acos(z[2] / 4) + (z[1] + 4)^(-4 / 3),
            ],
            (z, p) -> [
                1 / (4 * sqrt(1 - (z[1] / 4)^2)) (3 / 2) * (z[2] + 4)^(1 / 2)
                (-4 / 3) * (z[1] + 4)^(-7 / 3) -1 / (4 * sqrt(1 - (z[2] / 4)^2))
            ],
        ),
    ]
end

@testset "AcbInterpreter" begin
    @testset "$name" for (name, polys, vars, params) in acb_test_systems()
        F = System(polys; variables = vars, parameters = params)
        m, n, r = length(polys), length(vars), length(params)
        prec = 256

        eval_interp = AcbInterpreter(F._interp_f64.sequence; prec = prec)
        jac_interp = AcbInterpreter(F._interp_jac.sequence; prec = prec)
        @test occursin("AcbInterpreter", sprint(show, eval_interp))

        xv = ComplexF64[0.3 + 0.2im, -1.1 + 0.4im, 0.7 - 0.9im][1:n]
        pv = ComplexF64[1.7 - 0.5im, -0.6 + 1.3im][1:r]
        acb_p = r == 0 ? nothing : pv

        u = Arblib.AcbRefMatrix(m, 1; prec = prec)
        acb_execute!(u, eval_interp, xv, acb_p)
        truth = ComplexF64[mp_at(f, vars, xv, params, pv) for f in polys]
        @test ComplexF64.(u) ≈ reshape(truth, m, 1) rtol = 1.0e-12

        U = Arblib.AcbRefMatrix(m, n; prec = prec)
        acb_execute!(nothing, U, jac_interp, xv, acb_p)
        truth_J = ComplexF64[
            mp_at(DP.differentiate(polys[i], vars[j]), vars, xv, params, pv)
                for i in 1:m, j in 1:n
        ]
        @test ComplexF64.(U) ≈ truth_J rtol = 1.0e-12

        # The Jacobian can also be requested together with the value.
        u2 = Arblib.AcbRefMatrix(m, 1; prec = prec)
        U2 = Arblib.AcbRefMatrix(m, n; prec = prec)
        acb_execute!(u2, U2, jac_interp, xv, acb_p)
        @test ComplexF64.(U2) ≈ truth_J rtol = 1.0e-12
        @test ComplexF64.(u2) ≈ reshape(truth, m, 1) rtol = 1.0e-12

        @testset "low precision brackets high precision" begin
            low = AcbInterpreter(F._interp_f64.sequence; prec = 32)
            u_low = Arblib.AcbRefMatrix(m, 1; prec = 32)
            acb_execute!(u_low, low, xv, acb_p)
            for i in 1:m
                @test Arblib.contains(u_low[i], u[i])
            end

            radius(v, i) = Float64(Arblib.radius(Arblib.realref(v[i])))
            # Re-viewing the same tape at higher precision tightens the balls.
            setprecision!(low, prec)
            u_high = Arblib.AcbRefMatrix(m, 1; prec = prec)
            acb_execute!(u_high, low, xv, acb_p)
            for i in 1:m
                @test radius(u_high, i) <= radius(u_low, i)
                @test Arblib.contains(u_low[i], u_high[i])
            end
        end
    end

    @testset "non-polynomial tape: $name" for (name, exprs, vars, params, ref, jac_ref) in
        nonpolynomial_acb_systems()

        F = System(exprs; variables = vars, parameters = params)
        m, n, r = length(exprs), length(vars), length(params)
        prec = 256

        eval_interp = AcbInterpreter(F._interp_f64.sequence; prec = prec)
        jac_interp = AcbInterpreter(F._interp_jac.sequence; prec = prec)

        # Off the branch cut of `sqrt` and away from the poles.
        xv = ComplexF64[0.8 + 0.2im, 1.3 - 0.4im]
        pv = ComplexF64[1.7 - 0.5im, -0.6 + 1.3im]

        u = Arblib.AcbRefMatrix(m, 1; prec = prec)
        acb_execute!(u, eval_interp, xv, pv)
        @test ComplexF64.(u) ≈ reshape(ref(xv, pv), m, 1) rtol = 1.0e-12

        U = Arblib.AcbRefMatrix(m, n; prec = prec)
        acb_execute!(nothing, U, jac_interp, xv, pv)
        @test ComplexF64.(U) ≈ jac_ref(xv, pv) rtol = 1.0e-12

        # The ball at 32 bits must still contain the 256-bit value.
        low = AcbInterpreter(F._interp_f64.sequence; prec = 32)
        u_low = Arblib.AcbRefMatrix(m, 1; prec = 32)
        acb_execute!(u_low, low, xv, pv)
        for i in 1:m
            @test Arblib.contains(u_low[i], u[i])
        end
    end

    # The branch-cut rejections, which the systems above stay away from.
    @testset "transcendental op kernels" begin
        prec = 256
        ball(re, im, err) = (
            z = Arblib.Acb(re, im; prec);
            Arblib.add_error!(z, Arblib.Mag(err));
            z
        )
        t = Arblib.Acb(0; prec)
        for (op!, f) in (
                (acb_op_exp!, exp), (acb_op_sinh!, sinh), (acb_op_cosh!, cosh),
                (acb_op_tan!, tan), (acb_op_tanh!, tanh),
                (acb_op_asin!, asin), (acb_op_acos!, acos),
            )
            op!(t, ball(0.7, -1.3, 0.0), ())
            @test ComplexF64(t) ≈ f(0.7 - 1.3im) rtol = 1.0e-12
        end
        acb_op_pow!(t, ball(1.7, 0.4, 0.0), Arblib.Acb(1.5, 0.0; prec), ())
        @test ComplexF64(t) ≈ (1.7 + 0.4im)^1.5 rtol = 1.0e-12

        # These are the widths Arb itself still answers on, with a sound but
        # discontinuous enclosure the Krawczyk hypotheses cannot use.
        for op! in (acb_op_asin!, acb_op_acos!)
            op!(t, ball(2.0, 0.0, 1.0e-12), ())
            @test !isfinite(t)
            op!(t, ball(-2.0, 0.0, 1.0e-12), ())
            @test !isfinite(t)
            op!(t, ball(0.3, 0.0, 1.0e-12), ())
            @test isfinite(t)
        end
        for op! in (acb_op_sqrt!, (u, x, m) -> acb_op_pow!(u, x, Arblib.Acb(1.5), m))
            op!(t, ball(-2.0, 0.0, 1.0e-12), ())
            @test !isfinite(t)
            op!(t, ball(2.0, 0.0, 1.0e-12), ())
            @test isfinite(t)
        end
    end
end
