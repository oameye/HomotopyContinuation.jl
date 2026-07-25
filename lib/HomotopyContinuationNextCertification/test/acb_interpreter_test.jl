using Test
using HomotopyContinuationNext: @polyvar, System
using HomotopyContinuationNextCertification: AcbInterpreter, acb_execute!, setprecision!
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
end
