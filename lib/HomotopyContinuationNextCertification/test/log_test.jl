using Test
using HomotopyContinuationNext: @var, System
using HomotopyContinuationNextCertification:
    AcbInterpreter, acb_execute!, acb_op_log!,
    Certification, certify, ncertified, is_certified, certificates
import Arblib

@testset "Principal logarithm certification" begin
    prec = 256

    @testset "Acb tape value and Jacobian" begin
        @var x
        F = System([log(x + 2)]; variables = [x])
        xv = ComplexF64[0.3 + 0.2im]

        eval_interp = AcbInterpreter(F._interp_f64.sequence; prec = prec)
        jac_interp = AcbInterpreter(F._interp_jac.sequence; prec = prec)

        u = Arblib.AcbRefMatrix(1, 1; prec = prec)
        acb_execute!(u, eval_interp, xv, nothing)
        @test ComplexF64(u[1]) ≈ log(xv[1] + 2) rtol = 1.0e-12

        U = Arblib.AcbRefMatrix(1, 1; prec = prec)
        acb_execute!(nothing, U, jac_interp, xv, nothing)
        @test ComplexF64(U[1, 1]) ≈ inv(xv[1] + 2) rtol = 1.0e-12
    end

    @testset "Acb principal branch cut" begin
        ball(re, im, err) = (
            z = Arblib.Acb(re, im; prec = prec);
            Arblib.add_error!(z, Arblib.Mag(err));
            z
        )
        t = Arblib.Acb(0; prec = prec)

        acb_op_log!(t, ball(0.7, -1.3, 0.0), ())
        @test ComplexF64(t) ≈ log(0.7 - 1.3im) rtol = 1.0e-12

        acb_op_log!(t, ball(-2.0, 0.0, 1.0e-12), ())
        @test !isfinite(t)

        acb_op_log!(t, ball(-2.0, 0.2, 1.0e-12), ())
        @test isfinite(t)
    end

    @testset "end-to-end certification" begin
        @var x
        F = System([log(x)]; variables = [x])
        result = certify(
            F, [ComplexF64[1.0]],
            Certification(; show_progress = false),
        )
        @test ncertified(result) == 1
        @test is_certified(only(certificates(result)))
    end
end
