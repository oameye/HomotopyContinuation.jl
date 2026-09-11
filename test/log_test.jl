using Test
import HomotopyContinuationNext as Next
using HomotopyContinuationNext:
    Expression, System, CompileMode, @var, differentiate, subs,
    TaylorVector, ComplexDF64
using FixedSizeArrays: FixedSizeArray

const LogFSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const LogFSMat{T} = FixedSizeArray{T, 2, Memory{T}}

log_fsv(v) = LogFSVec{ComplexF64}(collect(ComplexF64, v))
log_fsm(m) = LogFSMat{ComplexF64}(collect(ComplexF64, m))

@testset "Principal logarithm" begin
    @var x

    @testset "Expression frontend" begin
        @test sprint(show, log(x)) == "log(x)"
        @test Next.expr_number(log(Expression(2))) ≈ log(2)
        @test differentiate(log(x), x) == inv(x)
        @test subs(log(x + 2), x => 1) == Expression(log(3))
        @test !Next.is_polynomial(log(x), [x])
        @test Next.degree(log(x), [x]) == -1
    end

    @testset "system backends" begin
        x0 = 0.3 + 0.2im
        z0 = x0 + 2
        truth = log(z0)
        jac_truth = inv(z0)

        for mode in (CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL)
            F = System([log(x + 2)]; variables = [x], compile = mode)

            u = log_fsv(zeros(1))
            U = log_fsm(zeros(1, 1))
            Next.evaluate_and_jacobian!(u, U, F.evaluator, log_fsv([x0]), log_fsv(ComplexF64[]))
            @test u[1] ≈ truth rtol = 1.0e-12
            @test U[1, 1] ≈ jac_truth rtol = 1.0e-12

            u_df64 = LogFSVec{ComplexDF64}(zeros(ComplexDF64, 1))
            x_df64 = LogFSVec{ComplexDF64}(ComplexDF64[x0])
            Next.evaluate!(u_df64, F.evaluator, x_df64, log_fsv(ComplexF64[]))
            @test ComplexF64(u_df64[1]) ≈ truth rtol = 1.0e-12

            # x(λ) = x0 + d1 λ + d2 λ² + d3 λ³. For z(λ) = 2 + x(λ),
            # [λ³] log(z) = d3/z0 - d1*d2/z0² + d1³/(3z0³).
            d1 = 0.2 - 0.1im
            d2 = -0.05 + 0.03im
            d3 = 0.01 + 0.02im
            xdata = LogFSMat{ComplexF64}(reshape(ComplexF64[x0, d1, d2, d3], 4, 1))
            taylor_out = log_fsv(zeros(1))
            Next.taylor!(
                taylor_out, Val(3), F.evaluator,
                TaylorVector{4, ComplexF64}(xdata), log_fsv(ComplexF64[]),
            )
            taylor_truth = d3 / z0 - d1 * d2 / z0^2 + d1^3 / (3z0^3)
            @test taylor_out[1] ≈ taylor_truth atol = 1.0e-12
        end
    end
end
