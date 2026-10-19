using Test
using HomotopyContinuation

@testset "Principal logarithm" begin
    @var x

    @testset "Expression frontend" begin
        @test sprint(show, log(x)) == "log(x)"
        @test log(Expression(2)) == Expression(log(2))
        @test differentiate(log(x), x) == inv(x)
        @test subs(log(x + 2), x => 1) == Expression(log(3))
        @test degree(log(x), [x]) == -1
    end

    @testset "value and Jacobian agree across compilation backends" begin
        x0 = 0.3 + 0.2im
        truth = log(x0 + 2)
        jac_truth = inv(x0 + 2)

        for mode in (CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL)
            F = System([log(x + 2)]; variables = [x], compile = mode)
            @test only(evaluate(F, [x0])) ≈ truth rtol = 1.0e-12
            @test jacobian(F, [x0])[1, 1] ≈ jac_truth rtol = 1.0e-12
        end
    end
end
