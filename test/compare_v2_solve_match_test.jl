# Compare that actual solution values match between HomotopyContinuationNext and HomotopyContinuation v2.

using Test

using HomotopyContinuationNext
using HomotopyContinuation

const Next = HomotopyContinuationNext
const HC = HomotopyContinuation

using DynamicPolynomials: @polyvar
using HomotopyContinuation.ModelKit: @var

@testset "Compare v2: solutions match" begin

    @testset "x²-1, y²-4 solutions agree" begin
        @polyvar nx ny
        r3 = Next.solve(Next.System([nx^2 - 1, ny^2 - 4]); show_progress = false)

        @var hx hy
        r2 = HC.solve(HC.ModelKit.System([hx^2 - 1, hy^2 - 4]); show_progress = false)

        sols3 = sort(Next.real_solutions(r3); by = s -> (s[1], s[2]))
        sols2 = sort(HC.real_solutions(r2); by = s -> (s[1], s[2]))

        @test length(sols3) == length(sols2)
        for (s3, s2) in zip(sols3, sols2)
            @test s3 ≈ s2 atol = 1.0e-8
        end
    end

    @testset "parameter homotopy: solutions agree" begin
        @polyvar nx ny na
        F3 = Next.System([nx^2 - na, ny^2 - na]; parameters = [na])
        F3_fixed = Next.System([nx^2 - 1, ny^2 - 1])
        starts3 = Next.solutions(Next.solve(F3_fixed; show_progress = false))
        r3 = Next.solve(F3, starts3, [1.0], [4.0]; show_progress = false)

        @var hx hy ha
        F2 = HC.ModelKit.System([hx^2 - ha, hy^2 - ha]; parameters = [ha])
        r2_init = HC.solve(HC.ModelKit.System([hx^2 - 1, hy^2 - 1]); show_progress = false)
        starts2 = HC.solutions(r2_init)
        r2 = HC.solve(F2, starts2; start_parameters = [1.0], target_parameters = [4.0], show_progress = false)

        @test Next.nresults(r3) == HC.nresults(r2)
        @test Next.nreal(r3) == HC.nreal(r2)

        sols3 = sort(Next.real_solutions(r3); by = s -> (s[1], s[2]))
        sols2 = sort(HC.real_solutions(r2); by = s -> (s[1], s[2]))
        @test length(sols3) == length(sols2)
        for (s3, s2) in zip(sols3, sols2)
            @test s3 ≈ s2 atol = 1.0e-6
        end
    end
end
