# Compare solve result counts between HomotopyContinuationNext and HomotopyContinuation v2.

using Test

using HomotopyContinuationNext
using HomotopyContinuation

const Next = HomotopyContinuationNext
const HC = HomotopyContinuation

using DynamicPolynomials: @polyvar
using HomotopyContinuation.ModelKit: @var

@testset "Compare v2: solve result counts" begin

    @testset "quadratic: x²+y-1, xy-2" begin
        @polyvar nx ny
        r3 = Next.solve(
            Next.System([nx^2 + ny - 1, nx * ny - 2]),
            Next.TotalDegree(; show_progress = false),
        )

        @var hx hy
        r2 = HC.solve(HC.ModelKit.System([hx^2 + hy - 1, hx * hy - 2]); start_system = :total_degree, show_progress = false)

        @test Next.nresults(r3) == HC.nresults(r2)
        @test Next.nsingular(r3) == HC.nsingular(r2)
        @test Next.nreal(r3) == HC.nreal(r2)
    end

    @testset "katsura-3" begin
        @polyvar nx0 nx1 nx2 nx3
        F3 = Next.System(
            [
                nx0 + 2nx1 + 2nx2 + 2nx3 - 1,
                nx0^2 + 2nx1^2 + 2nx2^2 + 2nx3^2 - nx0,
                2nx0 * nx1 + 2nx1 * nx2 + 2nx2 * nx3 - nx1,
                nx1^2 + 2nx0 * nx2 + 2nx1 * nx3 - nx2,
            ]
        )
        r3 = Next.solve(F3, Next.TotalDegree(; show_progress = false))

        @var hx0 hx1 hx2 hx3
        F2 = HC.ModelKit.System(
            [
                hx0 + 2hx1 + 2hx2 + 2hx3 - 1,
                hx0^2 + 2hx1^2 + 2hx2^2 + 2hx3^2 - hx0,
                2hx0 * hx1 + 2hx1 * hx2 + 2hx2 * hx3 - hx1,
                hx1^2 + 2hx0 * hx2 + 2hx1 * hx3 - hx2,
            ]
        )
        r2 = HC.solve(F2; start_system = :total_degree, show_progress = false)

        @test Next.nresults(r3) == HC.nresults(r2)
        @test Next.nreal(r3) == HC.nreal(r2)
    end

    @testset "at-infinity: 2 finite + 2 diverging" begin
        @polyvar nx ny
        r3 = Next.solve(
            Next.System([2.3nx^2 + 1.2ny^2 + 3nx - 2ny + 3, 2.3nx^2 + 1.2ny^2 + 5nx + 2ny - 5]),
            Next.TotalDegree(; show_progress = false),
        )

        @var hx hy
        r2 = HC.solve(
            HC.ModelKit.System([2.3hx^2 + 1.2hy^2 + 3hx - 2hy + 3, 2.3hx^2 + 1.2hy^2 + 5hx + 2hy - 5]);
            start_system = :total_degree, show_progress = false,
        )

        @test Next.nresults(r3) == HC.nresults(r2)
        @test Next.nat_infinity(r3) == HC.nat_infinity(r2)
    end

    @testset "singular (x-10)^2" begin
        @polyvar nx
        r3 = Next.solve(
            Next.System([(nx - 10)^2]),
            Next.TotalDegree(; seed = UInt32(42), show_progress = false),
        )

        @var hx
        r2 = HC.solve(HC.ModelKit.System([(hx - 10)^2]); start_system = :total_degree, seed = UInt32(42), show_progress = false)

        @test Next.nresults(r3) == HC.nresults(r2)
        @test Next.nsingular(r3) == HC.nsingular(r2)
    end

    @testset "Hyperbolic 6,6" begin
        @polyvar nx nz
        ny = 1
        r3 = Next.solve(
            Next.System(
                [
                    0.75nx^4 + 1.5nx^2 * ny^2 - 2.5nx^2 * nz^2 + 0.75ny^4 - 2.5ny^2 * nz^2 + 0.75nz^4,
                    10nx^2 * nz + 10ny^2 * nz - 6nz^3,
                ]
            ),
            Next.TotalDegree(; seed = UInt32(1), show_progress = false),
        )

        @var hx hz
        hy = 1
        r2 = HC.solve(
            HC.ModelKit.System(
                [
                    0.75hx^4 + 1.5hx^2 * hy^2 - 2.5hx^2 * hz^2 + 0.75hy^4 - 2.5hy^2 * hz^2 + 0.75hz^4,
                    10hx^2 * hz + 10hy^2 * hz - 6hz^3,
                ]
            );
            start_system = :total_degree, seed = UInt32(1), show_progress = false,
        )

        @test Next.nresults(r3) == HC.nresults(r2)
        @test Next.nsingular(r3) == HC.nsingular(r2)
    end

    @testset "singular multiplicity 3" begin
        @polyvar nx ny
        nz = 1
        r3 = Next.solve(
            Next.System(
                [
                    nx^2 + 2ny^2 + 2im * ny * nz,
                    (18 + 3im) * nx * ny + 7im * ny^2 - (3 - 18im) * nx * nz - 14ny * nz - 7im * nz^2,
                ]
            ),
            Next.TotalDegree(; seed = UInt32(12345), show_progress = false),
        )

        @var hx hy
        hz = 1
        r2 = HC.solve(
            HC.ModelKit.System(
                [
                    hx^2 + 2hy^2 + 2im * hy * hz,
                    (18 + 3im) * hx * hy + 7im * hy^2 - (3 - 18im) * hx * hz - 14hy * hz - 7im * hz^2,
                ]
            );
            start_system = :total_degree, seed = UInt32(12345), show_progress = false,
        )

        @test Next.nresults(r3) == HC.nresults(r2)
        @test Next.nsingular(r3) == HC.nsingular(r2)
        @test Next.nresults(r3; only_nonsingular = true) == HC.nresults(r2; only_nonsingular = true)
    end
end
