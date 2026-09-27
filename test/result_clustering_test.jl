using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar

@testset "public solution clustering" begin
    @testset "nearby regular roots stay separate under loose reclustering tolerance" begin
        @polyvar x
        F = System([(x - 1) * (x - 101 // 100)]; variables = [x])
        raw = solve(
            F,
            TotalDegree(; seed = UInt32(0x51a7), show_progress = false),
            Serial(),
        )

        @test nsolutions(raw) == 2
        @test count(is_success, path_results(raw)) == 2
        @test all(is_nonsingular, path_results(raw))
        @test sort([real(solution(pr)[1]) for pr in path_results(raw)]) ≈ [1.0, 1.01] atol = 1.0e-8

        # The roots are much closer than this tolerance, but regular endpoints
        # are distinct solutions and must not be merged by geometric proximity.
        r = recluster(raw; atol = 0.1, rtol = 0.0)
        @test nsolutions(r) == 2
        @test sort(length.(clusters(r))) == [1, 1]
        @test all(==(1), multiplicity.(path_results(r)))
    end

    @testset "recluster groups a symmetry orbit without changing root multiplicity" begin
        @polyvar x y
        F = System([x^2 - 1, y^2 - 1]; variables = [x, y])
        r = solve(
            F,
            TotalDegree(; seed = UInt32(0x2c71), show_progress = false),
            Serial(),
        )
        @test nsolutions(r) == 4
        @test sort(length.(clusters(r))) == [1, 1, 1, 1]

        swap(v) = ([v[2], v[1]],)
        rs = recluster(r; group_action = swap)
        @test sort(length.(clusters(rs))) == [1, 1, 2]
        @test all(==(1), multiplicity.(path_results(rs)))

        paths = path_results(rs)
        i = findfirst(pr -> isapprox(solution(pr), ComplexF64[1, -1]; atol = 1.0e-8), paths)
        j = findfirst(pr -> isapprox(solution(pr), ComplexF64[-1, 1]; atol = 1.0e-8), paths)
        @test i !== nothing
        @test j !== nothing
        @test cluster_of(rs, i) == cluster_of(rs, j)
        @test length(cluster_of(rs, i)) == 2

        # Fixed points of the swap remain singleton clusters.
        k = findfirst(pr -> isapprox(solution(pr), ComplexF64[1, 1]; atol = 1.0e-8), paths)
        @test k !== nothing
        @test length(cluster_of(rs, k)) == 1

        # Reapplying the same public action is idempotent.
        twice = recluster(rs; group_action = swap)
        @test sort(length.(clusters(twice))) == [1, 1, 2]
        @test all(==(1), multiplicity.(path_results(twice)))
    end

    @testset "a regular path onto a singular root joins its cluster" begin
        @polyvar x y
        # (1, 1) is both a total-degree start point and the double root, so one
        # path never moves and is flagged singular while the other arrives
        # through the regular tracker without the flag.
        r = solve(System([(x - 1)^2, y - 1]), TotalDegree(; seed = UInt32(1), show_progress = false))
        @test count(is_success, path_results(r)) == 2
        @test sort(length.(clusters(r))) == [2]
        @test all(is_singular, path_results(r))
        @test nsingular(r) == 1
        @test nsolutions(r) == 0
    end

    @testset "double root at x = 2 forms one cluster" begin
        @polyvar x y
        F = System([(x - 2)^2, y - 1])
        r = solve(F, TotalDegree(; seed = UInt32(1), show_progress = false), Serial())
        @test all(p -> maximum(abs.(solution(p) .- [2, 1])) < 1.0e-8, path_results(r))
        @test count(is_success, path_results(r)) == 2
        @test sort(length.(clusters(r))) == [2]
    end
end
