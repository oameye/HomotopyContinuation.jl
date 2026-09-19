using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar

function parameter_homotopy()
    @polyvar x y a b
    F = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
    return ParameterHomotopy(F, [1.0, 0.0], [2.0, 4.0])
end

@testset "public path iterator" begin
    H = parameter_homotopy()
    @test eltype(iterator(H, [1.0, 1.0], 1.0, 0.0)) ===
        Tuple{Vector{ComplexF64}, Float64}
    @test first(iterator(H, [1.0, 1.0], 1.0, 0.0)) isa
        Tuple{Vector{ComplexF64}, Float64}

    path = collect(iterator(H, [1.0, 1.0], 1.0, 0.0))
    @test first(path)[2] == 1.0
    @test last(path)[2] == 0.0
    @test issorted([t for (_, t) in path]; rev = true)
    @test last(path)[1] ≈ ComplexF64[sqrt(2), -sqrt(2)] atol = 1.0e-8

    @testset "a smaller step size yields more points" begin
        fine = iterator(
            parameter_homotopy(), [1.0, 1.0], 1.0, 0.0;
            tracker_options = TrackerOptions(; max_step_size = 0.01),
        )
        @test length(collect(fine)) >= 101
    end

    @testset "linear path steps at the requested size" begin
        @polyvar z c
        G = System([z - c]; variables = [z], parameters = [c])
        Hlinear = ParameterHomotopy(G, [1.0], [2.0])
        xs = Vector{ComplexF64}[]
        for (x, _) in iterator(
                Hlinear, [1.0], 1.0, 0.0;
                tracker_options = TrackerOptions(; max_step_size = 0.015625),
            )
            push!(xs, x)
        end
        @test length(xs) >= length(1:0.015625:2)
        @test only(last(xs)) ≈ 2.0 atol = 1.0e-10
    end

    @testset "complex endpoints yield complex t" begin
        iter = iterator(parameter_homotopy(), [1.0, 1.0], 1.0 + 0.0im, 0.0 + 0.0im)
        @test eltype(iter) === Tuple{Vector{ComplexF64}, ComplexF64}
        (_, t) = first(iter)
        @test t isa ComplexF64
    end
end

@testset "public path_info" begin
    H = parameter_homotopy()
    info = path_info(H, [1.0, 1.0], 1.0, 0.0)
    @test info isa PathInfo
    @test info isa AbstractVector{PathStep}
    @test is_success(info)
    @test !isempty(info)
    @test steps(info) == length(info) == length(collect(info))
    @test accepted_steps(info) + rejected_steps(info) == steps(info)
    @test info.n_factorizations > 0
    @test info.n_ldivs > 0
    @test first(info).s == 1.0
    @test last(info) === info[end] === info[length(info)]
    @test all(step -> step.cond >= 0, info)

    @test sprint(show, info) ==
        "PathInfo($(steps(info)) steps, $(accepted_steps(info)) ✓ / " *
        "$(rejected_steps(info)) ✗, $(info.return_code))"

    out = sprint(show, MIME("text/plain"), info)
    @test occursin("PathInfo:", out)
    @test occursin("TRACKER_SUCCESS", out)
    @test count(==('│'), out) == 14 * (steps(info) + 1)
    @test !isempty(sprint(path_table, info))

    @testset "a limited display elides the middle rows" begin
        long = path_info(
            parameter_homotopy(), [1.0, 1.0], 1.0, 0.0;
            tracker_options = TrackerOptions(; max_step_size = 0.01),
        )
        @test steps(long) > 20
        out = sprint(
            show, MIME("text/plain"), long;
            context = (:limit => true, :displaysize => (14, 200)),
        )
        @test occursin("⋮", out)
        @test count(==('│'), out) < 14 * (steps(long) + 1)
        @test !occursin("⋮", sprint(path_table, long))
    end

    @testset "tracking the same path twice gives the same table" begin
        again = path_info(parameter_homotopy(), [1.0, 1.0], 1.0, 0.0)
        @test map(step -> step.s, again) == map(step -> step.s, info)
        @test map(step -> step.accepted, again) == map(step -> step.accepted, info)
    end
end
