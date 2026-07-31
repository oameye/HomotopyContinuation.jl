using Test
using HomotopyContinuationNext
import HomotopyContinuationNext as HCN
using HomotopyContinuationNext: Serial, Threaded, System, TotalDegree, is_homogeneous,
    ParameterHomotopy, StraightLineHomotopy, AffineChartHomotopy, AbstractHomotopy,
    HomotopyEvaluator, _clone_system_evaluator, _clone_homotopy, fix_parameters,
    FixedParameterSystem, path_results, result_iterator, on_affine_chart, evaluate!,
    evaluate_and_jacobian!, taylor!, set_solution!, get_solution!, TaylorVector,
    ComplexDF64, FSVec, FSMat
using DynamicPolynomials: @polyvar
using CommonSolve: init, solve!
using LinearAlgebra: norm
using Random: MersenneTwister

# A caller's homotopy with no `_clone_homotopy` method. Results go through
# `scratch`, so a task sharing one with another task would race.
struct _BufferedHomotopy <: AbstractHomotopy
    inner::StraightLineHomotopy
    scratch::FSVec{ComplexF64}
end

_BufferedHomotopy(H::StraightLineHomotopy) =
    _BufferedHomotopy(H, FSVec{ComplexF64}(zeros(ComplexF64, first(size(H)))))

Base.size(H::_BufferedHomotopy) = size(H.inner)

function HCN.evaluate!(
        u::FSVec{ComplexF64}, H::_BufferedHomotopy,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )
    evaluate!(H.scratch, H.inner, x, t)
    copyto!(u, H.scratch)
    return nothing
end

function HCN.evaluate!(
        u::FSVec{ComplexF64}, H::_BufferedHomotopy,
        x::FSVec{ComplexDF64}, t::ComplexF64,
    )
    evaluate!(H.scratch, H.inner, x, t)
    copyto!(u, H.scratch)
    return nothing
end

HCN.evaluate_and_jacobian!(
    u::FSVec{ComplexF64}, U::FSMat{ComplexF64}, H::_BufferedHomotopy,
    x::FSVec{ComplexF64}, t::ComplexF64,
) = evaluate_and_jacobian!(u, U, H.inner, x, t)

HCN.taylor!(
    u::FSVec{ComplexF64}, ::Val{1}, H::_BufferedHomotopy,
    x::FSVec{ComplexF64}, t::ComplexF64,
) = taylor!(u, Val(1), H.inner, x, t)

HCN.taylor!(
    u::FSVec{ComplexF64}, ::Val{2}, H::_BufferedHomotopy,
    tx::TaylorVector{3, ComplexF64}, t::ComplexF64,
) = taylor!(u, Val(2), H.inner, tx, t)

HCN.taylor!(
    u::FSVec{ComplexF64}, ::Val{3}, H::_BufferedHomotopy,
    tx::TaylorVector{4, ComplexF64}, t::ComplexF64,
) = taylor!(u, Val(3), H.inner, tx, t)

HCN.set_solution!(
    x::FSVec{ComplexF64}, ::_BufferedHomotopy, y::FSVec{ComplexF64}, ::ComplexF64,
) = (copyto!(x, y); nothing)

HCN.get_solution!(
    out::FSVec{ComplexF64}, ::_BufferedHomotopy, x::FSVec{ComplexF64}, ::ComplexF64,
) = (copyto!(out, x); nothing)

# Projective representative scaled so that its largest entry is 1.
_normalize_projective(x) = x ./ x[argmax(abs.(x))]

# Largest distance from a tracked endpoint to the nearest reference solution.
function max_distance(tracked, reference)
    isempty(tracked) && return Inf
    return maximum(minimum(norm(a - b) for b in reference) for a in tracked)
end

@testset "Homotopy solve" begin

    @testset "start-target: parameter-fixed systems" begin
        @polyvar x y a b
        f = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
        G = fix_parameters(f, [1, 0])
        F = fix_parameters(f, [2, 4])
        r = solve(G, F, [[1.0, 1.0]], Continuation(; show_progress = false))
        @test nsolutions(r) == 1
        @test max_distance(solutions(r), [[sqrt(2), -sqrt(2)]]) < 1.0e-8
        @test r.tracked_paths == 1

        # A composition has no equations to substitute into, so it fixes its
        # parameters at the evaluator level and reaches the route that way.
        h = System([x^2 - a, y - b]; variables = [x, y], parameters = [a, b])
        C = fix_parameters(h ∘ System([x, y]; variables = [x, y]), [4, 3])
        @test C isa FixedParameterSystem
        rc = solve(
            fix_parameters(h, [1, 2]),
            C,
            [[1.0, 2.0]],
            Continuation(; show_progress = false),
        )
        @test nsolutions(rc) == 1
        @test max_distance(solutions(rc), [[2.0, 3.0]]) < 1.0e-8
    end

    @testset "start-target: agrees with a direct solve" begin
        @polyvar x y
        G = System([x^2 + y^2 - 5, x * y - 2]; variables = [x, y])
        F = System([x^2 + 3y^2 - 7, x * y + x - 3]; variables = [x, y])
        starts = solve(G, TotalDegree(; show_progress = false))
        @test nsolutions(starts) == 4

        reference = solutions(solve(F, TotalDegree(; show_progress = false)))
        r = solve(G, F, starts, Continuation(; show_progress = false))
        @test nsolutions(r) == 4
        @test max_distance(solutions(r), reference) < 1.0e-8

        # Start solutions in every accepted shape.
        @test nsolutions(solve(G, F, solutions(starts), Continuation(; show_progress = false))) == 4
        @test nsolutions(
            solve(
                G,
                F,
                result_iterator(G, TotalDegree()),
                Continuation(; show_progress = false),
            ),
        ) == 4
    end

    @testset "start-target: seed determines the paths" begin
        @polyvar x y
        G = System([x^2 + y^2 - 5, x * y - 2]; variables = [x, y])
        F = System([x^2 + 3y^2 - 7, x * y + x - 3]; variables = [x, y])
        starts = solutions(solve(G, TotalDegree(; show_progress = false)))
        seed = UInt32(0x1234)
        serial = solve(G, F, starts, Continuation(; seed = seed, show_progress = false), Serial())
        threaded = solve(G, F, starts, Continuation(; seed = seed, show_progress = false), Threaded())
        for (u, v) in zip(path_results(serial), path_results(threaded))
            for field in fieldnames(typeof(u))
                @test isequal(getfield(u, field), getfield(v, field))
            end
        end
        other = solve(
            G,
            F,
            starts,
            Continuation(; seed = UInt32(0x99), show_progress = false),
            Serial(),
        )
        @test nsolutions(other) == nsolutions(serial)
    end

    @testset "start-target: projective" begin
        @polyvar x y z
        mons = [x^2, x * y, x * z, y^2, y * z, z^2]
        rng = MersenneTwister(3)
        coeffs = [randn(rng, 6) for _ in 1:4]
        G = System([sum(coeffs[1] .* mons), sum(coeffs[2] .* mons)])
        F = System([sum(coeffs[3] .* mons), sum(coeffs[4] .* mons)])
        @test is_homogeneous(G) && is_homogeneous(F)

        starts = solutions(solve(G, TotalDegree(; show_progress = false)))
        @test length(starts) == 4
        # Any representative of the projective point is accepted: the route
        # places the start points on its chart.
        scaled = [(3.7 - 1.2im) .* s for s in starts]

        reference = _normalize_projective.(solutions(solve(F, TotalDegree(; show_progress = false))))
        r = solve(G, F, scaled, Continuation(; show_progress = false))
        @test nsolutions(r) == 4
        @test max_distance(_normalize_projective.(solutions(r)), reference) < 1.0e-8
    end

    @testset "start-target: overdetermined" begin
        @polyvar x y
        G = System([x^2 - 1, y^2 - 1, x - y]; variables = [x, y])
        F = System([x^2 - 4, y^2 - 4, x - y]; variables = [x, y])
        starts = [[1.0, 1.0], [-1.0, -1.0]]
        r = solve(G, F, starts, Continuation(; show_progress = false))
        @test nsolutions(r) == 2
        @test max_distance(solutions(r), [[2.0, 2.0], [-2.0, -2.0]]) < 1.0e-8
    end

    @testset "start-target: rejected input" begin
        @polyvar x y a
        parametric = System([x^2 - a, y - 1]; variables = [x, y], parameters = [a])
        square = System([x^2 - 1, y - 1]; variables = [x, y])
        err = try
            solve(parametric, square, [[1.0, 1.0]], Continuation(; show_progress = false))
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("fix_parameters", err.msg)

        err = try
            solve(
                square,
                System([x^2 - 4]; variables = [x]),
                [[1.0, 1.0]],
                Continuation(; show_progress = false),
            )
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("size", err.msg)

        err = try
            solve(
                System([x^2 - y^2]; variables = [x, y]),
                System([x^2 - y^2 - 1]; variables = [x, y]),
                [[1.0, 1.0]],
                Continuation(; show_progress = false),
            )
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("homogeneous", err.msg)

        underdetermined = System([x * y - 1]; variables = [x, y])
        err = try
            solve(
                underdetermined,
                System([x * y - 4]; variables = [x, y]),
                [[1.0, 1.0]],
                Continuation(; show_progress = false),
            )
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("positive-dimensional", err.msg)

        err = try
            solve(square, square, [[1.0]], Continuation(; show_progress = false))
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("start solution has length", err.msg)
    end

    @testset "solve(H, starts)" begin
        @polyvar x y a b
        f = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
        H = ParameterHomotopy(f, [1, 0], [2, 4])
        r = solve(H, [[1.0, 1.0]], Continuation(; show_progress = false))
        @test nsolutions(r) == 1
        @test max_distance(solutions(r), [[sqrt(2), -sqrt(2)]]) < 1.0e-8

        # The same paths the typed parameter-homotopy route tracks.
        typed = solve(
            f,
            [[1.0, 1.0]],
            [1, 0],
            [2, 4],
            Continuation(; show_progress = false),
            Serial(),
        )
        @test max_distance(solutions(r), solutions(typed)) < 1.0e-8

        @polyvar u v
        G = System([u^2 + v^2 - 5, u * v - 2]; variables = [u, v])
        F = System([u^2 + 3v^2 - 7, u * v + u - 3]; variables = [u, v])
        SL = StraightLineHomotopy(
            _clone_system_evaluator(G), _clone_system_evaluator(F); γ = complex(0.6, 0.8),
        )
        starts = solutions(solve(G, TotalDegree(; show_progress = false)))
        reference = solutions(solve(F, TotalDegree(; show_progress = false)))
        rsl = solve(SL, starts, Continuation(; show_progress = false))
        @test nsolutions(rsl) == 4
        @test max_distance(solutions(rsl), reference) < 1.0e-8
    end

    @testset "solve(H, starts): affine chart" begin
        @polyvar x y z
        mons = [x^2, x * y, x * z, y^2, y * z, z^2]
        rng = MersenneTwister(5)
        coeffs = [randn(rng, 6) for _ in 1:4]
        G = System([sum(coeffs[1] .* mons), sum(coeffs[2] .* mons)])
        F = System([sum(coeffs[3] .* mons), sum(coeffs[4] .* mons)])
        H = on_affine_chart(
            StraightLineHomotopy(
                _clone_system_evaluator(G), _clone_system_evaluator(F);
                γ = complex(0.6, 0.8),
            ),
        )
        @test H isa AffineChartHomotopy
        @test size(H) == (3, 3)

        starts = [(2.5 + 0.5im) .* s for s in solutions(solve(G, TotalDegree(; show_progress = false)))]
        reference = _normalize_projective.(solutions(solve(F, TotalDegree(; show_progress = false))))
        r = solve(H, starts, Continuation(; show_progress = false))
        @test nsolutions(r) == 4
        @test max_distance(_normalize_projective.(solutions(r)), reference) < 1.0e-8
    end

    @testset "solve(H, starts): threaded" begin
        @polyvar x y
        G = System([x^2 + y^2 - 5, x * y - 2]; variables = [x, y])
        F = System([x^2 + 3y^2 - 7, x * y + x - 3]; variables = [x, y])
        H = StraightLineHomotopy(
            _clone_system_evaluator(G), _clone_system_evaluator(F); γ = complex(0.6, 0.8),
        )
        starts = solutions(solve(G, TotalDegree(; show_progress = false)))

        serial = solve(H, starts, Continuation(; seed = UInt32(4), show_progress = false), Serial())
        threaded = solve(
            H,
            starts,
            Continuation(; seed = UInt32(4), show_progress = false),
            Threaded(),
        )
        @test nsolutions(threaded) == 4
        for (u, v) in zip(path_results(serial), path_results(threaded))
            for field in fieldnames(typeof(u))
                @test isequal(getfield(u, field), getfield(v, field))
            end
        end

        # Every task rebuilds the homotopy, so the caller's own buffers are
        # untouched and it can be tracked again afterwards.
        again = solve(H, starts, Continuation(; seed = UInt32(4), show_progress = false), Serial())
        @test solutions(again) == solutions(serial)
    end

    @testset "_clone_homotopy" begin
        @polyvar x y a b
        f = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
        for H in (
                ParameterHomotopy(f, [1, 0], [2, 4]),
                StraightLineHomotopy(
                    _clone_system_evaluator(System([x^2 - 1, x * y - 1])),
                    _clone_system_evaluator(System([x^2 - 4, x * y - 2]));
                    γ = complex(0.6, 0.8),
                ),
            )
            C = _clone_homotopy(H)
            @test typeof(C) === typeof(H)
            @test C !== H
            @test size(C) == size(H)
            u, v = FSVec{ComplexF64}(zeros(ComplexF64, size(H)[1])),
                FSVec{ComplexF64}(zeros(ComplexF64, size(H)[1]))
            z = FSVec{ComplexF64}(ComplexF64[0.7, -1.3])
            t = complex(0.4)
            evaluate!(u, H, z, t)
            evaluate!(v, C, z, t)
            @test u == v
        end

        # A chart wrapper clones through to the homotopy it wraps.
        H = on_affine_chart(
            StraightLineHomotopy(
                _clone_system_evaluator(System([x^2 - 1, x * y - 1])),
                _clone_system_evaluator(System([x^2 - 4, x * y - 2])); γ = complex(0.6, 0.8),
            ),
        )
        C = _clone_homotopy(H)
        @test C isa AffineChartHomotopy
        @test C.chart == H.chart
        @test C.homotopy !== H.homotopy
    end

    @testset "solve(H, starts): rejected input" begin
        @polyvar x y a b
        f = System([x^2 - a, x * y - a + b]; variables = [x, y], parameters = [a, b])
        H = ParameterHomotopy(f, [1, 0], [2, 4])

        err = try
            solve(H, [[1.0]], Continuation(; show_progress = false))
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("start solution has length", err.msg)

        underdetermined = System([x * y - a]; variables = [x, y], parameters = [a])
        err = try
            solve(
                ParameterHomotopy(underdetermined, [1], [4]),
                [[1.0, 1.0]],
                Continuation(; show_progress = false),
            )
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("positive-dimensional", err.msg)
    end

    @testset "solve(H, starts): homotopy with no rebuild rule" begin
        @polyvar x y
        G = System([x^2 + y^2 - 5, x * y - 2]; variables = [x, y])
        F = System([x^2 + 3y^2 - 7, x * y + x - 3]; variables = [x, y])
        H = _BufferedHomotopy(
            StraightLineHomotopy(
                _clone_system_evaluator(G), _clone_system_evaluator(F);
                γ = complex(0.6, 0.8),
            ),
        )
        starts = solutions(solve(G, TotalDegree(; show_progress = false)))
        reference = solutions(solve(F, TotalDegree(; show_progress = false)))

        # The default rebuild copies the buffers and rebuilds the tapes.
        C = _clone_homotopy(H)
        @test C isa _BufferedHomotopy
        @test C.scratch !== H.scratch
        @test C.inner !== H.inner
        @test C.inner.start !== H.inner.start
        @test C.inner.u_start !== H.inner.u_start
        @test C.inner.γ == H.inner.γ

        serial = solve(H, starts, Continuation(; seed = UInt32(9), show_progress = false), Serial())
        threaded = solve(
            H,
            starts,
            Continuation(; seed = UInt32(9), show_progress = false),
            Threaded(),
        )
        @test nsolutions(serial) == 4
        @test max_distance(solutions(serial), reference) < 1.0e-8
        @test max_distance(solutions(threaded), reference) < 1.0e-8
        for (u, v) in zip(path_results(serial), path_results(threaded))
            @test u.solution ≈ v.solution
            @test steps(u) == steps(v)
        end

        # A `HomotopyEvaluator` holds no rebuild thunk, so copying it is refused
        # instead of aliasing the buffers it closed over.
        @test_throws ArgumentError deepcopy(HomotopyEvaluator(H))
    end

    @testset "solve(build_homotopy, starts)" begin
        @polyvar x y
        G = System([x^2 + y^2 - 5, x * y - 2]; variables = [x, y])
        F = System([x^2 + 3y^2 - 7, x * y + x - 3]; variables = [x, y])
        # Each call builds its own evaluators, so the tasks share nothing mutable.
        build() = StraightLineHomotopy(
            _clone_system_evaluator(G), _clone_system_evaluator(F); γ = complex(0.6, 0.8),
        )
        starts = solutions(solve(G, TotalDegree(; show_progress = false)))
        reference = solutions(solve(F, TotalDegree(; show_progress = false)))

        serial = solve(build, starts, Continuation(; show_progress = false), Serial())
        threaded = solve(build, starts, Continuation(; show_progress = false), Threaded())
        @test nsolutions(serial) == 4
        @test max_distance(solutions(serial), reference) < 1.0e-8
        @test max_distance(solutions(threaded), reference) < 1.0e-8
        for (u, v) in zip(path_results(serial), path_results(threaded))
            @test u.solution ≈ v.solution
            @test steps(u) == steps(v)
        end

        err = try
            solve(() -> G, starts, Continuation(; show_progress = false), Serial())
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("AbstractHomotopy", err.msg)
    end

    @testset "init returns a reusable cache" begin
        @polyvar x y
        G = System([x^2 + y^2 - 5, x * y - 2]; variables = [x, y])
        F = System([x^2 + 3y^2 - 7, x * y + x - 3]; variables = [x, y])
        starts = solutions(solve(G, TotalDegree(; show_progress = false)))
        cache = init(
            G, F, starts,
            Continuation(; seed = UInt32(11), show_progress = false), Serial(),
        )
        r1 = solve!(cache)
        r2 = solve!(cache)
        @test solutions(r1) == solutions(r2)
        @test r1.seed == UInt32(11)
    end
end
