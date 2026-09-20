using Test
using HomotopyContinuation
using HomotopyContinuation: Serial, Polyhedral, System, CompileMode
using HomotopyContinuation: _SupportSystem, evaluate!, FSVec,
    nparameters, _distributed_solve!, _distributed_sweep_entries,
    _distributed_monodromy_solve!
using HomotopyContinuation: variable_groups, multi_degrees, is_homogeneous
using DynamicPolynomials: @polyvar
using Serialization: serialize, deserialize
using CommonSolve: init

# The builder cache is intentionally private; this suite owns that contract.
_inner_builder(b) = b._inner[]

roundtrip(x) = (io = IOBuffer(); serialize(io, x); seekstart(io); deserialize(io))

function evaluate_at(evaluator, x::Vector{ComplexF64}, p::Vector{ComplexF64})
    u = FSVec{ComplexF64}(zeros(ComplexF64, size(evaluator)[1]))
    xs = FSVec{ComplexF64}(copy(x))
    ps = FSVec{ComplexF64}(copy(p))
    evaluate!(u, evaluator, xs, ps)
    return collect(u)
end

@testset "Distributed serialization and extension kernels" begin
    @polyvar x y a b

    F = System([x^3 + y^2 - 2x * y + 1, x * y^2 - 3x + 2y - 1])
    F_param = System([x^2 + y^2 - a, x * y - b]; parameters = [a, b])

    @testset "system serialization" begin
        @testset "System, compile mode $mode" for mode in (
                CompileMode.INTERPRETED, CompileMode.COMPILED,
                CompileMode.COMPILED_ALL,
            )
            G = System([x^3 + y^2 - 2x * y + 1, x * y^2 - 3x + 2y - 1]; compile = mode)
            H = roundtrip(G)
            @test typeof(H) === typeof(G)
            @test H.degrees == G.degrees
            @test H.compile_mode === G.compile_mode
            @test size(H.evaluator) == size(G.evaluator)
            pt = ComplexF64[0.7, -1.3]
            @test evaluate_at(H.evaluator, pt, ComplexF64[]) ==
                evaluate_at(G.evaluator, pt, ComplexF64[])
        end

        @testset "parametric System" begin
            H = roundtrip(F_param)
            @test nparameters(H.evaluator) == 2
            pt, ps = ComplexF64[0.7, -1.3], ComplexF64[2.0, 5.0]
            @test evaluate_at(H.evaluator, pt, ps) ==
                evaluate_at(F_param.evaluator, pt, ps)
        end

        @testset "FixedParameterSystem" begin
            ps = ComplexF64[2.0, 5.0]
            G = FixedParameterSystem(F_param, ps)
            H = roundtrip(G)
            @test typeof(H) === typeof(G)
            @test H.parameters == G.parameters
            @test degrees(H) == degrees(G)
            pt = ComplexF64[0.7, -1.3]
            @test evaluate_at(H.evaluator, pt, ComplexF64[]) ==
                evaluate_at(G.evaluator, pt, ComplexF64[])
        end

        @testset "SystemEvaluator" begin
            # Only the rebuild thunk ships; the far side calls it.
            ev = roundtrip(F.evaluator)
            @test size(ev) == size(F.evaluator)
            pt = ComplexF64[0.7, -1.3]
            @test evaluate_at(ev, pt, ComplexF64[]) ==
                evaluate_at(F.evaluator, pt, ComplexF64[])
        end

        @testset "grouped System" begin
            @polyvar u v s t
            G = System(
                [u * s - 2v * t, u^2 - 4 * v^2]; variable_groups = [[u, v], [s, t]],
            )
            H = roundtrip(G)
            @test variable_groups(H) == variable_groups(G)
            @test multi_degrees(H) == multi_degrees(G)
            @test is_homogeneous(H) == is_homogeneous(G)
        end

        @testset "_SupportSystem" begin
            support_system =
                _inner_builder(init(F, Polyhedral(), Serial()).builder).support_system
            @test support_system isa _SupportSystem
            other = roundtrip(support_system)
            pt = ComplexF64[0.7, -1.3]
            ps = ComplexF64[0.3 + 0.1im * k for k in 1:nparameters(support_system.evaluator)]
            @test evaluate_at(other.evaluator, pt, ps) ==
                evaluate_at(support_system.evaluator, pt, ps)
        end

        @testset "CompositionSystem" begin
            f = System([y^2 + 2x + 3, x - 1])
            g = System([x + y * a, x - b]; parameters = [a, b])
            C = g ∘ f
            D = roundtrip(C)
            @test typeof(D) === typeof(C)
            @test length(D.stages) == length(C.stages)
            @test D.degrees == C.degrees
            @test D.is_homogeneous == C.is_homogeneous
            @test [s.degrees for s in D.stages] == [s.degrees for s in C.stages]
            pt, ps = ComplexF64[0.7, -1.3], ComplexF64[2.0, 5.0]
            @test evaluate_at(D.evaluator, pt, ps) == evaluate_at(C.evaluator, pt, ps)
        end
    end

    # Public distributed execution semantics live in test/distributed_test.jl.
    # Quality retains only the private extension-fallback contracts.
    @testset "extension hooks" begin
        @test Base.get_extension(
            HomotopyContinuation, :DistributedExt,
        ) !== nothing
        @test_throws ArgumentError _distributed_solve!(nothing)
        @test_throws ArgumentError _distributed_sweep_entries(
            nothing, [], Int[], nothing, identity, identity, nothing,
        )
        @test_throws ArgumentError _distributed_monodromy_solve!(
            nothing, nothing, nothing, nothing, nothing,
        )
    end
end
