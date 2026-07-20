using Test
using CheckConcreteStructs: all_concrete
using HomotopyContinuationNext
using HomotopyContinuationNext: Interpreter, TaylorVector, TruncatedTaylorSeries,
    DoubleF64, ComplexDF64

# Types from other packages (e.g. FunctionWrapper aliases) are filtered by parentmodule.
const _CONCRETE_SKIP = Set{Symbol}()

@testset "CheckConcreteStructs" begin
    # Non-parametric structs
    for name in names(HomotopyContinuationNext; all = true)
        name in _CONCRETE_SKIP && continue
        isdefined(HomotopyContinuationNext, name) || continue
        T = getfield(HomotopyContinuationNext, name)
        T isa Type || continue
        isabstracttype(T) && continue
        T isa UnionAll && continue
        isstructtype(T) || continue
        # Skip types defined in other packages (e.g. FunctionWrapper aliases)
        parentmodule(T) === HomotopyContinuationNext || continue
        @testset "$name" begin
            @test all_concrete(T; verbose = false)
        end
    end

    # Parametric structs — check concrete instantiations
    @testset "Interpreter{Vector{ComplexF64}}" begin
        @test all_concrete(Interpreter{Vector{ComplexF64}}; verbose = false)
    end
    @testset "Interpreter{Vector{ComplexDF64}}" begin
        @test all_concrete(Interpreter{Vector{ComplexDF64}}; verbose = false)
    end
    @testset "TaylorVector{ComplexF64}" begin
        @test all_concrete(TaylorVector{ComplexF64}; verbose = false)
    end
    @testset "TruncatedTaylorSeries{2,ComplexF64}" begin
        @test all_concrete(TruncatedTaylorSeries{2, ComplexF64}; verbose = false)
    end
    @testset "TruncatedTaylorSeries{4,ComplexF64}" begin
        @test all_concrete(TruncatedTaylorSeries{4, ComplexF64}; verbose = false)
    end
    @testset "RegenerationState concrete phase" begin
        system_type = System{
            Int, Int, CompileMode.INTERPRETED,
            HomotopyContinuationNext.SquareShape,
        }
        state_type = HomotopyContinuationNext.RegenerationState{Int, Int, system_type}
        @test all_concrete(state_type; verbose = false)
    end
    @testset "NumericalIrreducibleDecomposition concrete result" begin
        system_type = System{
            Int, Int, CompileMode.INTERPRETED,
            HomotopyContinuationNext.SquareShape,
        }
        witness_type = WitnessSet{system_type}
        @test all_concrete(
            NumericalIrreducibleDecomposition{witness_type}; verbose = false,
        )
    end
end
