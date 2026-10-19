using Test
using CheckConcreteStructs: all_concrete
using HomotopyContinuation
using DynamicPolynomials: @polyvar

include(joinpath(dirname(dirname(@__DIR__)), "workloads", "workloads.jl"))

# Types from other packages (e.g. FunctionWrapper aliases) are filtered by parentmodule.
const _CONCRETE_SKIP = Set{Symbol}()

# Concrete instantiations of package parametric structs reachable from a type:
# its parameters, union members and, for package types, its field types.
function parametric_instantiations!(found, T, seen)
    T in seen && return found
    push!(seen, T)
    if T isa Union
        parametric_instantiations!(found, T.a, seen)
        parametric_instantiations!(found, T.b, seen)
    elseif T isa DataType
        own = T.name.module === HomotopyContinuation
        own && !isempty(T.parameters) && isconcretetype(T) && isstructtype(T) &&
            !(T <: Function) && push!(found, T)
        foreach(p -> parametric_instantiations!(found, p, seen), T.parameters)
        own && isconcretetype(T) &&
            foreach(F -> parametric_instantiations!(found, F, seen), fieldtypes(T))
    end
    return found
end

# Instantiations in the signatures the workloads compiled and in the values they return.
function workload_instantiations()
    outputs = map(name -> Workloads.run(Val(name)), Workloads.workloads())
    found, seen = Set{DataType}(), Set{Any}()
    for name in names(HomotopyContinuation; all = true)
        isdefined(HomotopyContinuation, name) || continue
        for method in methods(getfield(HomotopyContinuation, name))
            method.module === HomotopyContinuation || continue
            for mi in Base.specializations(method)
                mi isa Core.MethodInstance && parametric_instantiations!(found, mi.specTypes, seen)
            end
        end
    end
    foreach(output -> parametric_instantiations!(found, typeof(output), seen), outputs)
    return found
end

@testset "CheckConcreteStructs" begin
    # Non-parametric structs
    for name in names(HomotopyContinuation; all = true)
        name in _CONCRETE_SKIP && continue
        isdefined(HomotopyContinuation, name) || continue
        T = getfield(HomotopyContinuation, name)
        T isa Type || continue
        isabstracttype(T) && continue
        T isa UnionAll && continue
        isstructtype(T) || continue
        # Skip types defined in other packages (e.g. FunctionWrapper aliases)
        parentmodule(T) === HomotopyContinuation || continue
        @testset "$name" begin
            @test all_concrete(T; verbose = false)
        end
    end

    # Parametric structs, at every concrete instantiation the workloads produce.
    @testset "$T" for T in sort!(collect(workload_instantiations()); by = string)
        @test all_concrete(T; verbose = false)
    end
end

@testset "concrete public result and option types" begin
    base = Monodromy(; show_progress = false)
    @test isconcretetype(typeof(base))
    @test isconcretetype(typeof(Monodromy(; dim = 1, show_progress = false)))
    @test isconcretetype(typeof(Monodromy(; codim = 1, show_progress = false)))
    @test isconcretetype(eltype(base.variables))
    @test isconcretetype(eltype(base.parameters))

    @polyvar x y
    F = System([x * y]; variables = [x, y])
    seed = UInt32(0x1234)
    witness_superset = solve(F, Regeneration(; seed, show_progress = false), Serial())
    @test isconcretetype(eltype(witness_superset))
    decomposition = solve(
        witness_superset, Decomposition(; seed, show_progress = false), Serial(),
    )
    @test isconcretetype(eltype(decomposition))
    @test eltype(decomposition) === eltype(witness_superset)
    nid = solve(F, Decomposition(; seed, show_progress = false), Serial())
    @test isconcretetype(typeof(nid))
end
