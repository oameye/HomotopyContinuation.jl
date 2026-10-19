using Test
using HomotopyContinuation
using DynamicPolynomials: @polyvar
using Distributed: Distributed, addprocs, rmprocs, workers, remotecall_eval
using Random: seed!
using Serialization: serialize, deserialize

function same_path_semantics(a, b)
    pa, pb = path_results(a), path_results(b)
    length(pa) == length(pb) || return false
    ntracked(a) == ntracked(b) || return false
    nsolutions(a) == nsolutions(b) || return false
    nfailed(a) == nfailed(b) || return false
    nat_infinity(a) == nat_infinity(b) || return false
    seed(a) == seed(b) || return false

    for (u, v) in zip(pa, pb)
        sprint(show, MIME("text/plain"), u) == sprint(show, MIME("text/plain"), v) ||
            return false
        path_number(u) == path_number(v) || return false
        is_success(u) == is_success(v) || return false
        is_finite(u) == is_finite(v) || return false
        is_at_infinity(u) == is_at_infinity(v) || return false
        is_excess_solution(u) == is_excess_solution(v) || return false
        is_singular(u) == is_singular(v) || return false
        is_real(u) == is_real(v) || return false
        winding_number(u) == winding_number(v) || return false
        multiplicity(u) == multiplicity(v) || return false
        steps(u) == steps(v) || return false
        accepted_steps(u) == accepted_steps(v) || return false
        rejected_steps(u) == rejected_steps(v) || return false
        isequal(solution(u), solution(v)) || return false
        isequal(start_solution(u), start_solution(v)) || return false
        isequal(last_path_point(u), last_path_point(v)) || return false
        isequal(accuracy(u), accuracy(v)) || return false
        isequal(residual(u), residual(v)) || return false
        isequal(condition_jacobian(u), condition_jacobian(v)) || return false
        isequal(valuation(u), valuation(v)) || return false
    end
    return true
end

_entry_result(entry::Result) = entry
_entry_result(entry::Tuple) = entry[1]

function same_sweep_semantics(a, b)
    length(a) == length(b) || return false
    return all(same_path_semantics(_entry_result(a[k]), _entry_result(b[k])) for k in eachindex(a))
end

include("../helpers/solution_sets.jl")


roundtrip(v) = (io = IOBuffer(); serialize(io, v); seekstart(io); deserialize(io))

# Worker processes receive systems through Serialization. A round trip must give
# back a system that evaluates and solves exactly like the original.
@testset "System serialization round trip" begin
    @polyvar x y a b
    pt = ComplexF64[0.7 - 0.2im, -1.3 + 0.4im]
    ps = ComplexF64[2.0 + 0.5im, 5.0 - 1.0im]

    @testset "System, compile mode $mode" for mode in (
            CompileMode.INTERPRETED, CompileMode.COMPILED, CompileMode.COMPILED_ALL,
        )
        F = System([x^3 + y^2 - 2x * y + 1, x * y^2 - 3x + 2y - 1]; compile = mode)
        G = roundtrip(F)
        @test typeof(G) === typeof(F)
        @test degrees(G) == degrees(F) == [3, 3]
        @test variables(G) == variables(F)
        u, v = pt
        @test evaluate(G, pt) == evaluate(F, pt)
        @test evaluate(G, pt) ≈ [u^3 + v^2 - 2u * v + 1, u * v^2 - 3u + 2v - 1]
        @test jacobian(G, pt) == jacobian(F, pt)
        @test jacobian(G, pt) ≈ [3u^2 - 2v 2v - 2u; v^2 - 3 2u * v + 2]

        alg = TotalDegree(; seed = UInt32(0x5e71), show_progress = false)
        rF, rG = solve(F, alg, Serial()), solve(G, alg, Serial())
        @test nsolutions(rG) == nsolutions(rF)
        @test isequal(solutions(rG), solutions(rF))
    end

    @testset "parametric System" begin
        F = System([x^2 + y^2 - a, x * y - b]; parameters = [a, b])
        G = roundtrip(F)
        @test nparameters(G) == 2
        @test parameters(G) == parameters(F)
        u, v = pt
        @test evaluate(G, pt, ps) == evaluate(F, pt, ps)
        @test evaluate(G, pt, ps) ≈ [u^2 + v^2 - ps[1], u * v - ps[2]]
        @test jacobian(G, pt, ps) == jacobian(F, pt, ps)
    end

    @testset "FixedParameterSystem" begin
        F = System([x^2 + y^2 - a, x * y - b]; parameters = [a, b])
        H = FixedParameterSystem(F, ps)
        G = roundtrip(H)
        @test typeof(G) === typeof(H)
        @test degrees(G) == degrees(H)
        @test evaluate(G, pt) == evaluate(H, pt)
        @test evaluate(G, pt) ≈ evaluate(F, pt, ps)
        @test jacobian(G, pt) == jacobian(H, pt)
    end

    @testset "grouped System" begin
        @polyvar u v s t
        F = System(
            [u * s - 2v * t, u^2 - 4 * v^2]; variable_groups = [[u, v], [s, t]],
        )
        G = roundtrip(F)
        @test variable_groups(G) == variable_groups(F)
        alg = TotalDegree(; seed = UInt32(0x6a09), show_progress = false)
        @test paths_to_track(G, alg) == paths_to_track(F, alg)
    end

    @testset "CompositionSystem" begin
        f = System([y^2 + 2x + 3, x - 1])
        g = System([x + y * a, x - b]; parameters = [a, b])
        C = g ∘ f
        D = roundtrip(C)
        @test typeof(D) === typeof(C)
        @test degrees(D) == degrees(C)
        u, v = pt
        f_pt = [v^2 + 2u + 3, u - 1]
        @test evaluate(D, pt, ps) == evaluate(C, pt, ps)
        @test evaluate(D, pt, ps) ≈ [f_pt[1] + f_pt[2] * ps[1], f_pt[1] - ps[2]]
        @test jacobian(D, pt, ps) == jacobian(C, pt, ps)
    end
end
