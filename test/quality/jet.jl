using Test
using JET: JET
using HomotopyContinuation: HomotopyContinuation
using Distributed: Distributed
using SemialgebraicSets: SemialgebraicSets
using Serialization: Serialization

include(joinpath(dirname(dirname(@__DIR__)), "workloads", "workloads.jl"))

const HC = HomotopyContinuation
const EXTENSIONS = (:DistributedExt, :SemialgebraicSetsExt)
const TARGETS = (HC, map(ext -> Base.get_extension(HC, ext), EXTENSIONS)...)

# Setup code that calls through a deliberately erased boundary, keyed by the method
# that makes the dynamic call. The tracking path behind these boundaries is concrete;
# StrictMode and AllocCheck prove that separately.
const BOUNDARIES = Dict(
    (:build_compiled_evaluator, "codegen.jl") => "FunctionWrapper erases the generated evaluator",
    (:build_taylor_fws, "codegen.jl") => "FunctionWrapper erases the generated Taylor kernels",
    (:dispatch_parameter_taylor_fws, "system.jl") => "@nospecialize dispatcher",
    (:invoke_polynomial_lowerer, "polynomial_input.jl") => "@nospecialize dispatcher",
    (:dispatch_solve_policy, "track_paths.jl") => "@nospecialize dispatcher",
    (:dispatch_monodromy_policy, "solve.jl") => "@nospecialize dispatcher",
    (:dispatch_start_pair_strategy, "start_pair.jl") => "@nospecialize dispatcher",
    (:init, "solve.jl") => "inferencebarrier selects one start-system route",
    (:init, "polyhedral.jl") => "inferencebarrier selects one start-system route",
    (:make_matrix_workspace, "linear_algebra.jl") => "inferencebarrier keeps the tall QR path lazy",
    (:DF64Installer, "system_evaluator.jl") => "inferencebarrier defers the DoubleF64 evaluator",
    (:clone_system_evaluator, "system_evaluator.jl") => "RefValue{Any} evaluator factory",
    (:fold_composition, "composition_system.jl") => "RefValue{Any} evaluator factory",
    (:PathBuilder, "builder.jl") => "RefValue{Any} path builder",
    (:retarget!, "sweep.jl") => "RefValue{Any} path worker",
    (:make_progress, "progress.jl") => "stdout is an abstractly typed global",
    (:make_many_progress, "progress.jl") => "stdout is an abstractly typed global",
)

# A keyword body `#f#12` is reported under `f`.
function site(report)
    method = report.vst[end].linfo.def
    method isa Method || return (:toplevel, "")
    name = string(method.name)
    startswith(name, "#") && (name = split(name, '#')[2])
    return (Symbol(name), basename(string(method.file)))
end

# Moshi's `@match` binds pattern variables in branches JET cannot prove exhaustive.
function moshi_binding(report)
    report isa JET.UndefVarErrorReport || return false
    var = report.var
    return startswith(string(var isa GlobalRef ? var.name : var), "##")
end

# `FixedParameterHomotopy` forwards to methods the wrapped user homotopy must define.
function extension_contract(report)
    report isa JET.MethodErrorReport || return false
    method = report.vst[end].linfo.def
    return method isa Method && occursin("FixedParameterHomotopy", string(method.sig))
end

report_text(report) = sprint(show, MIME("text/plain"), report)

function test_no_reports(reports)
    isempty(reports) || @info "unexpected JET reports" map(report_text, reports)
    @test isempty(reports)
    return
end

errors(result) = filter(r -> !moshi_binding(r) && !extension_contract(r), JET.get_reports(result))

@static if isempty(VERSION.prerelease)
    @testset "JET report_package" begin
        test_no_reports(errors(JET.report_package(HC; target_modules = (HC,))))
        @testset "$ext" for ext in EXTENSIONS
            m = Base.get_extension(HC, ext)
            @test m isa Module
            test_no_reports(errors(JET.report_package(m; target_modules = (m,))))
        end
    end

    # Every public feature, as built and run by its first-use workload, must be free
    # of errors and of runtime dispatch outside the boundaries above.
    @testset "JET workloads" begin
        hit = Set{Tuple{Symbol, String}}()
        @testset "$name" for name in Workloads.workloads()
            workload = Val(name)
            test_no_reports(errors(JET.report_call(Workloads.run, (typeof(workload),); target_modules = TARGETS)))
            dispatch = JET.get_reports(JET.report_opt(Workloads.run, (typeof(workload),); target_modules = TARGETS))
            foreach(r -> push!(hit, site(r)), dispatch)
            test_no_reports(filter(r -> !haskey(BOUNDARIES, site(r)), dispatch))
        end
        @testset "every boundary is still reached" begin
            test_no_reports(sort!([k for k in keys(BOUNDARIES) if k ∉ hit]))
        end
    end
end
