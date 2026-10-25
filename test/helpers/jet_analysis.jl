# JET analysis shared by the package and certification quality checks.

function package_method(frame, targets)
    method = frame.linfo.def
    return method isa Method && method.module in targets ? method : nothing
end

const REPOSITORY = dirname(dirname(@__DIR__))

# The package call site a report is attributed to: the innermost frame owned by
# `owners` whose source lies in a package `src/` or `ext/` directory, as
# `(function, path)` with the path relative to the repository. A dispatch inside a
# dependency is thereby charged to the package line that calls it, and a generator or
# macro-generated closure to the function around it. A keyword body `#f#12` is
# reported under `f`.
function call_site(report, owners)
    for frame in Iterators.reverse(report.vst)
        method = package_method(frame, owners)
        method === nothing && continue
        path = relpath(string(method.file), REPOSITORY)
        occursin(r"^(lib/[^/]+/)?(src|ext)/", path) || continue
        name = string(method.name)
        startswith(name, "#") && (name = split(name, '#')[2])
        return (Symbol(name), path)
    end
    return nothing
end

# Moshi's `@match` binds pattern variables in branches JET cannot prove exhaustive.
function moshi_binding(report)
    report isa JET.UndefVarErrorReport || return false
    var = report.var
    return startswith(string(var isa GlobalRef ? var.name : var), "##")
end

# Every lock's slow path reaches `Base.wait_forever`, the scheduler's last resort,
# which prints a fatal error through `display_error`. That dispatch belongs to Base
# and says nothing about the package line that took the lock.
scheduler_fatal_error(report) = any(report.vst) do frame
    method = frame.linfo.def
    method isa Method && method.module === Base && method.name === :wait_forever
end

report_text(report) = sprint(show, MIME("text/plain"), report)

function test_no_reports(reports)
    isempty(reports) || @info "unexpected JET reports" map(report_text, reports)
    @test isempty(reports)
    return
end

# Every specialization compiled so far for methods of `targets`. JET's own inference
# creates method instances too, so call this after running the workloads and before
# any analysis.
function compiled_specializations(targets)
    specializations = Core.MethodInstance[]
    seen = Set{Method}()
    for mod in targets, name in names(mod; all = true)
        isdefined(mod, name) || continue
        for method in methods(getfield(mod, name))
            method.module in targets && method ∉ seen || continue
            push!(seen, method)
            for mi in Base.specializations(method)
                mi isa Core.MethodInstance && Base.isdispatchtuple(mi.specTypes) &&
                    push!(specializations, mi)
            end
        end
    end
    return specializations
end

# The runtime dispatch reachable from `tt`, through dependencies included, that is
# charged to a call site of `owners` outside `exemptions`. A frame JET could not
# optimize says nothing about types unless it is the package's own.
function unexempted_dispatch(tt, owners, exemptions, hit = Set{Tuple{Symbol, String}}())
    return filter(JET.get_reports(JET.report_opt(tt))) do r
        r isa JET.OptimizationFailureReport &&
            package_method(r.vst[end], owners) === nothing && return false
        scheduler_fatal_error(r) && return false
        site = call_site(r, owners)
        site === nothing && return false
        push!(hit, site)
        return !haskey(exemptions, site)
    end
end

# Every workload, and every specialization the workloads compiled, must be free of
# errors and of runtime dispatch outside `exemptions`, a `(function, path) => reason`
# table. The specializations include code the workloads reach behind those
# exemptions, where analysis from the entry point stops. An entry that always throws
# is its own contract, so analysed alone it reports an uncaught exception that
# carries no information. An exemption that is no longer reached fails, so the
# table cannot go stale. Dispatch inside `inherited_owners`, the packages `targets`
# build on, is charged to their own call sites and accepted by `inherited`.
function test_jet_workloads(
        run, workloads, specializations; targets, exemptions,
        inherited_owners = (), inherited = Dict{Tuple{Symbol, String}, String}(), ignore = Returns(false),
    )
    hit = Set{Tuple{Symbol, String}}()
    owners = (targets..., inherited_owners...)
    accepted = merge(inherited, exemptions)
    errors(result) = filter(r -> !moshi_binding(r) && !ignore(r), JET.get_reports(result))
    @testset "JET workloads" begin
        @testset "$name" for name in workloads
            tt = Tuple{typeof(run), Val{name}}
            test_no_reports(errors(JET.report_call(tt; target_modules = targets)))
            test_no_reports(unexempted_dispatch(tt, owners, accepted, hit))
        end
    end
    @testset "JET compiled specializations" begin
        @test !isempty(specializations)
        failures = JET.InferenceErrorReport[]
        for mi in specializations
            tt = mi.specTypes
            calls = errors(JET.report_call(tt; target_modules = targets))
            append!(failures, filter(r -> !(r isa JET.UncaughtExceptionReport), calls))
            append!(failures, unexempted_dispatch(tt, owners, accepted, hit))
        end
        test_no_reports(failures)
    end
    @testset "every exemption is still reached" begin
        test_no_reports(sort!([k for k in keys(exemptions) if k ∉ hit]))
    end
    return
end
