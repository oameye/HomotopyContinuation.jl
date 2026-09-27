using Test
using HomotopyContinuation

const HC_PUBLIC_NAMES = Set(names(HomotopyContinuation; all = false, imported = false))
const QUALITY_CONTRACTS = Set(
    [
        "alloc_check_test.jl",
        "aqua_test.jl",
        "concrete_structs_test.jl",
        "dispatch_doctor_test.jl",
        "explicit_imports_test.jl",
        "instruction_count_test.jl",
        "jet_test.jl",
    ],
)

# Ordinary tests should exercise the supported accessor surface rather than opaque
# storage of Result, PathResult, System, evaluator, or monodromy implementation
# objects. Deliberately documented data carriers (for example NewtonResult,
# PathInfo, StartPair, and ExtrinsicDescription) keep their documented fields.
const HC_PRIVATE_STORAGE_PROPERTIES = Set(
    [
        :_inner,
        :accepted_steps,
        :builder,
        :cache,
        :clusters,
        :condition_jacobian,
        :evaluator,
        :last_path_point,
        :multiplicity,
        :path_number,
        :path_results,
        :polys,
        :rejected_steps,
        :returncode,
        :seed,
        :solution,
        :start_solution,
        :start_solutions,
        :statistics,
        :tracked_loops,
        :tracked_paths,
        :tree,
        :valuation,
        :winding_number,
        :worker,
    ],
)

function hc_aliases(source::String)
    aliases = Set(["HomotopyContinuation"])
    for m in eachmatch(r"\bHomotopyContinuation\s+as\s+([A-Za-z_][A-Za-z0-9_]*)", source)
        push!(aliases, m.captures[1])
    end
    for m in eachmatch(r"(?m)^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*HomotopyContinuation\s*$", source)
        push!(aliases, m.captures[1])
    end
    return aliases
end

function explicit_hc_imports(source::String)
    imported = Symbol[]
    lines = split(source, '\n')
    i = 1
    while i <= length(lines)
        line = lines[i]
        m = match(r"^\s*(?:using|import)\s+HomotopyContinuation\s*:\s*(.*)$", line)
        if m !== nothing
            chunk = m.captures[1]
            while i < length(lines) && (isempty(strip(chunk)) || endswith(strip(lines[i]), ","))
                i += 1
                chunk *= " " * strip(lines[i])
            end
            for token in split(chunk, ',')
                name = strip(first(split(strip(token), r"\s+as\s+"; limit = 2)))
                isempty(name) && continue
                push!(imported, Symbol(name))
            end
        end
        i += 1
    end
    return imported
end

function qualified_hc_references(source::String)
    refs = Symbol[]
    for alias in hc_aliases(source)
        pattern = Regex("\\b" * alias * "\\.(@?[A-Za-z_][A-Za-z0-9_!]*)")
        for m in eachmatch(pattern, source)
            name = Symbol(m.captures[1])
            name === :jl && continue
            push!(refs, name)
        end
    end
    return refs
end

# Reflection reaches storage and module internals that no accessor exposes, so
# the ordinary suite may not use it at all.
const REFLECTIVE_ACCESSORS = Set([:getfield, :getproperty, :setfield!, :setproperty!])
const EVAL_FUNCTIONS = Set([:eval, :include_string])

callee_name(f::Symbol) = f
callee_name(f::QuoteNode) = f.value isa Symbol ? f.value : nothing
callee_name(f::Expr) = f.head === :. && length(f.args) == 2 ? callee_name(f.args[2]) : nothing
callee_name(_) = nothing

is_hc_module(ex, aliases::Set{String}) = ex isa Symbol && string(ex) in aliases

function collect_reflection_references!(found::Set{String}, ex, aliases::Set{String})
    ex isa Expr || return found
    if ex.head === :call && !isempty(ex.args)
        name = callee_name(ex.args[1])
        if name in REFLECTIVE_ACCESSORS
            push!(found, string(name))
        elseif name in EVAL_FUNCTIONS && length(ex.args) >= 2 &&
                is_hc_module(ex.args[2], aliases)
            push!(found, string(name, "(", ex.args[2], ", …)"))
        end
    elseif ex.head === :macrocall && !isempty(ex.args) &&
            callee_name(ex.args[1]) === Symbol("@eval")
        module_args = filter(a -> !(a isa LineNumberNode), ex.args[2:end])
        length(module_args) >= 2 && is_hc_module(first(module_args), aliases) &&
            push!(found, string("@eval ", first(module_args)))
    end
    for arg in ex.args
        collect_reflection_references!(found, arg, aliases)
    end
    return found
end

function collect_property_references!(refs::Set{Symbol}, ex)
    ex isa Expr || return refs
    if ex.head === :. && length(ex.args) >= 2
        property = ex.args[2]
        name = property isa QuoteNode ? property.value : property
        name isa Symbol && push!(refs, name)
    end
    for arg in ex.args
        collect_property_references!(refs, arg)
    end
    return refs
end

function parsed_expressions(source::String)
    exprs = Any[]
    pos = firstindex(source)
    stop = ncodeunits(source)
    while pos <= stop
        ex, next = Meta.parse(source, pos; raise = false)
        ex === nothing && break
        ex isa Expr && ex.head === :error && error("failed to parse semantic-test source near byte $pos")
        push!(exprs, ex)
        next > pos || error("parser made no progress near byte $pos")
        pos = next
    end
    return exprs
end

property_references(exprs::Vector{Any}) =
    foldl(collect_property_references!, exprs; init = Set{Symbol}())

function reflection_references(exprs::Vector{Any}, aliases::Set{String})
    found = Set{String}()
    for ex in exprs
        collect_reflection_references!(found, ex, aliases)
    end
    return found
end

function semantic_test_files(root::String)
    files = String[]
    for (dir, _, names) in walkdir(root)
        rel = relpath(dir, root)
        startswith(rel, "extensive") && continue
        startswith(rel, "strict") && continue
        for name in names
            endswith(name, ".jl") || continue
            name == "public_surface_contract_test.jl" && continue
            name in QUALITY_CONTRACTS && continue
            push!(files, joinpath(dir, name))
        end
    end
    return sort(files)
end

function surface_violations(source::String)
    refs = union(explicit_hc_imports(source), qualified_hc_references(source))
    private_refs = sort!(collect(setdiff(Set(refs), HC_PUBLIC_NAMES)); by = string)
    exprs = parsed_expressions(source)
    private_properties = sort!(
        collect(intersect(property_references(exprs), HC_PRIVATE_STORAGE_PROPERTIES));
        by = string,
    )
    reflection = sort!(collect(reflection_references(exprs, hc_aliases(source))))
    return private_refs, private_properties, reflection
end

name_violations = Dict{String, Vector{Symbol}}()
property_violations = Dict{String, Vector{Symbol}}()
reflection_violations = Dict{String, Vector{String}}()
for file in semantic_test_files(@__DIR__)
    private_refs, private_properties, reflection = surface_violations(read(file, String))
    key = relpath(file, @__DIR__)
    isempty(private_refs) || (name_violations[key] = private_refs)
    isempty(private_properties) || (property_violations[key] = private_properties)
    isempty(reflection) || (reflection_violations[key] = reflection)
end

if !isempty(name_violations)
    println(stderr, "Non-public HomotopyContinuation references in the ordinary semantic suite:")
    for file in sort!(collect(keys(name_violations)))
        println(stderr, "  ", file, ": ", join(string.(name_violations[file]), ", "))
    end
end

if !isempty(property_violations)
    println(stderr, "Opaque HomotopyContinuation storage properties in the ordinary semantic suite:")
    for file in sort!(collect(keys(property_violations)))
        println(stderr, "  ", file, ": ", join(string.(property_violations[file]), ", "))
    end
end

if !isempty(reflection_violations)
    println(stderr, "Reflective HomotopyContinuation access in the ordinary semantic suite:")
    for file in sort!(collect(keys(reflection_violations)))
        println(stderr, "  ", file, ": ", join(reflection_violations[file], ", "))
    end
end

@testset "normal tests use only public HomotopyContinuation API" begin
    @test isempty(name_violations)
    @test isempty(property_violations)
    @test isempty(reflection_violations)
end

@testset "the guard rejects known bypasses" begin
    rejected(source) = any(!isempty, surface_violations(source))
    @test rejected("a = UniquePoints(3)\na.tree.triangle_inequality\n")
    @test rejected("cache.start_solutions[1]\n")
    @test rejected("getfield(r, :returncode)\n")
    @test rejected("Base.getproperty(r, :x)\n")
    @test rejected("@eval HomotopyContinuation f() = 1\n")
    @test rejected("Core.eval(HomotopyContinuation, :(x = 1))\n")
    @test rejected("HomotopyContinuation.eval(:(x = 1))\n")
    @test rejected("using HomotopyContinuation: HomotopyContinuation as HC\nHC.MonodromyCode\n")
    @test rejected("import HomotopyContinuation as HC\nHC.Tracker\n")
    @test !rejected("using HomotopyContinuation: HomotopyContinuation as HC\nHC.solve\n")
    @test !rejected("@eval f() = 1\ninfo.return_code\n")
end
