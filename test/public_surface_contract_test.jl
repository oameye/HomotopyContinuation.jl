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
        :accepted_steps,
        :clusters,
        :condition_jacobian,
        :evaluator,
        :last_path_point,
        :multiplicity,
        :path_number,
        :path_results,
        :polys,
        :rejected_steps,
        :seed,
        :solution,
        :start_solution,
        :statistics,
        :tracked_loops,
        :tracked_paths,
        :valuation,
        :winding_number,
    ],
)

function hc_aliases(source::String)
    aliases = Set(["HomotopyContinuation"])
    for m in eachmatch(r"(?m)^\s*(?:using|import)\s+HomotopyContinuation\s+as\s+([A-Za-z_][A-Za-z0-9_]*)", source)
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

function property_references(source::String)
    refs = Set{Symbol}()
    pos = firstindex(source)
    stop = ncodeunits(source)
    while pos <= stop
        ex, next = Meta.parse(source, pos; raise = false)
        ex === nothing && break
        ex isa Expr && ex.head === :error && error("failed to parse semantic-test source near byte $pos")
        collect_property_references!(refs, ex)
        next > pos || error("parser made no progress near byte $pos")
        pos = next
    end
    return refs
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

name_violations = Dict{String, Vector{Symbol}}()
property_violations = Dict{String, Vector{Symbol}}()
for file in semantic_test_files(@__DIR__)
    source = read(file, String)
    refs = union(explicit_hc_imports(source), qualified_hc_references(source))
    private_refs = sort!(collect(setdiff(Set(refs), HC_PUBLIC_NAMES)); by = string)
    isempty(private_refs) || (name_violations[relpath(file, @__DIR__)] = private_refs)

    private_properties = sort!(collect(intersect(property_references(source), HC_PRIVATE_STORAGE_PROPERTIES)); by = string)
    isempty(private_properties) ||
        (property_violations[relpath(file, @__DIR__)] = private_properties)
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

@testset "normal tests use only public HomotopyContinuation API" begin
    @test isempty(name_violations)
    @test isempty(property_violations)
end
