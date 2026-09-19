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

violations = Dict{String, Vector{Symbol}}()
for file in semantic_test_files(@__DIR__)
    source = read(file, String)
    refs = union(explicit_hc_imports(source), qualified_hc_references(source))
    private_refs = sort!(collect(setdiff(Set(refs), HC_PUBLIC_NAMES)); by = string)
    isempty(private_refs) || (violations[relpath(file, @__DIR__)] = private_refs)
end

if !isempty(violations)
    println(stderr, "Non-public HomotopyContinuation references in the ordinary semantic suite:")
    for file in sort!(collect(keys(violations)))
        println(stderr, "  ", file, ": ", join(string.(violations[file]), ", "))
    end
end

@testset "normal tests use only public HomotopyContinuation API" begin
    @test isempty(violations)
end
