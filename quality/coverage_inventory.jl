using Printf

repo_root = realpath(joinpath(@__DIR__, ".."))
coverage_dir = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(repo_root, "coverage")
output_dir = length(ARGS) >= 2 ? abspath(ARGS[2]) : coverage_dir
runtime_label = length(ARGS) >= 3 ? ARGS[3] : string(VERSION)
const PRODUCTION_ROOTS = ("src", "ext")

tracefiles = sort(filter(path -> endswith(path, ".info"), readdir(coverage_dir; join = true)))
isempty(tracefiles) && error("no LCOV tracefiles found in $coverage_dir")

function production_files()
    files = String[]
    for root in PRODUCTION_ROOTS
        absolute_root = joinpath(repo_root, root)
        isdir(absolute_root) || continue
        for (dir, _, names) in walkdir(absolute_root)
            for name in names
                endswith(name, ".jl") || continue
                rel = replace(relpath(joinpath(dir, name), repo_root), '\\' => '/')
                push!(files, rel)
            end
        end
    end
    return sort!(unique!(files))
end

function production_relative_path(path::AbstractString)
    absolute = isabspath(path) ? normpath(path) : normpath(joinpath(repo_root, path))
    rel = replace(relpath(absolute, repo_root), '\\' => '/')
    for root in PRODUCTION_ROOTS
        startswith(rel, root * "/") && return rel
    end
    return nothing
end

instrumented = Dict{String, Set{Int}}()
hit = Dict{String, Set{Int}}()
for tracefile in tracefiles
    current = nothing
    for line in eachline(tracefile)
        if startswith(line, "SF:")
            current = production_relative_path(line[4:end])
            if current !== nothing
                get!(instrumented, current, Set{Int}())
                get!(hit, current, Set{Int}())
            end
        elseif current !== nothing && startswith(line, "DA:")
            fields = split(line[4:end], ',')
            length(fields) >= 2 || continue
            line_number = tryparse(Int, fields[1])
            count = tryparse(Int, fields[2])
            (line_number === nothing || count === nothing) && continue
            push!(instrumented[current], line_number)
            count > 0 && push!(hit[current], line_number)
        elseif line == "end_of_record"
            current = nothing
        end
    end
end

isempty(instrumented) && error("LCOV traces contain no instrumented production lines")

function line_ranges(lines)
    xs = sort!(collect(lines))
    isempty(xs) && return ""
    parts = String[]
    first_line = last_line = xs[1]
    for x in @view xs[2:end]
        if x == last_line + 1
            last_line = x
        else
            push!(parts, first_line == last_line ? string(first_line) : "$first_line-$last_line")
            first_line = last_line = x
        end
    end
    push!(parts, first_line == last_line ? string(first_line) : "$first_line-$last_line")
    return join(parts, ",")
end

files = production_files()
observed = Set(keys(instrumented))
rows = NamedTuple[]
for file in files
    lines = get(instrumented, file, Set{Int}())
    reached = get(hit, file, Set{Int}())
    total = length(lines)
    covered = length(reached)
    missed = total - covered
    fraction = total == 0 ? 0.0 : covered / total
    push!(rows, (; file, total, covered, missed, fraction, observed = file in observed))
end

instrumented_files = filter(row -> row.total > 0, rows)
total_lines = sum(row.total for row in instrumented_files)
covered_lines = sum(row.covered for row in instrumented_files)
missed_lines = total_lines - covered_lines
fraction = total_lines == 0 ? 1.0 : covered_lines / total_lines
unobserved = filter(row -> !row.observed, rows)

mkpath(output_dir)
summary_path = joinpath(output_dir, "summary.md")
uncovered_path = joinpath(output_dir, "uncovered.txt")
inventory_path = joinpath(output_dir, "inventory.tsv")
state_path = joinpath(output_dir, "state.tsv")

open(summary_path, "w") do io
    println(io, "# Public semantic-suite coverage inventory — ", runtime_label)
    println(io)
    println(io, "Runtime: Julia $(VERSION)")
    println(io)
    println(io, "This is line/reachability evidence from the ordinary public-API-only semantic suite. It is not branch coverage and is not by itself a deletion decision.")
    println(io)
    @printf(io, "Instrumented production lines reached: **%d/%d (%.2f%%)**; %d unreached.\n\n", covered_lines, total_lines, 100 * fraction, missed_lines)
    println(io, "Production files absent from this runtime's LCOV trace: **", length(unobserved), "**.")
    println(io)
    println(io, "| Production file | Observed | Reached | Instrumented | Unreached | Coverage |")
    println(io, "| --- | --- | ---: | ---: | ---: | ---: |")
    for row in sort(rows; by = row -> (row.observed ? 1 : 0, row.fraction, row.file))
        coverage = row.total == 0 ? "n/a" : @sprintf("%.2f%%", 100 * row.fraction)
        println(io, "| `", row.file, "` | ", row.observed ? "yes" : "no", " | ", row.covered, " | ", row.total, " | ", row.missed, " | ", coverage, " |")
    end
end

open(uncovered_path, "w") do io
    for row in sort(rows; by = row -> row.file)
        if !row.observed
            println(io, row.file, ":NOT_IN_TRACE")
            continue
        end
        missed = setdiff(get(instrumented, row.file, Set{Int}()), get(hit, row.file, Set{Int}()))
        isempty(missed) || println(io, row.file, ':', line_ranges(missed))
    end
end

open(inventory_path, "w") do io
    println(io, "file\tobserved\treached\tinstrumented\tunreached\tcoverage")
    for row in sort(rows; by = row -> row.file)
        @printf(io, "%s\t%d\t%d\t%d\t%d\t%.8f\n", row.file, row.observed ? 1 : 0, row.covered, row.total, row.missed, row.fraction)
    end
    @printf(io, "TOTAL\t1\t%d\t%d\t%d\t%.8f\n", covered_lines, total_lines, missed_lines, fraction)
end

open(state_path, "w") do io
    println(io, "kind\tfile\tline\thit\truntime")
    for row in sort(rows; by = row -> row.file)
        println(io, "FILE\t", row.file, "\t0\t", row.observed ? 1 : 0, "\t", runtime_label)
        for line in sort!(collect(get(instrumented, row.file, Set{Int}())))
            println(io, "LINE\t", row.file, '\t', line, '\t', line in get(hit, row.file, Set{Int}()) ? 1 : 0, '\t', runtime_label)
        end
    end
end

println("coverage traces: ", length(tracefiles))
@printf("production reachability: %d/%d (%.2f%%), %d unreached; %d files absent from trace\n", covered_lines, total_lines, 100 * fraction, missed_lines, length(unobserved))
println("summary: ", summary_path)
println("state: ", state_path)
