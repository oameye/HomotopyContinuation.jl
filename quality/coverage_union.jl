using Printf

artifacts_root = length(ARGS) >= 1 ? abspath(ARGS[1]) : error("artifact root required")
output_dir = length(ARGS) >= 2 ? abspath(ARGS[2]) : joinpath(pwd(), "coverage-union")

statefiles = String[]
for (dir, _, names) in walkdir(artifacts_root)
    "state.tsv" in names && push!(statefiles, joinpath(dir, "state.tsv"))
end
sort!(statefiles)
isempty(statefiles) && error("no per-runtime state.tsv files found under $artifacts_root")

production_files = Set{String}()
observed = Dict{String, Bool}()
instrumented = Dict{String, Set{Int}}()
hit = Dict{String, Set{Int}}()
runtimes = Set{String}()

for statefile in statefiles
    for (i, line) in enumerate(eachline(statefile))
        i == 1 && continue
        fields = split(line, '\t')
        length(fields) == 5 || error("malformed state row in $statefile: $line")
        kind, file, line_field, hit_field, runtime = fields
        push!(runtimes, runtime)
        push!(production_files, file)
        if kind == "FILE"
            seen = parse(Int, hit_field) != 0
            observed[file] = get(observed, file, false) || seen
        elseif kind == "LINE"
            line_number = parse(Int, line_field)
            reached = parse(Int, hit_field) != 0
            push!(get!(instrumented, file, Set{Int}()), line_number)
            reached && push!(get!(hit, file, Set{Int}()), line_number)
        else
            error("unknown state row kind $kind in $statefile")
        end
    end
end

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

rows = NamedTuple[]
for file in sort!(collect(production_files))
    lines = get(instrumented, file, Set{Int}())
    reached = get(hit, file, Set{Int}())
    total = length(lines)
    covered = length(reached)
    missed = total - covered
    fraction = total == 0 ? 0.0 : covered / total
    push!(rows, (; file, total, covered, missed, fraction, observed = get(observed, file, false)))
end

total_lines = sum(row.total for row in rows)
covered_lines = sum(row.covered for row in rows)
missed_lines = total_lines - covered_lines
fraction = total_lines == 0 ? 1.0 : covered_lines / total_lines
unobserved = filter(row -> !row.observed, rows)

mkpath(output_dir)
summary_path = joinpath(output_dir, "summary.md")
uncovered_path = joinpath(output_dir, "uncovered.txt")
inventory_path = joinpath(output_dir, "inventory.tsv")

runtime_list = join(sort!(collect(runtimes)), ", ")
open(summary_path, "w") do io
    println(io, "# Public semantic-suite supported-runtime coverage union")
    println(io)
    println(io, "Runtime traces: ", runtime_list)
    println(io)
    println(io, "This is the union of line/reachability evidence across supported runtimes. A line is reached if any runtime reaches it. It is not branch coverage and is not by itself a deletion decision.")
    println(io)
    @printf(io, "Instrumented production lines reached: **%d/%d (%.2f%%)**; %d unreached on every collected runtime.\n\n", covered_lines, total_lines, 100 * fraction, missed_lines)
    println(io, "Production files absent from every runtime trace: **", length(unobserved), "**.")
    println(io)
    println(io, "| Production file | Observed anywhere | Reached | Instrumented union | Unreached everywhere | Coverage |")
    println(io, "| --- | --- | ---: | ---: | ---: | ---: |")
    for row in sort(rows; by = row -> (row.observed ? 1 : 0, row.fraction, row.file))
        coverage = row.total == 0 ? "n/a" : @sprintf("%.2f%%", 100 * row.fraction)
        println(io, "| `", row.file, "` | ", row.observed ? "yes" : "no", " | ", row.covered, " | ", row.total, " | ", row.missed, " | ", coverage, " |")
    end
end

open(uncovered_path, "w") do io
    for row in sort(rows; by = row -> row.file)
        if !row.observed
            println(io, row.file, ":NOT_IN_ANY_TRACE")
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

println("runtime states: ", length(statefiles), " [", runtime_list, "]")
@printf("supported-runtime production reachability: %d/%d (%.2f%%), %d unreached; %d files absent from every trace\n", covered_lines, total_lines, 100 * fraction, missed_lines, length(unobserved))
println("summary: ", summary_path)
println("uncovered everywhere: ", uncovered_path)
