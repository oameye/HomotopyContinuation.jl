## Bertini-format text serialization of solution lists and parameter vectors.

"""
    write_solutions(filename, solutions)

Write `solutions` to `filename` as plain text. The first line holds the number of
solutions; each solution follows after a blank line, one number per line as its
real and imaginary part separated by a space. Numbers are always written as
complex.

This is the file format used by [Bertini](https://bertini.nd.edu). Read it back
with [`read_solutions`](@ref).

## Example

```julia
julia> write_solutions("solutions.txt", [[1, 1], [-1, 2]])

shell> cat solutions.txt
2

1 0
1 0

-1 0
2 0

julia> read_solutions("solutions.txt")
2-element Vector{Vector{ComplexF64}}:
 [1.0 + 0.0im, 1.0 + 0.0im]
 [-1.0 + 0.0im, 2.0 + 0.0im]
```
"""
function write_solutions(
        filename::AbstractString,
        S::AbstractVector{<:AbstractVector{<:Number}},
    )::Nothing
    open(filename, "w") do io
        println(io, length(S))
        for s in S
            println(io)
            for z in s
                println(io, real(z), " ", imag(z))
            end
        end
    end
    return
end

"""
    read_solutions(filename)

Read the solutions written by [`write_solutions`](@ref).
"""
function read_solutions(filename::AbstractString)::Vector{Vector{ComplexF64}}
    out = Vector{ComplexF64}[]
    current = ComplexF64[]
    n = -1
    for line in eachline(filename)
        stripped = strip(line)
        if isempty(stripped)
            isempty(current) || push!(out, current)
            current = ComplexF64[]
        elseif n < 0
            n = parse(Int, stripped)
        else
            push!(current, _parse_complex(stripped, filename))
        end
    end
    isempty(current) || push!(out, current)
    _check_count(length(out), n, filename, "solutions")
    return out
end

"""
    write_parameters(filename, parameters)

Write `parameters` to `filename` as plain text. The first line holds the number
of values, then a blank line, then one value per line as its real and imaginary
part separated by a space.

This is the file format used by [Bertini](https://bertini.nd.edu). Read it back
with [`read_parameters`](@ref).

## Example

```julia
julia> write_parameters("parameters.txt", [2.0, -3.2 + 2im])

shell> cat parameters.txt
2

2.0 0.0
-3.2 2.0

julia> read_parameters("parameters.txt")
2-element Vector{ComplexF64}:
  2.0 + 0.0im
 -3.2 + 2.0im
```
"""
function write_parameters(filename::AbstractString, p::AbstractVector{<:Number})::Nothing
    open(filename, "w") do io
        println(io, length(p))
        println(io)
        for z in p
            println(io, real(z), " ", imag(z))
        end
    end
    return
end

"""
    read_parameters(filename)

Read the parameter values written by [`write_parameters`](@ref).
"""
function read_parameters(filename::AbstractString)::Vector{ComplexF64}
    out = ComplexF64[]
    n = -1
    for line in eachline(filename)
        stripped = strip(line)
        if isempty(stripped)
            continue
        elseif n < 0
            n = parse(Int, stripped)
        else
            push!(out, _parse_complex(stripped, filename))
        end
    end
    _check_count(length(out), n, filename, "parameter values")
    return out
end

function _parse_complex(line::AbstractString, filename::AbstractString)::ComplexF64
    fields = split(line)
    length(fields) <= 2 || throw(
        ArgumentError(
            string(
                filename, ": expected a real and an imaginary part, got ",
                length(fields), " fields in \"", line, "\"",
            ),
        ),
    )
    re = parse(Float64, fields[1])
    im = length(fields) == 2 ? parse(Float64, fields[2]) : 0.0
    return complex(re, im)
end

function _check_count(
        got::Int, declared::Int, filename::AbstractString, what::AbstractString,
    )::Nothing
    declared >= 0 || throw(ArgumentError(string(filename, " is empty")))
    got == declared || throw(
        ArgumentError(
            string(
                filename, " declares ", declared, " ", what, " but holds ", got,
            ),
        ),
    )
    return
end
