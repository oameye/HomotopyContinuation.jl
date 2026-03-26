## Result — aggregate result from solve().

struct Result
    path_results::Vector{PathResult}
    tracked_paths::Int
    seed::UInt32
end

function Base.show(io::IO, r::Result)
    n_success = count(is_success, r.path_results)
    n_real = count(is_real, r.path_results)
    print(io, "Result with ", r.tracked_paths, " tracked paths\n")
    print(io, " • ", n_success, " solutions (", n_real, " real)\n")
    n_failed = r.tracked_paths - n_success
    if n_failed > 0
        print(io, " • ", n_failed, " paths failed")
    end
    return nothing
end

function solutions(r::Result; only_real::Bool = false, real_tol::Float64 = DEFAULT_REAL_TOL)::Vector{Vector{ComplexF64}}
    filter_fn = if only_real
        pr -> is_success(pr) && is_real(pr; tol = real_tol)
    else
        is_success
    end
    return [pr.solution for pr in r.path_results if filter_fn(pr)]
end

function real_solutions(r::Result; tol::Float64 = DEFAULT_REAL_TOL)::Vector{Vector{Float64}}
    return [
        Float64.(real.(pr.solution)) for pr in r.path_results
            if is_success(pr) && is_real(pr; tol = tol)
    ]
end

nsolutions(r::Result)::Int = count(is_success, r.path_results)

nreal(r::Result; tol::Float64 = DEFAULT_REAL_TOL)::Int =
    count(pr -> is_success(pr) && is_real(pr; tol = tol), r.path_results)
