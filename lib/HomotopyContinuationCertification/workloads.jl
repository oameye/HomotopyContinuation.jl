module CertificationWorkloads

# One workload per public certification feature, written through the exported
# API only. Each workload builds and solves its own problem first.

using HomotopyContinuation: @polyvar, @var, System, TotalDegree, Monodromy,
    DuplicateCheck, Serial, Threaded, solve, solutions, result_iterator
using HomotopyContinuationCertification: Certification, IteratorCertification, certify,
    certificates, certified_solution_interval, distinct_certified_solutions, save,
    ncertified, ndistinct_real_certified, is_real

const SEED = UInt32(0x1234)

quiet() = Certification(; show_progress = false)

function circle_and_line()
    @polyvar x y
    F = System([x^2 + y^2 - 1, x - y]; variables = [x, y])
    return F, solve(F, TotalDegree(; seed = SEED, show_progress = false), Serial())
end

const WORKLOADS = (
    :certify_result,
    :certify_extended_threaded,
    :certify_parameters,
    :distinct_certified,
    :certificate_queries,
    :iterator_certification,
    :monodromy_certified_duplicates,
)

function run(::Val{:certify_result})
    F, result = circle_and_line()
    return certify(F, result, quiet())
end

function run(::Val{:certify_extended_threaded})
    F, result = circle_and_line()
    alg = Certification(; extended_certificate = true, show_progress = false)
    return certify(F, solutions(result), nothing, alg, Threaded(1))
end

function run(::Val{:certify_parameters})
    @polyvar x a
    F = System([x^2 - a]; variables = [x], parameters = [a])
    return certify(F, [[2.0 + 0im], [-2.0 + 0im]], [4.0 + 0im], quiet(), Serial())
end

function run(::Val{:distinct_certified})
    F, result = circle_and_line()
    S = solutions(result)
    return distinct_certified_solutions(F, [S; S], nothing, quiet())
end

function run(::Val{:certificate_queries})
    F, result = circle_and_line()
    cert = certify(F, result, quiet())
    c = first(certificates(cert))
    text = mktempdir() do dir
        path = joinpath(dir, "certificates")
        save(path, cert)
        return read(path, String)
    end
    return ncertified(cert), ndistinct_real_certified(cert), is_real(c),
        certified_solution_interval(c), text
end

function run(::Val{:iterator_certification})
    @var x y
    F = System([x^2 - 1, y - 1]; variables = [x, y])
    alg = IteratorCertification(; show_progress = false, boundaries = -3:3)
    return certify(F, result_iterator(F), nothing, alg)
end

function run(::Val{:monodromy_certified_duplicates})
    @polyvar x a
    F = System([x^2 - a]; variables = [x], parameters = [a])
    alg = Monodromy(;
        duplicate_check = DuplicateCheck.CERTIFIED, show_progress = false,
        target_solutions_count = 2, seed = SEED,
    )
    return solve(F, [[1.0 + 0im]], [1.0 + 0im], alg, Serial())
end

workloads() = WORKLOADS

end
