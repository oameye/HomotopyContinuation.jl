# First-use workload for the TTFX pull-request check. It runs after
# `using HomotopyContinuation` and covers the default user paths from the shared
# workload corpus: a total-degree solve, a polyhedral solve and a parameter
# continuation, all serial.
include(joinpath(@__DIR__, "..", "..", "workloads", "workloads.jl"))

for name in (
        :total_degree_interpreted_serial,
        :polyhedral_interpreted_serial,
        :parameter_interpreted_serial,
    )
    Workloads.run(name)
end
