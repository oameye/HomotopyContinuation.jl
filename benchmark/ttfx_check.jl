# First-use workload for the TTFX pull-request check. It runs after
# `using HomotopyContinuation` and covers the default user paths: a total-degree
# solve, a polyhedral solve and a parameter continuation, all serial.
@polyvar x y p
F = System([x^2 + y - 1, x * y - 0.5]; variables = [x, y])
solve(F, TotalDegree(; seed = UInt32(0x1234), show_progress = false), Serial())
solve(F, Polyhedral(; seed = UInt32(0x1234), show_progress = false), Serial())
G = System([x^2 - p]; variables = [x], parameters = [p])
solve(
    G,
    [[1.0 + 0.0im]],
    [1.0 + 0.0im],
    [9.0 + 0.0im],
    Continuation(; seed = UInt32(1), show_progress = false),
    Serial(),
)
