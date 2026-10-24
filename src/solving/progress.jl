mutable struct ProgressStats
    nonsingular::Int
    singular::Int
    nonsingular_real::Int
    singular_real::Int
end

ProgressStats() = ProgressStats(0, 0, 0, 0)

# Fold a finished path into the running tallies.
function record!(stats::ProgressStats, r::PathResult)::Nothing
    if is_success(r)
        rl = is_real(r)
        if counts_as_singular(r)
            stats.singular += 1
            rl && (stats.singular_real += 1)
        else
            stats.nonsingular += 1
            rl && (stats.nonsingular_real += 1)
        end
    end
    return nothing
end

function progress_showvalues(stats::ProgressStats, ntracked::Int)
    total = stats.nonsingular + stats.singular
    total_real = stats.nonsingular_real + stats.singular_real
    return (
        ("# paths tracked", ntracked),
        ("# non-singular solutions (real)", string(stats.nonsingular, " (", stats.nonsingular_real, ")")),
        ("# singular endpoints (real)", string(stats.singular, " (", stats.singular_real, ")")),
        ("# total solutions (real)", string(total, " (", total_real, ")")),
    )
end

# `stdout` is a global of abstract type `IO`, so a meter built with `output = stdout`
# has the runtime type of whatever stream it is, which inference cannot see. Storing
# the stream behind `IO` keeps the meter one concrete type.
const ProgressBar = ProgressMeter.Progress{IO}
const ProgressSpinner = ProgressMeter.ProgressUnknown{IO}

function stdout_progress_core(desc::String, dt::Float64)::ProgressMeter.ProgressCore{IO}
    core = ProgressMeter.ProgressCore{IO}(; output = stdout, dt = dt, desc = desc)
    core.tlast += 0.3
    return core
end

# `barlen` is left unset so ProgressMeter auto-sizes the bar to the terminal width.
make_progress(n::Int, desc::String = "Tracking $n paths... ")::ProgressBar =
    ProgressMeter.Progress(n, stdout_progress_core(desc, 0.2))

make_many_progress(n::Int)::ProgressBar =
    ProgressMeter.Progress(n, stdout_progress_core("Solving for $n targets... ", 0.2))

make_progress_spinner(desc::String, dt::Float64, spinner::Bool)::ProgressSpinner =
    ProgressMeter.ProgressUnknown(spinner, stdout_progress_core(desc, dt))

@inline update_many_progress!(::Nothing, ::Int, ::Int)::Nothing = nothing
function update_many_progress!(
        progress::ProgressBar, nsolved::Int, ntracked::Int,
    )::Nothing
    ProgressMeter.update!(
        progress, nsolved;
        showvalues = (("# targets solved", nsolved), ("# paths tracked", ntracked)),
    )
    return nothing
end

@inline update_progress!(::Nothing, ntracked::Int, ::ProgressStats, ::PathResult)::Nothing = nothing
function update_progress!(
        progress::ProgressBar, ntracked::Int, stats::ProgressStats, r::PathResult,
    )::Nothing
    record!(stats, r)
    ProgressMeter.update!(progress, ntracked; showvalues = progress_showvalues(stats, ntracked))
    return nothing
end

@inline next_progress!(::Nothing)::Nothing = nothing
function next_progress!(progress::ProgressBar)::Nothing
    ProgressMeter.next!(progress)
    return nothing
end
