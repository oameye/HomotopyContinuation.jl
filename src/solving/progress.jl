# **Mutable justification:** updated in place as each path completes; a single
# instance is shared (lock-guarded) across the threaded solve.
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
        if r.singular
            stats.singular += 1
            rl && (stats.singular_real += 1)
        else
            stats.nonsingular += 1
            rl && (stats.nonsingular_real += 1)
        end
    end
    return nothing
end

function _showvalues(stats::ProgressStats, ntracked::Int)
    total = stats.nonsingular + stats.singular
    total_real = stats.nonsingular_real + stats.singular_real
    return (
        ("# paths tracked", ntracked),
        ("# non-singular solutions (real)", string(stats.nonsingular, " (", stats.nonsingular_real, ")")),
        ("# singular endpoints (real)", string(stats.singular, " (", stats.singular_real, ")")),
        ("# total solutions (real)", string(total, " (", total_real, ")")),
    )
end

function make_progress(n::Int, show::Bool; delay::Float64 = 0.3, desc::String = "Tracking $n paths... ")
    show || return nothing
    # `barlen` is left unset so ProgressMeter auto-sizes the bar to the terminal
    # width (avoids depending on the internal `tty_width`).
    progress = ProgressMeter.Progress(n; dt = 0.2, desc = desc, output = stdout)
    progress.tlast += delay
    return progress
end

function make_many_progress(n::Int, show::Bool; delay::Float64 = 0.3)
    show || return nothing
    progress = ProgressMeter.Progress(
        n; dt = 0.2, desc = "Solving for $n targets... ", output = stdout,
    )
    progress.tlast += delay
    return progress
end

update_many_progress!(::Nothing, ::Int, ::Int)::Nothing = nothing
function update_many_progress!(
        progress::ProgressMeter.Progress, nsolved::Int, ntracked::Int,
    )::Nothing
    ProgressMeter.update!(
        progress, nsolved;
        showvalues = (("# targets solved", nsolved), ("# paths tracked", ntracked)),
    )
    return nothing
end

update_progress!(::Nothing, ntracked::Int, ::ProgressStats, ::PathResult)::Nothing = nothing
function update_progress!(
        progress::ProgressMeter.Progress, ntracked::Int, stats::ProgressStats, r::PathResult,
    )::Nothing
    record!(stats, r)
    ProgressMeter.update!(progress, ntracked; showvalues = _showvalues(stats, ntracked))
    return nothing
end
