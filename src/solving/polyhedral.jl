## Polyhedral — BKK-optimal start system via mixed subdivisions.
#
# Two-phase polyhedral homotopy:
#   Phase 1 (toric):       Track from binomial start solutions through ToricHomotopy (t: 0 -> 1)
#   Phase 2 (coefficient): Track from generic system to target through CoefficientHomotopy (t: 1 -> 0)

"""
    Polyhedral(; only_torus, early_stop_callback, excess_residual_tol,
                 tracker_options, endgame_options, seed, show_progress)

Algorithm that constructs a polyhedral (BKK-optimal) start system using mixed subdivisions.
The number of paths tracked equals the mixed volume, which is at most the Bezout bound.

`only_torus = true` computes only the solutions with all coordinates non-zero.
That start system tracks [`mixed_volume`](@ref) paths, fewer than the default,
which pads every support with the zero exponent vector so that the solutions on
the coordinate hyperplanes are found as well.

`early_stop_callback` is called with each successful [`PathResult`](@ref); return
`true` to stop. Paths already running still finish, so which extra results appear
is not reproducible under [`Threaded`](@ref) or [`DistributedExecutor`](@ref).

For an overdetermined system, an endpoint solving the squared-up system but not
the original one is reported as an excess solution. A positive
`excess_residual_tol` instead keeps such an endpoint when its residual on the
original system is at most that value, which is what a system consistent only up
to a measurement error needs. It is compared against the same quantity
[`residual`](@ref) reports, so it is read on the equations [`evaluate`](@ref)
gives; divide by [`equation_scales`](@ref) to state it in the units of a system
whose coefficients were normalized. Must be non-negative.

# Examples
```julia
@polyvar x y
result = solve(System([x^2 + y - 1, x*y - 2]), Polyhedral())

# Only the solutions in the torus
result = solve(F, Polyhedral(; only_torus = true))

# Tune tracker options
result = solve(F, Polyhedral(; tracker_options = TrackerOptions(; max_steps = 500)))
```
"""
struct Polyhedral <: AbstractAlgorithm
    common::CommonOptions
    early_stop::EarlyStop
    only_torus::Bool
    excess_residual_tol::Float64
end

Polyhedral(;
    only_torus::Bool = false,
    early_stop_callback = _never_stop,
    excess_residual_tol::Float64 = 0.0,
    tracker_options::TrackerOptions = TrackerOptions(),
    endgame_options::EndgameOptions = EndgameOptions(),
    seed::UInt32 = rand(Random.RandomDevice(), UInt32),
    show_progress::Bool = true,
) = Polyhedral(
    CommonOptions(tracker_options, endgame_options, seed, show_progress),
    _early_stop(early_stop_callback),
    only_torus,
    _checked_excess_residual_tol(excess_residual_tol),
)

early_stop_callback(alg::Polyhedral)::EarlyStop = alg.early_stop

excess_residual_tol(alg::Polyhedral)::Float64 = alg.excess_residual_tol

# Every option but `common` survives, so a new field reaches `_reseed`/`_quiet` here.
_with_common(alg::Polyhedral, common::CommonOptions)::Polyhedral = Polyhedral(
    common, alg.early_stop, alg.only_torus, alg.excess_residual_tol,
)

# A parent derives its children's seeds from its own, so one top-level seed
# reproduces every stage.
_reseed(alg::Polyhedral, seed::UInt32)::Polyhedral =
    _with_common(alg, _with_seed(alg.common, seed))

_quiet(alg::Polyhedral)::Polyhedral = _with_common(alg, _quiet(alg.common))

"""
    PolyhedralSolveCache

Holds pre-built trackers and start solutions for the two-phase polyhedral homotopy.
Created by `CommonSolve.init`.
"""
struct PolyhedralSolveCache{E <: AbstractExecutor, B <: PolyhedralBuilder, S, C}
    executor::E
    builder::B
    toric_tracker::Tracker
    coeff_tracker::EndgameTracker
    toric_homotopy::ToricHomotopy
    support::Vector{Matrix{Int32}}
    lifting::Vector{Vector{Int32}}
    start_solutions::Vector{Tuple{MixedSubdivisions.MixedCell, Vector{ComplexF64}}}
    seed::UInt32
    # carries a point from the toric phase to the coefficient phase
    x_buffer::Vector{ComplexF64}
    # GC roots for interpreters — must be kept alive for FunctionWrapper closures
    _support_system::S
    # ExcessSolutionChecker for overdetermined systems, Nothing for square ones
    excess_checker::C
    show_progress::Bool
    early_stop::EarlyStop
end

# This cache pairs every start point with the mixed cell it belongs to.
_start_points(
    starts::Vector{Tuple{MixedSubdivisions.MixedCell, Vector{ComplexF64}}},
)::Vector{Vector{ComplexF64}} = [copy(x) for (_, x) in starts]

# ── Helper: randomize support/coefficients for overdetermined systems ───────

"""
    _randomize_support(support, coeffs, A, perm) -> (support, coeffs)

Support and coefficients of the squared-up system G = [I A]·(F∘perm). Row i of G
is `F_perm[i] + Σ_j A[i,j]·F_perm[n+j]`, so its support is the union of the
combined supports with linearly combined coefficients. Duplicate monomials are
merged; exponent columns are sorted for a deterministic result.
"""
function _randomize_support(
        support::Vector{Matrix{Int32}},
        coeffs::Vector{Vector{ComplexF64}},
        A::FSMat{ComplexF64},
        perm::Vector{Int},
    )::Tuple{Vector{Matrix{Int32}}, Vector{Vector{ComplexF64}}}
    n, k = size(A)
    new_support = Vector{Matrix{Int32}}(undef, n)
    new_coeffs = Vector{Vector{ComplexF64}}(undef, n)
    for i in 1:n
        acc = Dict{Vector{Int32}, ComplexF64}()
        S_i = support[perm[i]]
        c_i = coeffs[perm[i]]
        for t in axes(S_i, 2)
            col = S_i[:, t]
            acc[col] = get(acc, col, zero(ComplexF64)) + c_i[t]
        end
        for j in 1:k
            S_j = support[perm[n + j]]
            c_j = coeffs[perm[n + j]]
            a = A[i, j]
            for t in axes(S_j, 2)
                col = S_j[:, t]
                acc[col] = get(acc, col, zero(ComplexF64)) + a * c_j[t]
            end
        end
        exponents = sort!(collect(keys(acc)))
        new_support[i] = reduce(hcat, exponents)
        new_coeffs[i] = ComplexF64[acc[e] for e in exponents]
    end
    return new_support, new_coeffs
end

# ── Helper: build coefficient-parametric evaluator from support ─────────

"""
    _SupportSystem

Minimal reconstruction data for `Fᵢ(x; p) = Σⱼ pᵢⱼ x^support[i][:,j]`.
Polyhedral tracking needs only an evaluator and immutable instruction sequences
for worker-local tapes; no symbolic variables or polynomial objects are needed.
"""
struct _SupportSystem
    evaluator::SystemEvaluator
    eval_sequence::InstructionSequence
    jacobian_sequence::InstructionSequence
end

# Sparse (variable_slot, exponent) key for the monomial x^support[:, term],
# optionally differentiated once wrt `derivative_variable` (0 = no derivative).
# Shares `MonomialCache` with the symbolic-free polynomial frontend so identical
# monomials collapse to a single tape slot.
function _support_monomial_key(
        compiler::TapeCompiler, support::Matrix{Int32},
        term::Int, derivative_variable::Int,
    )::MonomialKey
    data = Int32[]
    for variable in axes(support, 1)
        exponent = support[variable, term]
        variable == derivative_variable && (exponent -= one(Int32))
        iszero(exponent) && continue
        push!(data, compiler.var_slots[variable])
        push!(data, exponent)
    end
    return MonomialKey(data)
end

# Returns the (parameter_slot, monomial_slot) pair for one support term, folding
# an optional integer multiplier (the power-rule factor) into the tape.
function _support_term_parts!(
        cache::MonomialCache, parameter::Int, key::MonomialKey, multiplier::Int = 1,
    )::Tuple{Int32, Int32}
    monomial_slot = _monomial_slot!(cache, key)
    monomial_slot = _coeff_mul_slot!(cache.compiler, ComplexF64(multiplier), monomial_slot)
    return cache.compiler.param_slots[parameter], monomial_slot
end

function _compile_support_equation!(
        cache::MonomialCache, support::Matrix{Int32}, parameter_offset::Int,
    )::Int32
    terms = Vector{Tuple{Int32, Int32}}(undef, size(support, 2))
    for term in axes(support, 2)
        key = _support_monomial_key(cache.compiler, support, term, 0)
        terms[term] = _support_term_parts!(cache, parameter_offset + term, key)
    end
    return _compile_sum_products!(cache.compiler, terms)
end

function _compile_support_derivative!(
        cache::MonomialCache, support::Matrix{Int32},
        parameter_offset::Int, variable::Int,
    )::Int32
    terms = Tuple{Int32, Int32}[]
    for term in axes(support, 2)
        exponent = support[variable, term]
        iszero(exponent) && continue
        key = _support_monomial_key(cache.compiler, support, term, variable)
        push!(
            terms,
            _support_term_parts!(cache, parameter_offset + term, key, Int(exponent)),
        )
    end
    isempty(terms) && return _get_constant_slot!(cache.compiler, zero(ComplexF64))
    return _compile_sum_products!(cache.compiler, terms)
end

function _support_parameter_count(support::Vector{Matrix{Int32}})::Int
    count = 0
    for A in support
        count += size(A, 2)
    end
    return count
end

function _build_support_instruction_sequence(
        support::Vector{Matrix{Int32}}, include_jacobian::Bool,
    )::InstructionSequence
    nequations = length(support)
    nvariables = size(first(support), 1)
    nparams = _support_parameter_count(support)
    compiler = TapeCompiler(nvariables, nparams)
    _initialize_placeholder_slots!(compiler, nvariables, nparams)
    cache = MonomialCache(compiler)

    parameter_offsets = Vector{Int}(undef, nequations)
    offset = 0
    for equation in eachindex(support)
        parameter_offsets[equation] = offset
        offset += size(support[equation], 2)
    end

    result_slots = Int32[]
    sizehint!(result_slots, nequations * (include_jacobian ? nvariables + 1 : 1))
    for equation in eachindex(support)
        push!(
            result_slots,
            _compile_support_equation!(
                cache, support[equation], parameter_offsets[equation],
            ),
        )
    end
    if include_jacobian
        for variable in 1:nvariables
            for equation in eachindex(support)
                push!(
                    result_slots,
                    _compile_support_derivative!(
                        cache, support[equation], parameter_offsets[equation], variable,
                    ),
                )
            end
        end
    end

    return _finalize_compiler(
        Val(false), compiler, result_slots,
        nvariables, nparams, nequations,
    )
end

function _support_evaluator(
        eval_sequence::InstructionSequence,
        jacobian_sequence::InstructionSequence,
        nequations::Int,
        nvariables::Int,
        nparams::Int,
    )::SystemEvaluator
    # Polyhedral tracking evaluates order 1 directly and requests Taylor
    # coefficients only at orders 2 and 3, always with Taylor parameters.
    # Constructing the general System evaluator here eagerly compiled four
    # unreachable kernels (three scalar-parameter Taylor methods and the
    # order-1 parameter-Taylor method), including an otherwise-unused order-1
    # Taylor tape. Keep the concrete SystemEvaluator firewall, but make this
    # narrower internal capability explicit with fail-fast wrappers.
    interp_f64 = Interpreter(Vector{ComplexF64}, eval_sequence)
    interp_df64 = Interpreter(Vector{ComplexDF64}, eval_sequence)
    interp_jac = Interpreter(Vector{ComplexF64}, jacobian_sequence)
    interp_t2 = Interpreter(
        Vector{TruncatedTaylorSeries{3, ComplexF64}}, eval_sequence,
    )
    interp_t3 = Interpreter(
        Vector{TruncatedTaylorSeries{4, ComplexF64}}, eval_sequence,
    )
    evaluation_fws = _build_interpreted_evaluation_fws(
        interp_f64, interp_df64, interp_jac,
    )
    return SystemEvaluator(
        evaluation_fws...,
        SysTaylor1FW(_unsupported_support_taylor!),
        SysTaylor2FW(_unsupported_support_taylor!),
        SysTaylor3FW(_unsupported_support_taylor!),
        SysTaylor1ParamFW(_unsupported_support_taylor!),
        SysTaylor2ParamFW(
            (u, tx, tp) -> (execute_taylor!(u, Val(2), interp_t2, tx, tp); nothing),
        ),
        SysTaylor3ParamFW(
            (u, tx, tp) -> (execute_taylor!(u, Val(3), interp_t3, tx, tp); nothing),
        ),
        (nequations, nvariables),
        nparams,
        SystemFactory(
            _SupportEvaluatorCloner(
                eval_sequence, jacobian_sequence, nequations, nvariables, nparams,
            ),
        ),
    )
end

struct _SupportEvaluatorCloner
    eval_sequence::InstructionSequence
    jacobian_sequence::InstructionSequence
    nequations::Int
    nvariables::Int
    nparams::Int
end

(c::_SupportEvaluatorCloner)()::SystemEvaluator = _support_evaluator(
    c.eval_sequence, c.jacobian_sequence, c.nequations, c.nvariables, c.nparams,
)

function _unsupported_support_taylor!(::Any, ::Any, ::Any)::Nothing
    throw(
        ArgumentError(
            "the internal polyhedral support evaluator does not provide this Taylor mode",
        )
    )
end

function _support_system(support::Vector{Matrix{Int32}})::_SupportSystem
    eval_sequence = _build_support_instruction_sequence(support, false)
    jacobian_sequence = _build_support_instruction_sequence(support, true)
    evaluator = _support_evaluator(
        eval_sequence,
        jacobian_sequence,
        length(support),
        size(first(support), 1),
        _support_parameter_count(support),
    )
    return _SupportSystem(evaluator, eval_sequence, jacobian_sequence)
end

_clone_system_evaluator(system::_SupportSystem)::SystemEvaluator =
    _clone_system_evaluator(system.evaluator)

# ── CommonSolve.init: polys + Polyhedral ────────────────────────────────────

function _polyhedral_source_data(
        ::SquareShape, ::Random.MersenneTwister, F::System, ::Float64,
    )
    source_support, source_coeffs = support_coefficients(F)
    return source_support, source_coeffs, nothing
end

function _polyhedral_source_data(
        ::OverdeterminedShape, rng::Random.MersenneTwister, F::System,
        excess_residual_tol::Float64,
    )
    source_support, source_coeffs = support_coefficients(F)
    A, perm, checker = _square_up(rng, F, excess_residual_tol)
    randomized_support, randomized_coeffs =
        _randomize_support(source_support, source_coeffs, A, perm)
    return randomized_support, randomized_coeffs, checker
end

# `System` caches parameter-free supports as dense, nonnegative `Int32`
# matrices. MixedSubdivisions' public matrix entry point deliberately accepts
# arbitrary integer matrices, so it normalizes every support before building
# the regeneration traverser. That normalization is redundant here and pulls
# a broad reduction/broadcast graph into the cold polyhedral path.
#
# Constructing the traversers themselves is essential work. This narrow path
# only skips the public API's normalization, progress, and deprecated
# lifting-sampler branches. The exact typed public iterator already avoids its
# generic integer conversion method. Keep this constructor synchronized with
# `MixedSubdivisions.MixedCellIterator` (MixedSubdivisions 1.2.x).
#
# TODO(upstream): contribute a public non-normalizing entry point to
# MixedSubdivisions (e.g. `MixedCellIterator(support, lifting; normalize=false)`
# or a `RegenerationTraverser`-accepting `fine_mixed_cells`). The public
# `traverser(support)` unconditionally calls `normalize_supports`, which is a
# no-op for our already-nonnegative supports but costs ~1.5s of cold TTFX to
# compile. Once such an entry point exists upstream, replace this hand-rebuilt
# iterator (and its six internal-symbol dependencies) with the public call and
# drop the explicit-imports allowlist entries.
function _canonical_mixed_cell_iterator(
        support::Vector{Matrix{Int32}},
        lifting::Vector{Vector{Int32}},
    )
    n = length(support)

    # Calling RegenerationTraverser directly is the key fast path: its public
    # `traverser(support)` wrapper first calls `normalize_supports`.
    start_traverser = MixedSubdivisions.RegenerationTraverser(support)

    indices = fill((1, 2), n)
    indexing = MixedSubdivisions.CayleyIndexing(Int[size(A, 2) for A in support])
    cayley = MixedSubdivisions.cayley(support)
    target_cell = MixedSubdivisions.MixedCellTable(
        indices, cayley, indexing; fill_circuit_table = false,
    )

    n_lifts = sum(length, lifting)
    target_lifting = Vector{Int32}(undef, n_lifts)
    offset = 0
    for lift in lifting
        copyto!(target_lifting, offset + 1, lift, 1, length(lift))
        offset += length(lift)
    end
    target_traverser = MixedSubdivisions.MixedCellTableTraverser(
        target_cell,
        cayley,
        -target_lifting,
        MixedSubdivisions.LexicographicOrdering(),
    )

    return MixedSubdivisions.MixedCellIterator(
        start_traverser,
        target_traverser,
        support,
        lifting,
        MixedSubdivisions.MixedCell(n),
        zeros(n, n),
        zeros(n),
    )
end

function _fine_mixed_cells_canonical(
        support::Vector{Matrix{Int32}},
        lifting_sampler;
        max_tries::Int = 10,
    )::Tuple{Vector{MixedSubdivisions.MixedCell}, Vector{Vector{Int32}}}
    lifting = Vector{Int32}[Int32[] for _ in support]
    try
        for attempt in 1:max_tries
            for i in eachindex(support)
                lifting[i] = lifting_sampler(size(support[i], 2), attempt)::Vector{Int32}
            end

            cells = MixedSubdivisions.MixedCell[]
            all_valid = true
            for cell in _canonical_mixed_cell_iterator(support, lifting)
                if !MixedSubdivisions.is_fine(cell)
                    all_valid = false
                    break
                end
                push!(cells, copy(cell))
            end
            all_valid && return cells, lifting
        end
    catch err
        if !(err isa InexactError || err isa LinearAlgebra.SingularException)
            rethrow()
        end
    end
    _has_zero_mixed_volume(support) && return MixedSubdivisions.MixedCell[], lifting
    return _fine_mixed_cells_failure()
end

# The mixed cells of `support` sum to this, so it is also the number of paths the
# polyhedral start system tracks. A one-column support is a point: mixed volume 0 by
# multilinearity, but MixedSubdivisions cannot process it.
function _mixed_volume(support::Vector{Matrix{Int32}})::Int
    any(A -> size(A, 2) < 2, support) && return 0
    return Int(MixedSubdivisions.mixed_volume(support))
end

# A numeric failure is not zero volume.
function _has_zero_mixed_volume(support::Vector{Matrix{Int32}})::Bool
    try
        return iszero(_mixed_volume(support))
    catch err
        if err isa InexactError || err isa LinearAlgebra.SingularException
            return false
        end
        rethrow()
    end
end

@noinline _fine_mixed_cells_failure() =
    error("MixedSubdivisions could not compute fine mixed cells")

"""
    _polyhedral_system(F, alg) -> System

The system `_init_polyhedral` runs on, with every route's preparation applied: a
homogeneous system is put on an affine chart, a composition is substituted out.
Reading the support off this is what keeps `paths_to_track` from drifting away
from the paths a solve actually tracks.
"""
function _polyhedral_system(F::System, alg::Polyhedral)::System
    _check_parameter_free(F, "`Polyhedral`")
    _check_polynomial(F, "`Polyhedral`")
    if is_homogeneous(F)
        _check_single_group(F)
        _check_projective_determined(F, "`Polyhedral`")
        G = _polynomial_system(F)
        return _polyhedral_system(
            _sliced_solve_system(G, _full_subspace(nvariables(G)), _seed(alg)), alg,
        )
    end
    _check_square_or_overdetermined(F)
    return F
end

_polyhedral_system(C::CompositionSystem, alg::Polyhedral)::System =
    _polyhedral_system(System(C), alg)

# The support the mixed cells are computed from: squared up when the system is
# overdetermined, then padded unless only the torus solutions are wanted.
function _polyhedral_support(F::System, alg::Polyhedral)::Vector{Matrix{Int32}}
    rng = Random.MersenneTwister(_seed(alg))
    source_support, _, _ =
        _polyhedral_source_data(system_shape(F), rng, F, alg.excess_residual_tol)
    return Matrix{Int32}[_padded_support(A, alg.only_torus) for A in source_support]
end

# `only_torus` divides each equation by its lowest monomial; otherwise the zero exponent
# vector is added so the solutions on the coordinate hyperplanes are found too.
_padded_support(A::Matrix{Int32}, only_torus::Bool)::Matrix{Int32} =
    only_torus ? A .- minimum(A; dims = 2) :
    has_zero_column(A) ? A : hcat(A, zeros(Int32, size(A, 1)))

function CommonSolve.init(
        F::System, alg::Polyhedral,
        exec::AbstractExecutor = Threaded(),
    )::PolyhedralSolveCache
    # Dynamic call: specializes the body on the concrete `System` so that
    # `system_shape(F)` resolves statically instead of union-splitting.
    initializer = Base.inferencebarrier(_init_polyhedral)
    return initializer(_polyhedral_system(F, alg), alg, exec)::PolyhedralSolveCache
end

function _init_polyhedral(
        F::System, alg::Polyhedral, exec::AbstractExecutor,
    )::PolyhedralSolveCache
    show_progress = _show_progress(alg)
    seed = _seed(alg)
    n = F.nvars

    # Task-local RNG seeded from user seed — deterministic without mutating global state.
    rng = Random.MersenneTwister(seed)

    # 1. Get support + target coefficients from the System.
    #    Overdetermined systems are squared up first: the polyhedral machinery
    #    (mixed cells, parametric system, both homotopy phases) then operates on
    #    the support of G = [I A]·(F∘perm) and never sees the original system.
    source_support, source_coeffs, excess_checker =
        _polyhedral_source_data(system_shape(F), rng, F, alg.excess_residual_tol)

    # 2. Generate start coefficients for ORIGINAL support FIRST.
    #    Coefficients must be generated before zero column addition so the RNG
    #    stream is deterministic. cospi/sinpi give exact values at rational π.
    start_coeffs_orig = Vector{Vector{ComplexF64}}(undef, length(source_coeffs))
    for i in eachindex(source_coeffs)
        c = source_coeffs[i]
        nrm = LinearAlgebra.norm(c, Inf)
        start_coeffs_orig[i] = ComplexF64[
            let r = rand(rng), φ = rand(rng)
                    (0.9 + 0.2 * r) * complex(cospi(2φ), sinpi(2φ)) * nrm
            end
                for _ in 1:length(c)
        ]
    end

    # 3. Add zero columns to support and extend coefficients.
    #    Zero-column extensions use randn(ComplexF64) for start, 0.0 for target.
    #    `only_torus` instead divides each equation by its lowest monomial.
    support = Vector{Matrix{Int32}}(undef, length(source_support))
    target_coeffs = Vector{Vector{ComplexF64}}(undef, length(source_coeffs))
    start_coeffs = Vector{Vector{ComplexF64}}(undef, length(source_coeffs))
    for (i, A) in enumerate(source_support)
        pad = !alg.only_torus && !has_zero_column(A)
        support[i] = _padded_support(A, alg.only_torus)
        target_coeffs[i] = pad ?
            push!(copy(source_coeffs[i]), zero(ComplexF64)) : source_coeffs[i]
        start_coeffs[i] = pad ?
            vcat(start_coeffs_orig[i], randn(rng, ComplexF64)) : start_coeffs_orig[i]
    end

    # 4. Compute mixed cells via MixedSubdivisions
    # Custom lifting sampler that draws from our local rng for reproducibility.
    _lifting_sampler(nterms::Int, attempt::Int = 1) =
        rand(rng, Int32(-2^(10 + attempt)):Int32(2^(10 + attempt)), nterms)
    mixed_cells, lifting = _fine_mixed_cells_canonical(support, _lifting_sampler)

    # 5. Solve binomial systems for each mixed cell. No cell means mixed volume 0, so
    #    there is no path to track and the cache is built empty.
    max_d_hat = isempty(mixed_cells) ? 1 : maximum(c.volume for c in mixed_cells)
    BSS = BinomialSystemSolver(n; max_d_hat = max_d_hat)
    X = Matrix{ComplexF64}(undef, n, max_d_hat)

    all_starts = Tuple{MixedSubdivisions.MixedCell, Vector{ComplexF64}}[]
    for cell in mixed_cells
        d_hat = solve_binomial!(X, BSS, support, start_coeffs, cell)
        for j in 1:d_hat
            x = ComplexF64[X[i, j] for i in 1:n]
            push!(all_starts, (cell, x))
        end
    end

    # 6. Lower the cached support directly to the coefficient-parametric tape.
    #    This avoids a synthetic DynamicPolynomials -> System compiler round trip.
    support_system = _support_system(support)

    # 7. Toric phase (t goes from 0 to 1), on a conservative max_initial_step_size
    toric_opts = TrackerOptions(;
        max_steps = _tracker_options(alg).max_steps,
        max_step_size = _tracker_options(alg).max_step_size,
        max_initial_step_size = min(_tracker_options(alg).max_initial_step_size, 0.2),
        extended_precision = _tracker_options(alg).extended_precision,
        min_step_size = _tracker_options(alg).min_step_size,
        terminate_cond = _tracker_options(alg).terminate_cond,
        a = _tracker_options(alg).a,
        β_a = _tracker_options(alg).β_a,
        β_ω = _tracker_options(alg).β_ω,
        β_τ = _tracker_options(alg).β_τ,
        strict_β_τ = _tracker_options(alg).strict_β_τ,
    )

    # 8. Coefficient phase (t goes from 1 to 0)
    flat_start = reduce(vcat, start_coeffs)
    flat_target = reduce(vcat, target_coeffs)

    builder = PolyhedralBuilder(
        support_system, start_coeffs, flat_start, flat_target,
        toric_opts, _tracker_options(alg), _endgame_options(alg),
    )
    worker = builder()

    return PolyhedralSolveCache(
        exec, builder,
        worker.toric_tracker, worker.coeff_tracker, worker.toric_homotopy,
        support, lifting,
        all_starts, seed,
        worker.x_buffer,
        support_system,
        excess_checker,
        show_progress,
        early_stop_callback(alg),
    )
end

# ── Toric phase tracking with two-stage reparameterization ─────────────────

"""
    _init_toric!(tracker, x₀, t_start, t_end)

Initialize the toric tracker with ω=20 (optimistic initial Lipschitz),
μ=1e-12, and max_initial_step_size=0.2.
"""
function _init_toric!(
        tracker::Tracker,
        x₀::AbstractVector{<:Number},
        t_start::ComplexF64,
        t_end::ComplexF64,
    )::TrackerCode.T
    # Pass ω/μ directly so init! skips init_newton — the toric phase uses
    # empirically-tuned parameters since exact start solutions are known
    return init!(
        tracker, x₀, t_start, t_end;
        ω = 20.0, μ = 1.0e-12, max_initial_step_size = 0.2,
    )
end

"""
    _track_toric_phase!(tracker, H, x₀, min_weight, max_weight, support, lifting, cell)

Track the toric phase with two-stage reparameterization for large weights.
When max_weight >= 10, the path is split into two stages to avoid numerical
issues from t^w where w is large and t is near 1.
"""
function _track_toric_phase!(
        tracker::Tracker,
        H::ToricHomotopy,
        x₀::AbstractVector{ComplexF64},
        min_weight::Float64,
        max_weight::Float64,
        support::Vector{Matrix{Int32}},
        lifting::Vector{Vector{Int32}},
        cell::MixedSubdivisions.MixedCell,
    )::TrackerCode.T

    if max_weight < 10.0
        # Simple case: track directly from 0 to 1
        code = _init_toric!(tracker, x₀, complex(0.0), complex(1.0))
        if code != TrackerCode.TRACKING
            return code
        end
        while tracker.state.code == TrackerCode.TRACKING
            step!(tracker)
        end
        return tracker.state.code
    end

    # Two-stage reparameterization for large weights:
    # Stage 1: track from 0 to t₀
    t₀ = clamp(0.1^(10.0 / max_weight), 0.9, 1.0 - 1.0e-6)
    code = _init_toric!(tracker, x₀, complex(0.0), complex(t₀))
    if code != TrackerCode.TRACKING
        return code
    end
    while tracker.state.code == TrackerCode.TRACKING
        step!(tracker)
    end
    if tracker.state.code != TrackerCode.TRACKER_SUCCESS
        return tracker.state.code
    end

    # Stage 2: renormalize weights with max_weight=10, track from t_restart to 1
    saved_ω = tracker.state.ω
    saved_μ = tracker.state.μ

    new_min_w, _ = update_weights!(H, support, lifting, cell; max_weight = 10.0)
    t_restart = t₀^(1.0 / new_min_w)

    # Re-init tracker from current solution at t_restart to 1.0
    # Pass ω/μ from stage 1, keep step counters
    code = init!(
        tracker, tracker.state.x, complex(t_restart), complex(1.0);
        ω = saved_ω, μ = saved_μ,
        keep_steps = true,
    )

    if code != TrackerCode.TRACKING
        return code
    end
    while tracker.state.code == TrackerCode.TRACKING
        step!(tracker)
    end
    return tracker.state.code
end

# ── CommonSolve.solve!: serial two-phase path tracking ───────────────────

function CommonSolve.solve!(cache::PolyhedralSolveCache{Serial})::Result
    solver = cache.show_progress ?
        _solve_polyhedral_serial_with_progress :
        _solve_polyhedral_serial_without_progress
    solver = Base.inferencebarrier(solver)
    return _dispatch_solve_policy(solver, cache)
end


@noinline _solve_polyhedral_serial_without_progress(cache::PolyhedralSolveCache{Serial}) =
    _solve_polyhedral_serial(cache, nothing)
@noinline _solve_polyhedral_serial_with_progress(cache::PolyhedralSolveCache{Serial}) =
    _solve_polyhedral_serial(cache, make_progress(length(cache.start_solutions), true))

# One two-phase path: toric phase from the mixed cell, then the coefficient homotopy.
function _track_polyhedral_path!(
        toric_tracker::Tracker, coeff_tracker::EndgameTracker,
        toric_H::ToricHomotopy, support::Vector{Matrix{Int32}},
        lifting::Vector{Vector{Int32}}, x_buffer::Vector{ComplexF64},
        cell::MixedSubdivisions.MixedCell, x₀::Vector{ComplexF64}, k::Int,
    )::PathResult
    min_w, max_w = update_weights!(toric_H, support, lifting, cell; min_weight = 1.0)

    code = _track_toric_phase!(
        toric_tracker, toric_H, x₀, min_w, max_w, support, lifting, cell,
    )

    if code != TrackerCode.TRACKER_SUCCESS
        return PathResult(
            toric_tracker; path_number = k, start_solution = Vector{ComplexF64}(x₀),
        )
    end

    toric_accepted = toric_tracker.state.accepted_steps
    toric_rejected = toric_tracker.state.rejected_steps

    copyto!(x_buffer, toric_tracker.state.x)

    init!(coeff_tracker, x_buffer; μ = toric_tracker.state.μ)
    while coeff_tracker.state.code == EndgameCode.TRACKING
        step!(coeff_tracker)
    end

    return _add_steps(
        PathResult(
            coeff_tracker; path_number = k, start_solution = Vector{ComplexF64}(x₀),
        ),
        toric_accepted, toric_rejected,
    )
end

_track_polyhedral_path!(
    ws::PolyhedralWorkerState, support::Vector{Matrix{Int32}},
    lifting::Vector{Vector{Int32}}, cell::MixedSubdivisions.MixedCell,
    x₀::Vector{ComplexF64}, k::Int,
)::PathResult = _track_polyhedral_path!(
    ws.toric_tracker, ws.coeff_tracker, ws.toric_homotopy, support, lifting,
    ws.x_buffer, cell, x₀, k,
)

function _solve_polyhedral_serial(cache::PolyhedralSolveCache{Serial}, progress)::Result
    support = cache.support
    n_paths = length(cache.start_solutions)
    path_results = PathResult[]
    sizehint!(path_results, n_paths)

    stats = ProgressStats()

    stop = cache.early_stop
    for (k, (cell, x₀)) in enumerate(cache.start_solutions)
        pr = _track_polyhedral_path!(
            cache.toric_tracker, cache.coeff_tracker, cache.toric_homotopy,
            support, cache.lifting, cache.x_buffer, cell, x₀, k,
        )
        push!(path_results, pr)
        update_progress!(progress, k, stats, pr)
        is_success(pr) && stop(pr) && break
    end

    return _finalize_result(
        path_results, length(path_results), cache.seed, cache.excess_checker,
    )
end

# ── CommonSolve.solve!: threaded two-phase path tracking ─────────────────

function CommonSolve.solve!(cache::PolyhedralSolveCache{Threaded})::Result
    solver = cache.show_progress ?
        _solve_polyhedral_threaded_with_progress :
        _solve_polyhedral_threaded_without_progress
    solver = Base.inferencebarrier(solver)
    return _dispatch_solve_policy(solver, cache)
end

@noinline _solve_polyhedral_threaded_without_progress(cache::PolyhedralSolveCache{Threaded}) =
    _solve_polyhedral_threaded(cache, nothing)
@noinline _solve_polyhedral_threaded_with_progress(cache::PolyhedralSolveCache{Threaded}) =
    _solve_polyhedral_threaded(cache, make_progress(length(cache.start_solutions), true))

function _solve_polyhedral_threaded(cache::PolyhedralSolveCache{Threaded}, progress)::Result
    nt = cache.executor.ntasks
    starts = cache.start_solutions
    n_paths = length(starts)
    results = Vector{PathResult}(undef, n_paths)
    support = cache.support
    lifting = cache.lifting

    stats = ProgressStats()
    counter = Threads.Atomic{Int}(0)
    plock = ReentrantLock()

    stop = cache.early_stop
    stopped = Threads.Atomic{Bool}(false)

    @tasks for i in eachindex(starts)
        @set ntasks = nt
        @local ws = cache.builder()

        if !stopped[]
            cell, x₀ = starts[i]
            pr = _track_polyhedral_path!(ws, support, lifting, cell, x₀, i)
            results[i] = pr

            if progress !== nothing
                k = Threads.atomic_add!(counter, 1) + 1
                @lock plock update_progress!(progress, k, stats, pr)
            end
            is_success(pr) && stop(pr) && (stopped[] = true)
        end
    end

    tracked = _assigned_results(results)
    return _finalize_result(tracked, length(tracked), cache.seed, cache.excess_checker)
end

# ── CommonSolve.solve!: distributed (extension) ──────────────────────────

CommonSolve.solve!(cache::PolyhedralSolveCache{DistributedExecutor})::Result =
    _distributed_solve!(cache)
