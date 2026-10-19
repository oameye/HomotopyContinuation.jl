using Test
using AllocCheck: check_allocs
import HomotopyContinuation as HC
using HomotopyContinuation: System, StraightLineHomotopy, HomotopyEvaluator,
    Tracker, TrackerCode, TrackerOptions,
    NewtonCorrector, NewtonCode, NewtonCorrectorResult,
    Predictor, PredictionMethod,
    Jacobian, MatrixWorkspace, WeightedNorm,
    TaylorVector, EndgameTracker, EndgameCode, EndgameOptions,
    EndgameState, Valuation, FSVec, FSMat
using HomotopyContinuation: ParameterHomotopy, ComplexDF64, evaluate!,
    evaluate_and_jacobian!, taylor!, updated!
using DynamicPolynomials: @polyvar
using LinearAlgebra: ldiv!
using Random: Random, MersenneTwister, randn
@static if VERSION >= v"1.11"
    using FixedSizeArrays: FixedSizeVectorDefault, FixedSizeMatrixDefault
end

# ---------------------------------------------------------------------------
# Helper: filter out known FunctionWrappers false positives
#
# FunctionWrappers.jl is the type-erasure boundary (by design). AllocCheck
# sees dynamic dispatch and jl_f_apply_type inside reinit_wrapper, but these
# don't allocate at runtime — the ccall is to a pre-compiled function pointer.
# We filter these so the test only catches real allocation sites in our code.
# ---------------------------------------------------------------------------

function _is_functionwrapper_false_positive(a)
    s = string(a)
    return contains(s, "FunctionWrappers") || contains(s, "FunctionWrapper")
end

function _real_allocs(f, types; ignore_throw::Bool = true)
    allocs = check_allocs(f, types; ignore_throw)
    real = filter(!_is_functionwrapper_false_positive, allocs)
    if !isempty(real)
        println("  Found $(length(real)) real allocation site(s) ($(length(allocs) - length(real)) FunctionWrapper filtered):")
        for a in real
            println("    ", a)
            println()
        end
    end
    return real
end

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@testset "AllocCheck: zero-allocation hot paths" begin

    # ── Primitives: norms ────────────────────────────────────────────────

    @testset "Norms" begin
        @test isempty(_real_allocs(HC.inf_norm, (FSVec{ComplexF64},)))
        @test isempty(_real_allocs(HC.inf_norm, (FSVec{Float64},)))
        @test isempty(_real_allocs(HC.inf_distance, (FSVec{ComplexF64}, FSVec{ComplexF64})))
        @test isempty(_real_allocs(HC.weighted_norm, (FSVec{ComplexF64}, WeightedNorm)))
        @test isempty(_real_allocs(HC.weighted_distance, (FSVec{ComplexF64}, FSVec{ComplexF64}, WeightedNorm)))
        @test isempty(_real_allocs(HC.update!, (WeightedNorm, FSVec{ComplexF64})))
        @test isempty(_real_allocs(HC.init!, (WeightedNorm, FSVec{ComplexF64})))
    end

    # ── Primitives: linear algebra ───────────────────────────────────────

    @testset "Linear algebra" begin
        @test isempty(_real_allocs(HC.factorize!, (MatrixWorkspace,)))
    end

    # ── Predictor ────────────────────────────────────────────────────────

    @testset "Predictor" begin
        @test isempty(
            _real_allocs(
                HC.predict!, (FSVec{ComplexF64}, Predictor, ComplexF64),
            )
        )
        @test isempty(
            _real_allocs(
                HC.compute_local_error!, (Predictor, FSVec{ComplexF64}, FSVec{ComplexF64}, WeightedNorm, Float64),
            )
        )
        # Predictor.update! goes through HomotopyEvaluator (FunctionWrappers boundary)
        @test isempty(
            _real_allocs(
                HC.update!, (Predictor, HomotopyEvaluator, FSVec{ComplexF64}, ComplexF64, Jacobian, WeightedNorm),
            )
        )
    end

    # ── Newton corrector ─────────────────────────────────────────────────

    @testset "Newton corrector" begin
        # newton! goes through HomotopyEvaluator (FunctionWrappers boundary)
        @test isempty(
            _real_allocs(
                HC.newton!,
                (
                    FSVec{ComplexF64}, NewtonCorrector, HomotopyEvaluator, FSVec{ComplexF64},
                    ComplexF64, Jacobian, WeightedNorm, Float64, Float64, Bool, Bool,
                ),
            )
        )
    end

    # ── RandomizedSystem (overdetermined square-up) ──────────────────────
    # Inner evaluation goes through SystemEvaluator (FunctionWrappers
    # boundary, filtered); the randomization fold itself must not allocate.

    @testset "RandomizedSystem" begin
        @test isempty(
            _real_allocs(
                HC.evaluate!,
                (FSVec{ComplexF64}, HC.RandomizedSystem, FSVec{ComplexF64}, FSVec{ComplexF64}),
            )
        )
        @test isempty(
            _real_allocs(
                HC.evaluate!,
                (FSVec{ComplexF64}, HC.RandomizedSystem, FSVec{HC.ComplexDF64}, FSVec{ComplexF64}),
            )
        )
        @test isempty(
            _real_allocs(
                HC.evaluate!,
                (FSVec{HC.ComplexDF64}, HC.RandomizedSystem, FSVec{HC.ComplexDF64}, FSVec{ComplexF64}),
            )
        )
        @test isempty(
            _real_allocs(
                HC.evaluate_and_jacobian!,
                (
                    FSVec{ComplexF64}, FSMat{ComplexF64}, HC.RandomizedSystem,
                    FSVec{ComplexF64}, FSVec{ComplexF64},
                ),
            )
        )
        @test isempty(
            _real_allocs(
                HC.taylor!,
                (
                    FSVec{ComplexF64}, Val{1}, HC.RandomizedSystem,
                    TaylorVector{2, ComplexF64}, FSVec{ComplexF64},
                ),
            )
        )
        @test isempty(
            _real_allocs(
                HC.taylor!,
                (
                    FSVec{ComplexF64}, Val{2}, HC.RandomizedSystem,
                    TaylorVector{3, ComplexF64}, FSVec{ComplexF64},
                ),
            )
        )
        @test isempty(
            _real_allocs(
                HC.taylor!,
                (
                    FSVec{ComplexF64}, Val{3}, HC.RandomizedSystem,
                    TaylorVector{4, ComplexF64}, FSVec{ComplexF64},
                ),
            )
        )
        @test isempty(
            _real_allocs(
                HC.taylor!,
                (
                    FSVec{ComplexF64}, Val{1}, HC.RandomizedSystem,
                    TaylorVector{2, ComplexF64}, TaylorVector{2, ComplexF64},
                ),
            )
        )
        @test isempty(
            _real_allocs(
                HC.taylor!,
                (
                    FSVec{ComplexF64}, Val{2}, HC.RandomizedSystem,
                    TaylorVector{3, ComplexF64}, TaylorVector{3, ComplexF64},
                ),
            )
        )
        @test isempty(
            _real_allocs(
                HC.taylor!,
                (
                    FSVec{ComplexF64}, Val{3}, HC.RandomizedSystem,
                    TaylorVector{4, ComplexF64}, TaylorVector{4, ComplexF64},
                ),
            )
        )
    end

    # ── AffineChartSystem (projective chart row) ─────────────────────────

    @testset "AffineChartSystem" begin
        @test isempty(
            _real_allocs(
                HC.evaluate!,
                (
                    FSVec{ComplexF64}, HC.AffineChartSystem,
                    FSVec{ComplexF64}, FSVec{ComplexF64},
                ),
            )
        )
        @test isempty(
            _real_allocs(
                HC.evaluate_and_jacobian!,
                (
                    FSVec{ComplexF64}, FSMat{ComplexF64}, HC.AffineChartSystem,
                    FSVec{ComplexF64}, FSVec{ComplexF64},
                ),
            )
        )
        for (V, N) in ((Val{1}, 2), (Val{2}, 3), (Val{3}, 4))
            @test isempty(
                _real_allocs(
                    HC.taylor!,
                    (
                        FSVec{ComplexF64}, V, HC.AffineChartSystem,
                        TaylorVector{N, ComplexF64}, FSVec{ComplexF64},
                    ),
                )
            )
        end
    end

    # ── SlicedSystem (`[F; A x − b]` plus an optional chart row) ─────────

    @testset "SlicedSystem" begin
        for X in (ComplexF64, HC.ComplexDF64)
            @test isempty(
                _real_allocs(
                    HC.evaluate!,
                    (
                        FSVec{ComplexF64}, HC.SlicedSystem,
                        FSVec{X}, FSVec{ComplexF64},
                    ),
                )
            )
        end
        @test isempty(
            _real_allocs(
                HC.evaluate!,
                (
                    FSVec{HC.ComplexDF64}, HC.SlicedSystem,
                    FSVec{HC.ComplexDF64}, FSVec{ComplexF64},
                ),
            )
        )
        @test isempty(
            _real_allocs(
                HC.evaluate_and_jacobian!,
                (
                    FSVec{ComplexF64}, FSMat{ComplexF64}, HC.SlicedSystem,
                    FSVec{ComplexF64}, FSVec{ComplexF64},
                ),
            )
        )
        for (V, N) in ((Val{1}, 2), (Val{2}, 3), (Val{3}, 4))
            for P in (FSVec{ComplexF64}, TaylorVector{N, ComplexF64})
                @test isempty(
                    _real_allocs(
                        HC.taylor!,
                        (
                            FSVec{ComplexF64}, V, HC.SlicedSystem,
                            TaylorVector{N, ComplexF64}, P,
                        ),
                    )
                )
            end
        end
    end

    @testset "BoundParameterSystem" begin
        for X in (ComplexF64, HC.ComplexDF64)
            @test isempty(
                _real_allocs(
                    HC.evaluate!,
                    (
                        FSVec{ComplexF64}, HC.BoundParameterSystem,
                        FSVec{X}, FSVec{ComplexF64},
                    ),
                )
            )
        end
        @test isempty(
            _real_allocs(
                HC.evaluate!,
                (
                    FSVec{HC.ComplexDF64}, HC.BoundParameterSystem,
                    FSVec{HC.ComplexDF64}, FSVec{ComplexF64},
                ),
            )
        )
        @test isempty(
            _real_allocs(
                HC.evaluate_and_jacobian!,
                (
                    FSVec{ComplexF64}, FSMat{ComplexF64}, HC.BoundParameterSystem,
                    FSVec{ComplexF64}, FSVec{ComplexF64},
                ),
            )
        )
        for (V, N) in ((Val{1}, 2), (Val{2}, 3), (Val{3}, 4))
            for P in (FSVec{ComplexF64}, TaylorVector{N, ComplexF64})
                @test isempty(
                    _real_allocs(
                        HC.taylor!,
                        (
                            FSVec{ComplexF64}, V, HC.BoundParameterSystem,
                            TaylorVector{N, ComplexF64}, P,
                        ),
                    )
                )
            end
        end
    end

    # ── Tracker step ─────────────────────────────────────────────────────
    # This is the main hot-path entry point. FunctionWrapper dispatches are
    # expected (type-erasure boundary) but no other allocations should occur.

    @testset "Tracker step!" begin
        @test isempty(_real_allocs(HC.step!, (Tracker,)))
    end

    # ── Valuation ────────────────────────────────────────────────────────

    @testset "Valuation" begin
        @test isempty(_real_allocs(HC.update!, (Valuation, Predictor, Float64)))
        @test isempty(_real_allocs(HC.estimate_winding_number, (Valuation, Int, Int)))
    end

    # ── Endgame tracker ──────────────────────────────────────────────────
    # EndgameTracker.step! wraps Tracker.step!, so FunctionWrapper findings
    # propagate through. We filter those and check for our own allocations.

    @testset "Endgame tracker" begin
        @test isempty(_real_allocs(HC.step!, (EndgameTracker,)))
        @test isempty(_real_allocs(HC.check_finite!, (EndgameTracker,)))
        @test isempty(_real_allocs(HC.check_at_infinity!, (EndgameTracker,)))
        @test isempty(_real_allocs(HC.add_sample!, (EndgameTracker, Int)))
        @test isempty(
            _real_allocs(
                HC.cubic_hermite!,
                (
                    FSVec{ComplexF64}, TaylorVector{2, ComplexF64}, Float64,
                    TaylorVector{2, ComplexF64}, Float64, Float64,
                ),
            )
        )
        @test isempty(_real_allocs(HC.predict_endpoint!, (EndgameTracker,)))
    end

end

# ---------------------------------------------------------------------------
# Measured allocations of the mutating entry points
#
# AllocCheck proves the absence of allocation sites statically; these measure
# the shipped configuration at runtime. Before Julia 1.11 the buffers are
# ordinary Vectors and the counter is process wide, so only later versions
# assert zero.
# ---------------------------------------------------------------------------

@static if VERSION < v"1.11"
    buffer(v::AbstractVector{T}) where {T} = Vector{T}(v)
    buffer(M::AbstractMatrix{T}) where {T} = Matrix{T}(M)
else
    buffer(v::AbstractVector{T}) where {T} = FixedSizeVectorDefault{T}(v)
    buffer(M::AbstractMatrix{T}) where {T} = FixedSizeMatrixDefault{T}(M)
end

function taylor_vector(X::AbstractMatrix{ComplexF64})
    tv = TaylorVector{size(X, 1), ComplexF64}(size(X, 2))
    for i in axes(X, 2)
        tv[i] = Tuple(X[:, i])
    end
    return tv
end

# Measured inside a function, so that the global-scope `Val` construction and
# argument boxing of a testset are not charged to the call.
taylor_allocations(u, v::Val, H, x, t) = @allocated taylor!(u, v, H, x, t)
evaluate_allocations(u, H, x, t) = @allocated evaluate!(u, H, x, t)
jacobian_allocations(u, U, H, x, t) = @allocated evaluate_and_jacobian!(u, U, H, x, t)

@testset "zero allocations on the mutating homotopy entry points" begin
    @polyvar x y a
    F = System([x^2 + a * y - 1, x * y - a^2]; parameters = [a])
    H = ParameterHomotopy(F, [2.0], [1.0])
    u = buffer(zeros(ComplexF64, 2))
    U = buffer(zeros(ComplexF64, 2, 2))
    xv = buffer(ComplexF64[2.0, 3.0])
    xd = buffer(ComplexDF64.(ComplexF64[2.0, 3.0]))
    t = complex(0.5)
    tx2 = taylor_vector(randn(MersenneTwister(1), ComplexF64, 3, 2))
    tx3 = taylor_vector(randn(MersenneTwister(2), ComplexF64, 4, 2))

    evaluate_allocations(u, H, xv, t)
    evaluate_allocations(u, H, xd, t)
    jacobian_allocations(u, U, H, xv, t)
    taylor_allocations(u, Val(1), H, xv, t)
    taylor_allocations(u, Val(2), H, tx2, t)
    taylor_allocations(u, Val(3), H, tx3, t)

    @test evaluate_allocations(u, H, xv, t) == 0
    @test evaluate_allocations(u, H, xd, t) == 0
    @test jacobian_allocations(u, U, H, xv, t) == 0
    @test taylor_allocations(u, Val(1), H, xv, t) == 0
    @test taylor_allocations(u, Val(2), H, tx2, t) == 0
    @test taylor_allocations(u, Val(3), H, tx3, t) == 0
end

function refactorize_and_solve!(x, WS, b)
    updated!(WS)
    ldiv!(x, WS, b)
    return x
end

@testset "ldiv! does not allocate" begin
    Random.seed!(0x1357)
    for (m, n) in ((3, 3), (13, 13), (8, 5))
        WS = MatrixWorkspace(m, n)
        copyto!(WS, randn(ComplexF64, m, n))
        updated!(WS)
        b = buffer(randn(ComplexF64, m))
        x = buffer(zeros(ComplexF64, n))
        refactorize_and_solve!(x, WS, b)
        ldiv!(x, WS, b)
        if VERSION >= v"1.11"
            @test (@allocated refactorize_and_solve!(x, WS, b)) == 0
            @test (@allocated ldiv!(x, WS, b)) == 0
        end
    end
end
