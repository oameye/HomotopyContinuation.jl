using Test
using AllocCheck: check_allocs
import HomotopyContinuationNext as HC
using HomotopyContinuationNext: System, StraightLineHomotopy, HomotopyEvaluator,
    Tracker, TrackerCode, TrackerOptions,
    NewtonCorrector, NewtonCode, NewtonCorrectorResult,
    Predictor, PredictionMethod,
    Jacobian, MatrixWorkspace, WeightedNorm,
    TaylorVector, EndgameTracker, EndgameCode, EndgameOptions,
    EndgameState, Valuation
using DynamicPolynomials: @polyvar
using FixedSizeArrays: FixedSizeArray

const FSVec{T} = FixedSizeArray{T, 1, Memory{T}}
const FSMat{T} = FixedSizeArray{T, 2, Memory{T}}

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
