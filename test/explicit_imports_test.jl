using Test
using ExplicitImports
using HomotopyContinuationNext
using Distributed: Distributed
using Serialization: Serialization

const DISTRIBUTED_EXT = Base.get_extension(
    HomotopyContinuationNext, :HomotopyContinuationNextDistributedExt,
)

# Allow non-public but necessary accesses:
# - Base.RefValue: used for mutable cache scalars in immutable structs
# - Base.decompose: required for DoubleF64 <: AbstractFloat
# - LinearAlgebra.qrfactUnblocked!: needed for custom QR (LAPACK qr! returns QRCompactWY)
# - Base.literal_pow: extended so `expr^2` builds an EPow instead of a product
const QUALIFIED_ACCESS_IGNORE = (
    :RefValue, :decompose, :qrfactUnblocked!, :literal_pow, :TwicePrecision,
    # model_kit internal uses of non-public Base APIs:
    Symbol("@_inline_meta"), Symbol("@_propagate_inbounds_meta"),
    :FastMath, :div_fast, :inv_fast, :power_by_squaring,
    # CommonSolve interface (not declared public in CommonSolve.jl):
    :init, Symbol("solve!"), :solve,
    # Checked arithmetic for HNF in binomial_system.jl:
    :checked_add, :checked_mul,
    # Base.broadcastable is the documented broadcast customization
    # hook but is not declared public in Base:
    :broadcastable,
    # Deliberate construction-time compiler barrier used to isolate
    # mutually exclusive TTFX-heavy backends:
    :inferencebarrier,
    # MixedSubdivisions 1.2.x has no public "already normalized"
    # iterator constructor. The polyhedral canonical-support fast path
    # deliberately mirrors its iterator setup to avoid recompiling
    # normalize_supports for System's cached nonnegative Int32 data.
    :RegenerationTraverser, :CayleyIndexing, :cayley,
    :MixedCellTable, :MixedCellTableTraverser,
    :LexicographicOrdering,
    # Distributed extension. Asking whether a worker process has the package
    # loaded has no public spelling, and a custom `Serialization.serialize` method
    # has to write the type tag itself:
    :PkgId, :root_module_exists, :serialize_type,
)

@testset "ExplicitImports" begin
    # OpType and SFuncKind modules created by @enumx are not analyzable by ExplicitImports
    allow_unanalyzable = (
        HomotopyContinuationNext.OpType,
        HomotopyContinuationNext.SFuncKind,
        HomotopyContinuationNext.NewtonCode,
        HomotopyContinuationNext.NewtonReturnCode,
        HomotopyContinuationNext.PredictionMethod,
        HomotopyContinuationNext.TrackerCode,
        HomotopyContinuationNext.PathResultCode,
        HomotopyContinuationNext.CompileMode,  # @enumx module
        HomotopyContinuationNext.SExpr,  # Moshi @data module
        HomotopyContinuationNext.SUnaryKind,  # @enumx module
        HomotopyContinuationNext.SymExpr,  # Moshi @data module
        HomotopyContinuationNext.ExecInstruction,  # Moshi @data module
        HomotopyContinuationNext.EndgameCode,  # @enumx module
        HomotopyContinuationNext.MonodromyCode,  # @enumx module
        HomotopyContinuationNext.ReuseLoops,  # @enumx module
        HomotopyContinuationNext.Irreducibility,  # @enumx module
    )

    @test check_no_implicit_imports(HomotopyContinuationNext; allow_unanalyzable) == nothing
    @test check_all_explicit_imports_via_owners(HomotopyContinuationNext) == nothing
    # FunctionWrapper is the only public API of FunctionWrappers.jl but it is not
    # declared `public` in the package — ignore it here.
    @test check_all_explicit_imports_are_public(
        HomotopyContinuationNext;
        ignore = (
            :FunctionWrapper,
            Symbol("@data"),
            Symbol("@derive"),
            Symbol("@RuntimeGeneratedFunction"),
        ),
    ) == nothing
    @test check_no_stale_explicit_imports(HomotopyContinuationNext; allow_unanalyzable) == nothing
    @test check_all_qualified_accesses_via_owners(HomotopyContinuationNext) == nothing
    # The check's own filter chain, minus the extension's accesses into its parent:
    # `allow_internal_accesses` does not cover extensions and `ignore` cannot
    # exclude one, so `check_all_qualified_accesses_are_public` cannot be used here.
    @testset "qualified accesses are public" begin
        offenders = Tuple{Symbol, Symbol, Module}[]
        for (submodule, rows) in ExplicitImports.improper_qualified_accesses(
                HomotopyContinuationNext, pathof(HomotopyContinuationNext); skip = (),
            )
            for row in rows
                row.name in QUALIFIED_ACCESS_IGNORE && continue
                row.self_qualified && continue
                row.public_access && continue
                # The check's own default `skip = (Base => Core,)`.
                row.accessing_from === Base &&
                    ExplicitImports.public_or_exported(Core, row.name) && continue
                submodule === DISTRIBUTED_EXT &&
                    row.accessing_from === HomotopyContinuationNext && continue
                push!(offenders, (nameof(submodule), row.name, row.accessing_from))
            end
        end
        isempty(offenders) || foreach(println, offenders)
        @test isempty(offenders)
    end

    @test check_no_self_qualified_accesses(HomotopyContinuationNext) == nothing

    # Every other rule still applies to the extension.
    @testset "Distributed extension" begin
        @test DISTRIBUTED_EXT !== nothing
        @test check_no_implicit_imports(DISTRIBUTED_EXT) == nothing
        @test check_no_stale_explicit_imports(DISTRIBUTED_EXT) == nothing
        @test check_all_explicit_imports_are_public(DISTRIBUTED_EXT) == nothing
        @test check_all_explicit_imports_via_owners(DISTRIBUTED_EXT) == nothing
        @test check_all_qualified_accesses_via_owners(DISTRIBUTED_EXT) == nothing
        @test check_no_self_qualified_accesses(DISTRIBUTED_EXT) == nothing
    end
end
