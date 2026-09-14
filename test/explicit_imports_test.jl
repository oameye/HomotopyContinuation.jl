using Test
using ExplicitImports
using HomotopyContinuation
using Distributed: Distributed
using Serialization: Serialization

const DISTRIBUTED_EXT = Base.get_extension(
    HomotopyContinuation, :DistributedExt,
)

# An extension reaches into its parent by design, and whether one is loaded here
# depends on what else the worker imported, so the rules below must not turn on it.
_is_extension(m::Module)::Bool =
    Base.get_extension(HomotopyContinuation, nameof(m)) === m

# Allow non-public but necessary accesses:
const QUALIFIED_ACCESS_IGNORE = (
    :RefValue, :decompose, :qrfactUnblocked!, :literal_pow, :TwicePrecision,
    Symbol("@_inline_meta"), Symbol("@_propagate_inbounds_meta"),
    :FastMath, :div_fast, :inv_fast, :power_by_squaring,
    :init, Symbol("solve!"), :solve,
    :checked_add, :checked_mul,
    :broadcastable,
    :deepcopy_internal,
    :SizeUnknown,
    :Condition, :filter,
    :inferencebarrier,
    :RegenerationTraverser, :CayleyIndexing, :cayley,
    :MixedCellTable, :MixedCellTableTraverser,
    :LexicographicOrdering,
    :PkgId, :root_module_exists, :serialize_type,
    :AbstractAlgebraicSolver, :NoAlgorithm, :promote_for,
    Symbol("default_gröbner_basis_algorithm"),
)

@testset "ExplicitImports" begin
    allow_unanalyzable = (
        HomotopyContinuation.OpType,
        HomotopyContinuation.SFuncKind,
        HomotopyContinuation.NewtonCode,
        HomotopyContinuation.NewtonReturnCode,
        HomotopyContinuation.PredictionMethod,
        HomotopyContinuation.TrackerCode,
        HomotopyContinuation.PathResultCode,
        HomotopyContinuation.CompileMode,
        HomotopyContinuation.SExpr,
        HomotopyContinuation.SUnaryKind,
        HomotopyContinuation.SymExpr,
        HomotopyContinuation.ExecInstruction,
        HomotopyContinuation.EndgameCode,
        HomotopyContinuation.MonodromyCode,
        HomotopyContinuation.ReuseLoops,
        HomotopyContinuation.DuplicateCheck,
        HomotopyContinuation.AddSolutionCode,
        HomotopyContinuation.Irreducibility,
        HomotopyContinuation.EquationSorting,
    )

    @test check_no_implicit_imports(HomotopyContinuation; allow_unanalyzable) == nothing
    @test check_all_explicit_imports_via_owners(HomotopyContinuation) == nothing
    @static if VERSION >= v"1.11"
        @test check_all_explicit_imports_are_public(
            HomotopyContinuation;
            ignore = (
                :FunctionWrapper,
                Symbol("@data"),
                Symbol("@derive"),
                Symbol("@RuntimeGeneratedFunction"),
            ),
        ) == nothing
    end
    @test check_no_stale_explicit_imports(HomotopyContinuation; allow_unanalyzable) == nothing
    @test check_all_qualified_accesses_via_owners(HomotopyContinuation) == nothing
    @static if VERSION >= v"1.11"
        @testset "qualified accesses are public" begin
            offenders = Tuple{Symbol, Symbol, Module}[]
            for (submodule, rows) in ExplicitImports.improper_qualified_accesses(
                    HomotopyContinuation, pathof(HomotopyContinuation); skip = (),
                )
                for row in rows
                    row.name in QUALIFIED_ACCESS_IGNORE && continue
                    row.self_qualified && continue
                    row.public_access && continue
                    row.accessing_from === Base &&
                        ExplicitImports.public_or_exported(Core, row.name) && continue
                    _is_extension(submodule) &&
                        row.accessing_from === HomotopyContinuation && continue
                    push!(offenders, (nameof(submodule), row.name, row.accessing_from))
                end
            end
            isempty(offenders) || foreach(println, offenders)
            @test isempty(offenders)
        end
    end

    @test check_no_self_qualified_accesses(HomotopyContinuation) == nothing

    @testset "Distributed extension" begin
        @test DISTRIBUTED_EXT !== nothing
        @test check_no_implicit_imports(DISTRIBUTED_EXT) == nothing
        @test check_no_stale_explicit_imports(DISTRIBUTED_EXT) == nothing
        @static if VERSION >= v"1.11"
            @test check_all_explicit_imports_are_public(DISTRIBUTED_EXT) == nothing
        end
        @test check_all_explicit_imports_via_owners(DISTRIBUTED_EXT) == nothing
        @test check_all_qualified_accesses_via_owners(DISTRIBUTED_EXT) == nothing
        @test check_no_self_qualified_accesses(DISTRIBUTED_EXT) == nothing
    end
end
