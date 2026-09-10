using Test
using TOML

@testset "v2.22.4 parity ledger" begin
    ledger_path = joinpath(
        @__DIR__, "..", "implementation_docs", "v2_22_4_parity_ledger.toml",
    )
    ledger = TOML.parsefile(ledger_path)

    @test ledger["upstream_sha"] == "0cbf1e27d062d7eb01d962c3b948c39444c9a62c"
    @test ledger["v3_base_sha"] == "b5c7c2e81f372c22a29ff2963743ab5fad1fa9ed"
    @test ledger["upstream_testset_occurrences"] == 208

    entries = ledger["testset"]
    @test length(entries) == 208

    ids = [(entry["source"], entry["line"]) for entry in entries]
    @test length(unique(ids)) == length(ids)

    allowed = Set((
        "equivalent-test",
        "replacement-test",
        "architecture-obsolete",
        "deliberate-api-break",
        "upstream-disabled",
        "gap",
        "unaccounted",
    ))
    @test all(entry["classification"] in allowed for entry in entries)
    @test count(entry -> entry["classification"] == "unaccounted", entries) == 0
    @test count(entry -> entry["classification"] == "gap", entries) == 0

    for entry in entries
        @test !isempty(entry["raw"])
        @test !isempty(entry["rationale"])
        if entry["classification"] != "upstream-disabled"
            @test entry["evidence"] != "-"
        end
    end
end
