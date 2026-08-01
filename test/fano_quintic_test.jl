using Test
using HomotopyContinuationNext: System, degrees, nparameters, is_polynomial,
    evaluate, to_number, @var, dense_poly
using Random: MersenneTwister

include("test_systems.jl")

# Covers the `Expression` front end (`dense_poly` → `subs` → `to_dict` → `horner`) on a
# system large enough to matter. The 15625-path solve is in `test/extensive/`.
@testset "Lines on a quintic surface" begin
    equations, vars, params = fano_quintic_system()
    F = System(equations; variables = vars, parameters = params)

    @testset "shape" begin
        @test size(F) == (6, 6)
        @test degrees(F) == fill(5, 6)
        @test all(f -> is_polynomial(f, vars), equations)
        # 4 variables, degree 5: 126 monomials, the constant one fixed to 1.
        @test nparameters(F) == 125
    end

    # `horner` rewrote each equation, so check values against the definition, not shape.
    @testset "equations evaluate to the quintic restricted to the line" begin
        rng = MersenneTwister(7)
        abvals = randn(rng, ComplexF64, 6)
        qvals = randn(rng, ComplexF64, 125)
        u = evaluate(F, abvals, qvals)
        @test length(u) == 6

        # Rebuild the same quintic and substitute the line directly.
        @var y[1:4] t
        Q, c = dense_poly(y, 5; coeff_name = :c)
        Q = hc_subs(Q, c[end] => 1)
        line = [abvals[1:3] .* t .+ abvals[4:6]; t]
        along = hc_subs(hc_subs(Q, c[1:(end - 1)] => qvals), y => line)
        # Σ uₖ tᵏ must match `along` at any t.
        for tval in (0.0 + 0im, 1.0 + 0im, 0.5 - 0.25im)
            @test isapprox(
                sum(u[k + 1] * tval^k for k in 0:5),
                to_number(hc_subs(along, t => tval));
                rtol = 1.0e-10,
            )
        end
    end
end
