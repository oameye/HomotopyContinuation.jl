using Test
using HomotopyContinuationNext:
    @polyvar, System, solve, solutions, is_real, path_results,
    monodromy_solve, nsolutions
using HomotopyContinuationNextCertification:
    certify, certificates, is_certified, is_complex, is_positive, ncertified,
    nreal_certified, ncomplex_certified, ndistinct_certified,
    ndistinct_real_certified, ndistinct_complex_certified,
    certified_solution_interval, solution_candidate, solution_approximation,
    certificate_index, distinct_certificates, distinct_solutions,
    ExtendedSolutionCertificate, SolutionCertificate, CertificationResult,
    save, DistinctCertifiedSolutions, add_solution!, distinct_certified_solutions
import DynamicPolynomials as DP
import Arblib

# ─────────────────────────────────────────────────────────────────────────────
# Certification tests: certify(), the Krawczyk operator, the arbitrary-precision
# Arb fallback, the DistinctCertifiedSolutions accumulator, and the 3264-conics
# (steiner) regression.
# ─────────────────────────────────────────────────────────────────────────────

# The steiner (3264 conics) system. The 2×2 tangency determinant is written out
# by hand so the polynomial construction never touches LinearAlgebra's
# division-based `det`.
function steiner()
    @polyvar x[1:2] a[1:5] c[1:6] y[1:2, 1:5] v[1:6, 1:5]
    f = a[1] * x[1]^2 + a[2] * x[1] * x[2] + a[3] * x[2]^2 + a[4] * x[1] + a[5] * x[2] + 1
    ∇ = DP.differentiate(f, x)
    g = c[1] * x[1]^2 + c[2] * x[1] * x[2] + c[3] * x[2]^2 + c[4] * x[1] + c[5] * x[2] + c[6]
    ∇_2 = DP.differentiate(g, x)
    function Incidence(a₀, b₀, x₀)
        fᵢ = DP.subs(f, x => x₀, a => a₀)
        ∇ᵢ = DP.subs(∇, x => x₀, a => a₀)
        Cᵢ = DP.subs(g, x => x₀, c => b₀)
        ∇_Cᵢ = DP.subs(∇_2, x => x₀, c => b₀)
        detg = ∇ᵢ[1] * ∇_Cᵢ[2] - ∇ᵢ[2] * ∇_Cᵢ[1]
        return [fᵢ; Cᵢ; detg]
    end
    F = reduce(vcat, (Incidence(a, v[:, i], y[:, i]) for i in 1:5))
    return System(F; variables = [a; vec(y)], parameters = vec(v))
end

# Read the solution data file: a count line, then blank-line-separated blocks of
# "<real> <imag>" coordinate lines.
function read_solutions_txt(filename)
    sols = Vector{Vector{ComplexF64}}()
    cur = ComplexF64[]
    seen_count = false
    for line in eachline(filename)
        s = strip(line)
        if isempty(s)
            isempty(cur) || (push!(sols, cur); cur = ComplexF64[])
            continue
        end
        if !seen_count
            seen_count = true
            continue
        end
        p = split(s)
        push!(cur, parse(Float64, p[1]) + im * parse(Float64, p[2]))
    end
    isempty(cur) || push!(sols, cur)
    return sols
end

@testset "Certification" begin
    @testset "Simple: introduction example (18 solutions, 4 real)" begin
        @polyvar x y
        f₁ = (x^4 + y^4 - 1) * (x^2 + y^2 - 2) + x^5 * y
        f₂ = x^2 + 2x * y^2 - 2y^2 - 1 // 2
        F = System([f₁, f₂])
        result = solve(F)

        cert = certify(F, result; show_progress = false)
        @test cert isa CertificationResult
        @test ncertified(cert) == 18
        @test ndistinct_certified(cert) == 18
        @test nreal_certified(cert) == 4
        @test ncomplex_certified(cert) == 14
        @test ndistinct_real_certified(cert) == 4
        @test ndistinct_complex_certified(cert) == 14

        # save() writes a non-empty text representation
        fn = tempname()
        save(fn, cert)
        @test !isempty(read(fn, String))

        # threading = false path
        cert_nothread = certify(F, result; show_progress = false, threading = false)
        @test ncertified(cert_nothread) == 18

        # Double solutions: each true solution appears twice, grouped as duplicates
        S = solutions(result)
        cert2 = certify(F, [S; S]; extended_certificate = true, show_progress = false)
        @test ncertified(cert2) == 36
        @test ndistinct_certified(cert2) == 18
        @test nreal_certified(cert2) == 8
        @test ndistinct_real_certified(cert2) == 4
    end

    @testset "circle ∩ line (2 real)" begin
        @polyvar x y
        F = System([x^2 + y^2 - 1, x - y])
        res = solve(F)
        cert = certify(F, res; show_progress = false)
        @test cert isa CertificationResult
        @test ncertified(cert) == 2
        @test nreal_certified(cert) == 2
        @test ncomplex_certified(cert) == 0
        @test ndistinct_certified(cert) == 2
        for c in certificates(cert)
            @test is_certified(c)
            @test is_real(c)
            @test !is_complex(c)
            I = certified_solution_interval(c)
            @test I !== nothing
            v = solution_approximation(c)      # midpoint of the enclosure
            # the enclosed point satisfies the system to high accuracy
            @test abs(v[1]^2 + v[2]^2 - 1) < 1.0e-12
            @test abs(v[1] - v[2]) < 1.0e-12
        end
    end

    @testset "Reality Check" begin
        @polyvar x
        x0 = [1.0]
        # a solution just off the real axis is not certified real
        F = System([x - 1 + 1.0e-16 * im])
        certF = certify(F, [x0]; show_progress = false)
        @test is_real(certificates(certF)[1]) == false
        # a genuinely real solution is certified real
        G = System([x - 1])
        certG = certify(G, [x0]; show_progress = false)
        @test is_real(certificates(certG)[1]) == true
    end

    @testset "Parameters (Lagrange multipliers, 36 solutions)" begin
        @polyvar x y
        @polyvar λ[1:1] u[1:2]
        f = (x^4 + y^4 - 1) * (x^2 + y^2 - 2) + x^5 * y
        J = DP.differentiate(f, [x, y])                 # ∇f, length 2
        eqs = [[x, y] .- u .- (J .* λ[1]); f]            # 3 equations
        C = System(eqs; variables = [x, y, λ[1]], parameters = u)
        u₀ = [-0.32, -0.1]

        # Solve with u₀ substituted, then certify against the parametric system.
        eqs_sub = [DP.subs(e, u[1] => u₀[1], u[2] => u₀[2]) for e in eqs]
        res = solve(System(eqs_sub; variables = [x, y, λ[1]]))
        cands = solutions(res)
        @test length(cands) == 36

        # certify against the parametric system, positional parameters
        cert = certify(C, cands, u₀; show_progress = false)
        @test ncertified(cert) == 36
        @test ndistinct_certified(cert) == 36
        @test nreal_certified(cert) == 8
        @test ndistinct_real_certified(cert) == 8

        # and via the target_parameters keyword
        cert_kw = certify(C, cands; target_parameters = u₀, show_progress = false)
        @test ncertified(cert_kw) == 36

        # invalid solutions: random points certify (almost) none
        invalid = [100 .* randn(ComplexF64, 3) for _ in 1:10]
        cert_bad = certify(C, invalid, u₀; show_progress = false)
        @test ncertified(cert_bad) < 10
    end

    @testset "positive" begin
        @polyvar x y
        F = System([x^2 + y^2 - 1, x - y])
        res = solve(F)
        cert = certify(F, res; show_progress = false)
        # exactly one of ±(√½, √½) is positive in every coordinate
        @test count(is_positive, certificates(cert)) == 1
        @test count(s -> is_positive(s, 1), certificates(cert)) == 1
        @test count(is_real, certificates(cert)) == 2
    end

    @testset "3264" begin
        F = steiner()
        real_conics = [
            10124547 // 662488724,
            8554609 // 755781377,
            5860508 // 2798943247,
            -251402893 // 1016797750,
            -25443962 // 277938473,
            1 // 1,
            520811 // 1788018449,
            2183697 // 542440933,
            9030222 // 652429049,
            -12680955 // 370629407,
            -24872323 // 105706890,
            1 // 1,
            6537193 // 241535591,
            -7424602 // 363844915,
            6264373 // 1630169777,
            13097677 // 39806827,
            -29825861 // 240478169,
            1 // 1,
            13173269 // 2284890206,
            4510030 // 483147459,
            2224435 // 588965799,
            33318719 // 219393000,
            92891037 // 755709662,
            1 // 1,
            8275097 // 452566634,
            -19174153 // 408565940,
            5184916 // 172253855,
            -23713234 // 87670601,
            28246737 // 81404569,
            1 // 1,
        ]
        real_sols = read_solutions_txt(joinpath(@__DIR__, "data", "3264_real_sols.txt"))
        @test length(real_sols) == 3264

        cert = certify(F, real_sols, real_conics; show_progress = false)
        @test ncertified(cert) == 3264
        @test ndistinct_real_certified(cert) == 3264

        # certifying a parametric system without parameters is an error
        @test_throws ArgumentError certify(F, real_sols; show_progress = false)

        # streaming accumulator: add solutions one by one, keep only distinct
        dcs = DistinctCertifiedSolutions(F, real_conics)
        for s in real_sols
            add_solution!(dcs, s, 1)
        end
        @test length(solutions(dcs)) == 3264

        # and the batch entry point deduplicates repeated inputs
        dcs2 = distinct_certified_solutions(
            F, [real_sols; real_sols[1:100]], real_conics;
            threading = true, show_progress = false,
        )
        @test length(solutions(dcs2)) == 3264
    end

    @testset "duplicate detection" begin
        @polyvar x y
        F = System([x^2 + y^2 - 1, x - y])
        s = solutions(solve(F))[1]
        cert = certify(F, [s, s, s]; show_progress = false)
        @test ncertified(cert) == 3
        @test ndistinct_certified(cert) == 1
        @test length(distinct_certificates(cert)) == 1
        @test length(distinct_solutions(cert)) == 1
        @test length(cert.duplicates) == 1
        @test sort(only(cert.duplicates)) == [1, 2, 3]
    end

    @testset "single solution and PathResult inputs" begin
        @polyvar x y
        F = System([x^2 + y^2 - 1, x - y])
        res = solve(F)
        s = solutions(res)[1]
        cert = certify(F, s; show_progress = false)
        @test ncertified(cert) == 1
        @test certificate_index(certificates(cert)[1]) == 1

        pr = path_results(res)[1]
        certpr = certify(F, pr; show_progress = false)
        @test ncertified(certpr) == 1
    end

    @testset "extended certificate" begin
        @polyvar x y
        F = System([x^2 + y^2 - 1, x - y])
        res = solve(F)
        cert = certify(F, res; extended_certificate = true, show_progress = false)
        @test ncertified(cert) == 2
        for c in certificates(cert)
            @test c isa ExtendedSolutionCertificate
            @test certified_solution_interval(c) !== nothing
            @test c.I′ !== nothing
        end
    end

    @testset "certify uses complex inversion (MonodromyResult)" begin
        # Certify the monodromy result of a polynomial parametric system.
        @polyvar y[1:2] p[1:2]
        F = System(
            [y[1]^2 + y[2]^2 - p[1], y[1] + y[2] - p[2]];
            variables = y, parameters = p,
        )
        monres = monodromy_solve(F; seed = UInt32(4242), threading = false, show_progress = false)
        @test nsolutions(monres) == 2
        cert = certify(F, monres; show_progress = false)
        @test ndistinct_certified(cert) >= 1
        @test ncertified(cert) == nsolutions(monres)
    end

    @testset "extended-precision Arb fallback" begin
        # The arbitrary-precision Krawczyk driver is the fallback the Float64
        # path defers to. Drive it directly through its internal entry point on
        # well-conditioned roots so the test is deterministic (the public path
        # only reaches it on ill-conditioned inputs).
        import HomotopyContinuationNext as HCN
        import HomotopyContinuationNextCertification as HCNC
        @polyvar x y

        # Seed the Arb path exactly as the Float64 driver does: refine the
        # candidate, form the approximate inverse `C ≈ J(x̃)⁻¹`, then hand off.
        # `newton`/`execute!`/`solution` are core; the certification internals
        # (`CertificationCache`, `extended_prec_certify_solution`, …) are in the
        # certification package.
        function run_arb(F, candidate, params; extended = false)
            cache = HCNC.CertificationCache(F)
            cand = ComplexF64.(candidate)
            r = HCN.newton(
                F, cand; p = params, atol = 0.0, rtol = 8 * eps(),
                extended_precision = true, max_iters = 8, cache = cache.newton_cache,
            )
            x̃ = HCN.solution(r)
            if isempty(params)
                HCN.execute!(cache.u_C64, cache.J_C64, cache.jac_interpreter_C64, x̃)
            else
                HCN.execute!(cache.u_C64, cache.J_C64, cache.jac_interpreter_C64, x̃, params)
            end
            C = inv(cache.J_C64)
            cert_params = HCNC.certification_parameters(isempty(params) ? nothing : params)
            is_real_system = HCNC._is_real_system(F)
            CertT = extended ? HCNC.ExtendedSolutionCertificate : HCNC.SolutionCertificate
            return HCNC.extended_prec_certify_solution(
                F, cand, x̃, C, cert_params, cache, 1, is_real_system, CertT,
            )
        end

        # real solution of a real system → certified real, 128-bit Arb enclosure
        F = System([x^2 + y^2 - 1, x - y])
        c = run_arb(F, ComplexF64[-sqrt(0.5), -sqrt(0.5)], ComplexF64[])
        @test is_certified(c)
        @test is_real(c)
        @test !is_complex(c)
        @test precision(c) == 128
        @test certified_solution_interval(c) !== nothing
        v = solution_approximation(c)
        @test abs(v[1]^2 + v[2]^2 - 1) < 1.0e-14
        @test abs(v[1] - v[2]) < 1.0e-14

        # non-real complex solution → is_complex, not is_real
        G = System([x^2 + 1, x - y])
        cc = run_arb(G, ComplexF64[im, im], ComplexF64[])
        @test is_certified(cc)
        @test is_complex(cc)
        @test !is_real(cc)
        w = solution_approximation(cc)
        @test abs(w[1] - im) < 1.0e-14

        # parametric system through the Arb path
        @polyvar a b
        P = System([x^2 - a, y^2 - b]; variables = [x, y], parameters = [a, b])
        cp = run_arb(P, ComplexF64[sqrt(2), sqrt(3)], ComplexF64[2.0, 3.0])
        @test is_certified(cp)
        vp = solution_approximation(cp)
        @test abs(vp[1]^2 - 2) < 1.0e-14
        @test abs(vp[2]^2 - 3) < 1.0e-14

        # extended certificate through the Arb path populates all operator data
        ce = run_arb(F, ComplexF64[sqrt(0.5), sqrt(0.5)], ComplexF64[]; extended = true)
        @test ce isa ExtendedSolutionCertificate
        @test is_certified(ce)
        @test certified_solution_interval(ce) !== nothing
        @test ce.I′ !== nothing
        @test size(ce.Y) == (2, 2)
    end
end
