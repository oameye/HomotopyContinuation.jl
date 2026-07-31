using Test
import HomotopyContinuationNext as HCN
using HomotopyContinuationNext:
    @polyvar, @var, System, solve, solutions, is_real, path_results,
    nsolutions, Expression, differentiate, TotalDegree, Monodromy, Serial
using HomotopyContinuationNextCertification:
    Certification, certify, certificates, is_certified, is_complex, is_positive, ncertified,
    nreal_certified, ncomplex_certified, ndistinct_certified,
    ndistinct_real_certified, ndistinct_complex_certified,
    certified_solution_interval, solution_candidate, solution_approximation,
    certificate_index, distinct_certificates, distinct_solutions,
    ExtendedSolutionCertificate, SolutionCertificate, CertificationResult,
    save, DistinctCertifiedSolutions, add_solution!, distinct_certified_solutions
import DynamicPolynomials as DP
import Arblib
using LinearAlgebra: det

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
        result = solve(F, TotalDegree(; show_progress = false))

        cert = certify(F, result, Certification(; show_progress = false))
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

        # serial path
        cert_nothread = certify(
            F, result, nothing, Certification(; show_progress = false), Serial(),
        )
        @test ncertified(cert_nothread) == 18

        # Double solutions: each true solution appears twice, grouped as duplicates
        S = solutions(result)
        cert2 = certify(
            F, [S; S], nothing,
            Certification(; extended_certificate = true, show_progress = false),
        )
        @test ncertified(cert2) == 36
        @test ndistinct_certified(cert2) == 18
        @test nreal_certified(cert2) == 8
        @test ndistinct_real_certified(cert2) == 4
    end

    @testset "circle ∩ line (2 real)" begin
        @polyvar x y
        F = System([x^2 + y^2 - 1, x - y])
        res = solve(F, TotalDegree(; show_progress = false))
        cert = certify(F, res, Certification(; show_progress = false))
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
        certF = certify(F, [x0], Certification(; show_progress = false))
        @test is_real(certificates(certF)[1]) == false
        # a genuinely real solution is certified real
        G = System([x - 1])
        certG = certify(G, [x0], Certification(; show_progress = false))
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
        res = solve(
            System(eqs_sub; variables = [x, y, λ[1]]),
            TotalDegree(; show_progress = false),
        )
        cands = solutions(res)
        @test length(cands) == 36

        # certify against the parametric system at those parameter values
        cert = certify(C, cands, u₀, Certification(; show_progress = false))
        @test ncertified(cert) == 36
        @test ndistinct_certified(cert) == 36
        @test nreal_certified(cert) == 8
        @test ndistinct_real_certified(cert) == 8

        # invalid solutions: random points certify (almost) none
        invalid = [100 .* randn(ComplexF64, 3) for _ in 1:10]
        cert_bad = certify(C, invalid, u₀, Certification(; show_progress = false))
        @test ncertified(cert_bad) < 10
    end

    @testset "positive" begin
        @polyvar x y
        F = System([x^2 + y^2 - 1, x - y])
        res = solve(F, TotalDegree(; show_progress = false))
        cert = certify(F, res, Certification(; show_progress = false))
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

        cert = certify(
            F, real_sols, real_conics, Certification(; show_progress = false),
        )
        @test ncertified(cert) == 3264
        @test ndistinct_real_certified(cert) == 3264

        # certifying a parametric system without parameters is an error
        @test_throws ArgumentError certify(
            F, real_sols, nothing, Certification(; show_progress = false),
        )

        # streaming accumulator: add solutions one by one, keep only distinct
        dcs = DistinctCertifiedSolutions(F, real_conics)
        for s in real_sols
            add_solution!(dcs, s, 1)
        end
        @test length(solutions(dcs)) == 3264

        # and the batch entry point deduplicates repeated inputs
        dcs2 = distinct_certified_solutions(
            F, [real_sols; real_sols[1:100]], real_conics,
            Certification(; show_progress = false),
        )
        @test length(solutions(dcs2)) == 3264
    end

    @testset "duplicate detection" begin
        @polyvar x y
        F = System([x^2 + y^2 - 1, x - y])
        s = solutions(solve(F, TotalDegree(; show_progress = false)))[1]
        cert = certify(F, [s, s, s], Certification(; show_progress = false))
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
        res = solve(F, TotalDegree(; show_progress = false))
        s = solutions(res)[1]
        cert = certify(F, s, Certification(; show_progress = false))
        @test ncertified(cert) == 1
        @test certificate_index(certificates(cert)[1]) == 1

        pr = path_results(res)[1]
        certpr = certify(F, pr, Certification(; show_progress = false))
        @test ncertified(certpr) == 1
    end

    @testset "extended certificate" begin
        @polyvar x y
        F = System([x^2 + y^2 - 1, x - y])
        res = solve(F, TotalDegree(; show_progress = false))
        cert = certify(
            F, res, nothing,
            Certification(; extended_certificate = true, show_progress = false),
        )
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
        monres = solve(
            F, Monodromy(; seed = UInt32(4242), show_progress = false), Serial(),
        )
        @test nsolutions(monres) == 2
        cert = certify(F, monres, Certification(; show_progress = false))
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

    @testset "non-polynomial input" begin
        import HomotopyContinuationNextCertification as HCNC
        @var x y a b

        Fr = System([x / y - 2, x^2 + y^2 - 5])
        cr = certify(
            Fr, [ComplexF64[2, 1], ComplexF64[-2, -1]], nothing,
            Certification(; show_progress = false),
        )
        @test ncertified(cr) == 2
        @test nreal_certified(cr) == 2

        # A certificate at 53 bits means the interval `sqrt` carried it, not Arb.
        Fq = System([sqrt(a + b) * x^2 - y, x + y - 1]; parameters = [a, b])
        s5 = sqrt(5.0)
        xs = [(-1 + sqrt(1 + 4s5)) / (2s5), (-1 - sqrt(1 + 4s5)) / (2s5)]
        cq = certify(
            Fq, [ComplexF64[xi, 1 - xi] for xi in xs], ComplexF64[2.0, 3.0],
            Certification(; show_progress = false),
        )
        @test ncertified(cq) == 2
        @test nreal_certified(cq) == 2
        @test all(c -> precision(c) == 53, certificates(cq))

        Ft = System([sin(x) + y, cos(x) - y - 1]; variables = [x, y])
        ct = certify(
            Ft, [ComplexF64[0, 0]], nothing, Certification(; show_progress = false),
        )
        @test ncertified(ct) == 1
        @test precision(only(certificates(ct))) == 53

        Fc = System([x^2 + im * y, x - y])
        @test !HCNC._is_real_system(Fc)
        @test HCNC._is_real_system(Fr)
    end
end

# `d/dxⱼ Σᵢ sᵢ log(gᵢ)` over the 3x8 minors and conics: a sum of 76 rational
# terms, ill conditioned at the solution below.
@testset "rational log-derivative system needs extended precision" begin
    combos_from(lo, n, k) = k == 0 ? [Int[]] :
        isempty(lo:(n - k + 1)) ? Vector{Int}[] :
        reduce(vcat, [[[i; c] for c in combos_from(i + 1, n, k - 1)] for i in lo:(n - k + 1)])
    combos(n, k) = combos_from(1, n, k)

    @var x[1:8]
    M = Expression[
        1 0 0 1 1 1 1 1
        0 1 0 1 x[1] x[2] x[3] x[4]
        0 0 1 1 x[5] x[6] x[7] x[8]
    ]
    minors = filter(
        m -> !isempty(HCN.variables(m)), [det(M[:, c]) for c in combos(8, 3)],
    )
    V = Expression[zero(Expression) for _ in 1:6, _ in 1:8]
    for i in 1:8
        row = 1
        for j in 1:3, k in j:3
            V[row, i] = M[j, i] * M[k, i]
            row += 1
        end
    end
    g = [minors; [det(V[:, c]) for c in combos(8, 6)]]
    @test length(g) == 76

    @var s[1:76]
    F = System(
        [sum(s[i] * differentiate(g[i], x[j]) / g[i] for i in eachindex(g)) for j in 1:8];
        variables = collect(x), parameters = collect(s),
    )
    @test size(F) == (8, 8)

    p = ComplexF64[
        0.04694929498353116 - 0.9795943487338107im,
        0.03814936256080075 + 0.6036805123359278im,
        0.19811294615170094 - 0.04655749375473129im,
        -0.568612454485359 - 0.30035127049952426im,
        -0.15638090220760134 - 1.2654555749207392im,
        -0.41345309978217826 - 0.1784389293895055im,
        -1.1113178721109995 - 0.22606998166881093im,
        0.9982708200513642 + 0.8528947638503744im,
        0.20166791921434227 + 0.9988764609171762im,
        -0.8100994025378792 + 0.3969273031583972im,
        0.6071086706963922 - 1.4311940916205672im,
        -0.01611865898465837 - 0.4502118183049486im,
        -0.110443619721666 - 0.7045834618227158im,
        0.7234274346282669 + 0.6283366831106044im,
        0.36809026072491596 - 0.8203266151063194im,
        1.1813346349972564 + 0.3998055846562327im,
        0.7027850166072859 - 1.2954603858129299im,
        1.0399846079963362 - 0.08005516566457833im,
        0.9198982114121447 + 0.9781446557067801im,
        -0.6891065881783133 - 0.4053802660764455im,
        0.483503689861307 - 0.3225817056899272im,
        -1.3251128587644545 + 0.897858125440237im,
        0.0747608828119116 - 0.5836512862131934im,
        -0.17485527683038962 + 0.44459749365390716im,
        0.21960852538572917 - 0.5379498288177373im,
        -0.8014331739697617 + 0.059816621817923445im,
        -0.43460375391295053 - 0.2546402146687261im,
        -0.6046592521930908 - 0.2957968257500265im,
        -1.1053855142695876 + 1.590614119832034im,
        0.9840486357930686 + 0.7702769853904382im,
        0.16170642515220965 - 0.1222743301057992im,
        0.4795973979603369 + 0.2810297596371718im,
        -1.2000892295608054 - 0.8923206318916687im,
        -0.6077910779725929 + 0.26436888705825917im,
        -1.0376531919986127 - 0.33652306556055805im,
        -0.06733756067158703 - 0.639783597527494im,
        -0.37371460434888953 + 0.24901918161963316im,
        0.017508442680565478 + 0.32123450723164876im,
        1.4128224393512774 + 0.33400143986123293im,
        -0.0434837081190993 - 0.8985222271337256im,
        0.06449022830922942 + 0.04671222198604458im,
        0.24633610502585773 + 0.5801362854382477im,
        -0.6955336053945378 + 1.0796258697355416im,
        -0.6329313759525533 - 0.2354860006860427im,
        0.6109430855309662 - 0.6729029511956512im,
        -0.260922761581385 - 0.12434880231119286im,
        0.059517746441893456 + 0.38136326366327844im,
        0.12550005778394707 + 0.4949701677886087im,
        0.21111192677412854 + 0.6810392555780735im,
        0.6320528544930368 - 0.2833511182719277im,
        -0.6050405544591456 - 0.9440915895571896im,
        -1.4345871717374217 + 0.22556237643401553im,
        -0.6887743690027825 - 0.47027128695679876im,
        -0.6415806120384493 - 0.5014670765374285im,
        0.08502741985519593 - 0.6235217666945221im,
        0.21493766396179834 + 0.8552649062604912im,
        0.3046927960034058 - 1.590707502314377im,
        0.44767647377604874 + 0.8188582740588161im,
        1.1192822043579032 + 0.3234506727151911im,
        -0.008792767666757002 - 0.21480481868217485im,
        -1.1022743248740579 + 1.439084779398887im,
        -1.25202306664751 - 0.4856621088482818im,
        0.5609474151450039 - 0.06309637623414076im,
        -0.20050911884693357 + 0.2973185023874754im,
        -0.09243154166047889 - 0.49984234324259064im,
        -0.28169898290001166 + 0.9680337642860322im,
        1.8202758158992145 - 0.7826768803555774im,
        0.46305418751090566 + 0.19925413382595714im,
        -0.017534482018503578 - 0.6315561160225208im,
        0.28487129868590444 + 1.299813316065981im,
        -0.507158318173951 - 0.7613059977835542im,
        -1.5246336574768562 + 0.15842771871857747im,
        0.19964798257040645 - 0.028394263163747824im,
        -1.3150248862774019 + 0.43865160271978626im,
        -0.1447364027631027 + 0.04047743522664874im,
        0.18766444754121442 + 0.8548396241105418im,
    ]
    sol = ComplexF64[
        -267.34389183407836 + 90.16013119150394im,
        -0.5175728762869246 - 0.4762083613875728im,
        -0.5275621759154175 - 0.48081770228311727im,
        -0.5187465849951617 - 0.49134401394255406im,
        -0.14084183600441835 + 1.181750659496792im,
        -0.4215223684209179 + 0.1331890129058157im,
        -0.42154412086687626 + 0.13324039516624503im,
        -0.42153938328192286 + 0.13323037611055516im,
    ]

    r = certify(F, [sol], p, Certification(; show_progress = false))
    @test ncertified(r) == 1
    cert = only(certificates(r))
    @test is_certified(cert)
    @test precision(cert) > 53
end
