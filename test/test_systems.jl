# Shared collection of polynomial systems used by the evaluation sweep and the
# tracker regression cases. Each entry returns `(polys, variables, parameters)`.
using DynamicPolynomials: @polyvar, subs, monomials
using DynamicPolynomials: differentiate as mp_differentiate
using HomotopyContinuationNext: @var, System, dense_poly, to_dict, horner, differentiate
using HomotopyContinuationNext: subs as hc_subs
using MultivariatePolynomials: variables as mp_variables

# Cyclic roots: n sparse equations of degree n in n variables.
function cyclic_system(n::Int)
    @polyvar z[1:n]
    eqs = [
        sum(prod(z[(k - 1) % n + 1] for k in j:(j + m)) for j in 1:n)
            for m in 0:(n - 2)
    ]
    polys = [eqs; [prod(z) - 1]]
    return (polys, collect(z), typeof(z[1])[])
end

# Sigma factor B regulation network. 10 dense equations with float coefficients.
function bacillus_system()
    @polyvar w w2 w2v v w2v2 vP sigmaB w2sigmaB vPp phos
    polys = [
        (-1 * 0.7 * w + -2 * 3600.0 * (w^2 / 2) + 2 * 18.0 * w2) * (0.2 + sigmaB) +
            4.0 * 0.4 * (1 + 30.0sigmaB),
        -1 * 0.7 * w2 +
            3600.0 * (w^2 / 2) +
            -1 * 18.0 * w2 +
            -1 * 3600.0 * w2 * v +
            18.0w2v +
            36.0w2v +
            -1 * 3600.0 * w2 * sigmaB +
            18.0w2sigmaB,
        -1 * 0.7 * w2v +
            3600.0 * w2 * v +
            -1 * 18.0 * w2v +
            -1 * 3600.0 * w2v * v +
            18.0w2v2 +
            -1 * 36.0 * w2v +
            36.0w2v2 +
            1800.0 * w2sigmaB * v +
            -1 * 1800.0 * w2v * sigmaB,
        (
            -1 * 0.7 * v +
                -1 * 3600.0 * w2 * v +
                18.0w2v +
                -1 * 3600.0 * w2v * v +
                18.0w2v2 +
                -1 * 1800.0 * w2sigmaB * v +
                1800.0 * w2v * sigmaB +
                180.0vPp
        ) * (0.2 + sigmaB) + 4.5 * 0.4 * (1 + 30.0sigmaB),
        -1 * 0.7 * w2v2 + 3600.0 * w2v * v + -1 * 18.0 * w2v2 + -1 * 36.0 * w2v2,
        -1 * 0.7 * vP + 36.0w2v + 36.0w2v2 + -1 * 3600.0 * vP * phos + 18.0vPp,
        (
            -1 * 0.7 * sigmaB +
                -1 * 3600.0 * w2 * sigmaB +
                18.0w2sigmaB +
                1800.0 * w2sigmaB * v +
                -1 * 1800.0 * w2v * sigmaB
        ) * (0.2 + sigmaB) + 0.4 * (1 + 30.0sigmaB),
        -1 * 0.7 * w2sigmaB +
            3600.0 * w2 * sigmaB +
            -1 * 18.0 * w2sigmaB +
            -1 * 1800.0 * w2sigmaB * v +
            1800.0 * w2v * sigmaB,
        -1 * 0.7 * vPp + 3600.0 * vP * phos + -1 * 18.0 * vPp + -1 * 180.0 * vPp,
        (phos + vPp) - 2.0,
    ]
    vars = [w, w2, w2v, v, w2v2, vP, sigmaB, w2sigmaB, vPp, phos]
    return (polys, vars, typeof(w)[])
end

# Cyclooctane conformation space: 15 quadrics in 17 variables (underdetermined),
# with irrational constants.
function cyclo_system()
    c² = 2.0
    @polyvar z[1:3, 1:6]
    zero_p = 0.0 * z[1, 1]
    col(k) = Any[z[1, k], z[2, k], z[3, k]]
    Z = [
        Any[zero_p, zero_p, zero_p],
        col(1), col(2), col(3), col(4), col(5),
        Any[z[1, 6] + zero_p, z[2, 6] + zero_p, zero_p],
        Any[sqrt(c²) + zero_p, zero_p, zero_p],
    ]
    sqdist(u, v) = sum((u[k] - v[k])^2 for k in 1:3)
    F1 = [sqdist(Z[i], Z[i + 1]) - c² for i in 1:7]
    F2 = [sqdist(Z[i], Z[i + 2]) - 8c² / 3 for i in 1:6]
    F3 = sqdist(Z[7], Z[1]) - 8c² / 3
    F4 = sqdist(Z[8], Z[2]) - 8c² / 3
    polys = [F1; F2; F3; F4]
    return (polys, vec(z)[1:17], typeof(z[1, 1])[])
end

# Method of moments for a mixture of three Gaussians: 9 equations of degree up
# to 8 in 9 variables, with 9 parameters.
function moments3_system()
    @polyvar a[1:3] x[1:3] s[1:3] m[1:9]

    f0 = a[1] + a[2] + a[3]
    f1 = a[1] * x[1] + a[2] * x[2] + a[3] * x[3]
    f2 = a[1] * (x[1]^2 + s[1]) + a[2] * (x[2]^2 + s[2]) + a[3] * (x[3]^2 + s[3])
    f3 = a[1] * (x[1]^3 + 3 * s[1] * x[1]) +
        a[2] * (x[2]^3 + 3 * s[2] * x[2]) +
        a[3] * (x[3]^3 + 3 * s[3] * x[3])
    f4 = a[1] * (x[1]^4 + 6 * s[1] * x[1]^2 + 3 * s[1]^2) +
        a[2] * (x[2]^4 + 6 * s[2] * x[2]^2 + 3 * s[2]^2) +
        a[3] * (x[3]^4 + 6 * s[3] * x[3]^2 + 3 * s[3]^2)
    f5 = a[1] * (x[1]^5 + 10 * s[1] * x[1]^3 + 15 * x[1] * s[1]^2) +
        a[2] * (x[2]^5 + 10 * s[2] * x[2]^3 + 15 * x[2] * s[2]^2) +
        a[3] * (x[3]^5 + 10 * s[3] * x[3]^3 + 15 * x[3] * s[3]^2)
    f6 = a[1] * (x[1]^6 + 15 * s[1] * x[1]^4 + 45 * x[1]^2 * s[1]^2 + 15 * s[1]^3) +
        a[2] * (x[2]^6 + 15 * s[2] * x[2]^4 + 45 * x[2]^2 * s[2]^2 + 15 * s[2]^3) +
        a[3] * (x[3]^6 + 15 * s[3] * x[3]^4 + 45 * x[3]^2 * s[3]^2 + 15 * s[3]^3)
    f7 = a[1] *
        (x[1]^7 + 21 * s[1] * x[1]^5 + 105 * x[1]^3 * s[1]^2 + 105 * x[1] * s[1]^3) +
        a[2] *
        (x[2]^7 + 21 * s[2] * x[2]^5 + 105 * x[2]^3 * s[2]^2 + 105 * x[2] * s[2]^3) +
        a[3] *
        (x[3]^7 + 21 * s[3] * x[3]^5 + 105 * x[3]^3 * s[3]^2 + 105 * x[3] * s[3]^3)
    f8 = a[1] * (
        x[1]^8 + 28 * s[1] * x[1]^6 + 210 * x[1]^4 * s[1]^2 +
            420 * x[1]^2 * s[1]^3 + 105 * s[1]^4
    ) +
        a[2] * (
        x[2]^8 + 28 * s[2] * x[2]^6 + 210 * x[2]^4 * s[2]^2 +
            420 * x[2]^2 * s[2]^3 + 105 * s[2]^4
    ) +
        a[3] * (
        x[3]^8 + 28 * s[3] * x[3]^6 + 210 * x[3]^4 * s[3]^2 +
            420 * x[3]^2 * s[3]^3 + 105 * s[3]^4
    )

    polys = [f0, f1, f2, f3, f4, f5, f6, f7, f8] .- m
    return (polys, [a; x; s], collect(m))
end

# Six-revolute serial-link inverse kinematics: 12 quadrics in 12 variables.
function six_revolute_system()
    @polyvar z[1:4, 1:3]
    zero_p = 0.0 * z[1, 1]
    e1 = Any[1.0 + zero_p, zero_p, zero_p]
    row(i) = Any[z[i, 1], z[i, 2], z[i, 3]]
    # Z[i] mirrors the original z[i, :]; rows 1 and 6 are pinned to e1.
    Z = [e1, row(1), row(2), row(3), row(4), e1]

    p = [1.0, 1.0, 0.0]
    α = [
        0.9606655201575953,
        -0.12634585123792536,
        -0.821994270849398,
        -0.4612529209681106,
        0.1914597581898848,
    ]
    a = [
        -0.6508165773155992,
        -0.6252070694344862,
        1.2753203087262615,
        -0.30371050526688304,
        0.45398822665915634,
        -0.4042256983795828,
        -1.1485430048515646,
        -0.22145594990414993,
        0.9590486265536351,
    ]
    dot3(u, v) = sum(u[k] * v[k] for k in 1:3)
    cross3(u, v) = Any[
        u[2] * v[3] - u[3] * v[2],
        u[3] * v[1] - u[1] * v[3],
        u[1] * v[2] - u[2] * v[1],
    ]

    f = [dot3(Z[i], Z[i]) - 1 for i in 2:5]
    g = [dot3(Z[i], Z[i + 1]) - cos(α[i]) for i in 1:5]
    h = sum(a[i] .* cross3(Z[i], Z[i + 1]) for i in 1:3) +
        sum(a[i + 4] .* Z[i] for i in 2:5) .- p
    polys = [f; g; h]
    return (polys, vec(z), typeof(z[1, 1])[])
end

# Steiner's conic problem: 15 equations in 15 variables with 30 parameters.
function steiner_system()
    @polyvar x[1:2] a[1:5] c[1:6] y[1:2, 1:5] v[1:6, 1:5]

    f = a[1] * x[1]^2 + a[2] * x[1] * x[2] + a[3] * x[2]^2 +
        a[4] * x[1] + a[5] * x[2] + 1
    ∇f = [mp_differentiate(f, x[1]), mp_differentiate(f, x[2])]
    g = c[1] * x[1]^2 + c[2] * x[1] * x[2] + c[3] * x[2]^2 +
        c[4] * x[1] + c[5] * x[2] + c[6]
    ∇g = [mp_differentiate(g, x[1]), mp_differentiate(g, x[2])]

    polys = mapreduce(vcat, 1:5) do i
        x₀ = y[:, i]
        b₀ = v[:, i]
        fᵢ = subs(f, x => x₀)
        Cᵢ = subs(g, x => x₀, c => b₀)
        ∇ᵢ = [subs(d, x => x₀) for d in ∇f]
        ∇Cᵢ = [subs(d, x => x₀, c => b₀) for d in ∇g]
        [fᵢ, Cᵢ, ∇ᵢ[1] * ∇Cᵢ[2] - ∇ᵢ[2] * ∇Cᵢ[1]]
    end
    return (polys, [a; vec(y)], vec(v))
end

# Four-bar linkage design: 24 equations in 24 variables with 16 parameters.
function four_bar_system()
    @polyvar x a y b x_hat a_hat y_hat b_hat
    @polyvar gamma[1:8] gamma_hat[1:8] delta[1:8] delta_hat[1:8]

    D1 = [
        (a_hat * x - delta_hat[i] * x) * gamma[i] +
            (a * x_hat - delta[i] * x_hat) * gamma_hat[i] +
            (a_hat - x_hat) * delta[i] +
            (a - x) * delta_hat[i] - delta[i] * delta_hat[i]
            for i in 1:8
    ]
    D2 = [
        (b_hat * y - delta_hat[i] * y) * gamma[i] +
            (b * y_hat - delta[i] * y_hat) * gamma_hat[i] +
            (b_hat - y_hat) * delta[i] +
            (b - y) * delta_hat[i] - delta[i] * delta_hat[i]
            for i in 1:8
    ]
    D3 = [gamma[i] * gamma_hat[i] + gamma[i] + gamma_hat[i] for i in 1:8]

    vars = [x; a; y; b; x_hat; a_hat; y_hat; b_hat; gamma; gamma_hat]
    return ([D1; D2; D3], vars, [delta; delta_hat])
end

# Tritangent planes of a space curve: 12 equations in 12 variables with the 20
# coefficients of a dense cubic as parameters.
function tritangents_system()
    @polyvar h[1:3] x[1:3] y[1:3] z[1:3] c[1:20]

    mons = monomials(x, 0:3)
    length(mons) == 20 || error("expected 20 cubic monomials, got $(length(mons))")
    C = sum(c[i] * mons[i] for i in 1:20)
    Q = x[3] - x[1] * x[2]
    ∇Q = [mp_differentiate(Q, xi) for xi in x]
    ∇C = [mp_differentiate(C, xi) for xi in x]
    det3(c1, c2, c3) = c1[1] * (c2[2] * c3[3] - c2[3] * c3[2]) -
        c1[2] * (c2[1] * c3[3] - c2[3] * c3[1]) +
        c1[3] * (c2[1] * c3[2] - c2[2] * c3[1])

    P_x = [sum(h[k] * x[k] for k in 1:3) - 1, Q, C, det3(h, ∇Q, ∇C)]
    P_y = [subs(p, x => y) for p in P_x]
    P_z = [subs(p, x => z) for p in P_x]
    return ([P_x; P_y; P_z], [h; x; y; z], collect(c))
end

# The maximal minors of a 3 by 5 matrix: overdetermined, 10 by 3. Variables come off
# the equations; a fresh `@polyvar x y z` would declare new ones that only print the same.
function minors_system()
    polys = minors_polys()
    vars = collect(mp_variables(polys[1]))
    return (polys, vars, eltype(vars)[])
end

# The ED-discriminant system of a toric variety, for the twisted cubic exponent
# matrix: 21 solutions in 7 orbits of `toric_ed_roots_of_unity`. Returns the system
# and its parameters.
function toric_ed_system()
    A = [3 2 1 0; 0 1 2 3]
    d, n = size(A)
    @polyvar tv[1:d] yv[1:n] uv[1:n]
    φ = [prod(tv[i]^A[i, j] for i in 1:d) for j in 1:n]
    Dφ = [mp_differentiate(φ[j], tv[i]) for j in 1:n, i in 1:d]
    F = System(
        [φ .+ yv .- uv; transpose(Dφ) * yv];
        variables = [tv; yv], parameters = uv,
    )
    return F, uv
end

# The cube roots of unity acting on the first two coordinates, which is the group
# action of `toric_ed_system`.
function toric_ed_roots_of_unity(s)
    t = cis(π * 2 / 3)
    t² = t * t
    return (
        vcat(t * s[1], t * s[2], s[3:end]),
        vcat(t² * s[1], t² * s[2], s[3:end]),
    )
end

# (name, builder), the builder returning `(polys, variables, parameters)`. Lazy, so a
# file including this one for a single system does not build the rest.
const TEST_SYSTEM_COLLECTION = [
    ("cyclic5", () -> cyclic_system(5)),
    ("cyclic7", () -> cyclic_system(7)),
    ("minors", minors_system),
    ("bacillus", bacillus_system),
    ("cyclo", cyclo_system),
    ("moments3", moments3_system),
    ("six_revolute", six_revolute_system),
    ("steiner", steiner_system),
    ("four_bar", four_bar_system),
    ("tritangents", tritangents_system),
]

## ── Systems built through the `Expression` front end ────────────────────────

# Lines on a quintic surface in 3-space: a dense quintic in 4 variables restricted to
# `x = [a; 1]·t + [b; 0]`, one equation per power of `t`. 6 equations of degree 5 in
# `[a; b]`, one parameter per quintic coefficient bar the constant, fixed to 1.
function fano_quintic_system()
    @var x[1:4]
    F, q = dense_poly(x, 5; coeff_name = :q)
    F = hc_subs(F, q[end] => 1)
    q = q[1:(end - 1)]
    @var a[1:3] b[1:3] t
    coeffs_in_t = to_dict(hc_subs(F, x => [a; 1] .* t + [b; 0]), [t])
    return (horner.([coeffs_in_t[[k]] for k in 0:5]), [a; b], q)
end

## ── Non-polynomial systems ──────────────────────────────────────────────────
#
# Each entry carries a plain-Julia `ref` instead of the `MP.differentiate` /
# `MP.coefficient` ground truth the sweep over `TEST_SYSTEM_COLLECTION` uses.

# One rational equation in 6 variables with 8 parameters: division by variables
# and by differences of variables.
function small_rational_system()
    @var y[1:6] q[1:8]
    expr = q[1] / y[1] - q[2] / (-y[1] + y[2]) +
        q[5] * y[4] / (y[1] * y[4] - y[3] * y[2]) +
        q[8] * y[6] / (y[1] * y[6] - y[2] * y[5])
    ref = (x, p) -> [
        p[1] / x[1] - p[2] / (-x[1] + x[2]) +
            p[5] * x[4] / (x[1] * x[4] - x[3] * x[2]) +
            p[8] * x[6] / (x[1] * x[6] - x[2] * x[5]),
    ]
    return ([expr], collect(y), collect(q), ref)
end

# Two equations whose parameters enter under a square root.
function sqrt_parameters_system()
    @var x y a b
    exprs = [sqrt(a + b) * x^2 - y, (x * y + a - sqrt(b))^2 - 3]
    ref = (z, p) -> [
        sqrt(p[1] + p[2]) * z[1]^2 - z[2],
        (z[1] * z[2] + p[1] - sqrt(p[2]))^2 - 3,
    ]
    return (exprs, [x, y], [a, b], ref)
end

# A transcendental system mixing `sin` and `cos` of variables and parameters.
function trig_system()
    @var x y a
    exprs = [sin(a) * x + cos(y) - a, cos(x * y) + x^2 - 1]
    ref = (z, p) -> [
        sin(p[1]) * z[1] + cos(z[2]) - p[1],
        cos(z[1] * z[2]) + z[1]^2 - 1,
    ]
    return (exprs, [x, y], [a], ref)
end

# ∇_w Σ_k Σ_j (meas_j - ŵ_j/ŵ_3)² with ŵ = A[:,:,k] * [w; 1], the gradient
# `rigid_multiview_system` builds symbolically.
function reprojection_gradient(A, w, meas)
    T = promote_type(eltype(A), eltype(w), eltype(meas))
    g = zeros(T, 3)
    for k in axes(A, 3)
        ŵ = A[:, :, k] * [w; 1]
        for j in 1:2
            q = ŵ[j] / ŵ[3]
            c = -2 * (meas[j] - q) / ŵ[3]
            for m in 1:3
                g[m] += c * (A[j, m, k] - q * A[3, m, k])
            end
        end
    end
    return g
end

function rigid_multiview_ref(z, p)
    x, y, λ = z[1:3], z[4:6], z[7]
    A = reshape(p[6:29], 3, 4, 2)
    d = x .- y
    return [
        reprojection_gradient(A, x, p[1:2]) .+ 2λ .* d;
        reprojection_gradient(A, y, p[3:4]) .- 2λ .* d;
        sum(d .^ 2) - p[5]
    ]
end

# Two views of a rigid point pair: the gradient of the reprojection error of both
# points, with the squared distance between them constrained to δ. 7 rational
# equations in `[x; y; λ]`; the parameters are the two image points, δ and the two
# 3 by 4 cameras.
function rigid_multiview_system()
    @var A[1:3, 1:4, 1:2] x[1:3] y[1:3] u[1:2] v[1:2] δ λ
    x̃ = [A[:, :, k] * [x; 1] for k in 1:2]
    ỹ = [A[:, :, k] * [y; 1] for k in 1:2]
    r = sum((x .- y) .^ 2) - δ
    L = sum(sum((u .- z[1:2] ./ z[3]) .^ 2) for z in x̃) +
        sum(sum((v .- z[1:2] ./ z[3]) .^ 2) for z in ỹ) +
        λ * r
    return (
        [differentiate(L, [x; y]); r], [x; y; λ], [u; v; δ; vec(A)],
        rigid_multiview_ref,
    )
end

# (name, expressions, variables, parameters, reference implementation)
# Every remaining unary op and a fractional power, on arguments that stay clear of
# every branch cut and pole: `asin`/`acos` need |arg| ≤ 1 and `x + 2` keeps the
# power's base positive.
function transcendental_system()
    @var x y a
    exprs = [
        exp(x) + tan(y) - a,
        asin(x / 4) + acos(y / 4) + sinh(x) * cosh(y) + tanh(a * x) + (x + 2)^(3 // 2),
    ]
    ref = (z, p) -> [
        exp(z[1]) + tan(z[2]) - p[1],
        asin(z[1] / 4) + acos(z[2] / 4) + sinh(z[1]) * cosh(z[2]) + tanh(p[1] * z[1]) +
            (z[1] + 2)^(3 / 2),
    ]
    return (exprs, [x, y], [a], ref)
end

const NONPOLYNOMIAL_SYSTEM_COLLECTION = [
    ("small_rational", small_rational_system()...),
    ("sqrt_parameters", sqrt_parameters_system()...),
    ("trig", trig_system()...),
    ("transcendental", transcendental_system()...),
    ("rigid_multiview", rigid_multiview_system()...),
]
