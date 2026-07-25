# Shared collection of polynomial systems used by the evaluation sweep and the
# tracker regression cases. Each entry returns `(polys, variables, parameters)`.
using DynamicPolynomials: @polyvar, subs, differentiate, monomials

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
    ∇f = [differentiate(f, x[1]), differentiate(f, x[2])]
    g = c[1] * x[1]^2 + c[2] * x[1] * x[2] + c[3] * x[2]^2 +
        c[4] * x[1] + c[5] * x[2] + c[6]
    ∇g = [differentiate(g, x[1]), differentiate(g, x[2])]

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
    ∇Q = [differentiate(Q, xi) for xi in x]
    ∇C = [differentiate(C, xi) for xi in x]
    det3(c1, c2, c3) = c1[1] * (c2[2] * c3[3] - c2[3] * c3[2]) -
        c1[2] * (c2[1] * c3[3] - c2[3] * c3[1]) +
        c1[3] * (c2[1] * c3[2] - c2[2] * c3[1])

    P_x = [sum(h[k] * x[k] for k in 1:3) - 1, Q, C, det3(h, ∇Q, ∇C)]
    P_y = [subs(p, x => y) for p in P_x]
    P_z = [subs(p, x => z) for p in P_x]
    return ([P_x; P_y; P_z], [h; x; y; z], collect(c))
end

# (name, polys, variables, parameters)
const TEST_SYSTEM_COLLECTION = [
    ("cyclic5", cyclic_system(5)...),
    ("cyclic7", cyclic_system(7)...),
    ("bacillus", bacillus_system()...),
    ("cyclo", cyclo_system()...),
    ("moments3", moments3_system()...),
    ("six_revolute", six_revolute_system()...),
    ("steiner", steiner_system()...),
    ("four_bar", four_bar_system()...),
    ("tritangents", tritangents_system()...),
]
