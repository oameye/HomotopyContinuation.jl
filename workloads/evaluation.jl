# ModelKit expressions and direct System / Homotopy evaluation

const EVALUATION = (
    :expression_algebra,
    :polynomial_coefficients,
    :system_evaluate,
    :homotopy_evaluate,
    :symbolic_homotopy_solve,
)

function run(::Val{:expression_algebra})
    @var x y a
    f = (x + a * y)^3 + sin(x) / (1 + y^2)
    g = differentiate(f, [x, y])
    return expand(subs(g, a => 2)), horner(expand((x + y)^4)), num_den(f), to_number(subs(a, a => 1))
end

function run(::Val{:polynomial_coefficients})
    @var x y
    f, c = dense_poly([x, y], 3; coeff_name = :c)
    M, coeffs = exponents_coefficients(subs(f, c => 1:length(c)), [x, y])
    g = poly_from_exponents_coefficients(M, coeffs, [x, y])
    return (
        to_dict(g, [x, y]), monomials([x, y], 2), coefficients(g, [x, y]),
        coeffs_as_dense_poly(g, [x, y], 3), rand_poly(MersenneTwister(1), [x, y], 2),
    )
end

function run(::Val{:system_evaluate})
    F = square_system(CompileMode.INTERPRETED)
    x = ComplexF64[0.3, 0.7]
    return evaluate(F, x), jacobian(F, x)
end

function run(::Val{:homotopy_evaluate})
    @var x y t a
    H = Homotopy([x^2 - a * (1 + 3t), y - x * (1 + t)], [x, y], t; parameters = [a])
    x0, p = ComplexF64[2.0, 4.0], ComplexF64[1.0]
    return evaluate(H, x0, 0.5, p), jacobian(H, x0, 0.5, p)
end

function run(::Val{:symbolic_homotopy_solve})
    @var x y t
    H = Homotopy([x^2 - (1 + 3t), y - x * (1 + t)], [x, y], t)
    starts = [ComplexF64[2.0, 4.0], ComplexF64[-2.0, -4.0]]
    return solve(H, starts, Continuation(; seed = SEED, show_progress = false), Serial())
end
