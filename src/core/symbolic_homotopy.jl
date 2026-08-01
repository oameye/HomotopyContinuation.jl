## Homotopy — a user-written H(x, t) given as expressions.
#
# The equations compile to a system in `x` whose parameters are `[t; parameters]`.
# Taylor orders come from the interpreter's parameter convolution against the
# series `(t, 1, 0, …)`, which is exact for arbitrary dependence on `t`.

"""
    Homotopy(h, variables, t; parameters = [], compile = CompileMode.INTERPRETED)

The homotopy `H(x, t)` given by the expressions `h` in `variables` and the path
parameter `t`. Track it with `solve(H, starts)`; a homotopy with parameters needs
[`fix_parameters`](@ref) first.

`h` may hold [`Expression`](@ref)s or MultivariatePolynomials polynomials, and
`compile` selects the evaluation backend as it does for a [`System`](@ref).
Equations are normalized the same way too: one whose coefficients are far from
unit scale is divided by that scale, [`expressions`](@ref) reports the normalized
equations and [`equation_scales`](@ref) records the factors.

```julia
@var x y z t
H = Homotopy([x^2 + y + z + 2t, 4x^2 * z^2 * y + 4z - 6x * y * z^2], [x, y, z], t)
```
"""
struct Homotopy
    expressions::Vector{Expression}
    variables::Vector{Expression}
    t::Expression
    parameters::Vector{Expression}
    equation_scales::Vector{Float64}
    evaluator::SystemEvaluator
    compile_mode::CompileMode.T
end

function Homotopy(
        h::AbstractVector, vars::AbstractVector, t::Expression;
        parameters::AbstractVector = Expression[],
        compile::CompileMode.T = CompileMode.INTERPRETED,
    )::Homotopy
    exprs = _as_expressions(h)
    is_variable(t) ||
        throw(ArgumentError("the path parameter must be a variable, got `$t`"))
    declared_vars = collect(Expression, _as_variables(vars))
    params = collect(Expression, _as_variables(parameters))
    _contains_variable(declared_vars, t) &&
        throw(ArgumentError("the path parameter `$t` is also listed as a variable"))
    _contains_variable(params, t) &&
        throw(ArgumentError("the path parameter `$t` is also listed as a parameter"))

    declared = Expression[declared_vars; t; params]
    for v in variables(exprs)
        _contains_variable(declared, v) || throw(
            ArgumentError(
                "`$v` occurs in the homotopy but is neither a variable, the path " *
                    "parameter, nor a parameter",
            ),
        )
    end

    F = System(
        exprs; variables = declared_vars,
        parameters = Expression[t; params], compile = compile,
    )
    return Homotopy(
        polynomials(F), declared_vars, t, params,
        equation_scales(F), F.evaluator, compile,
    )
end

Homotopy(
    h::AbstractVector, vars::AbstractVector, t::MP.AbstractVariable;
    parameters::AbstractVector = Expression[],
    compile::CompileMode.T = CompileMode.INTERPRETED,
)::Homotopy = Homotopy(
    h, vars, convert(Expression, t); parameters = parameters, compile = compile,
)

Base.size(H::Homotopy)::Tuple{Int, Int} = (length(H.expressions), length(H.variables))
Base.size(H::Homotopy, i::Integer)::Int = size(H)[i]
Base.length(H::Homotopy)::Int = length(H.expressions)
nvariables(H::Homotopy)::Int = length(H.variables)
nparameters(H::Homotopy)::Int = length(H.parameters)
variables(H::Homotopy)::Vector{Expression} = H.variables
parameters(H::Homotopy)::Vector{Expression} = H.parameters
expressions(H::Homotopy)::Vector{Expression} = H.expressions
equation_scales(H::Homotopy)::Vector{Float64} = H.equation_scales

Base.:(==)(H::Homotopy, G::Homotopy)::Bool =
    H.expressions == G.expressions && H.variables == G.variables &&
    H.t == G.t && H.parameters == G.parameters

function Base.show(io::IO, H::Homotopy)
    if get(io, :compact, false)::Bool
        print(io, "[")
        join(io, H.expressions, ", ")
        print(io, "]")
        return
    end
    println(io, "Homotopy in ", H.t, " of length ", length(H.expressions))
    print(io, " ", length(H.variables), " variables: ")
    join(io, H.variables, ", ")
    if !isempty(H.parameters)
        print(io, "\n ", length(H.parameters), " parameters: ")
        join(io, H.parameters, ", ")
    end
    print(io, "\n\n")
    for (i, e) in enumerate(H.expressions)
        print(io, " ", e)
        i < length(H.expressions) && print(io, "\n")
    end
    return
end

"""
    fix_parameters(H::Homotopy, p::AbstractVector{<:Number}) -> Homotopy

`H` with its parameters substituted by the values `p`, ready to be tracked.
Substituting can push an equation back off unit scale, so
[`equation_scales`](@ref) of the result is the total factor relative to the
equations `H` was built from, not just the one applied here.

```julia
@var x y t a
H = Homotopy([x^2 - a * t - 1, y - t], [x, y], t; parameters = [a])
solve(fix_parameters(H, [3]), [[2.0 + 0im, 1]])
```
"""
function fix_parameters(H::Homotopy, p::AbstractVector{<:Number})::Homotopy
    values = _parameter_values(H, p)
    G = Homotopy(
        [subs(e, H.parameters => values) for e in H.expressions],
        H.variables, H.t; compile = H.compile_mode,
    )
    return Homotopy(
        G.expressions, G.variables, G.t, G.parameters,
        H.equation_scales .* G.equation_scales, G.evaluator, G.compile_mode,
    )
end

# `[t; p]` in the layout the compiled system takes its parameters in.
function _path_parameters(
        H::Homotopy, t::Number, p::AbstractVector{<:Number},
    )::Vector{ComplexF64}
    np = length(H.parameters)
    length(p) == np || throw(
        ArgumentError(
            string(
                "the homotopy has ", np, " parameter(s) but got ", length(p), " value(s)",
            ),
        ),
    )
    q = Vector{ComplexF64}(undef, np + 1)
    q[1] = t
    for i in 1:np
        q[i + 1] = p[i]
    end
    return q
end

"""
    evaluate(H::Homotopy, x, t, p = ComplexF64[])

Value of `H` at the point `x` and path parameter `t`, on the normalized equations
[`expressions`](@ref) reports (see [`equation_scales`](@ref)). The result is real
when every entry is.
"""
evaluate(
    H::Homotopy, x::AbstractVector{<:Number}, t::Number,
    p::AbstractVector{<:Number} = ComplexF64[],
) = evaluate(H.evaluator, x, _path_parameters(H, t, p))

"""
    jacobian(H::Homotopy, x, t, p = ComplexF64[])

Jacobian of `H` in its variables at the point `x` and path parameter `t`. The
result is real when every entry is.
"""
jacobian(
    H::Homotopy, x::AbstractVector{<:Number}, t::Number,
    p::AbstractVector{<:Number} = ComplexF64[],
) = jacobian(H.evaluator, x, _path_parameters(H, t, p))

(H::Homotopy)(x::AbstractVector{<:Number}, t::Number) = evaluate(H, x, t)
(H::Homotopy)(x::AbstractVector{<:Number}, t::Number, p::AbstractVector{<:Number}) =
    evaluate(H, x, t, p)

## The form a `Homotopy` is tracked through. The path parameter is the wrapped system's
## only parameter, so interpolating it from 1 at `t = 1` to 0 at `t = 0` reproduces `t`
## itself, and the parameter-Taylor convolution against `[t, 1, 0, …]` is exact for
## arbitrary dependence on `t`.

_as_homotopy(H::AbstractHomotopy) = H

# Every route that tracks a homotopy accepts both spellings and converts once.
const HomotopyLike = Union{AbstractHomotopy, Homotopy}

function _as_homotopy(H::Homotopy)::ParameterHomotopy
    np = length(H.parameters)
    np == 0 || throw(
        ArgumentError(
            "tracking requires a parameter-free homotopy, but this one has $np " *
                "parameter(s). Fix them first with `fix_parameters(H, p)`.",
        ),
    )
    return ParameterHomotopy(H.evaluator, ComplexF64[1], ComplexF64[0])
end
