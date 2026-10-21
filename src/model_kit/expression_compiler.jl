## Expression lowering
#
# Front-end glue from `Expression` input to the tape pipeline:
#   Expression -> SExpr -> CSE -> tape compiler
#
# The Jacobian is produced by symbolic differentiation on `Expression`, which
# covers division, non-integer and negative powers, and the unary functions that
# `MP.differentiate` cannot express.

function expression_index_map(values::AbstractVector{Expression})::Dict{Symbol, Int}
    index_map = Dict{Symbol, Int}()
    for (i, value) in enumerate(values)
        index_map[Symbol(value)] = i
    end
    return index_map
end

## Sums of monomials
#
# A polynomial written out as a sum of monomials, which is what
# MultivariatePolynomials input converts to, lowers term by term: each monomial as a
# product in variable order, the terms ordered by their monomials. The general
# lowering orders a sum by the full hash of its terms, coefficients included, which
# separates products the tape compiler would otherwise fuse into a multiply-add. The
# Jacobian of such a sum comes straight from its exponents.

# Coefficient of the monomial `e`, its exponents added into `exps`; `found` is
# `false` when `e` is not a monomial in the indexed symbols.
function monomial_coefficient!(
        exps::Vector{Int32}, e::Expression, index::Dict{Symbol, Int},
    )::ExprNumber
    return @match e begin
        SymExpr.ENum(val) => ExprNumber(true, val)
        SymExpr.EVar(name) => add_exponent!(exps, index, name, 1)
        SymExpr.EPow(SymExpr.EVar(name), k) =>
            k > 0 ? add_exponent!(exps, index, name, k) : ExprNumber(false, zero(ComplexF64))
        SymExpr.EMul(args) => begin
            coeff = one(ComplexF64)
            for a in args
                c = monomial_coefficient!(exps, a, index)
                c.found || return c
                coeff *= c.val
            end
            ExprNumber(true, coeff)
        end
        _ => ExprNumber(false, zero(ComplexF64))
    end
end

function add_exponent!(
        exps::Vector{Int32}, index::Dict{Symbol, Int}, name::Symbol, k::Int,
    )::ExprNumber
    i = get(index, name, 0)
    i == 0 && return ExprNumber(false, zero(ComplexF64))
    exps[i] += Int32(k)
    return ExprNumber(true, one(ComplexF64))
end

# The terms of `e` as columns of exponents and their coefficients. `found` is
# `false` when `e` is not a sum of monomials, possibly scaled by a number.
struct MonomialSum
    found::Bool
    exponents::Matrix{Int32}
    coefficients::Vector{ComplexF64}
end

not_a_monomial_sum(n::Int)::MonomialSum =
    MonomialSum(false, Matrix{Int32}(undef, n, 0), ComplexF64[])

function monomial_sum(e::Expression, index::Dict{Symbol, Int})::MonomialSum
    n = length(index)
    # A scaled sum `s * (t₁ + … + tₖ)` distributes its scale over the terms.
    scale = one(ComplexF64)
    body = e
    nsums = 0
    for a in expression_factors(e)
        v = expr_number(a)
        if v.found
            scale *= v.val
        elseif isa_variant(a, SymExpr.EAdd)
            nsums += 1
            body = a
        else
            nsums = 2
        end
    end
    if nsums != 1
        body = e
        scale = one(ComplexF64)
    end
    terms = expression_summands(body)
    E = zeros(Int32, n, length(terms))
    c = Vector{ComplexF64}(undef, length(terms))
    exps = zeros(Int32, n)
    for (k, t) in enumerate(terms)
        fill!(exps, Int32(0))
        v = monomial_coefficient!(exps, t, index)
        v.found || return not_a_monomial_sum(n)
        E[:, k] .= exps
        c[k] = scale * v.val
    end
    return MonomialSum(true, E, c)
end

# Symbols `1:nvars` are variables, the rest parameters.
function monomial_sum_sexpr(
        g::SGraph, E::AbstractMatrix{Int32}, c::Vector{ComplexF64}, nvars::Int,
    )::Int32
    terms = Int32[]
    for k in eachindex(c)
        coeff = c[k]
        iszero(coeff) && continue
        factors = Int32[]
        for i in axes(E, 1)
            e = Int(E[i, k])
            e == 0 && continue
            base = i <= nvars ? svar!(g, i) : sparam!(g, i - nvars)
            push!(factors, e == 1 ? base : spow!(g, base, e))
        end
        if isempty(factors)
            push!(terms, sconst!(g, coeff))
        elseif coeff == one(ComplexF64)
            push!(terms, length(factors) == 1 ? factors[1] : smul!(g, factors))
        else
            pushfirst!(factors, sconst!(g, coeff))
            push!(terms, smul!(g, factors))
        end
    end
    if length(terms) > 1
        keyed = Tuple{UInt, Int32}[(node_hash(g, monomial_of(g, t)), t) for t in terms]
        stable_sort!(keyed, isless)
        for i in eachindex(terms)
            terms[i] = keyed[i][2]
        end
    end
    return wrap_args(g, terms, sconst!(g, zero(ComplexF64)), sadd!)
end

# The term `t` of a sum without its leading coefficient.
function monomial_of(g::SGraph, t::Int32)::Int32
    if is_kind(g, t, SKind.S_MUL) && node_nargs(g, t) > 0 &&
            is_kind(g, node_arg(g, t, 1), SKind.S_CONST)
        rest = get_args(g, t)[2:end]
        return length(rest) == 1 ? rest[1] : smul!(g, rest)
    end
    return t
end

# Derivative of a sum of monomials in symbol `j`.
function differentiate_monomials(
        E::Matrix{Int32}, c::Vector{ComplexF64}, j::Int,
    )::Tuple{Matrix{Int32}, Vector{ComplexF64}}
    keep = findall(k -> E[j, k] > 0, eachindex(c))
    dE = E[:, keep]
    dc = ComplexF64[c[k] * E[j, k] for k in keep]
    dE[j, :] .-= Int32(1)
    return dE, dc
end

function build_instruction_sequence_from_expressions(
        exprs::AbstractVector{Expression},
        variables::AbstractVector{Expression},
        parameters::AbstractVector{Expression},
        include_jacobian::Bool,
    )::InstructionSequence
    nvars = length(variables)
    nparams = length(parameters)
    output_dim = length(exprs)

    var_to_idx = expression_index_map(variables)
    param_to_idx = expression_index_map(parameters)

    symbol_index = copy(var_to_idx)
    for (name, i) in param_to_idx
        symbol_index[name] = nvars + i
    end
    sums = MonomialSum[monomial_sum(e, symbol_index) for e in exprs]

    g = SGraph()
    all_exprs = Vector{Int32}(undef, output_dim)
    for i in eachindex(exprs)
        s = sums[i]
        all_exprs[i] = s.found ?
            monomial_sum_sexpr(g, s.exponents, s.coefficients, nvars) :
            expression_to_sexpr(g, exprs[i], var_to_idx, param_to_idx)
    end

    if include_jacobian
        # Column-major: all equations for variable 1, then variable 2, ...
        for (j, v) in enumerate(variables)
            sym = Symbol(v)
            for (i, f) in enumerate(exprs)
                s = sums[i]
                push!(
                    all_exprs, if s.found
                        dE, dc = differentiate_monomials(s.exponents, s.coefficients, j)
                        monomial_sum_sexpr(g, dE, dc, nvars)
                    else
                        expression_to_sexpr(g, differentiate_expression(f, sym), var_to_idx, param_to_idx)
                    end,
                )
            end
        end
    end

    replacements, reduced_exprs = cse(g, all_exprs)
    return compile_to_instructions(g, replacements, reduced_exprs, nvars, nparams, output_dim)
end
