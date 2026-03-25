## MP polynomial → Interpreter pipeline
#
# Converts DynamicPolynomials input into an InstructionSequence and Interpreter
# via: extract supports → build shared monomial table → build IR → compile

## Support extraction

"""
    extract_support(poly, vars::Vector{V}) where {V}

Extract the support (exponent matrix) and coefficients from a single polynomial.
Returns `(exponents::Matrix{Int}, coefficients::Vector{ComplexF64})` where
`exponents` has size `(nvars, nterms)`.
"""
function extract_support(
        poly::MP.AbstractPolynomialLike,
        vars::AbstractVector,
    )
    ts = MP.terms(poly)
    nterms = length(ts)
    nvars = length(vars)

    exponents = zeros(Int, nvars, nterms)
    coefficients = Vector{ComplexF64}(undef, nterms)

    for (j, term) in enumerate(ts)
        coefficients[j] = ComplexF64(MP.coefficient(term))
        mono = MP.monomial(term)
        for (i, v) in enumerate(vars)
            exponents[i, j] = MP.degree(mono, v)
        end
    end

    return exponents, coefficients
end

## Power decomposition for a single variable

const IR_ZERO = ComplexF64(0.0)
const IR_ONE = ComplexF64(1.0)
const IR_MINUS_ONE = ComplexF64(-1.0)

const IRValue = Union{ComplexF64, Symbol, IRStatementRef}
const PowerKey = Tuple{Symbol, Int}
const PowerTable = Dict{PowerKey, IRStatementRef}
const ScaledValueKey = Tuple{ComplexF64, IRValue}
const ScaledValueTable = Dict{ScaledValueKey, IRValue}

struct ProductTerm
    left::IRValue
    right::IRValue
    is_pair::Bool
end

ProductTerm(left::IRValue) = ProductTerm(left, IR_ONE, false)
ProductTerm(left::IRValue, right::IRValue) = ProductTerm(left, right, true)

_as_ir_value(x::IRValue)::IRValue = x
function _as_ir_value(x::IRStatementArg)::IRValue
    x === nothing && error("unexpected nothing in IR builder")
    return x
end

function new_ir_ref!(ref_counter::Base.RefValue{Int})::IRStatementRef
    ref_counter[] += 1
    return IRStatementRef(ref_counter[])
end

_is_ir_zero(x::IRStatementArg) = x isa ComplexF64 && iszero(x)
_is_ir_one(x::IRStatementArg) = x isa ComplexF64 && x == IR_ONE
_is_ir_minus_one(x::IRStatementArg) = x isa ComplexF64 && x == IR_MINUS_ONE

function _emit_unary!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        op::OpType.T,
        arg::IRStatementArg,
    )::IRStatementRef
    r = new_ir_ref!(ref_counter)
    push!(ir_stmts, IRStatement(op, r, arg))
    return r
end

function _emit_binary!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        op::OpType.T,
        a::IRStatementArg,
        b::IRStatementArg,
    )::IRStatementRef
    r = new_ir_ref!(ref_counter)
    push!(ir_stmts, IRStatement(op, r, a, b))
    return r
end

function _emit_ternary!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        op::OpType.T,
        a::IRStatementArg,
        b::IRStatementArg,
        c::IRStatementArg,
    )::IRStatementRef
    r = new_ir_ref!(ref_counter)
    push!(ir_stmts, IRStatement(op, r, a, b, c))
    return r
end

function _emit_quaternary!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        op::OpType.T,
        a::IRStatementArg,
        b::IRStatementArg,
        c::IRStatementArg,
        d::IRStatementArg,
    )::IRStatementRef
    r = new_ir_ref!(ref_counter)
    push!(ir_stmts, IRStatement(op, r, a, b, c, d))
    return r
end

function _negate_arg!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        arg::IRStatementArg,
    )::IRStatementArg
    arg === nothing && return nothing
    arg isa ComplexF64 && return -arg
    return _emit_unary!(ir_stmts, ref_counter, OpType.OP_NEG, arg)
end

function _mul_args!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        a::IRStatementArg,
        b::IRStatementArg,
    )::IRStatementArg
    (a === nothing || b === nothing) && return nothing
    _is_ir_zero(a) && return IR_ZERO
    _is_ir_zero(b) && return IR_ZERO
    _is_ir_one(a) && return b
    _is_ir_one(b) && return a
    _is_ir_minus_one(a) && return _negate_arg!(ir_stmts, ref_counter, b)
    _is_ir_minus_one(b) && return _negate_arg!(ir_stmts, ref_counter, a)
    return _emit_binary!(ir_stmts, ref_counter, OpType.OP_MUL, a, b)
end

function _add_args!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        a::IRStatementArg,
        b::IRStatementArg,
    )::IRStatementArg
    a === nothing && return b
    b === nothing && return a
    _is_ir_zero(a) && return b
    _is_ir_zero(b) && return a
    return _emit_binary!(ir_stmts, ref_counter, OpType.OP_ADD, a, b)
end

function _sub_args!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        a::IRStatementArg,
        b::IRStatementArg,
    )::IRStatementArg
    a === nothing && return _negate_arg!(ir_stmts, ref_counter, b)
    b === nothing && return a
    _is_ir_zero(b) && return a
    return _emit_binary!(ir_stmts, ref_counter, OpType.OP_SUB, a, b)
end

function _muladd_args!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        a::IRStatementArg,
        b::IRStatementArg,
        c::IRStatementArg,
    )::IRStatementArg
    c === nothing && return _mul_args!(ir_stmts, ref_counter, a, b)
    (a === nothing || b === nothing) && return c
    _is_ir_zero(a) && return c
    _is_ir_zero(b) && return c
    _is_ir_one(a) && return _add_args!(ir_stmts, ref_counter, b, c)
    _is_ir_one(b) && return _add_args!(ir_stmts, ref_counter, a, c)
    _is_ir_minus_one(a) && return _sub_args!(ir_stmts, ref_counter, c, b)
    _is_ir_minus_one(b) && return _sub_args!(ir_stmts, ref_counter, c, a)
    return _emit_ternary!(ir_stmts, ref_counter, OpType.OP_MULADD, a, b, c)
end

function _mulsub_args!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        a::IRStatementArg,
        b::IRStatementArg,
        c::IRStatementArg,
    )::IRStatementArg
    c === nothing && return _mul_args!(ir_stmts, ref_counter, a, b)
    (a === nothing || b === nothing) && return _negate_arg!(ir_stmts, ref_counter, c)
    _is_ir_zero(a) && return _negate_arg!(ir_stmts, ref_counter, c)
    _is_ir_zero(b) && return _negate_arg!(ir_stmts, ref_counter, c)
    _is_ir_one(a) && return _sub_args!(ir_stmts, ref_counter, b, c)
    _is_ir_one(b) && return _sub_args!(ir_stmts, ref_counter, a, c)
    return _emit_ternary!(ir_stmts, ref_counter, OpType.OP_MULSUB, a, b, c)
end

function _submul_args!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        a::IRStatementArg,
        b::IRStatementArg,
        c::IRStatementArg,
    )::IRStatementArg
    (a === nothing || b === nothing) && return c
    c === nothing && return _negate_arg!(ir_stmts, ref_counter, _mul_args!(ir_stmts, ref_counter, a, b))
    _is_ir_zero(a) && return c
    _is_ir_zero(b) && return c
    _is_ir_one(a) && return _sub_args!(ir_stmts, ref_counter, c, b)
    _is_ir_one(b) && return _sub_args!(ir_stmts, ref_counter, c, a)
    return _emit_ternary!(ir_stmts, ref_counter, OpType.OP_SUBMUL, a, b, c)
end

function _mulmuladd_args!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        a::IRStatementArg,
        b::IRStatementArg,
        c::IRStatementArg,
        d::IRStatementArg,
    )::IRStatementArg
    return _emit_quaternary!(ir_stmts, ref_counter, OpType.OP_MULMULADD, a, b, c, d)
end

function _mulmulsub_args!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        a::IRStatementArg,
        b::IRStatementArg,
        c::IRStatementArg,
        d::IRStatementArg,
    )::IRStatementArg
    return _emit_quaternary!(ir_stmts, ref_counter, OpType.OP_MULMULSUB, a, b, c, d)
end

function _get_constant_ref!(
        constants_list::Vector{ComplexF64},
        constants_map::Dict{ComplexF64, ComplexF64},
        c::ComplexF64,
    )::ComplexF64
    if haskey(constants_map, c)
        return constants_map[c]
    end
    push!(constants_list, c)
    constants_map[c] = c
    return c
end

function _is_real_negative(c::ComplexF64)::Bool
    return iszero(imag(c)) && real(c) < 0.0
end

function _copy_exponent_column(exponents::Matrix{Int}, j::Int)::ExponentVector
    nvars = size(exponents, 1)
    exponent = Vector{Int}(undef, nvars)
    @inbounds for i in 1:nvars
        exponent[i] = exponents[i, j]
    end
    return exponent
end

function _exponent_key(
        exponent::AbstractVector{Int},
        ::Type{<:NTuple{N, Int}},
    )::NTuple{N, Int} where {N}
    return ntuple(i -> exponent[i], Val(N))
end

function _single_factor_key(
        ::Type{<:NTuple{N, Int}},
        idx::Int,
        exponent::Int,
    )::NTuple{N, Int} where {N}
    return ntuple(i -> i == idx ? exponent : 0, Val(N))
end

function _combined_factor_key(
        a::NTuple{N, Int},
        b::NTuple{N, Int},
    )::NTuple{N, Int} where {N}
    return ntuple(i -> a[i] + b[i], Val(N))
end

function _remainder_key(
        exponent::AbstractVector{Int},
        candidate::NTuple{N, Int},
    )::NTuple{N, Int} where {N}
    return ntuple(i -> exponent[i] - candidate[i], Val(N))
end

_create_power_table() = PowerTable()

function _create_monomial_table(nvars::Int)
    K = NTuple{nvars, Int}
    return Dict{K, IRValue}()
end

_create_scaled_value_table() = ScaledValueTable()

function _scale_ir_value!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        scaled_value_table::ScaledValueTable,
        constants_list::Vector{ComplexF64},
        constants_map::Dict{ComplexF64, ComplexF64},
        coeff::ComplexF64,
        arg::IRValue,
    )::IRValue
    coeff == IR_ONE && return arg
    key = (coeff, arg)
    haskey(scaled_value_table, key) && return scaled_value_table[key]

    scaled = if coeff == ComplexF64(2.0)
        _as_ir_value(_add_args!(ir_stmts, ref_counter, arg, arg))
    else
        c_ref = _get_constant_ref!(constants_list, constants_map, coeff)
        _as_ir_value(_mul_args!(ir_stmts, ref_counter, c_ref, arg))
    end
    scaled_value_table[key] = scaled
    return scaled
end

function _best_cached_submonomial(
        monomial_table::Dict{K, IRValue},
        exponent::Vector{Int},
    )::Union{Nothing, K} where {N, K <: NTuple{N, Int}}
    best = nothing
    best_has_cached_remainder = false
    best_degree = 0
    best_balance = -1
    total_degree = sum(exponent)

    for candidate in keys(monomial_table)
        degree = 0
        valid = true
        @inbounds for i in eachindex(exponent)
            c = candidate[i]
            e = exponent[i]
            if c > e
                valid = false
                break
            end
            degree += c
        end
        (!valid || degree == 0 || degree == total_degree) && continue
        remainder = _remainder_key(exponent, candidate)
        has_cached_remainder = haskey(monomial_table, remainder)
        balance = min(degree, total_degree - degree)
        if has_cached_remainder && !best_has_cached_remainder
            best = candidate
            best_has_cached_remainder = true
            best_degree = degree
            best_balance = balance
        elseif has_cached_remainder == best_has_cached_remainder
            if degree > best_degree || (degree == best_degree && balance > best_balance)
                best = candidate
                best_degree = degree
                best_balance = balance
            end
        elseif !best_has_cached_remainder && balance > best_balance
            best = candidate
            best_degree = degree
            best_balance = balance
        end
    end

    return best
end

function _split_exponent_balanced(
        exponent::Vector{Int},
    )::Tuple{Vector{Int}, Vector{Int}}
    total_degree = sum(exponent)
    total_degree > 1 || error("balanced split requires total degree > 1")

    target_degree = fld(total_degree, 2)
    left = zeros(Int, length(exponent))
    right = copy(exponent)
    remaining_target = target_degree

    @inbounds for i in eachindex(exponent)
        e = exponent[i]
        if e == 0 || remaining_target == 0
            continue
        end
        take = min(e, remaining_target)
        left[i] = take
        right[i] -= take
        remaining_target -= take
    end

    if all(iszero, left) || all(iszero, right)
        left .= 0
        right .= exponent
        taken = false
        @inbounds for i in eachindex(exponent)
            if exponent[i] > 0
                if !taken
                    left[i] = 1
                    right[i] -= 1
                    taken = true
                end
            end
        end
    end

    return left, right
end

function _build_monomial_pair!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        monomial_table::Dict{K, IRValue},
        power_table::PowerTable,
        exponent::Vector{Int},
        var_refs::Vector{Symbol},
    )::ProductTerm where {N, K <: NTuple{N, Int}}
    exponent_key = _exponent_key(exponent, K)
    if haskey(monomial_table, exponent_key)
        return ProductTerm(monomial_table[exponent_key])
    end

    if sum(exponent) <= 1 || count(!iszero, exponent) <= 1
        mono_ref = build_monomial_ref!(
            ir_stmts, ref_counter, monomial_table, power_table, exponent, var_refs,
        )
        return ProductTerm(mono_ref)
    end

    candidate = _best_cached_submonomial(monomial_table, exponent)
    if !isnothing(candidate)
        remainder_key = _remainder_key(exponent, candidate)
        if !all(iszero, remainder_key)
            candidate_ref = monomial_table[candidate]
            remainder_ref = if haskey(monomial_table, remainder_key)
                monomial_table[remainder_key]
            else
                remainder = collect(remainder_key)
                build_monomial_ref!(
                    ir_stmts, ref_counter, monomial_table, power_table, remainder, var_refs,
                )
            end
            return ProductTerm(candidate_ref, remainder_ref)
        end
    end

    left_exp, right_exp = _split_exponent_balanced(exponent)
    if all(iszero, left_exp) || all(iszero, right_exp)
        mono_ref = build_monomial_ref!(
            ir_stmts, ref_counter, monomial_table, power_table, exponent, var_refs,
        )
        return ProductTerm(mono_ref)
    end

    left_ref = build_monomial_ref!(
        ir_stmts, ref_counter, monomial_table, power_table, left_exp, var_refs,
    )
    right_ref = build_monomial_ref!(
        ir_stmts, ref_counter, monomial_table, power_table, right_exp, var_refs,
    )
    return ProductTerm(left_ref, right_ref)
end

"""
    build_power_instructions!(ir_stmts::Vector{IRStatement}, ref_counter::Base.RefValue{Int},
        base_ref::IRStatementArg, exp::Int) -> IRStatementArg

Build instructions for computing `base^exp` using an efficient decomposition:
- exp=1: identity (return base_ref)
- exp=2: OP_SQR
- exp=3: OP_CB
- exp=4: OP_SQR(OP_SQR)
- exp=5: OP_SQR * OP_CB
- exp>5: OP_POW_INT
"""
function build_power_instructions!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        power_table::PowerTable,
        base_ref::Symbol,
        exp::Int,
    )::IRValue
    exp == 0 && error("exponent 0 should not reach build_power_instructions!")
    exp == 1 && return base_ref
    key = (base_ref, exp)
    haskey(power_table, key) && return power_table[key]

    result::IRStatementRef = if exp == 2
        _emit_unary!(ir_stmts, ref_counter, OpType.OP_SQR, base_ref)
    elseif exp == 3
        _emit_unary!(ir_stmts, ref_counter, OpType.OP_CB, base_ref)
    elseif exp == 4
        sq = build_power_instructions!(ir_stmts, ref_counter, power_table, base_ref, 2)
        _emit_unary!(ir_stmts, ref_counter, OpType.OP_SQR, sq)
    elseif exp == 5
        sq = build_power_instructions!(ir_stmts, ref_counter, power_table, base_ref, 2)
        cb = build_power_instructions!(ir_stmts, ref_counter, power_table, base_ref, 3)
        _as_ir_value(_mul_args!(ir_stmts, ref_counter, sq, cb))::IRStatementRef
    else
        _emit_binary!(ir_stmts, ref_counter, OpType.OP_POW_INT, base_ref, ComplexF64(exp))
    end

    power_table[key] = result
    return result
end

## Monomial table and instruction building

const ExponentVector = Vector{Int}

"""
    build_monomial_ref!(ir_stmts, ref_counter, monomial_table, exponent, var_refs)

Get or create the IR ref for a monomial given by its exponent vector.
Uses the monomial_table for CSE — if this monomial was already built, reuse it.
"""
function build_monomial_ref!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        monomial_table::Dict{K, IRValue},
        power_table::PowerTable,
        exponent::ExponentVector,
        var_refs::Vector{Symbol},
    )::IRValue where {N, K <: NTuple{N, Int}}
    exponent_key = _exponent_key(exponent, K)
    # Check cache
    haskey(monomial_table, exponent_key) && return monomial_table[exponent_key]

    # Collect variable powers with nonzero exponent
    factors = IRValue[]
    factor_exponents = K[]
    sizehint!(factors, length(var_refs))
    sizehint!(factor_exponents, length(var_refs))
    for (i, e) in enumerate(exponent)
        e == 0 && continue
        power_ref =
            build_power_instructions!(ir_stmts, ref_counter, power_table, var_refs[i], e)
        push!(factors, power_ref)
        factor_exp = _single_factor_key(K, i, e)
        monomial_table[factor_exp] = power_ref
        push!(factor_exponents, factor_exp)
    end

    if length(factors) == 1
        result = factors[1]
        monomial_table[exponent_key] = result
        return result
    end

    candidate = _best_cached_submonomial(monomial_table, exponent)
    if !isnothing(candidate)
        candidate_ref = monomial_table[candidate]
        remainder = collect(_remainder_key(exponent, candidate))
        remainder_ref = build_monomial_ref!(
            ir_stmts, ref_counter, monomial_table, power_table, remainder, var_refs,
        )
        result = _as_ir_value(_mul_args!(ir_stmts, ref_counter, candidate_ref, remainder_ref))
        monomial_table[exponent_key] = result
        return result
    end

    # Multiply factors together
    local result::IRValue
    if isempty(factors)
        # Constant monomial (all exponents zero) — should not normally happen
        # since we handle constant terms separately, but be safe
        error("zero exponent vector should not reach build_monomial_ref!")
    elseif length(factors) == 1
        result = factors[1]
    elseif length(factors) == 2
        result = _as_ir_value(_mul_args!(ir_stmts, ref_counter, factors[1], factors[2]))
    elseif length(factors) == 3
        result = _emit_ternary!(
            ir_stmts, ref_counter, OpType.OP_MUL3, factors[1], factors[2], factors[3],
        )
    elseif length(factors) == 4
        result = _emit_quaternary!(
            ir_stmts, ref_counter, OpType.OP_MUL4, factors[1], factors[2], factors[3], factors[4],
        )
    else
        acc = factors[1]
        acc_exp = factor_exponents[1]
        for k in 2:length(factors)
            next_exp = factor_exponents[k]
            combined_key = _combined_factor_key(acc_exp, next_exp)
            if haskey(monomial_table, combined_key)
                acc = monomial_table[combined_key]
            else
                acc = _as_ir_value(_mul_args!(ir_stmts, ref_counter, acc, factors[k]))
                monomial_table[combined_key] = acc
            end
            acc_exp = combined_key
        end
        result = acc
    end

    monomial_table[exponent_key] = result
    return result
end

function _sum_product_terms!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        terms::Vector{ProductTerm},
    )::IRValue
    isempty(terms) && error("sum of product terms requires at least one term")

    singles = IRValue[]
    pairs = ProductTerm[]
    sizehint!(singles, length(terms))
    sizehint!(pairs, length(terms))

    @inbounds for term in terms
        if term.is_pair
            push!(pairs, term)
        else
            push!(singles, term.left)
        end
    end

    n = length(pairs)
    k = 1
    while k < n
        t1 = pairs[k]
        t2 = pairs[k + 1]
        push!(singles, _mulmuladd_args!(ir_stmts, ref_counter, t1.left, t1.right, t2.left, t2.right))
        k += 2
    end

    if isodd(n)
        term = pairs[end]
        if isempty(singles)
            return _as_ir_value(_mul_args!(ir_stmts, ref_counter, term.left, term.right))
        end
        c = pop!(singles)
        push!(singles, _as_ir_value(_muladd_args!(ir_stmts, ref_counter, term.left, term.right, c)))
    end

    while length(singles) > 1
        if length(singles) >= 4
            d = pop!(singles)
            c = pop!(singles)
            b = pop!(singles)
            a = pop!(singles)
            push!(singles, _emit_quaternary!(ir_stmts, ref_counter, OpType.OP_ADD4, a, b, c, d))
        elseif length(singles) == 3
            c = pop!(singles)
            b = pop!(singles)
            a = pop!(singles)
            push!(singles, _emit_ternary!(ir_stmts, ref_counter, OpType.OP_ADD3, a, b, c))
        else
            b = pop!(singles)
            a = pop!(singles)
            push!(singles, _as_ir_value(_add_args!(ir_stmts, ref_counter, a, b)))
        end
    end

    return singles[1]
end

## Polynomial summation with fusion

"""
    build_polynomial_sum!(ir_stmts, ref_counter, monomial_table, exponents, coeffs, var_refs,
        constants_list, constants_map)

Build IR instructions to evaluate a polynomial: sum of coeff_j * monomial_j.
Returns the IRStatementArg for the result.

Handles constant terms (all-zero exponent), coefficient=1, coefficient=-1 optimizations,
and fused multiply-add patterns.
"""
function build_polynomial_sum!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        monomial_table::Dict{K, IRValue},
        power_table::PowerTable,
        scaled_value_table::ScaledValueTable,
        exponents::Matrix{Int},
        coeffs::Vector{ComplexF64},
        var_refs::Vector{Symbol},
        constants_list::Vector{ComplexF64},
        constants_map::Dict{ComplexF64, ComplexF64},
    )::IRValue where {K}
    _, nterms = size(exponents)
    positives = ProductTerm[]
    negatives = ProductTerm[]
    sizehint!(positives, nterms)
    sizehint!(negatives, nterms)

    for j in 1:nterms
        c = coeffs[j]
        iszero(c) && continue

        is_negative = _is_real_negative(c)
        abs_c = is_negative ? -c : c
        exponent = _copy_exponent_column(exponents, j)
        is_constant_term = all(iszero, exponent)

        if is_constant_term
            cref = _get_constant_ref!(constants_list, constants_map, abs_c)
            push!(is_negative ? negatives : positives, ProductTerm(cref))
        else
            term = _build_monomial_pair!(
                ir_stmts, ref_counter, monomial_table, power_table, exponent, var_refs,
            )
            if abs_c == IR_ONE
                push!(is_negative ? negatives : positives, term)
            else
                scaled_left = _scale_ir_value!(
                    ir_stmts,
                    ref_counter,
                    scaled_value_table,
                    constants_list,
                    constants_map,
                    abs_c,
                    term.left,
                )
                if term.is_pair
                    push!(is_negative ? negatives : positives, ProductTerm(scaled_left, term.right))
                else
                    push!(is_negative ? negatives : positives, ProductTerm(scaled_left))
                end
            end
        end
    end

    if isempty(positives) && isempty(negatives)
        return _get_constant_ref!(constants_list, constants_map, IR_ZERO)
    end
    isempty(negatives) && return _sum_product_terms!(ir_stmts, ref_counter, positives)
    if isempty(positives)
        neg_sum = _sum_product_terms!(ir_stmts, ref_counter, negatives)
        # Always emit OP_NEG instruction — cannot use arithmetic negation on
        # ComplexF64 constants here because the negated value would not have
        # a tape slot registered in arg_index_map.
        return _emit_unary!(ir_stmts, ref_counter, OpType.OP_NEG, neg_sum)
    end

    if length(positives) == 1 && length(negatives) == 1
        pos = positives[1]
        neg = negatives[1]
        if pos.is_pair && neg.is_pair
            return _mulmulsub_args!(ir_stmts, ref_counter, pos.left, pos.right, neg.left, neg.right)
        elseif pos.is_pair
            return _as_ir_value(_mulsub_args!(ir_stmts, ref_counter, pos.left, pos.right, neg.left))
        elseif neg.is_pair
            return _as_ir_value(_submul_args!(ir_stmts, ref_counter, neg.left, neg.right, pos.left))
        else
            return _as_ir_value(_sub_args!(ir_stmts, ref_counter, pos.left, neg.left))
        end
    elseif length(negatives) == 1
        pos_sum = _sum_product_terms!(ir_stmts, ref_counter, positives)
        neg = negatives[1]
        if neg.is_pair
            return _as_ir_value(_submul_args!(ir_stmts, ref_counter, neg.left, neg.right, pos_sum))
        else
            return _as_ir_value(_sub_args!(ir_stmts, ref_counter, pos_sum, neg.left))
        end
    end

    pos_sum = _sum_product_terms!(ir_stmts, ref_counter, positives)
    neg_sum = _sum_product_terms!(ir_stmts, ref_counter, negatives)
    return _as_ir_value(_sub_args!(ir_stmts, ref_counter, pos_sum, neg_sum))
end

## Main public API

"""
    build_interpreter(polys; parameters=[], variables=...)

Build an `Interpreter{Vector{ComplexF64}}` from a vector of MP polynomials.
"""
function build_interpreter(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    )
    seq = _build_instruction_sequence(polys, variables, parameters; include_jacobian = false)
    var_syms = Symbol[Symbol(v) for v in variables]
    param_syms = Symbol[Symbol(p) for p in parameters]
    return Interpreter(Vector{ComplexF64}, seq; variables = var_syms, parameters = param_syms)
end

"""
    build_jacobian_interpreter(polys; parameters=[], variables=...)

Build an `Interpreter{Vector{ComplexF64}}` that evaluates both F and its Jacobian.
"""
function build_jacobian_interpreter(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    )
    seq = _build_instruction_sequence(polys, variables, parameters; include_jacobian = true)
    var_syms = Symbol[Symbol(v) for v in variables]
    param_syms = Symbol[Symbol(p) for p in parameters]
    return Interpreter(Vector{ComplexF64}, seq; variables = var_syms, parameters = param_syms)
end

"""
    build_taylor_interpreter(polys, ::Val{K}; parameters=[], variables=...)

Build an `Interpreter{Vector{TruncatedTaylorSeries{K+1,ComplexF64}}}`.
"""
function build_taylor_interpreter(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        ::Val{K};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    ) where {K}
    seq = _build_instruction_sequence(polys, variables, parameters; include_jacobian = false)
    var_syms = Symbol[Symbol(v) for v in variables]
    param_syms = Symbol[Symbol(p) for p in parameters]
    return Interpreter(
        Vector{TruncatedTaylorSeries{K + 1, ComplexF64}}, seq;
        variables = var_syms, parameters = param_syms,
    )
end

"""
    build_df64_interpreter(polys; parameters=[], variables=...)

Build an `Interpreter{Vector{ComplexDF64}}`.
"""
function build_df64_interpreter(
        polys::AbstractVector{<:MP.AbstractPolynomialLike};
        parameters::AbstractVector = _empty_vars(polys),
        variables::AbstractVector = _effective_variables(polys, parameters),
    )
    seq = _build_instruction_sequence(polys, variables, parameters; include_jacobian = false)
    var_syms = Symbol[Symbol(v) for v in variables]
    param_syms = Symbol[Symbol(p) for p in parameters]
    return Interpreter(Vector{ComplexDF64}, seq; variables = var_syms, parameters = param_syms)
end

## Internal pipeline

# NOTE: MP.variables is only defined on concrete polynomial types (DynamicPolynomials),
# not on the abstract MP.AbstractPolynomialLike. JET flags calls on abstract types.
# We call it on individual polynomials (which are always concrete at runtime).
# The @noinline prevents JET from analyzing through the abstract dispatch.
@inline function _variable_creation_id(v)
    if hasfield(typeof(v), :variable_order)
        variable_order = getfield(v, :variable_order)
        if hasfield(typeof(variable_order), :order)
            order = getfield(variable_order, :order)
            if hasfield(typeof(order), :id)
                return getfield(order, :id)
            end
        end
    end
    return nothing
end

@inline function _lt_variable(a, b)
    id_a = _variable_creation_id(a)
    id_b = _variable_creation_id(b)
    if !(isnothing(id_a) || isnothing(id_b))
        return id_a < id_b
    end
    return string(Symbol(a)) < string(Symbol(b))
end

@noinline function _collect_variables(polys)
    all_vars = empty(MP.variables(first(polys)))
    for poly in polys
        append!(all_vars, MP.variables(poly))
    end
    unique!(all_vars)
    sort!(all_vars; lt = _lt_variable)
    return all_vars
end

# Return an empty vector with the correct variable element type
@noinline function _empty_vars(polys)
    return empty(_collect_variables(polys))
end

@noinline function _effective_variables(polys, parameters)
    all_vars = _collect_variables(polys)
    if isempty(parameters)
        return collect(all_vars)
    end
    param_set = Set(parameters)
    return [v for v in all_vars if v ∉ param_set]
end

"""
    _build_instruction_sequence(polys, variables, parameters; include_jacobian)

Core pipeline: extract supports, build monomial table, build IR, compile.
"""
function _build_instruction_sequence(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
        parameters::AbstractVector;
        include_jacobian::Bool,
    )::InstructionSequence
    monomial_table = _create_monomial_table(length(variables) + length(parameters))
    power_table = _create_power_table()
    return _build_instruction_sequence_impl(
        polys,
        variables,
        parameters,
        include_jacobian,
        monomial_table,
        power_table,
    )
end

function _build_instruction_sequence_impl(
        polys::AbstractVector{<:MP.AbstractPolynomialLike},
        variables::AbstractVector,
        parameters::AbstractVector,
        include_jacobian::Bool,
        monomial_table::Dict{K, IRValue},
        power_table::PowerTable,
    )::InstructionSequence where {K}
    nvars = length(variables)
    nparams = length(parameters)
    npolys = length(polys)

    # Variable and parameter symbols
    var_syms = Symbol[Symbol(v) for v in variables]
    param_syms = Symbol[Symbol(p) for p in parameters]
    all_sym_vars = vcat(var_syms, param_syms)

    # Extract supports from F
    all_mp_vars = vcat(variables, parameters)
    f_supports = [extract_support(p, all_mp_vars) for p in polys]

    # Extract supports from Jacobian if needed
    # Column-major order: iterate variables (columns) first, then polynomials (rows)
    # so that J[k] maps to linear index k in column-major U matrix
    jac_supports = Tuple{Matrix{Int}, Vector{ComplexF64}}[]
    if include_jacobian
        for v in variables
            for p in polys
                dp = MP.differentiate(p, v)
                push!(jac_supports, extract_support(dp, all_mp_vars))
            end
        end
    end

    # Build IR
    ir_stmts = IRStatement[]
    ref_counter = Ref(0)
    constants_list = ComplexF64[]
    constants_map = Dict{ComplexF64, ComplexF64}()
    scaled_value_table = _create_scaled_value_table()

    # Variable refs are symbols
    all_var_refs = Symbol[s for s in all_sym_vars]

    # Build IR for F polynomials
    f_result_refs = IRStatementArg[]
    for (exps, coeffs) in f_supports
        ref = build_polynomial_sum!(
            ir_stmts, ref_counter, monomial_table, power_table, scaled_value_table, exps, coeffs,
            all_var_refs, constants_list, constants_map,
        )
        push!(f_result_refs, ref)
    end

    # Build IR for Jacobian polynomials
    jac_result_refs = IRStatementArg[]
    if include_jacobian
        for (exps, coeffs) in jac_supports
            ref = build_polynomial_sum!(
                ir_stmts, ref_counter, monomial_table, power_table, scaled_value_table, exps, coeffs,
                all_var_refs, constants_list, constants_map,
            )
            push!(jac_result_refs, ref)
        end
    end

    # Ensure all result refs are IRStatementRefs (not raw constants/symbols).
    # The IR assignment system requires IRStatementRef targets.
    assigned_stmt_refs = Dict{IRStatementRef, Int}()
    function ensure_stmt_ref(ref::IRStatementArg)::IRStatementRef
        if ref isa IRStatementRef
            seen = get(assigned_stmt_refs, ref, 0)
            if seen == 0
                assigned_stmt_refs[ref] = 1
                return ref
            end
            assigned_stmt_refs[ref] = seen + 1
            ref_counter[] += 1
            r = IRStatementRef(ref_counter[])
            push!(ir_stmts, IRStatement(OpType.OP_IDENTITY, r, ref))
            return r
        end
        # Wrap constant or symbol in an OP_IDENTITY instruction
        ref_counter[] += 1
        r = IRStatementRef(ref_counter[])
        push!(ir_stmts, IRStatement(OpType.OP_IDENTITY, r, ref))
        return r
    end

    # Build assignments: F outputs are indices 1..npolys,
    # Jacobian outputs are npolys+1..npolys+npolys*nvars
    assignments = Tuple{Int, IRStatementArg}[]
    for (i, ref) in enumerate(f_result_refs)
        push!(assignments, (i, ensure_stmt_ref(ref)))
    end
    for (k, ref) in enumerate(jac_result_refs)
        push!(assignments, (npolys + k, ensure_stmt_ref(ref)))
    end

    ir = IntermediateRepresentation(ir_stmts, assignments, npolys)

    # Compile IR to InstructionSequence
    return build_instruction_sequence_from_ir(
        ir;
        nvars = nvars,
        nparams = nparams,
        nconstants = length(constants_list),
        constants = constants_list,
        variables = var_syms,
        parameters = param_syms,
    )
end
