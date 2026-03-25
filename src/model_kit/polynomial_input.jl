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
        base_ref::IRStatementArg,
        exp::Int,
    )::IRStatementArg
    exp == 0 && error("exponent 0 should not reach build_power_instructions!")
    exp == 1 && return base_ref

    function new_ref()::IRStatementRef
        ref_counter[] += 1
        return IRStatementRef(ref_counter[])
    end

    if exp == 2
        r = new_ref()
        push!(ir_stmts, IRStatement(OpType.OP_SQR, r, base_ref))
        return r
    elseif exp == 3
        r = new_ref()
        push!(ir_stmts, IRStatement(OpType.OP_CB, r, base_ref))
        return r
    elseif exp == 4
        # sqr(sqr(x))
        r1 = new_ref()
        push!(ir_stmts, IRStatement(OpType.OP_SQR, r1, base_ref))
        r2 = new_ref()
        push!(ir_stmts, IRStatement(OpType.OP_SQR, r2, r1))
        return r2
    elseif exp == 5
        # sqr(x) * cb(x)
        r_sq = new_ref()
        push!(ir_stmts, IRStatement(OpType.OP_SQR, r_sq, base_ref))
        r_cb = new_ref()
        push!(ir_stmts, IRStatement(OpType.OP_CB, r_cb, base_ref))
        r = new_ref()
        push!(ir_stmts, IRStatement(OpType.OP_MUL, r, r_sq, r_cb))
        return r
    else
        # General case: use OP_POW_INT
        r = new_ref()
        push!(ir_stmts, IRStatement(OpType.OP_POW_INT, r, base_ref, ComplexF64(exp)))
        return r
    end
end

## Monomial table and instruction building

const ExponentVector = Vector{Int}
const MonomialTable = Dict{ExponentVector, IRStatementArg}

"""
    build_monomial_ref!(ir_stmts, ref_counter, monomial_table, exponent, var_refs)

Get or create the IR ref for a monomial given by its exponent vector.
Uses the monomial_table for CSE — if this monomial was already built, reuse it.
"""
function build_monomial_ref!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        monomial_table::MonomialTable,
        exponent::ExponentVector,
        var_refs::Vector{Symbol},
    )::IRStatementArg
    # Check cache
    haskey(monomial_table, exponent) && return monomial_table[exponent]

    function new_ref()::IRStatementRef
        ref_counter[] += 1
        return IRStatementRef(ref_counter[])
    end

    # Collect variable powers with nonzero exponent
    factors = IRStatementArg[]
    for (i, e) in enumerate(exponent)
        e == 0 && continue
        power_ref = build_power_instructions!(ir_stmts, ref_counter, var_refs[i], e)
        push!(factors, power_ref)
    end

    # Multiply factors together
    local result::IRStatementArg
    if isempty(factors)
        # Constant monomial (all exponents zero) — should not normally happen
        # since we handle constant terms separately, but be safe
        error("zero exponent vector should not reach build_monomial_ref!")
    elseif length(factors) == 1
        result = factors[1]
    elseif length(factors) == 2
        r = new_ref()
        push!(ir_stmts, IRStatement(OpType.OP_MUL, r, factors[1], factors[2]))
        result = r
    elseif length(factors) == 3
        r = new_ref()
        push!(ir_stmts, IRStatement(OpType.OP_MUL3, r, factors[1], factors[2], factors[3]))
        result = r
    elseif length(factors) == 4
        r = new_ref()
        push!(
            ir_stmts,
            IRStatement(OpType.OP_MUL4, r, factors[1], factors[2], factors[3], factors[4]),
        )
        result = r
    else
        # Chain OP_MUL for more than 4 factors
        acc = factors[1]
        for k in 2:length(factors)
            r = new_ref()
            push!(ir_stmts, IRStatement(OpType.OP_MUL, r, acc, factors[k]))
            acc = r
        end
        result = acc
    end

    monomial_table[exponent] = result
    return result
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
        monomial_table::MonomialTable,
        exponents::Matrix{Int},
        coeffs::Vector{ComplexF64},
        var_refs::Vector{Symbol},
        constants_list::Vector{ComplexF64},
        constants_map::Dict{ComplexF64, IRStatementArg},
    )::IRStatementArg
    nvars, nterms = size(exponents)

    function new_ref()::IRStatementRef
        ref_counter[] += 1
        return IRStatementRef(ref_counter[])
    end

    function get_constant_ref(c::ComplexF64)::IRStatementArg
        if haskey(constants_map, c)
            return constants_map[c]
        end
        push!(constants_list, c)
        constants_map[c] = c
        return c
    end

    # Build term refs: coeff * monomial
    term_refs = IRStatementArg[]
    for j in 1:nterms
        exp_vec = exponents[:, j]
        c = coeffs[j]
        iszero(c) && continue

        is_constant_term = all(iszero, exp_vec)

        if is_constant_term
            # Pure constant term
            push!(term_refs, get_constant_ref(c))
        else
            mono_ref = build_monomial_ref!(ir_stmts, ref_counter, monomial_table, exp_vec, var_refs)
            if c == ComplexF64(1.0)
                push!(term_refs, mono_ref)
            elseif c == ComplexF64(-1.0)
                r = new_ref()
                push!(ir_stmts, IRStatement(OpType.OP_NEG, r, mono_ref))
                push!(term_refs, r)
            else
                c_ref = get_constant_ref(c)
                r = new_ref()
                push!(ir_stmts, IRStatement(OpType.OP_MUL, r, c_ref, mono_ref))
                push!(term_refs, r)
            end
        end
    end

    # Sum all terms with fusion
    length(term_refs) == 0 && error("polynomial has no nonzero terms")
    length(term_refs) == 1 && return term_refs[1]

    return _fused_sum!(ir_stmts, ref_counter, term_refs)
end

"""
    _fused_sum!(ir_stmts, ref_counter, terms) -> IRStatementArg

Sum a list of term refs using fused operations (ADD, ADD3, ADD4).
"""
function _fused_sum!(
        ir_stmts::Vector{IRStatement},
        ref_counter::Base.RefValue{Int},
        terms::Vector{IRStatementArg},
    )::IRStatementArg
    function new_ref()::IRStatementRef
        ref_counter[] += 1
        return IRStatementRef(ref_counter[])
    end

    n = length(terms)
    n == 1 && return terms[1]

    if n == 2
        r = new_ref()
        push!(ir_stmts, IRStatement(OpType.OP_ADD, r, terms[1], terms[2]))
        return r
    elseif n == 3
        r = new_ref()
        push!(ir_stmts, IRStatement(OpType.OP_ADD3, r, terms[1], terms[2], terms[3]))
        return r
    elseif n == 4
        r = new_ref()
        push!(
            ir_stmts,
            IRStatement(OpType.OP_ADD4, r, terms[1], terms[2], terms[3], terms[4]),
        )
        return r
    else
        # Chain: first 4 via ADD4, then ADD each remaining
        r = new_ref()
        push!(
            ir_stmts,
            IRStatement(OpType.OP_ADD4, r, terms[1], terms[2], terms[3], terms[4]),
        )
        acc = r
        for k in 5:n
            r2 = new_ref()
            push!(ir_stmts, IRStatement(OpType.OP_ADD, r2, acc, terms[k]))
            acc = r2
        end
        return acc
    end
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
@noinline function _collect_variables(polys)
    all_vars = MP.variables(first(polys))
    for i in 2:length(polys)
        for v in MP.variables(polys[i])
            if v ∉ all_vars
                all_vars = vcat(all_vars, [v])
            end
        end
    end
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
    nvars = length(variables)
    nparams = length(parameters)
    npolys = length(polys)

    # Variable and parameter symbols
    var_syms = Symbol[Symbol(v) for v in variables]
    param_syms = Symbol[Symbol(p) for p in parameters]
    all_sym_vars = vcat(var_syms, param_syms)

    # Extract supports from F
    f_supports = [extract_support(p, vcat(variables, parameters)) for p in polys]

    # Extract supports from Jacobian if needed
    # Column-major order: iterate variables (columns) first, then polynomials (rows)
    # so that J[k] maps to linear index k in column-major U matrix
    jac_supports = Tuple{Matrix{Int}, Vector{ComplexF64}}[]
    if include_jacobian
        for v in variables
            for p in polys
                dp = MP.differentiate(p, v)
                push!(jac_supports, extract_support(dp, vcat(variables, parameters)))
            end
        end
    end

    # Build IR
    ir_stmts = IRStatement[]
    ref_counter = Ref(0)
    monomial_table = MonomialTable()
    constants_list = ComplexF64[]
    constants_map = Dict{ComplexF64, IRStatementArg}()

    # Variable refs are symbols
    all_var_refs = Symbol[s for s in all_sym_vars]

    # Build IR for F polynomials
    f_result_refs = IRStatementArg[]
    for (exps, coeffs) in f_supports
        ref = build_polynomial_sum!(
            ir_stmts, ref_counter, monomial_table, exps, coeffs,
            all_var_refs, constants_list, constants_map,
        )
        push!(f_result_refs, ref)
    end

    # Build IR for Jacobian polynomials
    jac_result_refs = IRStatementArg[]
    if include_jacobian
        for (exps, coeffs) in jac_supports
            ref = build_polynomial_sum!(
                ir_stmts, ref_counter, monomial_table, exps, coeffs,
                all_var_refs, constants_list, constants_map,
            )
            push!(jac_result_refs, ref)
        end
    end

    # Ensure all result refs are IRStatementRefs (not raw constants/symbols).
    # The IR assignment system requires IRStatementRef targets.
    function ensure_stmt_ref(ref::IRStatementArg)::IRStatementRef
        if ref isa IRStatementRef
            return ref
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
