## Code generation helpers

"""
    nested_ifs(cond_body, elsebranch=nothing)

Build a nested if-elseif-else expression from `[(condition, body), ...]` pairs.
Used at code-generation time to build dispatch chains.
"""
function nested_ifs(cond_body::Vector, elsebranch = nothing)
    orig_expr = Expr(:if, cond_body[1][1], cond_body[1][2])
    expr = orig_expr
    for i in 2:length(cond_body)
        push!(expr.args, Expr(:elseif, cond_body[i][1], cond_body[i][2]))
        expr = expr.args[end]
    end
    if !isnothing(elsebranch)
        push!(expr.args, elsebranch)
    end
    return orig_expr
end

@enumx OpType::Int8 begin
    # Arity 0
    OP_STOP

    # Arity 1
    OP_CB # a ^ 3
    OP_COS # cos(a)
    OP_INV # 1 / a
    OP_INV_NOT_ZERO # a ≠ 0 ? 1 / a : a
    OP_INVSQR # 1 / a^2
    OP_NEG # -a
    OP_SIN # sin(a)
    OP_SQR # a ^ 2
    OP_SQRT # √(a)
    OP_IDENTITY # a

    # Arity 2
    OP_ADD # a + b
    OP_DIV # a / b
    OP_MUL # a * b
    OP_SUB # a - b
    OP_POW_INT # a ^ p where p isa Integer

    # Arity 3
    OP_ADD3 # a + b + c
    OP_MUL3 # a * b * c
    OP_MULADD # a * b + c
    OP_MULSUB # a * b - c
    OP_SUBMUL # c - a * b

    # Arity 4
    OP_ADD4 # a + b + c + d
    OP_MUL4 # a * b * c * d
    OP_MULMULADD # a * b + c * d
    OP_MULMULSUB # a * b - c * d

    # Out of arity order: `execute_instructions!` emits its switch in declaration
    # order, and a rare op ahead of `OP_ADD`/`OP_MUL` lengthens every hot tape walk.
    # Arity 1
    OP_ACOS # cos⁻¹(a)
    OP_ASIN # sin⁻¹(a)
    OP_COSH # cosh(a)
    OP_EXP # e^a
    OP_LOG # log(a), principal branch
    OP_SINH # sinh(a)
    OP_TAN # tan(a)
    OP_TANH # tanh(a)

    # Arity 2
    OP_POW # a ^ b where b is a non-integer number
end

const _OP_ARITY = (
    0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2,
    3, 3, 3, 3, 3,
    4, 4, 4, 4,
    1, 1, 1, 1, 1, 1, 1, 1,
    2,
)

const _OP_CALL = (
    :op_stop,
    :op_cb, :op_cos, :op_inv, :op_inv_not_zero, :op_invsqr, :op_neg, :op_sin, :op_sqr,
    :op_sqrt, :op_identity,
    :op_add, :op_div, :op_mul, :op_sub, :op_pow_int,
    :op_add3, :op_mul3, :op_muladd, :op_mulsub, :op_submul,
    :op_add4, :op_mul4, :op_mulmuladd, :op_mulmulsub,
    :op_acos, :op_asin, :op_cosh, :op_exp, :op_log, :op_sinh, :op_tan, :op_tanh,
    :op_pow,
)

# `OP_POW` takes its exponent from the tape rather than the instruction, since an
# `Instruction` input is an `Int32` slot and a real exponent does not fit one.
const _OP_IMMEDIATE_INPUT = (
    0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 2,
    0, 0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0,
)

@inline _op_index(op::OpType.T) = Int(op) + 1

@inline arity(op::OpType.T)::Int = @inbounds _OP_ARITY[_op_index(op)]
@inline op_call(op::OpType.T)::Symbol = @inbounds _OP_CALL[_op_index(op)]

@inline function should_use_index_not_reference(op::OpType.T, index::Int)::Bool
    return @inbounds _OP_IMMEDIATE_INPUT[_op_index(op)] == index
end

# arity 0
@inline op_stop() = nothing

# arity 1

# Generic fallback: 2 complex multiplications (8 real muls)
@inline op_cb(x) = x * x * x
# Specialized for Complex: use op_sqr (2 real muls via Karatsuba) + 1 complex mul (4 real muls) = 6 real muls total
@inline function op_cb(z::Complex)
    x, y = reim(z)
    a = (x + y) * (x - y)  # real part of z²
    b = (x + x) * y         # imag part of z²
    return Complex(a * x - b * y, a * y + b * x)
end

@inline op_cos(x) = cos(x)
@inline op_identity(x) = identity(x)
@inline op_inv(x) = inv(x)
@inline op_inv(x::Complex) = Base.FastMath.inv_fast(x)

"""
    op_inv_not_zero(x)

Invert x unless it is 0, then return 0.
"""
@inline op_inv_not_zero(x) = iszero(x) ? x : op_inv(x)

@inline op_invsqr(x) = op_sqr(op_inv(x))
@inline op_neg(x) = -x
@inline op_sin(x) = sin(x)

# Generic fallback
@inline op_sqr(x) = x * x
# Specialized for Complex: Karatsuba avoids one real mul (2 real muls instead of 4)
@inline function op_sqr(z::Complex)
    x, y = reim(z)
    return Complex((x + y) * (x - y), (x + x) * y)
end

@inline op_sqrt(x) = sqrt(x)

# arity 2
@inline op_add(a, b) = a + b
@inline op_div(a, b) = Base.FastMath.div_fast(a, b)
@inline op_mul(a, b) = a * b
@inline op_sub(a, b) = a - b

@inline op_pow_int(x, p::Integer) =
    p > 0 ? Base.power_by_squaring(x, p) : op_inv(Base.power_by_squaring(x, -p))

# arity 3
@inline op_add3(x, y, z) = x + y + z
@inline op_mul3(x, y, z) = x * y * z
# Generic: plain multiply-add
@inline op_muladd(x, y, z) = x * y + z
# For real floats, use muladd which maps to a hardware FMA instruction when available
@inline op_muladd(x::T, y::T, z::T) where {T <: AbstractFloat} = muladd(x, y, z)
@inline op_mulsub(x, y, z) = x * y - z
@inline op_submul(x, y, z) = z - x * y

# arity 4
@inline op_add4(a, b, c, d) = a + b + c + d
@inline op_mul4(a, b, c, d) = a * b * c * d
@inline op_mulmuladd(a, b, c, d) = a * b + c * d
@inline op_mulmulsub(a, b, c, d) = a * b - c * d

# transcendental
@inline op_acos(x) = acos(x)
@inline op_asin(x) = asin(x)
@inline op_cosh(x) = cosh(x)
@inline op_exp(x) = exp(x)
@inline op_log(x) = log(x)
@inline op_sinh(x) = sinh(x)
@inline op_tan(x) = tan(x)
@inline op_tanh(x) = tanh(x)
@inline op_pow(x, r) = x^r
