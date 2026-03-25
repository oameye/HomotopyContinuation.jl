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
end

function arity(op::OpType.T)::Int
    op_val = op
    return if op_val === OpType.OP_STOP
        0
    elseif op_val == OpType.OP_CB ||
            op_val == OpType.OP_COS ||
            op_val == OpType.OP_IDENTITY ||
            op_val == OpType.OP_INV ||
            op_val == OpType.OP_INV_NOT_ZERO ||
            op_val == OpType.OP_INVSQR ||
            op_val == OpType.OP_NEG ||
            op_val == OpType.OP_SIN ||
            op_val == OpType.OP_SQR ||
            op_val == OpType.OP_SQRT
        1
    elseif op_val == OpType.OP_ADD ||
            op_val == OpType.OP_DIV ||
            op_val == OpType.OP_MUL ||
            op_val == OpType.OP_SUB ||
            op_val == OpType.OP_POW_INT
        2
    elseif op_val == OpType.OP_ADD3 ||
            op_val == OpType.OP_MUL3 ||
            op_val == OpType.OP_MULADD ||
            op_val == OpType.OP_MULSUB ||
            op_val == OpType.OP_SUBMUL
        3
    elseif op_val == OpType.OP_ADD4 ||
            op_val == OpType.OP_MUL4 ||
            op_val == OpType.OP_MULMULADD ||
            op_val == OpType.OP_MULMULSUB
        4
    else
        error("Unexpected OpType $(op_val)")
    end
end

function op_call(op::OpType.T)::Symbol
    op_val = op
    return if op_val == OpType.OP_STOP
        :op_stop

        # Arity 1
    elseif op_val == OpType.OP_CB
        :op_cb
    elseif op_val == OpType.OP_COS
        :op_cos
    elseif op_val == OpType.OP_IDENTITY
        :op_identity
    elseif op_val == OpType.OP_INV
        :op_inv
    elseif op_val == OpType.OP_INV_NOT_ZERO
        :op_inv_not_zero
    elseif op_val == OpType.OP_INVSQR
        :op_invsqr
    elseif op_val == OpType.OP_NEG
        :op_neg
    elseif op_val == OpType.OP_SIN
        :op_sin
    elseif op_val == OpType.OP_SQR
        :op_sqr
    elseif op_val == OpType.OP_SQRT
        :op_sqrt

        # Arity 2
    elseif op_val == OpType.OP_ADD
        :op_add
    elseif op_val == OpType.OP_DIV
        :op_div
    elseif op_val == OpType.OP_MUL
        :op_mul
    elseif op_val == OpType.OP_SUB
        :op_sub
    elseif op_val == OpType.OP_POW_INT
        :op_pow_int

        # Arity 3
    elseif op_val == OpType.OP_ADD3
        :op_add3
    elseif op_val == OpType.OP_MUL3
        :op_mul3
    elseif op_val == OpType.OP_MULADD
        :op_muladd
    elseif op_val == OpType.OP_MULSUB
        :op_mulsub
    elseif op_val == OpType.OP_SUBMUL
        :op_submul

        # Arity 4
    elseif op_val == OpType.OP_ADD4
        :op_add4
    elseif op_val == OpType.OP_MUL4
        :op_mul4
    elseif op_val == OpType.OP_MULMULADD
        :op_mulmuladd
    elseif op_val == OpType.OP_MULMULSUB
        :op_mulmulsub
    else
        error("Unexpected OpType $(op_val)")
    end
end

function should_use_index_not_reference(op::OpType.T, index::Int)::Bool
    return op == OpType.OP_POW_INT && index == 2
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
@inline op_inv_not_zero(x) = ifelse(iszero(x), x, op_inv(x))

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
