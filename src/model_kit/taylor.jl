## TruncatedTaylorSeries, TaylorVector, and taylor_op_* functions
#
# TruncatedTaylorSeries{N,T} stores N Taylor coefficients (orders 0 to N-1).
# Indexed 0-based externally: t[k] returns the k-th order coefficient.
#
# TaylorVector{N,T} is a vector of TruncatedTaylorSeries backed by an FSMat{T}.
# The matrix is N×n: N rows for coefficient orders, n columns for elements.

struct TruncatedTaylorSeries{N, T}
    val::NTuple{N, T}
end

# Construct from tuple (promotes element types)
TruncatedTaylorSeries(v::Tuple) = TruncatedTaylorSeries(promote(v...))

# Construct order-0 series from a scalar (N=1 inferred from value)
TruncatedTaylorSeries(x::Number) = TruncatedTaylorSeries((x,))

# Construct fixed-N series from a scalar: TruncatedTaylorSeries{N,T}(x)
# places x at order 0 and fills remaining orders with zero(T).
function TruncatedTaylorSeries{N, T}(x::S) where {N, T, S <: Number}
    return convert(TruncatedTaylorSeries{N, T}, x)
end

Base.show(io::IO, x::TruncatedTaylorSeries) = show(io, x.val)
Base.eltype(::TruncatedTaylorSeries{N, T}) where {N, T} = T
Base.length(x::TruncatedTaylorSeries{N, T}) where {N, T} = N

# 0-indexed externally
Base.getindex(x::TruncatedTaylorSeries{N, T}, k::Integer) where {N, T} = x.val[k + 1]

Base.iterate(x::TruncatedTaylorSeries) = iterate(x.val)
Base.iterate(x::TruncatedTaylorSeries, s) = iterate(x.val, s)

Base.:(==)(tx::TruncatedTaylorSeries, ty::TruncatedTaylorSeries) = tx.val == ty.val

Base.zero(::TruncatedTaylorSeries{N, T}) where {N, T} =
    convert(TruncatedTaylorSeries{N, T}, zero(T))
Base.zero(::Type{TruncatedTaylorSeries{N, T}}) where {N, T} =
    convert(TruncatedTaylorSeries{N, T}, zero(T))
Base.one(::TruncatedTaylorSeries{N, T}) where {N, T} =
    convert(TruncatedTaylorSeries{N, T}, one(T))
Base.one(::Type{TruncatedTaylorSeries{N, T}}) where {N, T} =
    convert(TruncatedTaylorSeries{N, T}, one(T))

# Convert from same-or-smaller TTS (zero-padding) or scalar
function Base.convert(
        ::Type{TruncatedTaylorSeries{N, T}},
        x::TruncatedTaylorSeries{K, S},
    ) where {N, T, K, S}
    return TruncatedTaylorSeries(
        ntuple(Val(N)) do i
            if i <= K
                convert(T, x.val[i])
            else
                zero(T)
            end
        end
    )
end

function Base.convert(
        ::Type{TruncatedTaylorSeries{N, T}},
        x::S,
    ) where {N, T, S <: Number}
    return TruncatedTaylorSeries(
        ntuple(Val(N)) do i
            if i == 1
                convert(T, x)
            else
                zero(T)
            end
        end
    )
end

function Base.convert(
        ::Type{TruncatedTaylorSeries{N, T}},
        x::Tuple,
    ) where {N, T}
    return convert(TruncatedTaylorSeries{N, T}, TruncatedTaylorSeries(x))
end

# Arithmetic with scalars
Base.:*(c::Number, x::TruncatedTaylorSeries{N, T}) where {N, T} =
    TruncatedTaylorSeries(ntuple(i -> c * x.val[i], Val(N)))
Base.:*(x::TruncatedTaylorSeries{N, T}, c::Number) where {N, T} =
    TruncatedTaylorSeries(ntuple(i -> x.val[i] * c, Val(N)))

Base.:+(x::TruncatedTaylorSeries{N, T}, y::TruncatedTaylorSeries{N, T}) where {N, T} =
    TruncatedTaylorSeries(ntuple(i -> x.val[i] + y.val[i], Val(N)))
Base.:-(x::TruncatedTaylorSeries{N, T}, y::TruncatedTaylorSeries{N, T}) where {N, T} =
    TruncatedTaylorSeries(ntuple(i -> x.val[i] - y.val[i], Val(N)))
Base.:-(x::TruncatedTaylorSeries{N, T}) where {N, T} =
    TruncatedTaylorSeries(ntuple(i -> -x.val[i], Val(N)))

## TaylorVector

"""
    TaylorVector{N,T} <: AbstractVector{TruncatedTaylorSeries{N,T}}

A vector of `TruncatedTaylorSeries{N,T}` backed by an `FSMat{T}`.

The backing matrix has shape `N × n`: `N` rows for Taylor coefficient orders
(0 to N-1) and `n` columns for vector elements.
"""
struct TaylorVector{N, T} <: AbstractVector{TruncatedTaylorSeries{N, T}}
    data::FSMat{T}
end

"""
    TaylorVector{N,T}(n::Integer)

Allocate a `TaylorVector` of length `n` with element type `T` and `N` Taylor orders.
All coefficients are initialised to zero.
"""
function TaylorVector{N, T}(n::Integer) where {N, T}
    return TaylorVector{N, T}(FSMat{T}(zeros(T, N, n)))
end

Base.length(tv::TaylorVector) = size(tv.data, 2)
Base.size(tv::TaylorVector) = (length(tv),)
Base.eltype(::Type{TaylorVector{N, T}}) where {N, T} = TruncatedTaylorSeries{N, T}
Base.IndexStyle(::Type{<:TaylorVector}) = IndexLinear()

Base.fill!(tv::TaylorVector, x) = (fill!(tv.data, x); tv)

"""
    vectors(tv::TaylorVector{N}) -> NTuple{N, AbstractVector}

Split `tv` into `N` views — one per Taylor coefficient order.
`vectors(tv)[k]` is the view of all order-`(k-1)` coefficients across elements.
"""
@generated function vectors(tv::TaylorVector{N}) where {N}
    return Expr(:tuple, (:(view(tv.data, $k, :)) for k in 1:N)...)
end

@generated function Base.getindex(tv::TaylorVector{N, T}, i::Integer) where {N, T}
    return quote
        Base.@_propagate_inbounds_meta
        x = tv.data
        TruncatedTaylorSeries($(Expr(:tuple, (:(x[$k, i]) for k in 1:N)...)))
    end
end

function Base.setindex!(tv::TaylorVector{N, T}, x, i::Integer) where {N, T}
    return setindex!(tv, convert(TruncatedTaylorSeries{N, T}, x), i)
end

@generated function Base.setindex!(
        tv::TaylorVector{N, T},
        x::TruncatedTaylorSeries{N, T},
        i::Integer,
    ) where {N, T}
    return quote
        Base.@_propagate_inbounds_meta
        d = tv.data
        $((:(d[$k, i] = x.val[$k]) for k in 1:N)...)
        x
    end
end

## taylor_op_* functions
#
# Each function computes the Taylor expansion of the corresponding scalar op_*.
# Input/output: TruncatedTaylorSeries{N,T} with N coefficients (orders 0 to N-1).
# Coefficient-wise ops operate independently per order.
# Convolution-based ops (mul, div, inv, sqrt, pow_int) use recurrence relations.

# OP_IDENTITY # a
@inline function taylor_op_identity(
        a::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    return a
end

# OP_NEG # -a
@generated function taylor_op_neg(a::TruncatedTaylorSeries{N, T}) where {N, T}
    terms = Expr[:(-(a.val[$k])) for k in 1:N]
    return quote
        Base.@_inline_meta
        TruncatedTaylorSeries($(Expr(:tuple, terms...)))
    end
end

# OP_ADD # a + b
@generated function taylor_op_add(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    terms = Expr[:(a.val[$k] + b.val[$k]) for k in 1:N]
    return quote
        Base.@_inline_meta
        TruncatedTaylorSeries($(Expr(:tuple, terms...)))
    end
end

# OP_SUB # a - b
@generated function taylor_op_sub(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    terms = Expr[:(a.val[$k] - b.val[$k]) for k in 1:N]
    return quote
        Base.@_inline_meta
        TruncatedTaylorSeries($(Expr(:tuple, terms...)))
    end
end

# OP_MUL # a * b  (Cauchy product)
# c[k] = Σ_{j=0}^{k} a[j] * b[k-j]  (0-indexed), i.e., val[k+1] = Σ_{j=1}^{k+1} a.val[j]*b.val[k+2-j]
@generated function taylor_op_mul(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    exprs = Vector{Expr}(undef, N)
    for k in 1:N
        # k-th 1-indexed coefficient = Σ_{j=1}^{k} a.val[j] * b.val[k+1-j]
        terms = Expr[]
        for j in 1:k
            push!(terms, :(a.val[$j] * b.val[$(k + 1 - j)]))
        end
        acc = terms[1]
        for i in 2:length(terms)
            acc = :($(acc) + $(terms[i]))
        end
        exprs[k] = acc
    end
    return quote
        Base.@_inline_meta
        TruncatedTaylorSeries($(Expr(:tuple, exprs...)))
    end
end

# Helper: fold an Expr vector by addition (no closures)
function _expr_sum(terms::Vector{Expr})::Expr
    acc = terms[1]
    for i in 2:length(terms)
        acc = Expr(:call, :+, acc, terms[i])
    end
    return acc
end

# OP_DIV # a / b  (quotient rule recurrence)
# c[0] = a[0] / b[0]
# c[k] = (a[k] - Σ_{j=0}^{k-1} c[j] * b[k-j]) / b[0]
@generated function taylor_op_div(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    # Build sequential computation: c1, c2, ..., cN
    stmts = Expr[]
    push!(stmts, :(c1 = a.val[1] / b.val[1]))
    for k in 2:N
        # Σ_{j=1}^{k-1} c[j] * b[k+1-j]  (1-indexed: c[j] = c_{j-1} in 0-indexed)
        sum_terms = Expr[]
        for j in 1:(k - 1)
            push!(sum_terms, :($(Symbol(:c, j)) * b.val[$(k + 1 - j)]))
        end
        sum_expr = _expr_sum(sum_terms)
        push!(stmts, :($(Symbol(:c, k)) = (a.val[$k] - $sum_expr) / b.val[1]))
    end
    cvars = Symbol[Symbol(:c, k) for k in 1:N]
    return quote
        Base.@_inline_meta
        $(stmts...)
        TruncatedTaylorSeries($(Expr(:tuple, cvars...)))
    end
end

# OP_INV # 1 / a  (special case of div with numerator = 1)
# c[0] = 1 / a[0]
# c[k] = -(Σ_{j=0}^{k-1} c[j] * a[k-j]) / a[0]  for k >= 1
@generated function taylor_op_inv(a::TruncatedTaylorSeries{N, T}) where {N, T}
    stmts = Expr[]
    push!(stmts, :(c1 = inv(a.val[1])))
    for k in 2:N
        # Σ_{j=1}^{k-1} c[j] * a[k+1-j]
        sum_terms = Expr[]
        for j in 1:(k - 1)
            push!(sum_terms, :($(Symbol(:c, j)) * a.val[$(k + 1 - j)]))
        end
        sum_expr = _expr_sum(sum_terms)
        push!(stmts, :($(Symbol(:c, k)) = -($sum_expr * c1)))
    end
    cvars = Symbol[Symbol(:c, k) for k in 1:N]
    return quote
        Base.@_inline_meta
        $(stmts...)
        TruncatedTaylorSeries($(Expr(:tuple, cvars...)))
    end
end

# OP_SQR # a^2  (optimized Cauchy product exploiting symmetry)
# c[k] = Σ_{j=0}^{k} a[j]*a[k-j]
#       = 2 * Σ_{j=0}^{floor((k-1)/2)} a[j]*a[k-j]  + (iseven(k) ? a[k/2]^2 : 0)
@generated function taylor_op_sqr(a::TruncatedTaylorSeries{N, T}) where {N, T}
    exprs = Vector{Expr}(undef, N)
    for k in 1:N
        # k is 1-indexed, so 0-indexed order is ord = k-1
        ord = k - 1
        if ord == 0
            # c[0] = a[0]^2 = a.val[1]^2
            exprs[k] = :(a.val[1] * a.val[1])
        elseif isodd(ord)
            half = div(ord, 2)
            if half == 0
                # ord == 1: c[1] = 2 * a[0]*a[1] = 2 * a.val[1] * a.val[2]
                exprs[k] = :(2 * a.val[1] * a.val[2])
            else
                # c[ord] = 2 * Σ_{j=1}^{half} a.val[j] * a.val[ord+2-j]
                terms = Expr[]
                for j in 1:half
                    push!(terms, :(a.val[$j] * a.val[$(ord + 2 - j)]))
                end
                sum_expr = _expr_sum(terms)
                exprs[k] = :(2 * ($sum_expr))
            end
        else
            # ord is even, ord >= 2
            half = div(ord, 2)
            mid = half + 1  # 1-indexed index for a[half] (0-indexed)
            terms = Expr[]
            for j in 1:half
                push!(terms, :(a.val[$j] * a.val[$(ord + 2 - j)]))
            end
            if isempty(terms)
                exprs[k] = :(a.val[$mid] * a.val[$mid])
            else
                sum_expr = _expr_sum(terms)
                exprs[k] = :(2 * ($sum_expr) + a.val[$mid] * a.val[$mid])
            end
        end
    end
    return quote
        Base.@_inline_meta
        TruncatedTaylorSeries($(Expr(:tuple, exprs...)))
    end
end

# OP_CB # a^3 = sqr(a) * a
@inline function taylor_op_cb(a::TruncatedTaylorSeries{N, T}) where {N, T}
    return taylor_op_mul(taylor_op_sqr(a), a)
end

# OP_INV_NOT_ZERO # a ≠ 0 ? 1/a : a
@inline function taylor_op_inv_not_zero(a::TruncatedTaylorSeries{N, T}) where {N, T}
    iszero(a.val[1]) && return a
    return taylor_op_inv(a)
end

# OP_INVSQR # 1 / a^2
@inline function taylor_op_invsqr(a::TruncatedTaylorSeries{N, T}) where {N, T}
    return taylor_op_inv(taylor_op_sqr(a))
end

# OP_SQRT # √(a)
# c[0] = sqrt(a[0])
# c[1] = a[1] / (2 * c[0])
# c[k] = (a[k] - Σ_{j=1}^{k-1} c[j]*c[k-j]) / (2*c[0])  for k >= 2
@generated function taylor_op_sqrt(a::TruncatedTaylorSeries{N, T}) where {N, T}
    stmts = Expr[]
    push!(stmts, :(c1 = sqrt(a.val[1])))
    push!(stmts, :(two_c1 = c1 + c1))
    if N >= 2
        push!(stmts, :(c2 = a.val[2] / two_c1))
    end
    for k in 3:N
        # 0-indexed: c[k-1] = (a[k-1] - Σ_{j=1}^{k-2} c[j]*c[k-1-j]) / (2*c[0])
        # 1-indexed (m = j+1): Σ_{m=2}^{k-1} c_m * c_{k+1-m}
        sum_terms = Expr[]
        for j in 2:(k - 1)
            push!(sum_terms, :($(Symbol(:c, j)) * $(Symbol(:c, k + 1 - j))))
        end
        sum_expr = _expr_sum(sum_terms)
        push!(stmts, :($(Symbol(:c, k)) = (a.val[$k] - $sum_expr) / two_c1))
    end
    cvars = Symbol[Symbol(:c, k) for k in 1:N]
    return quote
        Base.@_inline_meta
        $(stmts...)
        TruncatedTaylorSeries($(Expr(:tuple, cvars...)))
    end
end

# OP_POW_INT # a^r where r isa Integer
# Uses logarithmic differentiation recurrence:
# w[0] = a[0]^r
# w[k] = (1/k) * (1/a[0]) * Σ_{j=1}^{k} (r*j - (k-j)) * a[j] * w[k-j]
# This is the standard recurrence from Griewank & Walther (Chapter 13).
@generated function taylor_op_pow_int(
        a::TruncatedTaylorSeries{N, T},
        r::I,
    ) where {N, T, I <: Integer}
    stmts = Expr[]
    push!(stmts, :(w1 = op_pow_int(a.val[1], r)))
    if N >= 2
        push!(stmts, :(a0_inv = inv(a.val[1])))
    end
    for k in 2:N
        # w[k-1] in 0-indexed = w_k in 1-indexed
        # Σ_{j=1}^{k-1} ((r*j - (k-1-j)) * a.val[j+1] * w_{k-j})
        sum_terms = Expr[]
        for j in 1:(k - 1)
            # coeff = j*r - (k-1-j): j is compile-time, r is runtime
            push!(
                sum_terms,
                :(
                    $j * r * a.val[$(j + 1)] * $(Symbol(:w, k - j)) -
                        $(k - 1 - j) * a.val[$(j + 1)] * $(Symbol(:w, k - j))
                ),
            )
        end
        sum_expr = _expr_sum(sum_terms)
        push!(stmts, :($(Symbol(:w, k)) = a0_inv * ($sum_expr) / $(k - 1)))
    end
    wvars = Symbol[Symbol(:w, k) for k in 1:N]
    return quote
        Base.@_inline_meta
        $(stmts...)
        TruncatedTaylorSeries($(Expr(:tuple, wvars...)))
    end
end

# OP_SIN and OP_COS
# Coupled recurrence (0-indexed):
# s[0] = sin(a[0]),  c[0] = cos(a[0])
# s[k] = (1/k) * Σ_{j=1}^{k} j * a[j] * c[k-j]
# c[k] = -(1/k) * Σ_{j=1}^{k} j * a[j] * s[k-j]
@generated function taylor_op_sin(a::TruncatedTaylorSeries{N, T}) where {N, T}
    stmts = Expr[]
    if N == 1
        push!(stmts, :(s1 = sin(a.val[1])))
    else
        push!(stmts, :((s1, c1) = sincos(a.val[1])))
        for k in 2:N
            # s_k in 1-indexed naming = s[k-1] in 0-indexed
            # s[k-1] = (1/(k-1)) * Σ_{j=1}^{k-1} j * a[j] * c[k-1-j]  (0-indexed)
            # 1-indexed: Σ_{j=1}^{k-1} j * a.val[j+1] * c_{k-j}
            sum_terms = Expr[]
            for j in 1:(k - 1)
                push!(sum_terms, :($j * a.val[$(j + 1)] * $(Symbol(:c, k - j))))
            end
            sum_expr = _expr_sum(sum_terms)
            push!(stmts, :($(Symbol(:s, k)) = ($sum_expr) / $(k - 1)))
            # c[k-1] in 0-indexed: c_k = -(1/(k-1)) * Σ_{j=1}^{k-1} j*a[j]*s[k-j]
            csum_terms = Expr[]
            for j in 1:(k - 1)
                push!(csum_terms, :($j * a.val[$(j + 1)] * $(Symbol(:s, k - j))))
            end
            csum_expr = _expr_sum(csum_terms)
            push!(stmts, :($(Symbol(:c, k)) = -($csum_expr) / $(k - 1)))
        end
    end
    svars = Symbol[Symbol(:s, k) for k in 1:N]
    return quote
        Base.@_inline_meta
        $(stmts...)
        TruncatedTaylorSeries($(Expr(:tuple, svars...)))
    end
end

@generated function taylor_op_cos(a::TruncatedTaylorSeries{N, T}) where {N, T}
    stmts = Expr[]
    if N == 1
        push!(stmts, :(c1 = cos(a.val[1])))
    else
        push!(stmts, :((s1, c1) = sincos(a.val[1])))
        for k in 2:N
            sum_terms = Expr[]
            for j in 1:(k - 1)
                push!(sum_terms, :($j * a.val[$(j + 1)] * $(Symbol(:c, k - j))))
            end
            sum_expr = _expr_sum(sum_terms)
            push!(stmts, :($(Symbol(:s, k)) = ($sum_expr) / $(k - 1)))
            csum_terms = Expr[]
            for j in 1:(k - 1)
                push!(csum_terms, :($j * a.val[$(j + 1)] * $(Symbol(:s, k - j))))
            end
            csum_expr = _expr_sum(csum_terms)
            push!(stmts, :($(Symbol(:c, k)) = -($csum_expr) / $(k - 1)))
        end
    end
    cvars = Symbol[Symbol(:c, k) for k in 1:N]
    return quote
        Base.@_inline_meta
        $(stmts...)
        TruncatedTaylorSeries($(Expr(:tuple, cvars...)))
    end
end

# OP_MULADD # a * b + c
@generated function taylor_op_muladd(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
        c::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    exprs = Vector{Expr}(undef, N)
    for k in 1:N
        # Cauchy product at order k (1-indexed), then add c.val[k]
        terms = Expr[]
        for j in 1:k
            push!(terms, :(a.val[$j] * b.val[$(k + 1 - j)]))
        end
        acc = terms[1]
        for i in 2:length(terms)
            acc = :($(acc) + $(terms[i]))
        end
        exprs[k] = :($(acc) + c.val[$k])
    end
    return quote
        Base.@_inline_meta
        TruncatedTaylorSeries($(Expr(:tuple, exprs...)))
    end
end

# OP_MULSUB # a * b - c
@generated function taylor_op_mulsub(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
        c::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    exprs = Vector{Expr}(undef, N)
    for k in 1:N
        terms = Expr[]
        for j in 1:k
            push!(terms, :(a.val[$j] * b.val[$(k + 1 - j)]))
        end
        acc = terms[1]
        for i in 2:length(terms)
            acc = :($(acc) + $(terms[i]))
        end
        exprs[k] = :($(acc) - c.val[$k])
    end
    return quote
        Base.@_inline_meta
        TruncatedTaylorSeries($(Expr(:tuple, exprs...)))
    end
end

# OP_SUBMUL # c - a * b
@generated function taylor_op_submul(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
        c::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    exprs = Vector{Expr}(undef, N)
    for k in 1:N
        terms = Expr[]
        for j in 1:k
            push!(terms, :(a.val[$j] * b.val[$(k + 1 - j)]))
        end
        acc = terms[1]
        for i in 2:length(terms)
            acc = :($(acc) + $(terms[i]))
        end
        exprs[k] = :(c.val[$k] - $(acc))
    end
    return quote
        Base.@_inline_meta
        TruncatedTaylorSeries($(Expr(:tuple, exprs...)))
    end
end

# OP_ADD3 # a + b + c
@inline function taylor_op_add3(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
        c::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    return taylor_op_add(taylor_op_add(a, b), c)
end

# OP_ADD4 # a + b + c + d
@inline function taylor_op_add4(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
        c::TruncatedTaylorSeries{N, T},
        d::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    return taylor_op_add(taylor_op_add(a, b), taylor_op_add(c, d))
end

# OP_MUL3 # a * b * c
@inline function taylor_op_mul3(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
        c::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    return taylor_op_mul(taylor_op_mul(a, b), c)
end

# OP_MUL4 # a * b * c * d
@inline function taylor_op_mul4(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
        c::TruncatedTaylorSeries{N, T},
        d::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    return taylor_op_mul(taylor_op_mul(a, b), taylor_op_mul(c, d))
end

# OP_MULMULADD # a * b + c * d
@inline function taylor_op_mulmuladd(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
        c::TruncatedTaylorSeries{N, T},
        d::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    return taylor_op_add(taylor_op_mul(a, b), taylor_op_mul(c, d))
end

# OP_MULMULSUB # a * b - c * d
@inline function taylor_op_mulmulsub(
        a::TruncatedTaylorSeries{N, T},
        b::TruncatedTaylorSeries{N, T},
        c::TruncatedTaylorSeries{N, T},
        d::TruncatedTaylorSeries{N, T},
    ) where {N, T}
    return taylor_op_sub(taylor_op_mul(a, b), taylor_op_mul(c, d))
end
