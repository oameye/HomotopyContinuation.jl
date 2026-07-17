## FunctionWrapper type aliases for SystemEvaluator
const SysEvalFW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, FSVec{ComplexF64}, FSVec{ComplexF64},
    },
}
const SysEvalDF64FW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, FSVec{ComplexDF64}, FSVec{ComplexF64},
    },
}
# DF64 output variant: consumers that combine several system evaluations
# (homotopy mixing, randomization fold) need the unrounded residual, since the
# cancellation between the combined terms is exactly what DF64 is there for.
const SysEvalDF64OutFW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexDF64}, FSVec{ComplexDF64}, FSVec{ComplexF64},
    },
}
const SysEvalJacFW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, FSMat{ComplexF64},
        FSVec{ComplexF64}, FSVec{ComplexF64},
    },
}
const SysTaylor1FW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, TaylorVector{2, ComplexF64}, FSVec{ComplexF64},
    },
}
const SysTaylor2FW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, TaylorVector{3, ComplexF64}, FSVec{ComplexF64},
    },
}
const SysTaylor3FW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, TaylorVector{4, ComplexF64}, FSVec{ComplexF64},
    },
}
# Taylor with TaylorVector parameters (Cauchy product convolution for parametric homotopies)
const SysTaylor1ParamFW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, TaylorVector{2, ComplexF64}, TaylorVector{2, ComplexF64},
    },
}
const SysTaylor2ParamFW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, TaylorVector{3, ComplexF64}, TaylorVector{3, ComplexF64},
    },
}
const SysTaylor3ParamFW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, TaylorVector{4, ComplexF64}, TaylorVector{4, ComplexF64},
    },
}

"""
    SystemEvaluator

Concrete, monomorphic wrapper around any `AbstractSystem`. Uses `FunctionWrapper`
closures to erase the system type — the tracker never sees the original system type,
eliminating runtime dispatch on hot paths.
"""
struct SystemEvaluator
    _evaluate!::SysEvalFW
    _evaluate_df64!::SysEvalDF64FW
    _evaluate_df64_out!::SysEvalDF64OutFW
    _evaluate_and_jacobian!::SysEvalJacFW
    _taylor_1!::SysTaylor1FW
    _taylor_2!::SysTaylor2FW
    _taylor_3!::SysTaylor3FW
    _taylor_1_param!::SysTaylor1ParamFW
    _taylor_2_param!::SysTaylor2ParamFW
    _taylor_3_param!::SysTaylor3ParamFW
    _size::Tuple{Int, Int}
    _nparameters::Int
end

## Dispatch methods — forward to FW closures

Base.size(S::SystemEvaluator) = S._size
nparameters(S::SystemEvaluator)::Int = S._nparameters

function evaluate!(
        u::FSVec{ComplexF64}, S::SystemEvaluator,
        x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._evaluate!(u, x, p)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, S::SystemEvaluator,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._evaluate_df64!(u, x, p)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexDF64}, S::SystemEvaluator,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._evaluate_df64_out!(u, x, p)
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64}, S::SystemEvaluator,
        x::FSVec{ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._evaluate_and_jacobian!(u, U, x, p)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, S::SystemEvaluator,
        tx::TaylorVector{2, ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._taylor_1!(u, tx, p)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, S::SystemEvaluator,
        tx::TaylorVector{3, ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._taylor_2!(u, tx, p)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, S::SystemEvaluator,
        tx::TaylorVector{4, ComplexF64}, p::FSVec{ComplexF64},
    )::Nothing
    S._taylor_3!(u, tx, p)
    return nothing
end

## Dispatch: taylor! with TaylorVector parameters (Cauchy product convolution)

function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, S::SystemEvaluator,
        tx::TaylorVector{2, ComplexF64}, tp::TaylorVector{2, ComplexF64},
    )::Nothing
    S._taylor_1_param!(u, tx, tp)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, S::SystemEvaluator,
        tx::TaylorVector{3, ComplexF64}, tp::TaylorVector{3, ComplexF64},
    )::Nothing
    S._taylor_2_param!(u, tx, tp)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, S::SystemEvaluator,
        tx::TaylorVector{4, ComplexF64}, tp::TaylorVector{4, ComplexF64},
    )::Nothing
    S._taylor_3_param!(u, tx, tp)
    return nothing
end

## Constructor from AbstractSystem

function SystemEvaluator(F::AbstractSystem)
    return SystemEvaluator(
        SysEvalFW((u, x, p) -> (evaluate!(u, F, x, p); nothing)),
        SysEvalDF64FW((u, x, p) -> (evaluate!(u, F, x, p); nothing)),
        SysEvalDF64OutFW((u, x, p) -> (evaluate!(u, F, x, p); nothing)),
        SysEvalJacFW((u, U, x, p) -> (evaluate_and_jacobian!(u, U, F, x, p); nothing)),
        SysTaylor1FW((u, tx, p) -> (taylor!(u, Val(1), F, tx, p); nothing)),
        SysTaylor2FW((u, tx, p) -> (taylor!(u, Val(2), F, tx, p); nothing)),
        SysTaylor3FW((u, tx, p) -> (taylor!(u, Val(3), F, tx, p); nothing)),
        SysTaylor1ParamFW((u, tx, tp) -> (taylor!(u, Val(1), F, tx, tp); nothing)),
        SysTaylor2ParamFW((u, tx, tp) -> (taylor!(u, Val(2), F, tx, tp); nothing)),
        SysTaylor3ParamFW((u, tx, tp) -> (taylor!(u, Val(3), F, tx, tp); nothing)),
        size(F),
        nparameters(F),
    )
end
