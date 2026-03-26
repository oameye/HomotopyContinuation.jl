## FunctionWrapper type aliases for HomotopyEvaluator
const HomEvalFW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64,
    },
}
const HomEvalDF64FW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, FSVec{ComplexDF64}, ComplexF64,
    },
}
const HomEvalJacFW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, FSMat{ComplexF64},
        FSVec{ComplexF64}, ComplexF64,
    },
}
const HomTaylor1FW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64,
    },
}
const HomTaylor2FW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, TaylorVector{3, ComplexF64}, ComplexF64, Bool,
    },
}
const HomTaylor3FW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, TaylorVector{4, ComplexF64}, ComplexF64, Bool,
    },
}
const HomSetSolFW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64,
    },
}
const HomGetSolFW = FunctionWrapper{
    Nothing, Tuple{
        FSVec{ComplexF64}, FSVec{ComplexF64}, ComplexF64,
    },
}
const HomParamsFW = FunctionWrapper{Nothing, Tuple{FSVec{ComplexF64}}}

"""
    HomotopyEvaluator

Concrete, monomorphic wrapper around any `AbstractHomotopy`. Uses `FunctionWrapper`
closures to erase the homotopy type — the tracker never sees the original homotopy type,
eliminating runtime dispatch on hot paths.
"""
struct HomotopyEvaluator
    _evaluate!::HomEvalFW
    _evaluate_df64!::HomEvalDF64FW
    _evaluate_and_jacobian!::HomEvalJacFW
    _taylor_1!::HomTaylor1FW
    _taylor_2!::HomTaylor2FW
    _taylor_3!::HomTaylor3FW
    _set_solution!::HomSetSolFW
    _get_solution!::HomGetSolFW
    _start_parameters!::HomParamsFW
    _target_parameters!::HomParamsFW
    _size::Tuple{Int, Int}
end

## Dispatch methods — forward to FW closures

Base.size(H::HomotopyEvaluator) = H._size

function evaluate!(
        u::FSVec{ComplexF64}, H::HomotopyEvaluator,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    H._evaluate!(u, x, t)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexF64}, H::HomotopyEvaluator,
        x::FSVec{ComplexDF64}, t::ComplexF64,
    )::Nothing
    H._evaluate_df64!(u, x, t)
    return nothing
end

function evaluate_and_jacobian!(
        u::FSVec{ComplexF64}, U::FSMat{ComplexF64}, H::HomotopyEvaluator,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    H._evaluate_and_jacobian!(u, U, x, t)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{1}, H::HomotopyEvaluator,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    H._taylor_1!(u, x, t)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{2}, H::HomotopyEvaluator,
        tx::TaylorVector{3, ComplexF64}, t::ComplexF64;
        incremental::Bool = false,
    )::Nothing
    H._taylor_2!(u, tx, t, incremental)
    return nothing
end

function taylor!(
        u::FSVec{ComplexF64}, ::Val{3}, H::HomotopyEvaluator,
        tx::TaylorVector{4, ComplexF64}, t::ComplexF64;
        incremental::Bool = false,
    )::Nothing
    H._taylor_3!(u, tx, t, incremental)
    return nothing
end

function set_solution!(
        x::FSVec{ComplexF64}, H::HomotopyEvaluator,
        y::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    H._set_solution!(x, y, t)
    return nothing
end

function get_solution!(
        out::FSVec{ComplexF64}, H::HomotopyEvaluator,
        x::FSVec{ComplexF64}, t::ComplexF64,
    )::Nothing
    H._get_solution!(out, x, t)
    return nothing
end

function start_parameters!(H::HomotopyEvaluator, p::FSVec{ComplexF64})::Nothing
    H._start_parameters!(p)
    return nothing
end

function target_parameters!(H::HomotopyEvaluator, p::FSVec{ComplexF64})::Nothing
    H._target_parameters!(p)
    return nothing
end

## Constructor from AbstractHomotopy

function HomotopyEvaluator(H::AbstractHomotopy)
    return HomotopyEvaluator(
        HomEvalFW((u, x, t) -> (evaluate!(u, H, x, t); nothing)),
        HomEvalDF64FW((u, x, t) -> (evaluate!(u, H, x, t); nothing)),
        HomEvalJacFW((u, U, x, t) -> (evaluate_and_jacobian!(u, U, H, x, t); nothing)),
        HomTaylor1FW((u, x, t) -> (taylor!(u, Val(1), H, x, t); nothing)),
        HomTaylor2FW((u, tx, t, inc) -> (taylor!(u, Val(2), H, tx, t; incremental = inc); nothing)),
        HomTaylor3FW((u, tx, t, inc) -> (taylor!(u, Val(3), H, tx, t; incremental = inc); nothing)),
        HomSetSolFW((x, y, t) -> (set_solution!(x, H, y, t); nothing)),
        HomGetSolFW((out, x, t) -> (get_solution!(out, H, x, t); nothing)),
        HomParamsFW((p) -> (start_parameters!(H, p); nothing)),
        HomParamsFW((p) -> (target_parameters!(H, p); nothing)),
        size(H),
    )
end
