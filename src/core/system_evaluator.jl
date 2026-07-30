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
const SysEvalDF64Pair = Tuple{SysEvalDF64FW, SysEvalDF64OutFW}
const SysDF64InstallFW = FunctionWrapper{Nothing, Tuple{}}
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
    _df64::LazyRef{SysEvalDF64Pair}
    _install_df64!::SysDF64InstallFW
    _evaluate_and_jacobian!::SysEvalJacFW
    _taylor_1!::SysTaylor1FW
    _taylor_2!::SysTaylor2FW
    _taylor_3!::SysTaylor3FW
    _taylor_1_param!::SysTaylor1ParamFW
    _taylor_2_param!::SysTaylor2ParamFW
    _taylor_3_param!::SysTaylor3ParamFW
    _size::Tuple{Int, Int}
    _nparameters::Int
    _clone::FunctionWrapper{SystemEvaluator, Tuple{}}
end

const SystemFactory = FunctionWrapper{SystemEvaluator, Tuple{}}

# An evaluator equal to `S` with its own tapes, usable from another task.
_clone_system_evaluator(S::SystemEvaluator)::SystemEvaluator = S._clone()

# A copied `FunctionWrapper` keeps a raw pointer to the original closure, so a
# copied evaluator would silently run the original tapes. Rebuild instead.
Base.deepcopy_internal(S::SystemEvaluator, stackdict::IdDict)::SystemEvaluator =
    get!(() -> S._clone(), stackdict, S)

struct _EvaluatorCloner{F <: AbstractSystem}
    system::F
end

(c::_EvaluatorCloner)()::SystemEvaluator = SystemEvaluator(_clone_system(c.system))

## Lazy extended-precision wrappers

"""
    _lazy_df64(make) -> (cache, installer)

Pair a fresh empty cache with a wrapper that fills it by calling `make`, which
must return `(evaluate_df64!, evaluate_df64_out!)`. `make` is invoked through a
dynamic call so that compiling the installer does not compile `make`.
"""
function _lazy_df64(make::F)::Tuple{LazyRef{SysEvalDF64Pair}, SysDF64InstallFW} where {F}
    cache = LazyRef{SysEvalDF64Pair}()
    return cache, SysDF64InstallFW(_DF64Installer(cache, make))
end

struct _DF64Installer{F}
    cache::LazyRef{SysEvalDF64Pair}
    make::F
end

function (installer::_DF64Installer)()::Nothing
    Base.inferencebarrier(_install_df64!)(installer.cache, installer.make)
    return nothing
end

@noinline function _install_df64!(
        cache::LazyRef{SysEvalDF64Pair}, make,
    )::Nothing
    Base.@nospecialize make
    install!(cache, make()::SysEvalDF64Pair)
    return nothing
end

# Two threads racing here install equivalent wrappers; the duplicate work is the
# only cost, since `LazyRef` publishes the pair as a single atomic store.
function _df64_evaluators(S::SystemEvaluator)::SysEvalDF64Pair
    cache = S._df64
    is_installed(cache) || S._install_df64!()
    return cache[]
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
    first(_df64_evaluators(S))(u, x, p)
    return nothing
end

function evaluate!(
        u::FSVec{ComplexDF64}, S::SystemEvaluator,
        x::FSVec{ComplexDF64}, p::FSVec{ComplexF64},
    )::Nothing
    last(_df64_evaluators(S))(u, x, p)
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
        _lazy_df64(
            () -> (
                SysEvalDF64FW((u, x, p) -> (evaluate!(u, F, x, p); nothing)),
                SysEvalDF64OutFW((u, x, p) -> (evaluate!(u, F, x, p); nothing)),
            ),
        )...,
        SysEvalJacFW((u, U, x, p) -> (evaluate_and_jacobian!(u, U, F, x, p); nothing)),
        SysTaylor1FW((u, tx, p) -> (taylor!(u, Val(1), F, tx, p); nothing)),
        SysTaylor2FW((u, tx, p) -> (taylor!(u, Val(2), F, tx, p); nothing)),
        SysTaylor3FW((u, tx, p) -> (taylor!(u, Val(3), F, tx, p); nothing)),
        SysTaylor1ParamFW((u, tx, tp) -> (taylor!(u, Val(1), F, tx, tp); nothing)),
        SysTaylor2ParamFW((u, tx, tp) -> (taylor!(u, Val(2), F, tx, tp); nothing)),
        SysTaylor3ParamFW((u, tx, tp) -> (taylor!(u, Val(3), F, tx, tp); nothing)),
        size(F),
        nparameters(F),
        SystemFactory(_EvaluatorCloner(F)),
    )
end
