# HomotopyContinuationNext.jl

This is a fork of [HomotopyContinuation.jl](https://github.com/JuliaHomotopyContinuation/HomotopyContinuation.jl), with the goals to make the package more type-stable and reduce inference time (TTFX). The eventual goal would be for the package to become HomotopyContinuation v3.

## TODO

- **Direct monomial evaluator for polyhedral homotopy**: Currently `_build_parametric_system` in `polyhedral.jl` roundtrips through DynamicPolynomials to build a parametric `SystemEvaluator` from support matrices (creating string-named variables as coefficient parameters). A cleaner approach: build an `InstructionSequence` directly from the support, computing `F(x;c) = Σ_j c_j · x^{A[:,j]}`, Jacobian, and Taylor orders 1-3 without the MP roundtrip. Non-trivial (needs Taylor machinery), but eliminates the DynamicPolynomials dependency at init time.


- I don't like the default union stuff. Can we have a nice interface?
- Abstract type for caches?
- Default seed?
- Where should the solver optoins go?
- where is the :none, :mixed, :all options?

- @kwdef struct Polyhedral
    tracker_options::TrackerOptions = TrackerOptions()
    seed::Union{Nothing, UInt32} = nothing
end
Should be parametrically typed or UInt32 default

- Add ConcreteStructs tests

- No allocation tests

- review type system and API