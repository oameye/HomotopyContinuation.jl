# HomotopyContinuationNext.jl

This is a fork of [HomotopyContinuation.jl](https://github.com/JuliaHomotopyContinuation/HomotopyContinuation.jl), with the goals to make the package more type-stable and reduce inference time (TTFX). The eventual goal would be for the package to become HomotopyContinuation v3.

## TODO

- **Direct monomial evaluator for polyhedral homotopy**: Currently `_build_parametric_system` in `polyhedral.jl` roundtrips through DynamicPolynomials to build a parametric `SystemEvaluator` from support matrices (creating string-named variables as coefficient parameters). A cleaner approach: build an `InstructionSequence` directly from the support, computing `F(x;c) = Σ_j c_j · x^{A[:,j]}`, Jacobian, and Taylor orders 1-3 without the MP roundtrip. Non-trivial (needs Taylor machinery), but eliminates the DynamicPolynomials dependency at init time.


- **Benchmark interpreter vs v2 compiled mode**: v3 uses interpreter-only evaluation (no RuntimeGeneratedFunctions). Verify that steady-state performance of the tape-based `Interpreter` matches or beats v2's `compile=:all` mode on standard benchmarks (katsura, cyclic, etc.). If there's a gap, profile the hot loop in `execute!` and consider SIMD or batched evaluation.


- No allocation tests
- ~~Add ConcreteStructs tests~~ Done: `test/concrete_structs_test.jl`

- review type system and API