using PrecompileTools: @compile_workload, @setup_workload

# Compile the reusable evaluator and tracker pipelines during package
# precompilation. Runtime-generated polynomial kernels remain system-specific,
# but the surrounding construction and solve machinery is shared by all
# DynamicPolynomials systems of the same representation.
@setup_workload begin
    @polyvar precompile_x precompile_y precompile_parameter
    precompile_polys = [
        precompile_x^2 + precompile_y - 1,
        precompile_x * precompile_y - 2,
    ]

    @compile_workload begin
        interpreted_system = System(precompile_polys; compile = CompileMode.INTERPRETED)
        System(precompile_polys; compile = CompileMode.COMPILED)

        algorithm = TotalDegree(; seed = UInt32(1))
        solve(interpreted_system, algorithm, Serial(); show_progress = false)
        solve(interpreted_system, algorithm, Threaded(1); show_progress = false)

        solve(
            interpreted_system, Polyhedral(; seed = UInt32(1)), Serial();
            show_progress = false,
        )

        overdetermined_system = System(
            [
                precompile_x^2 + precompile_y^2 - 1,
                precompile_x - precompile_y,
                precompile_x * precompile_y - 0.25,
            ]
        )
        solve(overdetermined_system, algorithm, Serial(); show_progress = false)

        parameter_system = System(
            [precompile_x^2 - precompile_parameter];
            variables = [precompile_x], parameters = [precompile_parameter],
        )
        find_start_pair(parameter_system)
        solve(
            parameter_system, [ComplexF64[1]], Serial();
            start_parameters = ComplexF64[1],
            target_parameters = ComplexF64[2],
            seed = UInt32(1),
            show_progress = false,
        )
        monodromy_solve(
            parameter_system, [ComplexF64[1]], ComplexF64[1];
            target_solutions_count = 2,
            seed = UInt32(1),
            threading = false,
            show_progress = false,
        )

        subspace = rand_subspace(4; dim = 2)
        rebuilt_subspace = LinearSubspace(intrinsic(subspace))
        geodesic_distance(subspace, rebuilt_subspace)
        unique = UniquePoints(2)
        add!(unique, ComplexF64[1, 2], 1, 1.0e-8)
    end
end
