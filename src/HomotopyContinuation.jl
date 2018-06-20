__precompile__()

module HomotopyContinuation

    import DynamicPolynomials: @polyvar, PolyVar
    export @polyvar, PolyVar

    export AffinePatches,
        Correctors,
        Endgaming,
        Homotopies,
        PatchSwitching,
        PathTracking,
        Predictors,
        Input,
        Problems,
        ProjectiveVectors,
        Solving,
        StepLength,
        Systems,
        Utilities

    include("utilities.jl")
    include("parallel.jl")
    include("projective_vectors.jl")
    include("affine_patches.jl")

    include("systems_base.jl")
    include("homotopies_base.jl")
    include("systems.jl")
    include("homotopies.jl")
    include("input.jl")
    include("problems.jl")
    include("predictors.jl")
    include("correctors.jl")
    include("prediction_correction.jl")

    include("step_length.jl")

    include("path_tracking.jl")
    include("endgaming.jl")

    include("patch_switching.jl")

    include("solving.jl")
    include("solve.jl")


    import .Solving: AffineResult, ProjectiveResult, PathResult, solution,
        residual, startsolution, issuccess,
        isfailed, isaffine, isprojective,
        isatinfinity, issingular, isnonsingular,
        nresults, nfinite, nsingular, natinfinity, nfailed, nnonsingular,
        finite, results, solutions, failed, atinfinity, singular, nonsingular, seed

    export AffineResult, ProjectiveResult, PathResult, solution,
        residual, startsolution, issuccess,
        isfailed, isaffine, isprojective,
        isatinfinity, issingular, isnonsingular,
        nresults, nfinite, nsingular, natinfinity, nfailed, nnonsingular,
        finite, results, solutions, failed, atinfinity, singular, nonsingular, seed

    import .Homotopies: StraightLineHomotopy, FixedPointHomotopy
    export StraightLineHomotopy, FixedPointHomotopy

    import .Systems: FPSystem, SPSystem
    export FPSystem, SPSystem


end #
