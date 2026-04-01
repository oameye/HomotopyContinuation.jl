# Compare primitives: inf_norm, LU+ldiv!
# Standalone: julia --project=benchmark benchmark/compare/primitives.jl

if !@isdefined(print_header)
    include(joinpath(@__DIR__, "common.jl"))
end

print_header("Primitives: Next vs HC v2")

println("\n── inf_norm ──")
for n in [4, 16, 64]
    x_fs = FSVec{ComplexF64}(rand(ComplexF64, n))
    x_vec = Vector{ComplexF64}(collect(x_fs))
    hc_inf = HC.InfNorm()
    # Warmup
    Next.inf_norm(x_fs)
    hc_inf(x_vec)
    t_next = @belapsed Next.inf_norm($x_fs)
    t_hc = @belapsed $hc_inf($x_vec)
    print_row("inf_norm n=$n", t_next, t_hc)
end

println("\n── LU + ldiv! (well-conditioned) ──")
for n in [4, 8, 16]
    A_data = rand(ComplexF64, n, n) + 5.0I
    b_data = rand(ComplexF64, n)

    WS_next = Next.MatrixWorkspace(n, n)
    copyto!(WS_next.A, A_data)
    Next.updated!(WS_next)
    x_next = FSVec{ComplexF64}(zeros(ComplexF64, n))
    b_next = FSVec{ComplexF64}(b_data)
    # Warmup
    copyto!(WS_next.A, A_data); Next.updated!(WS_next); ldiv!(x_next, WS_next, b_next)
    t_next = @belapsed begin
        copyto!($WS_next.A, $A_data)
        Next.updated!($WS_next)
        ldiv!($x_next, $WS_next, $b_next)
    end

    WS_hc = HC.MatrixWorkspace(n, n)
    copyto!(WS_hc.A, A_data)
    HC.updated!(WS_hc)
    x_hc = zeros(ComplexF64, n)
    b_hc = Vector{ComplexF64}(b_data)
    # Warmup
    copyto!(WS_hc.A, A_data); HC.updated!(WS_hc); ldiv!(x_hc, WS_hc, b_hc)
    t_hc = @belapsed begin
        copyto!($WS_hc.A, $A_data)
        HC.updated!($WS_hc)
        ldiv!($x_hc, $WS_hc, $b_hc)
    end
    print_row("lu_ldiv n=$n", t_next, t_hc)
end
