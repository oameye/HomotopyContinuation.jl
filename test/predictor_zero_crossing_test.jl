using Test
import HomotopyContinuationNext as HCN

@testset "predictor trust region through zero" begin
    pred = HCN.Predictor(1, 1)
    fill!(pred.tx3.data, 0)
    pred.tx3.data[2, 1] = 1
    pred.tx_norm = (0.0, 1.0, 0.0, 0.0)
    pred.local_error = NaN

    HCN._compute_trust_region!(pred)

    @test pred.trust_region == 1.0
    @test pred.local_error == 1.0
    @test isfinite(pred.trust_region)
    @test pred.trust_region > 0

    # A derivative-based radius remains authoritative when the Taylor data
    # determine one: |x₂| / |x₃| = 2 / 4.
    pred2 = HCN.Predictor(1, 1)
    fill!(pred2.tx3.data, 0)
    pred2.tx3.data[2, 1] = 1
    pred2.tx3.data[3, 1] = 2
    pred2.tx3.data[4, 1] = 4
    pred2.tx_norm = (0.0, 1.0, 2.0, 4.0)
    pred2.local_error = NaN

    HCN._compute_trust_region!(pred2)

    @test pred2.trust_region ≈ 0.5
    @test isfinite(pred2.local_error)
end
