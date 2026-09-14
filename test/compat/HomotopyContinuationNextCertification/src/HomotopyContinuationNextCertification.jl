module HomotopyContinuationNextCertification

# Transitional test-only forwarding layer for the certification-package rename.
import HomotopyContinuationCertification

const _HCC = HomotopyContinuationCertification

for name in names(_HCC; all = true, imported = true)
    name in (:eval, :include, :HomotopyContinuationCertification) && continue
    startswith(String(name), "#") && continue
    isdefined(@__MODULE__, name) && continue
    value = getfield(_HCC, name)
    Core.eval(@__MODULE__, Expr(:const, Expr(:(=), name, value)))
end

for name in names(_HCC; all = false, imported = false)
    Core.eval(@__MODULE__, Expr(:export, name))
end

end # module
