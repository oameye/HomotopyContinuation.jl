module HomotopyContinuationNext

# Transitional test-only compatibility layer for the v3 package-identity rename.
# Production code has no HomotopyContinuationNext module. Existing tests are
# migrated independently while this package forwards every core binding to the
# renamed HomotopyContinuation module.
import HomotopyContinuation

const _HC = HomotopyContinuation

for name in names(_HC; all = true, imported = true)
    name in (:eval, :include, :HomotopyContinuation) && continue
    startswith(String(name), "#") && continue
    isdefined(@__MODULE__, name) && continue
    value = getfield(_HC, name)
    Core.eval(@__MODULE__, Expr(:const, Expr(:(=), name, value)))
end

for name in names(_HC; all = false, imported = false)
    Core.eval(@__MODULE__, Expr(:export, name))
end

end # module
