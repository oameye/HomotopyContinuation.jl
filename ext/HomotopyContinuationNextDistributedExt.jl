module HomotopyContinuationNextDistributedExt

import Distributed
import Serialization
using Random: Random
using MixedSubdivisions: MixedSubdivisions

using HomotopyContinuationNext: HomotopyContinuationNext
const HCN = HomotopyContinuationNext

include("HomotopyContinuationNextDistributedExt/serialization.jl")
include("HomotopyContinuationNextDistributedExt/distributed_map.jl")
include("HomotopyContinuationNextDistributedExt/solve.jl")
include("HomotopyContinuationNextDistributedExt/monodromy.jl")

end
