module HomotopyContinuationNextDistributedExt

import Distributed
import Serialization
using MixedSubdivisions: MixedSubdivisions

using HomotopyContinuationNext: HomotopyContinuationNext
const HCN = HomotopyContinuationNext

include("HomotopyContinuationNextDistributedExt/serialization.jl")
include("HomotopyContinuationNextDistributedExt/distributed_map.jl")
include("HomotopyContinuationNextDistributedExt/solve.jl")

end
