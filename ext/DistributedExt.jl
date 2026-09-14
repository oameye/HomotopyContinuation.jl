module DistributedExt

import Distributed
import Serialization
using Random: Random
using MixedSubdivisions: MixedSubdivisions

using HomotopyContinuation: HomotopyContinuation
const HCN = HomotopyContinuation

include("HomotopyContinuationNextDistributedExt/serialization.jl")
include("HomotopyContinuationNextDistributedExt/distributed_map.jl")
include("HomotopyContinuationNextDistributedExt/solve.jl")
include("HomotopyContinuationNextDistributedExt/monodromy.jl")

end
