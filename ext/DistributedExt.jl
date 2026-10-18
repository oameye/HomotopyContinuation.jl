module DistributedExt

import Distributed
import Serialization
using Random: Random
using MixedSubdivisions: MixedSubdivisions

using HomotopyContinuation: HomotopyContinuation
const HCN = HomotopyContinuation

include("DistributedExt/serialization.jl")
include("DistributedExt/distributed_map.jl")
include("DistributedExt/solve.jl")
include("DistributedExt/monodromy.jl")

end
