module HomotopyContinuationDistributedExt

import Distributed
import Serialization
using Random: Random
using MixedSubdivisions: MixedSubdivisions

using HomotopyContinuation: HomotopyContinuation
const HCN = HomotopyContinuation

include("HomotopyContinuationDistributedExt/serialization.jl")
include("HomotopyContinuationDistributedExt/distributed_map.jl")
include("HomotopyContinuationDistributedExt/solve.jl")
include("HomotopyContinuationDistributedExt/monodromy.jl")

end
