# SPDX-License-Identifier: AGPL-3.0
# Copyright © 2025 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

module PatchWorkKriging

using LinearAlgebra, SparseArrays, Random

import SpatialTrees as ST

include("kernels.jl")
include("pseudo_obs.jl")
include("engine.jl")

end # module PatchWorkKriging
