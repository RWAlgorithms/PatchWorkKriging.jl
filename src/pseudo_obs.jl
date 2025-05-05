# SPDX-License-Identifier: AGPL-3.0
# Copyright © 2025 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

##### Assumes only one hyperplane.

function compute_projection(x::AbstractVector, a::AbstractVector, b::Real)
    return x - (dot(a, x) - b) / dot(a, a) .* a
end

# returns N randomly selected projection points on the hyperplane.
function generate_pseudo_obs(rng::AbstractRNG, X::AbstractVector, N::Integer, tree::ST.PCTree)

    a = tree.root.data.hp.v
    b = tree.root.data.hp.c

    inds = randperm(rng, length(X))[1:N]
    boundary_pts = collect(compute_projection(X[i], a, b) for i in inds)
    return boundary_pts
end

# returns all projection points on the hyperplane, sorted by the first coordinate in ascending order.
function generate_pseudo_obs(X::AbstractVector, tree::ST.PCTree)

    a = tree.root.data.hp.v
    b = tree.root.data.hp.c

    boundary_pts = collect(compute_projection(x, a, b) for x in X)

    # sorted along the first coordinate.
    inds = sortperm(map(xx -> first(xx), boundary_pts))
    boundary_pts = boundary_pts[inds]

    return boundary_pts
end
