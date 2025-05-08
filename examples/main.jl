# SPDX-License-Identifier: AGPL-3.0
# Copyright © 2025 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

# run this once: include("a.jl")

import Colors
include("helpers/visualization.jl")

PLT.close("all")
fig_num = 1

const T = Float64
const D = 2

rng = Random.Xoshiro(0)
N = 500
min_t = T(-5.0)
max_t = T(5.0)
max_N_t = 5000

X = collect(randn(rng, T, D) for _ in 1:N)

# # Levels: 1
levels = 2 # 2^(levels-1) leaf nodes. Must be larger than 1.
s_tree = ST.PCTree(X, levels)
X_parts, X_parts_inds = ST.label_leaf_nodes(s_tree, X)

centroid = Statistics.mean(Statistics.mean(x) for x in X_parts)
max_dist = maximum(maximum(norm(xj - centroid) for xj in x) for x in X_parts) * 1.1
y_set = Vector{Vector{T}}(undef, 0)
t_set = Vector{Vector{T}}(undef, 0)
ST.get_partition_lines!(y_set, t_set, s_tree, min_t, max_t, max_N_t, centroid, max_dist)
fig_num, _ = visualize2Dpartition(X_parts, y_set, t_set, fig_num, "levels = $(levels)")

boundary_pts = PK.generate_pseudo_obs(X, s_tree)
p_obs = boundary_pts[1:25:end] # definite discontinuity.
PLT.plot(map(xx -> first(xx), p_obs), map(xx -> last(xx), p_obs), "x", label = "pseudo obs")
PLT.legend()

PLT.xlabel("x1")
PLT.ylabel("x2")

####### oracle
f_oracle = xx -> sinc(norm(xx))
y1 = f_oracle.(X_parts[1])
y2 = f_oracle.(X_parts[2])
y = vcat(y1, y2)

#### patchwork kriging


# use the same square exponential bandwidth of 20 for both regions.
θs = [PK.SqExp(T(20)); PK.SqExp(T(20))]
ps = PK.GPParameters(X_parts, s_tree, θs)

K_XX = PK.compute_K_XX(ps)
K_Xδ = PK.compute_K_Xδ(p_obs, ps)
K_δδ = PK.compute_K_δδ(p_obs, ps)

# # sanity-check
xq = X[34]
#xq = randn(rng, T, D)
K_Xq = PK.compute_K_Xq(xq, ps)
K_δq = PK.compute_K_δq(xq, p_obs, ps)
K_qq = PK.compute_K_qq(xq, ps)

K_qX = K_Xq'
K_qδ = K_δq'

tmp = K_δδ \ (K_Xδ')
tmp1 = (K_qX - K_qδ * tmp)
tmp2 = (K_XX - K_Xδ * tmp)
mq = tmp1 * (tmp2 \ y)
vq = K_qq - dot(K_qδ, K_δδ \ K_δq) - dot(tmp1, tmp2 \ (tmp1'))

mq_gpr = K_qX * (K_XX \ y)

if xq == X[34]
    println("mq should be similar to fq, vq should be zero.")
end
@show f_oracle(xq), mq, mq_gpr
@show vq

# package up query routine.
model = PK.Model(p_obs, ps)
out = PK.query(xq, y, model, ps)
@show isapprox(out[1], mq), isapprox(out[2], vq)

### demonstrate continuity break.

function query_line(lb::T, ub::T, y_val::T, Nq::Integer, y, model, ps) where {T}
    xs = LinRange(lb, ub, Nq)

    mqs = Memory{T}(undef, Nq)
    vqs = Memory{T}(undef, Nq)
    xqs = Memory{Vector{T}}(undef, Nq)
    for i in eachindex(mqs, vqs, xs)
        xqs[i] = [xs[i]; y_val]
        mqs[i], vqs[i] = PK.query(xqs[i], y, model, ps)
    end
    return mqs, vqs, xqs
end

Nq = 100
y_val = T(2)
#y_val = zero(T)
lb, ub = -one(T), one(T)
mqs, vqs, xqs = query_line(lb, ub, y_val, Nq, y, model, ps)

xqs1 = map(xx -> first(xx), xqs)
xqs2 = map(xx -> last(xx), xqs)
PLT.scatter(xqs1, xqs2, marker = "x", label = "query points")
PLT.legend()

PLT.figure(fig_num)
fig_num += 1
PLT.plot(LinRange(lb, ub, Nq), mqs)
PLT.plot(LinRange(lb, ub, Nq), mqs, "*", label = "query points")
PLT.legend()
PLT.title("Patchwork kriging results")

# #

function query_line_rkhs(lb::T, ub::T, y_val::T, Nq::Integer, model) where {T}
    xs = LinRange(lb, ub, Nq)

    mqs = Memory{T}(undef, Nq)
    vqs = Memory{T}(undef, Nq)
    xqs = Memory{Vector{T}}(undef, Nq)
    for i in eachindex(mqs, vqs, xs)
        xqs[i] = [xs[i]; y_val]
        mqs[i], vqs[i] = PK.query_rkhs(xqs[i], model)
    end
    return mqs, vqs, xqs
end

y_rkhs = f_oracle.(X)
σ² = T(1.0e-8)
θ = θs[1]
model_rkhs = PK.RKHSModel(X, y_rkhs, σ², θ)
mqs_rkhs, vqs_rkhs, _ = query_line_rkhs(lb, ub, y_val, Nq, model_rkhs)

PLT.figure(fig_num)
fig_num += 1
PLT.plot(LinRange(lb, ub, Nq), mqs_rkhs)
PLT.plot(LinRange(lb, ub, Nq), mqs_rkhs, "*", label = "query points")
PLT.legend()
PLT.title("Conventional RKHS results")

fqs = f_oracle.(xqs)
PLT.figure(fig_num)
fig_num += 1
PLT.plot(LinRange(lb, ub, Nq), fqs)
PLT.title("Oracle function")


PLT.figure(fig_num)
fig_num += 1
PLT.plot(LinRange(lb, ub, Nq), mqs_rkhs, "*", label = "RKHS")
PLT.plot(LinRange(lb, ub, Nq), mqs, "*", label = "Patchwork kriging")
PLT.legend()
PLT.title("Query results, y = $(y_val)")

nothing
