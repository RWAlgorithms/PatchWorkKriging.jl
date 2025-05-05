# SPDX-License-Identifier: AGPL-3.0
# Copyright © 2025 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

# B is the set of pseudo observations.
# X is the set of observations.
# q is one query point.
# implemented only for the single boundary case.

struct GPParameters{AV <: AbstractVector, RT, KT <: Kernel}
    X_parts::Vector{AV}
    tree::RT
    θs::Memory{KT}

    function GPParameters(X::Vector{AV}, tree::RT, θs::AbstractVector{KT}) where {AV, RT, KT}
        return new{AV, RT, KT}(X, tree, Memory{KT}(θs))
    end
end

function compute_K_XX(ps::GPParameters)
    # just the block diagonal of K_XX. The others are 0 by Assumption 1 in Park's manuscript.
    Ks = collect(
        compute_K(ps.X_parts[k], ps.X_parts[k], ps.θs[k])
            for k in eachindex(ps.X_parts, ps.θs)
    )
    N = sum(length(x) for x in ps.X_parts)
    K = zeros(eltype(Ks[1]), N, N)

    K11 = compute_K(ps.X_parts[1], ps.X_parts[1], ps.θs[1])
    K22 = compute_K(ps.X_parts[2], ps.X_parts[2], ps.θs[2])

    K0 = compute_K(ps.X_parts[1], ps.X_parts[2], ps.θs[1])
    fill!(K0, 0)

    return [K11 K0; K0' K22]
end


# Eq. 3, 5, Park 2018.
function compute_K_Xδ(p_obs::AbstractVector, ps::GPParameters)

    K1 = compute_K(ps.X_parts[1], p_obs, ps.θs[1])
    K2 = compute_K(ps.X_parts[2], p_obs, ps.θs[2])
    return vcat(K1, -K2)
end


# only implement the case for the same boundary; k==u, l==v, Eq. 4, Park 2018.
function compute_K_δδ(p_obs::AbstractVector, ps::GPParameters)
    return sum(compute_K(p_obs, p_obs, θ) for θ in ps.θs)
end

function compute_K_Xq(xq::AbstractVector{T}, ps::GPParameters) where {T <: Real}
    i = ST.find_partition(xq, ps.tree)

    kq1 = zeros(T, length(ps.X_parts[1]))
    if i == 1
        kq1 = compute_K(ps.X_parts[1], xq, ps.θs[1])
    end

    kq2 = zeros(T, length(ps.X_parts[2]))
    if i == 2
        kq2 = compute_K(ps.X_parts[2], xq, ps.θs[2])
    end

    return vcat(kq1, kq2)
end

# Eq. 3, Park 2018.
function compute_K_δq(xq::AbstractVector, p_obs::AbstractVector, ps::GPParameters)

    j = ST.find_partition(xq, ps.tree)
    if j == 1
        return compute_K(p_obs, xq, ps.θs[1])
    end

    kq = compute_K(p_obs, xq, ps.θs[2])
    return -kq
end

function compute_K_qq(xq::AbstractVector{T}, ps::GPParameters) where {T <: Real}
    i = ST.find_partition(xq, ps.tree)
    return evalkernel(xq, xq, ps.θs[i])
end

struct Model{T}
    K_XX::Matrix{T}
    K_Xδ::Matrix{T}
    K_δδ::Matrix{T}
    p_obs::Vector{Vector{T}}

    function Model(p_obs::Vector{Vector{T}}, ps::GPParameters) where {T}
        K_XX = compute_K_XX(ps)
        K_Xδ = compute_K_Xδ(p_obs, ps)
        K_δδ = compute_K_δδ(p_obs, ps)
        return new{T}(K_XX, K_Xδ, K_δδ, copy(p_obs))
    end
end

function query(xq::AbstractVector, y::AbstractVector, model::Model, ps::GPParameters)
    K_XX, K_Xδ, K_δδ = model.K_XX, model.K_Xδ, model.K_δδ

    K_Xq = compute_K_Xq(xq, ps)
    K_δq = compute_K_δq(xq, model.p_obs, ps)
    K_qq = compute_K_qq(xq, ps)

    K_qX = K_Xq'
    K_qδ = K_δq'

    tmp = K_δδ \ (K_Xδ')
    tmp1 = (K_qX - K_qδ * tmp)
    tmp2 = (K_XX - K_Xδ * tmp)
    mq = tmp1 * (tmp2 \ y)
    vq = K_qq - dot(K_qδ, K_δδ \ K_δq) - dot(tmp1, tmp2 \ (tmp1'))

    return mq, vq
end


struct RKHSModel{T, KT <: Kernel}
    X::Vector{Vector{T}}
    K_plus_σ²::Matrix{T}
    c::Memory{T}
    θ::KT

    function RKHSModel(X::Vector{Vector{T}}, y::AbstractVector, σ²::T, θ::Kernel) where {T}
        K = compute_K(X, X, θ)
        for j in eachindex(axes(K, 2), axes(K, 1))
            K[j, j] += σ²
        end
        return new{T, typeof(θ)}(X, K, Memory{T}(K \ y), θ)
    end
end

function query_rkhs(xq::AbstractVector, model::RKHSModel)
    model.X, model.K_plus_σ²
    K_Xq = compute_K(model.X, xq, model.θ)
    mq = dot(K_Xq, model.c)
    vq = evalkernel(xq, xq, model.θ) - dot(K_Xq, model.K_plus_σ² \ K_Xq)
    return mq, vq
end
