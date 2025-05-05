# SPDX-License-Identifier: AGPL-3.0
# Copyright © 2025 Roy Chih Chung Wang <roy.c.c.wang@proton.me>

### elemntary Kernels.
# # All kernels are normalized to have maximum value of 1
abstract type Kernel end

abstract type Stationary <: Kernel end
# interface requirement: evalkernel(), get_num_params, update_kernel!, get_params

abstract type SquaredDistanceKernel <: Stationary end
abstract type DistanceKernel <: Stationary end

# no error checking on length of inputs.
function devectorizel2normsq(x, y::AbstractVector{T}) where {T <: AbstractFloat}

    out = zero(T)
    for i in eachindex(x, y)
        out += (x[i] - y[i])^2
    end
    return out
end

function evalkernel(x1, x2::AbstractVector{T}, θ::SquaredDistanceKernel) where {T <: Number}

    # # the if-else might introduce divergence flow slow down.
    # if pointer(x1) == pointer(x2)
    #     return evalkernel(zero(T), θ)
    # end

    τ_sq = devectorizel2normsq(x1, x2) # same as norm(x1-x2)^2
    return evalkernel(τ_sq, θ)
end

# 1D version. For speed.
function evalkernel(x1, x2::T, θ::SquaredDistanceKernel) where {T <: Number}

    return evalkernel((x1 - x2)^2, θ)
end


function devectorizel2norm(x, y::AbstractVector{T}) where {T <: AbstractFloat}

    out = zero(T)
    for i in eachindex(x)
        out += (x[i] - y[i])^2
    end
    return sqrt(out)
end

function evalkernel(x1, x2::AbstractVector, θ::DistanceKernel)

    τ = devectorizel2norm(x1, x2) # same as norm(x1-x2)
    return evalkernel(τ, θ)
end

# 1D version. For speed.
function evalkernel(x1, x2::AbstractFloat, θ::DistanceKernel)

    return evalkernel(abs(x1 - x2), θ)
end


############# the different types of common kernels.


# Matern kernel.
# k(x,z) = b*(1 + a*norm(x-z))*exp(-a*norm(x-z))
struct Matern3Halfs{T <: AbstractFloat} <: DistanceKernel

    a::Memory{T} # single-entry, above 0.
    b::T

    function Matern3Halfs(a::T, b::T) where {T <: AbstractFloat}
        p = Memory{T}(undef, 1)
        p[begin] = a
        return new{T}(p, b)
    end
end

function evalkernel(τ::T, θ::Matern3Halfs) where {T <: Real}
    a, b = θ.a[1], θ.b[1]
    tmp = a * τ
    return b * (one(T) + tmp) * exp(-tmp)
end


# # Stationary kernels with squared distance as input.

# ## Eq. 4.19, pg 86, GPML 2006.
struct RQ{T <: AbstractFloat, ET <: Real} <: SquaredDistanceKernel
    #a::T # 1/(2*n*l^2) from GPML 2006.
    a::Memory{T}
    n::ET # exponent.

    function RQ(a::T, n::ET) where {T <: AbstractFloat, ET <: Real}
        p = Memory{T}(undef, 1)
        p[begin] = a
        return new{T, ET}(p, n)
    end
end


function evalkernel(τ_sq::T, θ::RQ) where {T <: Real}
    return (one(T) + θ.a[begin] * τ_sq)^(θ.n)
end

# ## Eq. 4.9, pg 83, GPML 2006.
struct SqExp{T <: AbstractFloat} <: SquaredDistanceKernel
    #a::T # 1/(2*l^2) from GPML 2006.
    a::Memory{T}

    function SqExp(a::T) where {T <: AbstractFloat}
        p = Memory{T}(undef, 1)
        p[begin] = a
        return new{T}(p)
    end
end

# This is the canonical kernel: 1D Gaussian.
function evalkernel(τ_sq::T, θ::SqExp{T}) where {T <: Real}

    return exp(-θ.a[begin] * τ_sq)
end


# # Updates and constructions by parsing supplied hyperparams.
# used in hyperparameter optimization or inference.

# for select kernels with scalar-valued hyperparameters.

#const SingleParameterKernels = Union{SqExp, WendlandSpline, RQ, Matern3Halfs}
const SingleParameterKernels = Union{SqExp, RQ, Matern3Halfs}

function update_kernel!(
        θ::SingleParameterKernels,
        p::Union{AbstractVector{T}, T}
    ) where {T <: AbstractFloat}

    θ.a[begin] = p[begin]
    return nothing
end

function get_num_params(::SingleParameterKernels)
    return 1
end

function get_params(θ::SingleParameterKernels)
    return θ.a
end

###### kernel matrices


function compute_K(
        X::AbstractVector{<:AbstractVector{T}},
        Z::AbstractVector{<:AbstractVector{T}},
        ps::Kernel,
    ) where {T <: AbstractFloat}

    K = zeros(T, length(X), length(Z))
    for j in eachindex(axes(K, 2), Z)
        for i in eachindex(axes(K, 1), X)
            K[i, j] = evalkernel(X[i], Z[j], ps)
        end
    end
    return K
end

function compute_K(
        X::AbstractVector{<:AbstractVector{T}},
        z::AbstractVector{T},
        ps::Kernel,
    ) where {T <: AbstractFloat}

    kq = zeros(T, length(X))
    for i in eachindex(kq, X)
        kq[i] = evalkernel(X[i], z, ps)
    end
    return kq
end
