module Interpolation

using StaticArrays, LinearAlgebra

export linear_interpolation, cubic_interpolation

struct Linear{N, T}
    x::SVector{N, T}
    y::SVector{N, T}
    x_min::T
    x_max::T
end

function linear_interpolation(x::Vector{<:Real}, y::Vector{<:Real})
    if ! issorted(x)
        error("x must be sorted")
    end
    return Linear{length(x), eltype(x)}(x, y, x[1], x[end])
end

function (f::Linear)(x)
    if x < f.x_min || x > f.x_max
        error("x is out of range")
    elseif x ∈ f.x
        return f.y[findfirst(f.x .== x)]
    else
        lub = findfirst(f.x .> x)
        glb = lub - 1
        @views grad = (f.y[lub] - f.y[glb]) / (f.x[lub] - f.x[glb])
        return f.y[glb] + grad * (x - f.x[glb])
    end
end

struct Spline{N, T}
    x::SVector{N, T}
    y::SVector{N, T}
    ydp::SVector{N, T}
end

function cubic_interpolation(x::Vector{<:Real}, y::Vector{<:Real})
    if ! issorted(x)
        error("x must be sorted")
    end
    n = length(x)
    dl = zeros(n)
    d = zeros(n)
    du = zeros(n)
    r = zeros(n)

    dl[1] = 0
    dl[n] = -1
    d[1] = 1
    d[n] = 1
    du[1] = -1
    du[n] = 0
    r[1] = 0
    r[n] = 0

    for i ∈ 2:n-1
        dl[i] = (x[i] - x[i-1]) / 6
        d[i] = (x[i+1] - x[i-1]) / 3
        du[i] = (x[i+1] - x[i]) / 6
        r[i] = (y[i+1] - y[i])/(x[i+1] - x[i])
            - (y[i] - y[i-1])/(x[i] - x[i-1])
    end

    vdp = Tridiagonal(dl, d, du) \ r
    return Spline{n, eltype(x)}(x, y, vdp)
end

function (f::Spline)(x)
    if x < f.x[1] || x > f.x[end]
        error("x is out of range")
    elseif x ∈ f.x
        return f.y[findfirst(f.x .== x)]
    else
        lub = findfirst(f.x .> x)
        glb = lub - 1
        @views h = f.x[lub] - f.x[glb]
        @views a = (f.x[lub] - x) / h
        @views b = (x - f.x[glb]) / h

        @views y_hat = (
            (a * f.y[glb] + b * f.y[lub])
            + ((a^3 - a) * f.ydp[glb] + (b^3 - b) * f.ydp[lub]) * h^2 / 6
        )
        return y_hat
    end
end

end # module
