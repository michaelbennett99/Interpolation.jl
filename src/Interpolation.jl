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
    k::SVector{N, T}
end

function cubic_interpolation(x::Vector{<:Real}, y::Vector{<:Real})
    if ! issorted(x)
        error("x must be sorted")
    end
    n = length(x)
    dl = zeros(n-1)
    d = zeros(n)
    du = zeros(n-1)
    r = zeros(n)

    dl[1] = 1/(x[2] - x[1])
    du[1] = 1/(x[2] - x[1])
    d[1] = 2/(x[2] - x[1])
    d[n] = 2/(x[n] - x[n-1])
    r[1] = 3 * (y[2] - y[1]) / (x[2] - x[1])^2
    r[n] = 3 * (y[n] - y[n-1]) / (x[n] - x[n-1])^2

    for i ∈ 2:n-1
        dx1 = x[i] - x[i-1]
        dx2 = x[i+1] - x[i]
        dy1 = y[i] - y[i-1]
        dy2 = y[i+1] - y[i]
        d[i] = 2 * (1/dx1 + 1/dx2)
        dl[i] = 1/dx2
        du[i] = 1/dx2
        r[i] = 3 * (dy1/dx1^2 + dy2/dx2^2)
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
        try
            glb = lub - 1
        catch
            println(x)
            println(f.x)
            error("x is out of range")
        @views t = (x - f.x[glb]) / (f.x[lub] - f.x[glb])
        @views a = f.k[glb] * (f.x[lub] - f.x[glb]) - (f.y[lub] - f.y[glb])
        @views b = -f.k[lub] * (f.x[lub] - f.x[glb]) + (f.y[lub] - f.y[glb])
        @views y_hat = (((1 - t) * f.y[glb]) + (t * f.y[lub])
            + (t * (1 - t) * (a * (1 - t) + b * t)))
        return y_hat
    end
end

end # module
