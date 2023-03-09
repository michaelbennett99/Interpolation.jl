module Interpolation

using StaticArrays

struct Linear{N, T}
    x::SVector{N, T}
    y::SVector{N, T}
    x_min::T
    x_max::T
end

function linear_iterpolation(x::Vector{<:Real}, y::Vector{<:Real})
    if ! issorted(x)
        error("x must be sorted")
    end
    return Linear{length(x), eltype(x)}(x, y)
end

function (f::Linear)(x)
    if x < f.x_min || x > f.x_max
        error("x is out of range")
    end
    lub = findfirst(f.x .> x)
    glb = lub - 1
    @views grad = (f.y[lub] - f.y[glb]) / (f.x[lub] - f.x[glb])
    return f.y[glb] + grad * (x - f.x[glb])
end

end # module
