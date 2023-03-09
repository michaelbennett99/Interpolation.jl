module Interpolation

using StaticArrays

struct Linear{N, T}
    x::SVector{N, T}
    y::SVector{N, T}
    x_min::T
    x_max::T
end

function linear_iterpolation(x, y)
    return Linear{length(x), eltype(x)}(x, y)
end

function (f::Linear)(x)
    if x < f.x_min || x > f.x_max
        error("x is out of range")
    end
    for i in 1:length(f.x)-1
        if x < f.x[i+1]
            @views grad = (f.y[i+1] - f.y[i]) / (f.x[i+1] - f.x[i])
            @views return f.y[i] + grad * (x - f.x[i])
        end
    end
end

end # module
