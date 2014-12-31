module LazyMap

import Base.length

export LazyMapping, getindex, length

immutable LazyMapping
    arr
    f :: Function
end

getindex(lm :: LazyMapping, ix :: Int) = lm.f(lm.arr[ix])
length(lm :: LazyMapping) = length(lm.arr)

end