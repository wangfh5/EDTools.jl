"""
    creation(i::Int; α=I, ϕ=nothing)
creation operator at site `i`, a†ᵢ
"""
function creation(i::Int; α=I, ϕ=nothing)
    return OpTerm(uuLadder(i); α=α, ϕ=ϕ)
end
"""
    annihilation(i::Int; α=I, ϕ=nothing)
annihilation operator at site `i`, aᵢ
"""
function annihilation(i::Int; α=I, ϕ=nothing)
    return OpTerm(ddLadder(i); α=α, ϕ=ϕ)
end
"""
    hopping(i::Int,j::Int; α=I, ϕ=nothing)
hopping operator from site `j` to site `i`, a†ᵢaⱼ
"""
function hopping(i::Int,j::Int; α=I, ϕ=nothing)
    return OpTerm(ddLadder(j),uuLadder(i); α=α, ϕ=ϕ)
end
"""
    pairing(i::Int,j::Int; α=I, ϕ=nothing)
pairing operator between site `i` and site `j`, Δ†ᵢⱼ = a†ᵢa†ⱼ
"""
function pairing(i::Int,j::Int; α=I, ϕ=nothing)
    return OpTerm(uuLadder(j),uuLadder(i); α=α, ϕ=ϕ)
end
"""
    densities(i::Int; α=I, ϕ=nothing)
densities operator at site `i`, nᵢ=a†ᵢaᵢ
"""
function densities(i::Int; α=I, ϕ=nothing)
    return OpTerm(ddLadder(i),uuLadder(i); α=α, ϕ=ϕ)
end
"""
    densities(i::Int,j::Int; α=I, ϕ=nothing)
interaction operator between site `i` and site `j`, nᵢnⱼ
"""
function densities(i::Int,j::Int; α=I, ϕ=nothing)
    return OpTerm(ddLadder(j),uuLadder(j),ddLadder(i), uuLadder(i); α=α, ϕ=ϕ)
end