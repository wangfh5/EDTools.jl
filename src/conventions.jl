"""
    creation(i::Int; flr::Int=1, α=I, ϕ=nothing)
creation operator at site `i`, a†ᵢ
"""
function creation(i::Int; flr::Int=1, α=I, ϕ=nothing)
    return OpTerm(uuLadder(i; f=flr); α=α, ϕ=ϕ)
end
"""
    annihilation(i::Int; flr::Int=1, α=I, ϕ=nothing)
annihilation operator at site `i`, aᵢ
"""
function annihilation(i::Int; flr::Int=1, α=I, ϕ=nothing)
    return OpTerm(ddLadder(i; f=flr); α=α, ϕ=ϕ)
end
"""
    hopping(i::Int,j::Int; flr::NTuple{2,Int}=(1,1), α=I, ϕ=nothing)
hopping operator from site `j` to site `i`, a†ᵢaⱼ
"""
function hopping(i::Int,j::Int; flr::NTuple{2,Int}=(1,1), α=I, ϕ=nothing)
    return OpTerm(ddLadder(j; f=flr[2]),uuLadder(i; f=flr[1]); α=α, ϕ=ϕ)
end
"""
    pairing(i::Int,j::Int; flr::NTuple{2,Int}=(1,1), dag::Bool=true, α=I, ϕ=nothing)
pairing operator between site `i` and site `j`, Δ†ᵢⱼ = a†ᵢa†ⱼ (`dag`=true) or Δᵢⱼ = aᵢaⱼ (`dag`=false)
"""
function pairing(i::Int,j::Int; flr::NTuple{2,Int}=(1,1), dag::Bool=true, α=I, ϕ=nothing)
    if dag
        return OpTerm(uuLadder(j; f=flr[2]),uuLadder(i; f=flr[1]); α=α, ϕ=ϕ)
    else
        return OpTerm(ddLadder(j; f=flr[2]),ddLadder(i; f=flr[1]); α=α, ϕ=ϕ)
    end
end
"""
    densities(i::Int; flr::Int=1, α=I, ϕ=nothing)
densities operator at site `i`, nᵢ=a†ᵢaᵢ
"""
function densities(i::Int; flr::Int=1, α=I, ϕ=nothing)
    return OpTerm(ddLadder(i; f=flr),uuLadder(i; f=flr); α=α, ϕ=ϕ)
end
"""
    densities(i::Int,j::Int; flr::NTuple{2,Int}=(1,1), α=I, ϕ=nothing)
interaction operator between site `i` and site `j`, nᵢnⱼ
"""
function densities(i::Int,j::Int; flr::NTuple{2,Int}=(1,1), α=I, ϕ=nothing)
    return OpTerm(ddLadder(j; f=flr[2]),uuLadder(j; f=flr[2]),ddLadder(i; f=flr[1]), uuLadder(i; f=flr[1]); α=α, ϕ=ϕ)
end
"""
    hubbard(i::Int; α=I, ϕ=nothing)
Hubbard interaction operator at site `i` for two flavors, nᵢ↑nᵢ↓
"""
function hubbard(i::Int; α=I, ϕ=nothing)
    return OpTerm(ddLadder(i; f=2),uuLadder(i; f=2),ddLadder(i; f=1),uuLadder(i; f=1); α=α, ϕ=ϕ)
end