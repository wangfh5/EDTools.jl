"""
    density_matrix(ψ)
Turn a state vector into a pure-state density matrix.
"""
@inline function density_matrix(ψ)
    return Hermitian(ψ * ψ')
end

"""
    rdm_generator(ϕ::FBbasis, Asites::Vector{Int})
Generate a function used to calculates the reduced density matrix of a subsystem A, 
according to the basis `ϕ` and the sites `Asites` in subsystem A.
"""
function rdm_generator(ϕ::FBbasis, Asites::Vector{Int}; pure::Bool=true, newbasis::Bool=false)
    @assert Asites == unique(Asites)
    Nflr = ϕ.Nflr
    Ns = ϕ.Ns
    Hdim = ϕ.Hdim
    # setup subspace A and B
    A = falses(ϕ.Ndim)
    for i ∈ 1:Nflr
        A[Ns*(i-1) .+ Asites] .= true
    end
    B = (A .== false)
    ketsA = unique(k[A] for k ∈ ϕ.kets)
    ketsB = unique(k[B] for k ∈ ϕ.kets)
    idA = Dict(ketsA[i] => i for i ∈ eachindex(ketsA))
    idB = Dict(ketsB[i] => i for i ∈ eachindex(ketsB))
    AHdim = length(ketsA)
    # map the |ket⟩⟨bra| composite indices to subspace A
    ψids = Tuple{Int,Int}[(idA[k[A]], idB[k[B]]) for k ∈ ϕ.kets]
    ρids = LinearIndices((AHdim,AHdim))
    mapids = Tuple{Int,Int,Int}[]
    pA = qA = pB = qB = 0
    @inbounds for p ∈ 1:Hdim, q ∈ 1:Hdim
        pA, pB = ψids[p]
        qA, qB = ψids[q]
        if pB == qB # ⟨ketsB|p⟩⟨q|ketsB⟩ ≠ 0
            push!(mapids, (p,q,ρids[pA,qA]))
        end
    end
    function ρA_pure(ψ)::Matrix{Complex{Float64}}
        @assert length(ψ) == Hdim
        ρA = zeros(ComplexF64, AHdim, AHdim)
        for (i,j,k) ∈ mapids
            ρA[k] += ψ[i] * ψ[j]'
        end
        return ρA
    end
    function ρA_mixed(ρ::Matrix{Complex{Float64}})::Matrix{Complex{Float64}}
        @assert size(ρ,1) == Hdim
        ρA = zeros(ComplexF64, AHdim, AHdim)
        for (i,j,k) ∈ mapids
            ρA[k] += ρ[i,j]
        end
        return ρA
    end
    if newbasis
        ϕA = FBbasis(ketsA, ϕ.stype, ϕ.conservation; Nflr=Nflr)
        if pure
            return ρA_pure, ϕA
        else
            return ρA_mixed, ϕA
        end
    else
        return pure ? ρA_pure : ρA_mixed
    end
end

xlogx(x::Real) = x>0 ? x*log(x) : 0.0
"""
    vonNeumann_entropy(ρ::Matrix{Complex{Float64}})
Calculate the von Neumann entropy of a density matrix `ρ`. 
S = -tr(ρ*log(ρ))
"""
function vonNeumann_entropy(ρ::Matrix{Complex{Float64}})
    λ = eigvals(ρ)
    S = -sum(xlogx, λ)
    return S
end

"""
    Renyi_entropy(ρ::Matrix{Complex{Float64}}, α::Float64)
Calculate the Renyi entropy of a density matrix `ρ` with the order `α`.
S(α) = 1/(1-α) * log(tr(ρ^α))
"""
function Renyi_entropy(ρ::Matrix{Complex{Float64}}, α::Float64)
    if α == 1.0
        return vonNeumann_entropy(ρ)
    else
        λ = eigvals(ρ)
        S = log(sum(λ.^α))
        S = S/(1-α)
        return S
    end
end