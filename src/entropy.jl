using RandomMatrices
"""
    density_matrix(ψ)
Turn a state vector into a pure-state density matrix.
"""
@inline function density_matrix(ψ)
    return Hermitian(ψ * ψ')
end
"""
    random_density_matrix(ϕ::FBbasis; pure::Bool=true, rank::Int=ϕ.Hdim)
Generate a random density matrix of dimension `Hdim` with the option of being pure or mixed.
"""
function random_density_matrix(ϕ::FBbasis; pure::Bool=true, rank::Int=ϕ.Hdim)
    Hdim = ϕ.Hdim
    if pure
        ψ = randn(ComplexF64, Hdim)
        ψ /= norm(ψ)
        return density_matrix(ψ)
    else
        U = rand(Haar(2),Hdim)
        Apos = U'*Diagonal( [rand(rank);zeros(Hdim-rank)] )*U
        return Hermitian(Apos/tr(Apos))
    end
end

"""
    rdm_generator(ϕ::FBbasis, Asites::Vector{Int})
Generate a function used to calculates the reduced density matrix of a subsystem A, 
according to the basis `ϕ` and the sites `Asites` in subsystem A.
"""
function rdm_generator(ϕ::FBbasis, Asites::Union{Nothing,Vector{Int}}; pure::Bool=true, newbasis::Bool=false)
    @assert isnothing(Asites) ? true : (Asites == unique(Asites))
    Nflr = ϕ.Nflr
    Ns = ϕ.Ns
    Hdim = ϕ.Hdim
    # setup subspace A and B
    A = falses(ϕ.Ndim)
    if ~isnothing(Asites)
        for i ∈ 1:Nflr
            A[Ns*(i-1) .+ Asites] .= true
        end
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
    if (ϕ.stype == :fermion)
        phases = []
    end
    pA = qA = pB = qB = 0
    @inbounds for p ∈ 1:Hdim, q ∈ 1:Hdim
        pA, pB = ψids[p]
        qA, qB = ψids[q]
        if pB == qB # ⟨ketsB|p⟩⟨q|ketsB⟩ ≠ 0
            push!(mapids, (p,q,ρids[pA,qA]))
            # for fermions, figure out the phase for move all the Asites to the left of the B sites
            # |B⋯B A₁ B⋯B A₂ B⋯B⟩ → (-1)^[(occupied B left to A₁)*A₁ + (occupied B left to A₂)*A₂] |A₁ A₂ B⋯B B⋯B⟩
            if ϕ.stype == :fermion
                phase = 1.0
                for ket in (ϕ.kets[p], ϕ.kets[q])
                    ltip = count = 0
                    for i in filter(x -> A[x], 1:ϕ.Ndim)
                        count += sum(ket[ltip+1:i-1])
                        phase *= (-1.0)^(count * ket[i])
                        ltip = i
                    end
                end
                push!(phases, phase)
            end
        end
    end
    function ρA_pure(ψ)::Matrix{Complex{Float64}}
        @assert length(ψ) == Hdim
        ρA = zeros(ComplexF64, AHdim, AHdim)
        it = 0
        for (i,j,k) ∈ mapids
            it += 1
            ρA[k] += ψ[i] * ψ[j]' * (ϕ.stype == :fermion ? phases[it] : 1.0)
        end
        return ρA
    end
    function ρA_mixed(ρ::AbstractMatrix{T})::Matrix{Complex{Float64}} where T<:Union{Float64, ComplexF64}
        @assert size(ρ,1) == Hdim
        ρA = zeros(ComplexF64, AHdim, AHdim)
        it = 0
        for (i,j,k) ∈ mapids
            it += 1
            ρA[k] += ρ[i,j] * (ϕ.stype == :fermion ? phases[it] : 1.0)
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
    vonNeumann_entropy(ρ::AbstractMatrix{T}) where T<:Union{Float64, ComplexF64}
Calculate the von Neumann entropy of a density matrix `ρ`. 
S = -tr(ρ*log(ρ))
"""
function vonNeumann_entropy(ρ::AbstractMatrix{T}) where T<:Union{Float64, ComplexF64}
    λ = eigvals(ρ)
    S = -sum(xlogx, λ)
    return S
end

"""
    Renyi_entropy(ρ::AbstractMatrix{T}, α::Float64) where T<:Union{Float64, ComplexF64}
Calculate the Renyi entropy of a density matrix `ρ` with the order `α`.
S(α) = 1/(1-α) * log(tr(ρ^α))
"""
function Renyi_entropy(ρ::AbstractMatrix{T}, α::Float64) where T<:Union{Float64, ComplexF64}
    if α == 1.0
        return vonNeumann_entropy(ρ)
    else
        λ = eigvals(ρ)
        λ = map(x -> ((abs(x) < 1e-15) ? 0.0 : x), λ)
        S = log(sum(λ.^α))
        S = S/(1-α)
        return S
    end
end