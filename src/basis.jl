struct FBbasis
    indexdict::Dict{BitVector,Int}  # Basis ket => index Dict, provide an ordering of basis kets
    kets::Vector{BitVector} # Basis kets
    Ns::Int                 # total site number
    Np::Int                 # occupied site number / total particles
    Hdim::Int               # the dimension of the Hilbert space
    ε::Int                  # sign, +1 for hard core boson / -1 for fermion
    stype::Symbol           # :hcboson / :fermion
    conservation::Bool      # particle number conservation
end

"""
    bvector(m::Int, n::Int)::BitVector
    bvector(bstring::String)::BitVector
convert the integer `m` to a bit vector `bv` of length `n` according to its binary value,
or convert the string `bstring` to a bit vector.
"""
function bvector(m::Int, n::Int)::BitVector
    bv = falses(n)
    @inbounds for i ∈ 1:n
        bv[i] = isodd(m)
        m >>= 1
    end
    return bv
end
function bvector(bstring::String)::BitVector
    return [x for x ∈ bstring] .== '1'
end

"""
    FBbasis(kets::Vector{BitVector}, stype::Symbol, conservation = false)
Construct a `FBbasis` object with the given basis kets. 
`stype`: `:hcboson` for hard core boson | `:fermion` for spinless fermion;
`conservation`: true for the subspace with fixed particle number, false for the full Fock space
"""
function FBbasis(kets::Vector{BitVector}, stype::Symbol, conservation::Bool = false)
    Ns = length(kets[1])
    Hdim = length(kets)
    Np = 0 # no explict particle number conservation
    indexdict = Dict(kets[i] => i for i ∈ eachindex(kets))
    ε = (stype == :fermion) ? (-1) : (+1)
    return FBbasis(indexdict, kets, Ns, Np, Hdim, ε, stype, conservation)
end

"""
    FBbasis(n::Int, k::Int, stype::Symbol, conservation::Bool = true)
Construct a `FBbasis` object with the given site number `Ns` and particle number `Np`.
`stype`: `:hcboson` for hard core boson | `:fermion` for spinless fermion;
"""
function FBbasis(Ns::Int, Np::Int, stype::Symbol, conservation::Bool=true)
    @assert stype == :hcboson || stype == :fermion
    @assert (Np == 0) ⊻ conservation
    # --------------------------------------------
    kets = if conservation
        BitVector[
            let v = falses(Ns)
                v[q] .= true
                v
            end for q ∈ CoolLexCombinations(Ns, Np)
        ]
    else
        BitVector[bvector(m, Ns) for m ∈ 0:(2^Ns-1)]
    end
    sort!(kets,by=count)
    indexdict = Dict(kets[i] => i for i ∈ eachindex(kets))
    Hdim = length(kets)
    ε = (stype == :fermion) ? (-1) : (+1)
    return FBbasis(indexdict, kets, Ns, Np, Hdim, ε, stype, conservation)
end

"""
    wf_from_ket(ϕ::FBbasis, k::BitVector)::Vector{ComplexF64}
Convert a basis ket (BitVector `k`) to a wavefunction (complex vector `ψ`) in the many-body basis `ϕ`.
"""
function wf_from_ket(ϕ::FBbasis, k::BitVector)::Vector{ComplexF64}
    @assert length(k) == ϕ.Ns
    q = ϕ.indexdict[k]
    ψ = zeros(ComplexF64, ϕ.Hdim)
    ψ[q] = 1
    return ψ
end
wf_from_ket(ϕ::FBbasis, k::Vector{Bool}) = wf_from_ket(ϕ::FBbasis, BitVector(k))
wf_from_ket(ϕ::FBbasis, s::String) = wf_from_ket(ϕ, bvector(s))