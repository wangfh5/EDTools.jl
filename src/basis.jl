"""
    struct FBbasis
A struct to store the basis information of the Hilbert space for doing exact diagonalization.
The convention of the basis kets: 
1. The structure of a basis ket is blocked according to flavor number `Nflr`, 
   e.g., for Nflr=2, the basis ket is like `|a₁↑ a₂↑ ... aₙ↑ a₁↓ a₂↓ ... aₙ↓⟩`.
   This convention is important for the construction of operators. 
2. The order of the bases may be not important since there is a dictionary `FBbasis.indexdict` to store the index of each basis ket.
#TODO: Different Np for different flavors (spin imbalance). 
"""
struct FBbasis
    indexdict::Dict{BitVector,Int}  # Basis ket => index Dict, provide an ordering of basis kets
    kets::Vector{BitVector} # Basis kets
    Ns::Int                 # total site number
    Nflr::Int               # flavor number (e.g., different spin, 1 for spin up, 2 for spin down)
    Np::Int                 # occupied site number / total particles per flavor
    Ndim::Int               # total degree of freedom, Ns*Nflr
    Hdim::Int               # the dimension of the Hilbert space; for Fock space Hdim = 2^Ndim
    ε::Int                  # sign, +1 for hard core boson / -1 for fermion
    stype::Symbol           # :hcboson / :fermion
    conservation::Bool      # particle number conservation
    parity::Int             # 0 for even, 1 for odd, -1 for mixed
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
    FBbasis(kets::Vector{BitVector}, stype::Symbol, conservation::Bool=false, parity::Int=-1; Nflr::Int=1)
Construct a `FBbasis` object with the given basis kets. 
- `stype`: `:hcboson` for hard core boson | `:fermion` for spinless fermion;
- `conservation`: true for the subspace with fixed particle number, false for the full Fock space;
- `parity`: 0 for even, 1 for odd, -1 for mixed. 
"""
function FBbasis(kets::Vector{BitVector}, stype::Symbol, conservation::Bool=false, parity::Int=-1; Nflr::Int=1)
    @assert stype == :hcboson || stype == :fermion "The stype should be :hcboson or :fermion!"
    @assert parity ∈ [-1,0,1] "The parity should be -1 for mixed, 0 for even, 1 for odd!"
    Ndim = length(kets[1])
    Ns = Int(Ndim/Nflr)
    Hdim = length(kets)
    if conservation
        Np = count(kets[1])
        @assert all(count(k) == Np for k ∈ kets) "The particle number of the input kets is not conserved!"
        parity = (Np % 2 == 0) ? 0 : 1
    else
        Np = 0
    end
    indexdict = Dict(kets[i] => i for i ∈ eachindex(kets))
    ε = (stype == :fermion) ? (-1) : (+1)
    return FBbasis(indexdict, kets, Ns, Nflr, Np, Ndim, Hdim, ε, stype, conservation, parity)
end

"""
    FBbasis(Ns::Int, Np::Int, stype::Symbol, conservation::Bool = true)
Construct a `FBbasis` object with the given site number `Ns` and particle number `Np`.
- `stype`: `:hcboson` for hard core boson | `:fermion` for spinless fermion;
- `conservation`: true for the subspace with fixed particle number, false for the full Fock space;
- `parity`: 0 for even, 1 for odd, -1 for mixed. Only has effect when `conservation=false`.
"""
function FBbasis(Ns::Int, Np::Int, stype::Symbol, conservation::Bool=true, parity::Int=-1; Nflr::Int = 1)
    @assert stype == :hcboson || stype == :fermion "The stype should be :hcboson or :fermion!"
    @assert (Np == 0) ⊻ conservation "The particle number should be zero if the conservation is false!"
    @assert parity ∈ [-1,0,1] "The parity should be -1 for mixed, 0 for even, 1 for odd!"
    # --------------------------------------------
    Ndim = Ns*Nflr
    if conservation
        kets_flr = BitVector[
                    let v = falses(Ns)
                        v[q] .= true
                        v
                    end for q ∈ CoolLexCombinations(Ns, Np)
                    ]
        if Nflr > 1
            all_combinations = Iterators.product(ntuple(_ -> kets_flr, Nflr)...)
            kets = []
            for combination in all_combinations
                combined_bitvector = vcat(combination...)
                push!(kets, combined_bitvector)
            end
        else
            kets = kets_flr
        end
        parity = (Np*Nflr % 2 == 0) ? 0 : 1
    else
        if parity == -1
            kets = BitVector[bvector(m, Ndim) for m ∈ 0:(2^Ndim-1)]
        else
            kets = BitVector[bvector(m, Ndim) for m ∈ 0:(2^Ndim-1) if count(bvector(m, Ndim)) % 2 == parity]
        end
    end
    sort!(kets,by=count)
    indexdict = Dict(kets[i] => i for i ∈ eachindex(kets))
    Hdim = length(kets)
    ε = (stype == :fermion) ? (-1) : (+1)
    return FBbasis(indexdict, kets, Ns, Nflr, Np, Ndim, Hdim, ε, stype, conservation, parity)
end

"""
    wf_from_ket(ϕ::FBbasis, k::BitVector)::Vector{ComplexF64}
Convert a basis ket (BitVector `k`) to a wavefunction (complex vector `ψ`) in the many-body basis `ϕ`.
"""
function wf_from_ket(ϕ::FBbasis, k::BitVector)::Vector{ComplexF64}
    @assert length(k) == ϕ.Ndim
    q = ϕ.indexdict[k]
    ψ = zeros(ComplexF64, ϕ.Hdim)
    ψ[q] = 1
    return ψ
end
wf_from_ket(ϕ::FBbasis, k::Vector{Bool}) = wf_from_ket(ϕ, BitVector(k))
wf_from_ket(ϕ::FBbasis, s::String) = wf_from_ket(ϕ, bvector(s))