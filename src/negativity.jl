"""
    ketinFock(old_ket::Matrix,ϕ::FBbasis)
Convert a ket in the given basis `ϕ` to the corresponding ket in the Fock space.
"""
function ketinFock(old_ket::Matrix,ϕ::FBbasis)
    @assert length(old_ket) == ϕ.Hdim
    Fockϕ = FBbasis(ϕ.Ns, 0, ϕ.stype, false; Nflr=ϕ.Nflr)
    new_ket = zeros(ComplexF64, Fockϕ.Hdim)
    @inbounds for k ∈ ϕ.kets
        new_ket[Fockϕ.indexdict[k]] = old_ket[ϕ.indexdict[k]]
    end
    return new_ket
end

"""
    rhoinFock(ρ::Matrix{Complex{Float64}},ϕ::FBbasis)
Convert a density matrix in the given basis `ϕ` to the corresponding density matrix in the Fock space.
"""
function rhoinFock(ρ::Matrix{Complex{Float64}},ϕ::FBbasis)
    @assert size(ρ) == (ϕ.Hdim, ϕ.Hdim)
    Fockϕ = FBbasis(ϕ.Ns, 0, ϕ.stype, false; Nflr=ϕ.Nflr)
    new_ρ = zeros(ComplexF64, Fockϕ.Hdim, Fockϕ.Hdim)
    @inbounds for k ∈ ϕ.kets, l ∈ ϕ.kets
        new_ρ[Fockϕ.indexdict[k], Fockϕ.indexdict[l]] = ρ[ϕ.indexdict[k], ϕ.indexdict[l]]
    end
    return new_ρ
end


"""
    ptdm_generator(ϕ::FBbasis, Asites::Vector{Int}; pure::Bool=true)
Generate a function used to calculates the partially tranposed density matrix of a subsystem A, or the partial tranpose of ρ w.r.t. subsystem B. 
`ϕ`: the basis of the full Hilbert space.
`Asites`: an array of indices of subsystem A, starting from 1. 
"""
function ptdm_generator(ϕ::FBbasis, Asites::Union{Nothing,Vector{Int}}; pure::Bool=true)
    @assert isnothing(Asites) ? true : (Asites == unique(Asites))
    if (ϕ.Hdim < 2^(ϕ.Ndim)) 
        print("The partial transpose should be perfromed in the Fock space! We first change to the Fock space basis and then generate the ptdm function.\n")
        ϕ = FBbasis(ϕ.Ns, 0, ϕ.stype, false; Nflr=ϕ.Nflr)
    end
    Nflr = ϕ.Nflr
    Ns = ϕ.Ns
    Hdim = ϕ.Hdim
    # setup the masks for subspace A and B
    A = falses(ϕ.Ndim)
    if ~isnothing(Asites)
        for i ∈ 1:Nflr
            A[Ns*(i-1) .+ Asites] .= true
        end
    end
    B = (A .== false)
    # map the |ket⟩⟨bra| composite indices before and after partial transpose w.r.t. subsystem B
    mapids = Tuple{Int,Int,Int}[]
    if ϕ.stype == :fermion
        phases = []
    end
    ρids = LinearIndices((Hdim,Hdim))
    # (p,q)=(pApB,qAqB) => (i,j)=(pAqB,qApB)
    @inbounds for p ∈ 1:Hdim, q ∈ 1:Hdim
        ketp = ϕ.kets[p]
        ketq = ϕ.kets[q]
        ketpA = ketp[A]
        ketqA = ketq[A]
        ketpB = ketp[B]
        ketqB = ketq[B]
        keti = copy(ketp)
        ketj = copy(ketq)
        keti[B] = ketqB
        ketj[B] = ketpB
        i = ϕ.indexdict[keti]
        j = ϕ.indexdict[ketj]
        push!(mapids, (p,q,ρids[i,j]))

        # calculate phase factor
        if ϕ.stype == :fermion
            τ_pA = sum(ketpA)
            τ_qA = sum(ketqA)
            τ_pB = sum(ketpB)
            τ_qB = sum(ketqB)
            parity_factor = mod(τ_pB + τ_qB, 2)/2 + (τ_pA + τ_qA)*(τ_pB + τ_qB)
            push!(phases, parity_factor)
        end

        # if ((p+1) % 5000 == 0 && (q+1) % 5000 == 0) || (p==0 && q==0)
        #     print("p: $(i);q: $(j)\n")
        # end
    end
    function ρTB_pure(ψ)::Matrix{Complex{Float64}}
        @assert length(ψ) == Hdim "The input state should be in the Fock space! (You can use function `ketinFock` to convert first.)"
        ρTB = zeros(ComplexF64, Hdim, Hdim)
        it = 0
        for (i,j,k) ∈ mapids
            it += 1
            ρTB[k] = ψ[i] * ψ[j]'
            if ϕ.stype == :fermion
                ρTB[k] *= (-1+0im)^phases[it]
            end
        end
        return ρTB
    end
    function ρTB_mixed(ρ::Matrix{Complex{Float64}})::Matrix{Complex{Float64}}
        @assert size(ρ) == (Hdim,Hdim) "The input state should be in the Fock space! (You can use function `rhoinFock` to convert first.)"
        ρTB = zeros(ComplexF64, Hdim, Hdim)
        it = 0
        for (i,j,k) ∈ mapids
            it += 1
            ρTB[k] = ρ[i,j]
            if ϕ.stype == :fermion
                ρTB[k] *= (-1+0im)^phases[it]
            end
        end
        return ρTB
    end
    return pure ? ρTB_pure : ρTB_mixed
end

"""
    log_negativity(ρ::Matrix{Complex{Float64}})
Calculate the logarithmic negativity of a density matrix `ρ`.
E_N = log(tr(sqrt(ρ†ρ)))
"""
function log_negativity(ρ::Matrix{Complex{Float64}})
    q = svdvals(ρ)
    return log(sum(q))
end

"""
    Renyi_negativity(ρ::Matrix{Complex{Float64}}, α::Float64)
Calculate the Renyi negativity of a density matrix `ρ` with the order `α`.
E_N(α) = 1/(1-α) * log(tr(ρ^α))
"""
@inline function Renyi_negativity(ρ::Matrix{Complex{Float64}}, α::Float64)
    if α == 1.0
        return log_negativity(ρ)
    else
        λ = @inline eigvals(ρ)
        # check if the eigenvalues are real
        if all(isapprox.(imag.(λ), 0))
            λ = real.(λ)
        else
            print("Not all the eigenvalues of rho_FPT are real!")
        end
        Renyi_neg = log(sum(λ.^α))
        Renyi_neg = Renyi_neg/(1-α)
        # if !(imag(Renyi_neg) < 1e-10)
        #     print("Renyi negativity $(Renyi_neg) is not real!")
        #     # output the eigenvalues (sort by real part)
        #     λ = sort(λ, by=abs)
        #     for i in eachindex(λ)
        #         print("Eigenvalue $(@sprintf("%4i", i)): $(@sprintf("%16.8e", real(λ[i]))) $(@sprintf("%16.8e", imag(λ[i])))")
        #     end
        #     print("Sum of eigenvalues: $(sum(λ))")
        # end
        return Renyi_neg
    end
end