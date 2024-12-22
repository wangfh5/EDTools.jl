"""
    ketinFock(old_ket::AbstractMatrix{T},ϕ::FBbasis) where T<:Union{Float64, ComplexF64}
Convert a ket in the given basis `ϕ` (which may be a subspace of the Fock space, e.g., number conserving or with parity) 
to the corresponding ket in the Fock space.
"""
function ketinFock(old_ket::AbstractMatrix{T},ϕ::FBbasis) where T<:Union{Float64, ComplexF64}
    @assert length(old_ket) == ϕ.Hdim
    Fockϕ = FBbasis(ϕ.Ns, 0, ϕ.stype, false; Nflr=ϕ.Nflr)
    new_ket = zeros(ComplexF64, Fockϕ.Hdim)
    @inbounds for k ∈ ϕ.kets
        new_ket[Fockϕ.indexdict[k]] = old_ket[ϕ.indexdict[k]]
    end
    return new_ket
end

"""
    rhoinFock(ρ::AbstractMatrix{T},ϕ::FBbasis) where T<:Union{Float64, ComplexF64}
Convert a density matrix in the given basis `ϕ` (which may be a subspace of the Fock space, e.g., number conserving or with parity) 
to the corresponding density matrix in the Fock space.
"""
function rhoinFock(ρ::AbstractMatrix{T},ϕ::FBbasis) where T<:Union{Float64, ComplexF64}
    @assert size(ρ) == (ϕ.Hdim, ϕ.Hdim)
    Fockϕ = FBbasis(ϕ.Ns, 0, ϕ.stype, false; Nflr=ϕ.Nflr)
    new_ρ = zeros(ComplexF64, Fockϕ.Hdim, Fockϕ.Hdim)
    @inbounds for k ∈ ϕ.kets, l ∈ ϕ.kets
        new_ρ[Fockϕ.indexdict[k], Fockϕ.indexdict[l]] = ρ[ϕ.indexdict[k], ϕ.indexdict[l]]
    end
    return new_ρ
end


"""
    ptdm_generator(ϕ::FBbasis, Asites::Vector{Int}; pure::Bool=true, twisted::Bool=false)
Generate a function used to calculates the partially tranposed density matrix of a subsystem A, or the partial tranpose of ρ w.r.t. subsystem B. 
`ϕ`: the basis of the full Hilbert space.
`Asites`: an array of indices of subsystem A, starting from 1. 
`pure`: If true, the returned function takes a ket in the full Hilbert space as input; otherwise, it takes a density matrix.
`twisted`: If true, the twisted partial transpose is considered for fermions. Otherwise, use untwisted partial transpose.
`BPT`: If true, the conventional bosonic partial transpose is chosen, even for fermions.
"""
function ptdm_generator(ϕ::FBbasis, Asites::Union{Nothing,Vector{Int}}; pure::Bool=true, twisted::Bool=false, BPT::Bool=false)
    @assert isnothing(Asites) ? true : (Asites == unique(Asites))
    if (ϕ.Hdim < 2^(ϕ.Ndim)) 
        @warn "The partial transpose should be perfromed in the Fock space! We first change to the Fock space basis and then generate the ptdm function."
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
    if (ϕ.stype == :fermion && !BPT)
        phases = []
    end
    ρids = LinearIndices((Hdim,Hdim))
    # |p⟩⟨q|=|pA,pB⟩⟨qA,qB| => |i⟩⟨j|=|pA,qB⟩⟨qA,pB|
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
        if (ϕ.stype == :fermion && !BPT)
            τ_pA = sum(ketpA)
            τ_qA = sum(ketqA)
            τ_pB = sum(ketpB)
            τ_qB = sum(ketqB)
            parity_factor = mod(τ_pB + τ_qB, 2)/2 + (τ_pA + τ_qA)*(τ_pB + τ_qB)
            if twisted
                parity_factor += τ_pB
            end
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
            if (ϕ.stype == :fermion && !BPT)
                ρTB[k] *= (-1+0im)^phases[it]
            end
        end
        return ρTB
    end
    function ρTB_mixed(ρ::AbstractMatrix{T})::Matrix{Complex{Float64}} where T<:Union{Float64, ComplexF64}
        @assert size(ρ) == (Hdim,Hdim) "The input state should be in the Fock space! (You can use function `rhoinFock` to convert first.)"
        ρTB = zeros(ComplexF64, Hdim, Hdim)
        it = 0
        for (i,j,k) ∈ mapids
            it += 1
            ρTB[k] = ρ[i,j]
            if (ϕ.stype == :fermion && !BPT)
                ρTB[k] *= (-1+0im)^phases[it]
            end
        end
        return ρTB
    end
    return pure ? ρTB_pure : ρTB_mixed
end

"""
    log_negativity(ρ::AbstractMatrix{T}) where T<:Union{Float64, ComplexF64}
Calculate the logarithmic negativity of a density matrix `ρ`.
E_N = log(tr(sqrt(ρ†ρ)))
"""
function log_negativity(ρ::AbstractMatrix{T}) where T<:Union{Float64, ComplexF64}
    q = svdvals(ρ)
    return log(sum(q))
end

"""
    Renyi_negativity(ρ::AbstractMatrix{T}, α::Float64) where T<:Union{Float64, ComplexF64}
Calculate the Renyi negativity of a density matrix `ρ` with the order `α`.
E_N(α) = 1/(1-α) * log(tr(ρ^α))
"""
@inline function Renyi_negativity(ρ::AbstractMatrix{T}, α::Float64;lcheck::Bool=false) where T<:Union{Float64, ComplexF64}
    if α == 1.0
        return log_negativity(ρ)
    else
        λ = @inline eigvals(ρ)
        # check if the eigenvalues are real
        if all(abs.(imag.(λ)) .< 1e-10 )
            λ = real.(λ)
        else
            print("Not all the eigenvalues of rho_FPT are real!\n")
        end
        if lcheck
            print("Sum of eigenvalues: $(sum(λ))\n")
            print("Sum of eigenvalues^2: $(sum(λ.^2.0))\n")
            print("Sum of eigenvalues^3: $(sum(λ.^3.0))\n")
            print("Sum of eigenvalues^4: $(sum(λ.^4.0))\n")
        end
        trρα = sum(λ.^α)
        if abs(imag(trρα)) > 1e-10 
            @warn "The trace of ρ^$(α) is not real!\n"
        elseif abs(real(trρα)) < 1e-10
            @warn "The trace of ρ^$(α) is nearly zero, trρα = $(trρα), the log can not be done. \n"
            trρα = complex(real(trρα), 0)
        elseif real(trρα) < 0
            @warn "The trace of ρ^$(α) is negative, trρα = $(trρα), the log can not be done. \n"
            trρα = complex(real(trρα), 0)
        else
            trρα = real(trρα)
        end
        Renyi_neg = log(trρα)/(1-α)
        # if !(abs(imag(Renyi_neg)) < 1e-10)
        #     print("Renyi negativity $(Renyi_neg) is not real!\n")
        #     # output the eigenvalues (sort by real part)
        #     λ = sort(λ, by=abs)
        #     for i in eachindex(λ)
        #         print("Eigenvalue $(@sprintf("%4i", i)): $(@sprintf("%16.8e", real(λ[i]))) $(@sprintf("%16.8e", imag(λ[i])))\n")
        #     end
        #     print("Sum of eigenvalues: $(sum(λ))")
        # end
        return Renyi_neg
    end
end
@inline function Renyi_negativity(ρ::AbstractMatrix{T}, αs::Vector{Float64};lcheck::Bool=false) where T<:Union{Float64, ComplexF64}
    λ = @inline eigvals(ρ)
    # check if the eigenvalues are real
    if all(abs.(imag.(λ)) .< 1e-10 )
        λ = real.(λ)
    else
        print("Not all the eigenvalues of rho_FPT are real!\n")
    end
    if lcheck
        # output the eigenvalues
        print_eigenvalues_with_degeneracy(λ)
        for rnk in 1:16
            print("Sum of eigenvalues^$(rnk): $(sum(λ.^rnk))\n")
        end
    end
    Renyi_negs = zeros(ComplexF64, length(αs))
    for (i,α) ∈ enumerate(αs)
        if α == 1.0
            Renyi_negs[i] = log_negativity(ρ)
        else
            trρα = sum(λ.^α)
            if abs(imag(trρα)) > 1e-10 
                @warn "The trace of ρ^$(α) is not real!\n"
            elseif abs(real(trρα)) < 1e-10
                @warn "The trace of ρ^$(α) is nearly zero, trρα = $(trρα), the log can not be done. \n"
                trρα = complex(real(trρα), 0)
            elseif real(trρα) < 0
                @warn "The trace of ρ^$(α) is negative, trρα = $(trρα), the log can not be done. \n"
                trρα = complex(real(trρα), 0)
            else
                trρα = real(trρα)
            end
            Renyi_negs[i] = log(trρα)/(1-α)
        end
    end
    return Renyi_negs
end

function print_eigenvalues_with_degeneracy(eigenvalues::Array{T, 1}; precision::Int = 6) where T<:Union{Float64, ComplexF64}
    eigenvalue_dict = Dict{T, Int}()
    for eigval in eigenvalues
        # 按照指定精度舍入本征值作为字典的键
        key = round(eigval, digits=precision)
        eigenvalue_dict[key] = get(eigenvalue_dict, key, 0) + 1
    end

    # for (eigenvalue, degeneracy) in eigenvalue_dict
    #     println("Eigenvalue: $(@sprintf("%16.8e", real(eigenvalue))) $(@sprintf("%16.8e", imag(eigenvalue))), Degeneracy: $(degeneracy)")
    # end
    # sort by absolute value
    eigenvalue_dict = sort(collect(eigenvalue_dict), by=x->abs(real(x[1])))
    for (eigenvalue, degeneracy) in eigenvalue_dict
        println("Eigenvalue: $(@sprintf("%16.8e", real(eigenvalue))) $(@sprintf("%16.8e", imag(eigenvalue))), Degeneracy: $(degeneracy)")
    end
end