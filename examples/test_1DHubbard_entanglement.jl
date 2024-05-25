# Calculate the Renyi entropy and negativity of 1D Hubbard model

using Pkg, Revise
Pkg.activate("/home/wangfh5/Projects/ed_nega/EDTools.jl")
using EDTools
using LinearAlgebra, SparseArrays, Arpack
using DelimitedFiles

L = 6
Ns = L
Np = Int(Ns/2)
Nflr = 2

t = 1.0
U = 1.0
μ = U/2

## ED the ground state
ϕ = FBbasis(Ns, Np, :fermion, true; Nflr=Nflr) # number-conserved fermion basis
# ϕ = FBbasis(Ns, 0, :fermion, false; Nflr=Nflr) # Fock space, necessary for partial transpose
H_hopping  = sum(hopping(i,mod1(i+1,Ns);flr=(1,1),α=-t, ϕ=ϕ) for i in 1:Ns) # up hops right
H_hopping += sum(hopping(i,mod1(i-1,Ns);flr=(1,1),α=-t, ϕ=ϕ) for i in 1:Ns) # up hops left
H_hopping += sum(hopping(i,mod1(i+1,Ns);flr=(2,2),α=-t, ϕ=ϕ) for i in 1:Ns) # down hops right
H_hopping += sum(hopping(i,mod1(i-1,Ns);flr=(2,2),α=-t, ϕ=ϕ) for i in 1:Ns) # down hops left
H_potential  = sum(densities(i;flr=1, α=-μ, ϕ=ϕ) for i in 1:Ns)
H_potential += sum(densities(i;flr=2, α=-μ, ϕ=ϕ) for i in 1:Ns)
H_hubbard = sum(hubbard(i; α=U, ϕ=ϕ) for i in 1:Ns)
H = H_hopping + H_potential + H_hubbard
# H_dense = H |> Matrix
# ishermitian(H_dense)
E_GS, psi_GS = eigs(H, nev=1, which=:SR)
# writedlm("GS_L$(L)_U$(U)_jl.txt", psi_GS)

## Partial trace and Renyi entropy
# read the ground state from file
# psi_GS = readdlm("GS_L$(L)_U$(U)_jl.txt")
EE = zeros(ComplexF64, Ns+1)
EE[1] = 0.0
file_path = "entropy.out"
io = open(file_path, "w")
for LA ∈ 1:L
    # LA = 8
    rhoA_frompure = rdm_generator(ϕ, collect(1:LA))
    ρA = rhoA_frompure(psi_GS)
    EE[LA+1] = Renyi_entropy(ρA, 2.0)
    write(io, "LA = $LA, Renyi entropy = $(EE[LA+1])\n")
    flush(io)
end
close(io)

## Partial transpose and negativity
psi_GS = ketinFock(psi_GS,ϕ)
EN2 = zeros(ComplexF64, Ns+1)
file_path = "negativity.out"
io = open(file_path, "w")
for LA ∈ 0:L
    Asites = (LA == 0) ? nothing : collect(1:LA)
    ρTB_frompure = ptdm_generator(ϕ, Asites)
    ρTB = ρTB_frompure(psi_GS)
    EN2[LA+1] = Renyi_negativity(ρTB,2.0)
    write(io, "LA = $LA, Negativity = $(EN2[LA+1])\n")
    flush(io)
end
close(io)
