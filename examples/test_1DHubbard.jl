# Benchmark 1D Hubbard model with quspin

using Pkg, Revise
Pkg.activate("/home/wangfh5/Projects/ed_nega/EDTools.jl")
using EDTools
using LinearAlgebra, SparseArrays
using Arpack

L = 4
Ns = L
Np = Int(Ns/2)
Nflr = 2

t = 1.0
U = 1.0
μ = U/2

## EDTools calculations

ϕ = FBbasis(Ns, Np, :fermion, true; Nflr=Nflr)
H_hopping  = sum(hopping(i,mod1(i+1,Ns);flr=(1,1),α=-t, ϕ=ϕ) for i in 1:Ns) # up hops right
H_hopping += sum(hopping(i,mod1(i-1,Ns);flr=(1,1),α=-t, ϕ=ϕ) for i in 1:Ns) # up hops left
H_hopping += sum(hopping(i,mod1(i+1,Ns);flr=(2,2),α=-t, ϕ=ϕ) for i in 1:Ns) # down hops right
H_hopping += sum(hopping(i,mod1(i-1,Ns);flr=(2,2),α=-t, ϕ=ϕ) for i in 1:Ns) # down hops left
H_potential  = sum(densities(i;flr=1, α=-μ, ϕ=ϕ) for i in 1:Ns)
H_potential += sum(densities(i;flr=2, α=-μ, ϕ=ϕ) for i in 1:Ns)
H_hubbard = sum(hubbard(i; α=U, ϕ=ϕ) for i in 1:Ns)
H = H_hopping + H_potential + H_hubbard
H_dense = H |> Matrix
H_eigen, H_eigvec = eigen(H_dense)

## quspin calculation
using PyCall

py"""
import numpy as np
from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spinful_fermion_basis_1d

# basis = spinful_fermion_basis_1d($L) # the whole Fock space
N_up = $(L)//2 + $(L) % 2 # number of fermions with spin up
N_down = $(L)//2 # number of fermions with spin down
basis = spinful_fermion_basis_1d($L,Nf=(N_up,N_down)) # the subspace with fixed particle number

# define site-coupling lists
hop_right=[[-$t,i,(i+1)%$L] for i in range($L)] #PBC
hop_left= [[+$t,i,(i+1)%$L] for i in range($L)] #PBC 
pot=[[-$μ,i] for i in range($L)] # -\mu \sum_j n_{j \sigma}
interact=[[$U,i,i] for i in range($L)] # U \sum_j n_{j,up} n_{j,down}
# define static and dynamic lists
static=[
        ['+-|',hop_left],  # up hops left
        ['-+|',hop_right], # up hops right
        ['|+-',hop_left],  # down hops left
        ['|-+',hop_right], # down hops right
        ['n|',pot],        # up on-site potention
        ['|n',pot],        # down on-site potention
        ['n|n',interact]   # up-down interaction
                                ]
dynamic=[]
# build Hamiltonian
# no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
# H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64,**no_checks)
H=hamiltonian(static,dynamic,basis=basis,dtype=np.float64)

# diagonalise H
E_all, psi_all=H.eigh() 
"""

## compare the results
size(H_eigen) == size(py"E_all")
print(H_eigen)
print(sort(py"E_all"))
for i in eachindex(H_eigen)
    if isapprox(sort(H_eigen)[i], sort(py"E_all")[i])
        println("Eigenvalue $i is the same")
    else
        println("Eigenvalue $i is different, with" * 
            " EDTools: $(sort(H_eigen)[i]), quspin: $(sort(py"E_all")[i])")
    end
end
