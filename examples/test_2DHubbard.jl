# 2D Hubbard model
# benchmark with QuantumLattices.jl and ExactDiagonalization.jl

using Pkg, Revise
Pkg.activate("/home/wangfh5/Projects/ed_nega/EDTools.jl")
using EDTools
using LinearAlgebra, SparseArrays, Arpack
using DelimitedFiles

## 1. using EDTools
L = (2,2)
Ns = prod(L)
Np = Int(Ns/2)
Nflr = 2

t = 1.0
U = 1.0
μ = 0.0

# lattice information
Lids = LinearIndices(L)
nn_vec = [(1,0),(-1,0),(0,1),(0,-1)]
# PBC
nn_list = find_nn_list(L, nn_vec; obc=false)
# OBC
# nn_list = find_nn_list(L, nn_vec; obc=true)

# ED using EDTools.jl
ϕ = FBbasis(Ns, Np, :fermion, true; Nflr=Nflr)
# ϕ = FBbasis(Ns, 0, :fermion, false; Nflr=Nflr)
H_hopping  = sum(hopping(i,j;flr=(1,1),α=-t, ϕ=ϕ) for i ∈ 1:Ns for j ∈ nn_list[i] )
H_hopping += sum(hopping(i,j;flr=(2,2),α=-t, ϕ=ϕ) for i ∈ 1:Ns for j ∈ nn_list[i] )
H_potential  = sum(densities(i;flr=1, α=-μ, ϕ=ϕ) for i ∈ 1:Ns)
H_potential += sum(densities(i;flr=2, α=-μ, ϕ=ϕ) for i ∈ 1:Ns)
H_hubbard = sum(hubbard(i; α=U, ϕ=ϕ) for i ∈ 1:Ns)
H = H_hopping + H_potential + H_hubbard
H_dense = H |> Matrix
ishermitian(H_dense)
E_GS, psi_GS = eigs(H, nev=1, which=:SR)

## 2. benchmark with QuantumLattices.jl and ExactDiagonalization.jl
# script from https://quantum-many-body.github.io/ExactDiagonalization.jl/dev/examples/HubbardModel/
begin
    using QuantumLattices
    using ExactDiagonalization
    using LinearAlgebra: eigen

    # define the unitcell of the square lattice
    unitcell = Lattice([0.0, 0.0]; name=:Square, vectors=[[1.0, 0.0], [0.0, 1.0]])

    # define a finite 2×2 cluster of the square lattice
    lattice = Lattice(unitcell, (2, 2), ('P', 'P')) # periodic boundary condition
    # lattice = Lattice(unitcell, (2, 2), ('O', 'O')) # open boundary condition

    # define the Hilbert space (single-orbital spin-1/2 complex fermion)
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site=1:length(lattice))

    # define the quantum number of the sub-Hilbert space in which the computation to be carried out
    # here the particle number is set to be `length(lattice)` and Sz is set to be 0
    quantumnumber = SpinfulParticle(length(lattice), 0)

    # define the terms, i.e. the nearest-neighbor hopping and the Hubbard interaction
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 1.0)

    # define the exact diagonalization algorithm for the Fermi Hubbard model
    ed = ED(lattice, hilbert, (t, U), quantumnumber)

    # find the ground state and its energy
    eigensystem = eigen(ed; nev=1)

    # Ground state energy should be -4.913259209075605
    print(eigensystem.values)
end