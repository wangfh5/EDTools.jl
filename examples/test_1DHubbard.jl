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
ishermitian(H_dense)
H_eigen, H_eigvec = eigen(H_dense)

## quspin calculation
using PyCall
np = pyimport("numpy")
quspin = pyimport("quspin.operators")
spinful_fermion_basis_1d = pyimport("quspin.basis").spinful_fermion_basis_1d

N_up = div(L, 2) + mod(L, 2)
N_down = div(L, 2)

basis = spinful_fermion_basis_1d(L, Nf=(N_up, N_down))

hop_right = [(-t, i, (i + 1) % L) for i in 0:L-1]
hop_left = [(+t, i, (i + 1) % L) for i in 0:L-1]
pot = [(-μ, i) for i in 0:L-1]
interact = [(U, i, i) for i in 0:L-1]
# turn the list of tuples into a list of python lists (which quspin expects)
hop_right_py = [pybuiltin("list")(x) for x in hop_right]
hop_left_py = [pybuiltin("list")(x) for x in hop_left]
pot_py = [pybuiltin("list")(x) for x in pot]
interact_py = [pybuiltin("list")(x) for x in interact]

static = pybuiltin("list")([
    pybuiltin("list")(["+-|", hop_left_py]),  # up hops left
    pybuiltin("list")(["-+|", hop_right_py]), # up hops right
    pybuiltin("list")(["|+-", hop_left_py]),  # down hops left
    pybuiltin("list")(["|-+", hop_right_py]), # down hops right
    pybuiltin("list")(["n|", pot_py]),        # up on-site potential
    pybuiltin("list")(["|n", pot_py]),        # down on-site potential
    pybuiltin("list")(["n|n", interact_py])   # up-down interaction
])
dynamic = pybuiltin("list")([])

H = quspin.hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
E_all, psi_all = H.eigh()

## compare the results
size(H_eigen) == size(E_all)
print(H_eigen)
print(sort(E_all))
for i in eachindex(H_eigen)
    if isapprox(sort(H_eigen)[i], sort(E_all)[i])
        println("Eigenvalue $i is the same")
    else
        println("Eigenvalue $i is different, with" * 
            " EDTools: $(sort(H_eigen)[i]), quspin: $(sort(E_all)[i])")
    end
end
