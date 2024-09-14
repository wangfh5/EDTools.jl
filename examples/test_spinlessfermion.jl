# For local development
using Pkg, Revise
Pkg.activate("/home/wangfh5/Projects/ed_nega/EDTools.jl/examples")
Pkg.develop(path="/home/wangfh5/Projects/ed_nega/EDTools.jl")

using EDTools

L = 4
Ns = L
Np = 0

## test basis and ladder operators
ϕ = FBbasis(Ns, 2, :fermion, true)
ϕ = FBbasis(Ns, Np, :fermion, false)
# Define the BitVectors
v = ϕ.kets[3]
vt = BitVector(undef, length(v))
# Convert to complex vector
ψ = wf_from_ket(ϕ, v)
# Define the Ladder objects
ladderU3 = uuLadder(3)
ladderD2 = ddLadder(2)
# Define a tuple of Ladder objects (assuming Ladder is a defined type)
S = (ladderD2, ladderU3)  # These are Ladder objects
S = (ladderU3, ladderD2)  # These are Ladder objects
Ssorted, sign = normal_order(S)
# Apply ladders
result = apply_Ladders!(vt, v, S, ϕ.ε)
println("Original vt: ", v)
println("Coefficient: ", result)
println("Modified vt: ", vt)

## test the matrix operations of operator terms
# Define the operator terms
SopU3 = OpTerm(ladderU3; ϕ=ϕ)
SopD2 = OpTerm(ladderD2; ϕ=ϕ)
Sop32 = OpTerm(ladderU3,ladderD2; ϕ=ϕ)
Sop23 = OpTerm(Ssorted; ϕ=ϕ)
Matrix(SopU3)
Matrix(SopD2)
Matrix(Sop23)
Matrix(Sop23) == Matrix(SopU3) * Matrix(SopD2)
Matrix(Sop32) == Matrix(SopD2) * Matrix(SopU3)
# Matrix operations
α = 2.0
Sopt = α * Sop23
Sopt = Sop23 * α
ψ2 = Sopt * ψ
Matrix(Sop32) == SopD2 * SopU3
Matrix(adjoint(adjoint(Sop32))) == Matrix(Sop32)
Sop_p = SopU3 + SopD2
Sop_m = SopU3 - SopD2

## test some built-in operators
Matrix(SopU3) == Matrix(creation(3; ϕ=ϕ))
Matrix(SopD2) == Matrix(annihilation(2; ϕ=ϕ))
Matrix(Sop23) == Matrix(hopping(3,2; ϕ=ϕ))
Matrix(pairing(3,2; ϕ=ϕ)) == creation(3; ϕ=ϕ) * creation(2; ϕ=ϕ)
Matrix(densities(3; ϕ=ϕ)) == creation(3; ϕ=ϕ) * annihilation(3; ϕ=ϕ)
Matrix(densities(3,2; ϕ=ϕ)) == creation(3; ϕ=ϕ) * annihilation(3; ϕ=ϕ) * creation(2; ϕ=ϕ) * annihilation(2; ϕ=ϕ)