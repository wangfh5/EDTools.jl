# Benchmark the two-point reduced density matrix of free spinless fermions with quspin

# For local development
using Pkg, Revise
Pkg.activate("/home/wangfh5/Projects/ed_nega/EDTools.jl/examples")
Pkg.develop(path="/home/wangfh5/Projects/ed_nega/EDTools.jl")

using EDTools
using LinearAlgebra, SparseArrays, Arpack
using DelimitedFiles

## parameters

L = 6
Ns = L
Np = Int(Ns/2)
Nflr = 1

t = 1.0
μ = 0.5

# we benchmark the two-point RDM of the following two sites
s1 = 2
s2 = 4
Asites = [s1, s2]

## ##################### using EDTools #####################
begin # calculation of ground state
    # ϕ = FBbasis(Ns, Np, :fermion, true; Nflr=Nflr) # number-conserved fermion basis
    ϕ = FBbasis(Ns, 0, :fermion, false; Nflr=Nflr) # Fock space, necessary for partial transpose
    H_hopping  = sum(hopping(i,mod1(i+1,Ns);flr=(1,1),α=-t, ϕ=ϕ) for i in 1:Ns) # hops right
    H_hopping += sum(hopping(i,mod1(i-1,Ns);flr=(1,1),α=-t, ϕ=ϕ) for i in 1:Ns) # hops left
    H_potential  = sum(densities(i;flr=1, α=-μ, ϕ=ϕ) for i in 1:Ns)
    H = H_hopping + H_potential
    # H_dense = H |> Matrix
    # ishermitian(H_dense)
    E_GS, psi_GS = eigs(H, nev=1, which=:SR)
end

# two-point RDM
rhoA_frompure,ϕA = rdm_generator(ϕ, Asites;newbasis=true)
ρA = rhoA_frompure(psi_GS)
# related quantities
Gr = (psi_GS'*hopping(s1,s2;ϕ=ϕ)*psi_GS)[1,1]
ns1 = (psi_GS'*densities(s1;ϕ=ϕ)*psi_GS)[1,1]
ns2 = (psi_GS'*densities(s2;ϕ=ϕ)*psi_GS)[1,1]
nnr = (psi_GS'*densities(s1;ϕ=ϕ)*densities(s2;ϕ=ϕ)*psi_GS)[1,1]


## ##################### using quspin #####################
begin # calculation ground state
    using PyCall
    np = pyimport("numpy")
    quspin = pyimport("quspin.operators")
    spinless_fermion_basis_1d = pyimport("quspin.basis").spinless_fermion_basis_1d

    basis = spinless_fermion_basis_1d(L)

    hop_right = [(-t, i, (i + 1) % L) for i in 0:L-1]
    hop_left = [(+t, i, (i + 1) % L) for i in 0:L-1]
    pot = [(-μ, i) for i in 0:L-1]
    # turn the list of tuples into a list of python lists (which quspin expects)
    hop_right_py = [pybuiltin("list")(x) for x in hop_right]
    hop_left_py = [pybuiltin("list")(x) for x in hop_left]
    pot_py = [pybuiltin("list")(x) for x in pot]

    static = pybuiltin("list")([
        pybuiltin("list")(["+-", hop_left_py]),  # hops left
        pybuiltin("list")(["-+", hop_right_py]), # hops right
        pybuiltin("list")(["n", pot_py]),        # on-site potential
    ])
    dynamic = pybuiltin("list")([])

    H = quspin.hamiltonian(static, dynamic, basis=basis, dtype=np.float64)
    E_GS_quspin, psi_GS_quspin  = H.eigsh(k=1,which="SA",maxiter=1E4)
end

# two-point RDM
sub_sys_A = np.asarray(Asites)
ρA_quspin = basis.partial_trace(psi_GS_quspin,sub_sys_A=sub_sys_A)
# related quantities
function compute_greens_function(i::Int, j::Int, ground_state, basis)
    op = [pybuiltin("list")((-t, i, j))]
    op_str = pybuiltin("list")([
        pybuiltin("list")(["+-", op]),
        ])
    op_dict = Dict("op" => op_str)
    operator = quspin.quantum_operator(op_dict, basis=basis, dtype=np.complex128, check_herm=false)
    greens_function = operator.expt_value(ground_state)
    return greens_function
end
function compute_average_number(i::Int, ground_state, basis)
    op_str = pybuiltin("list")([
        pybuiltin("list")(["+-", pybuiltin("list")([(1.0, i, i)])])
    ])
    op_dict = Dict("op" => op_str)
    operator = quspin.quantum_operator(op_dict, basis=basis, dtype=np.complex128, check_herm=false)
    average_number = operator.expt_value(ground_state)
    return average_number
end
function compute_nn_correlation(i::Int, j::Int, ground_state, basis)
    op_str = pybuiltin("list")([
        pybuiltin("list")(["+-+-", pybuiltin("list")([(1.0, i, i, j, j)])])
    ])
    op_dict = Dict("op" => op_str)
    operator = quspin.quantum_operator(op_dict, basis=basis, dtype=np.complex128, check_herm=false)
    nn_correlation = operator.expt_value(ground_state)
    return nn_correlation
end
Gr_quspin = compute_greens_function(s1, s2, psi_GS_quspin, basis)[1]
ns1_quspin = compute_average_number(s1, psi_GS_quspin, basis)[1]
ns2_quspin = compute_average_number(s2, psi_GS_quspin, basis)[1]
nnr_quspin = compute_nn_correlation(s1, s2, psi_GS_quspin, basis)[1]

## compare the two 4x4 reduced density matrices
function isequ(x, y; atol=1e-10, rtol=1e-5)
    if abs(x) < atol && abs(y) < atol
        return true
    else
        return isapprox(x, y; atol=atol, rtol=rtol)
    end
end

display(ρA)
display(ρA_quspin)
idmap = Dict(1=>4,2=>2,3=>3,4=>1)
for i in 1:4
    for j in 1:4
        lequ = isequ(ρA[i,j], ρA_quspin[idmap[i],idmap[j]])
        println("ρA[$i,$j] ≈ ρA_quspin[$(idmap[i]),$(idmap[j])] = ", lequ) 
        if !lequ
            println("ρA[$i,$j] = ", ρA[i,j])
            println("ρA_quspin[$(idmap[i]),$(idmap[j])] = ", ρA_quspin[idmap[i],idmap[j]])
        end
    end
end

## check theoretical relation between two-point correlation functions and RDM, e.g., ⟨01|ρA|10⟩ = ⟨c†_1 c_2⟩. (see https://arxiv.org/abs/2310.15273)
begin
    @show 1-ns1-ns2+nnr ≈ ρA[1,1]
    @show ns2-nnr ≈ ρA[2,2]
    @show ns1-nnr ≈ ρA[3,3]
    @show nnr ≈ ρA[4,4]
    @show isequ(Gr, ρA[2,3])

    @show 1-ns1_quspin-ns2_quspin+nnr_quspin ≈ ρA_quspin[idmap[1],idmap[1]]
    @show ns2_quspin-nnr_quspin ≈ ρA_quspin[idmap[2],idmap[2]]
    @show ns1_quspin-nnr_quspin ≈ ρA_quspin[idmap[3],idmap[3]]
    @show nnr_quspin ≈ ρA_quspin[idmap[4],idmap[4]]
    @show isequ(Gr_quspin, ρA_quspin[idmap[2],idmap[3]])
    # I currently think that there may be bug in quspin due to this discrepancy happening from time to time. 
    # But it can also be a difference in some conventions.
end

## check the analytical expressions of ground-state energy and the Green's function
function free_fermion_spectrum(L::Int64, t::Float64=1.0, μ::Float64=0.0)
    k = collect(0:L-1)
    E = -2*t*cos.(2π*k/L) .- μ
    sort!(E)
    # sum the negative energy states
    Eg = 0.0
    for i in 1:L
        if E[i] < 0
            Eg += E[i] 
        end
    end
    return E, Eg
end
function free_fermion_GreenFunction(L::Int64, t::Float64=1.0, μ::Float64=0.0, i::Int64=1, j::Int64=1)
    # ⟨c†i cj⟩
    H = zeros(ComplexF64, L, L)
    for k in 1:L
        H[k, mod1(k+1,L)] = -t
        H[mod1(k+1,L), k] = -t
        H[k, k] = -μ
    end
    E, V = eigen(H)
    G = 0.0 + 0.0im
    num = 0
    for k in 1:L
        if E[k] <= 0
            G += conj(V[i,k])*V[j,k]
            num += 1
        end
    end
    println("Number of negative energy states: ", num)
    return G
end
begin
    E, Eg = free_fermion_spectrum(L, t, μ)
    @show Eg ≈ E_GS[1]
    @show Eg ≈ E_GS_quspin[1]
    Gr_ana = free_fermion_GreenFunction(L, t, μ, s1, s2)
    @show isequ(Gr, Gr_ana)
    @show isequ(Gr_quspin, Gr_ana)
end
