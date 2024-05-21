abstract type Qop end
"""
    mutable struct OpTerm{N} <: Qop
Mutable struct definition with a type parameter `N`.
"""
mutable struct OpTerm{N} <: Qop
    ladders::NTuple{N,Ladder}             # Operator representation: Tuple of N Ladder objects (note that the action of `ladders` starts with `ladders[1]` and ends with `ladders[N]`)
    α::UniformScaling                     # Scaling coefficient
    ϕ::Union{Nothing,FBbasis}             # Optional basis
    table::Union{Nothing,AbstractVector}  # Optional table representation
    matrix::Union{Nothing,AbstractMatrix} # Optional sparse matrix representation
end
"""
    OpTerm(ladders...; α=I, ϕ=nothing)
`OpTerm` object constructor, with input arguments: 
- a series of `Ladder` objects or a Tuple of `Ladder` objects `ladders`, 
- (optional) a scaling coefficient `α`, 
- and (optional) an optional basis `ϕ`.
"""
function OpTerm(ladders... ;α=I, ϕ=nothing)
    return OpTerm((ladders...,),α,ϕ,nothing,nothing)
end
function OpTerm(ladders::NTuple{N,Ladder};α=I, ϕ=nothing) where N
    return OpTerm(ladders,α,ϕ,nothing,nothing)
end

"""
    table!(op::OpTerm)
Construct the table representation of the operator `op` in terms of its operator expression `ladders` and basis `ϕ`.
Act on each ket in the basis `ϕ` with the operator expression `ladders`, 
and then record the (the index of result ket `i`, the index of original ket `j`, the coefficient `s` after acting). 
Mathmatically, each element `(i, j, s)` provides us with `s` = ⟨`i`|`op`|`j`⟩, i.e., the matrix element of `op` in the basis `ϕ`.
"""
function table!(op::OpTerm)
    @assert ~isnothing(op.ϕ) "The basis is not assigned!"
    if ~isnothing(op.table) return nothing end
    ϕ = op.ϕ
    ε::Int = ϕ.ε
    s::Int = 0
    i::Int = 0
    vt = falses(ϕ.Ndim)
    op.table = Tuple{Int,Int,Int}[]
    for j ∈ eachindex(ϕ.kets)
        s = apply_Ladders!(vt, ϕ.kets[j], op.ladders, ε, nflr=ϕ.Nflr)
        if s ≠ 0
            i = get(ϕ.indexdict, vt, 0)
            if i ≠ 0 # the final state exists in the interested Hilbert space
                push!(op.table, (i, j, s))
            end
        end
    end
    return nothing
end

"""
    matrix!(op::OpTerm)
Construct the matrix representation of the operator `op` in terms of its table representation. 
"""
function matrix!(op::OpTerm)
    if ~isnothing(op.matrix) return nothing end
    table!(op)
    N = op.ϕ.Hdim
    op.matrix = sparse(
        [x[1] for x ∈ op.table], [x[2] for x ∈ op.table],
        [x[3] for x ∈ op.table], N,N)
    # if S.M |> isdiag
    #     S.M = S.M |> diag |> Vector |> Diagonal
    # end
    return nothing
end

## Overload the matrix operations 

import Base.Matrix
import LinearAlgebra.Matrix
"""
    Matrix(op::OpTerm)
Obtain the `Matrix` variable of the operator `op` by constructing the matrix representation of `op` and then multiplying it by the scaling coefficient `α`.
"""
function Matrix(op::OpTerm)
    matrix!(op)
    return op.α * op.matrix
end

import Base.*
"""
    *(op::OpTerm, ψ::AbstractVector)
Apply the operator `op` to the vector `ψ`, return a vector. 
"""
function *(op::OpTerm, ψ::AbstractVector)
    matrix!(op)
    return op.α * (op.matrix * ψ)
end
"""
    *(op1::OpTerm, op2::OpTerm)
Multiply two operators `op1` and `op2`, return the matrix of the new operator.
#TODO the return type should be struct `OpTerm`
#TODO check the basis of input OpTerms. 
"""
function *(op1::OpTerm, op2::OpTerm)
    matrix!(op1)
    matrix!(op2)
    return (op1.α * op2.α) * (op1.matrix * op2.matrix)
end
"""
    *(op::OpTerm, A::Union{Number, UniformScaling, AbstractMatrix})
Multiply the operator `op` by a scalar/matrix `A`, return the matrix of the new operator.
"""
function *(S::OpTerm, A::Union{Number, UniformScaling, AbstractMatrix})
    matrix!(S)
    return (S.α * S.matrix) * A
end
"""
    *(A::Union{Number, UniformScaling, AbstractMatrix}, op::OpTerm)
Multiply a scalar/matrix `A` by the operator `op`, return the matrix of the new operator.
"""
function *(A::Union{Number,UniformScaling,AbstractMatrix}, S::OpTerm)
    matrix!(S)
    return A * (S.α * S.matrix)
end

import Base.adjoint
import LinearAlgebra.adjoint
"""
    adjoint(op::OpTerm)
Return the adjoint (Hermitian conjugate) operator of `op`.
"""
function adjoint(s::OpTerm)::OpTerm
    s_conj = OpTerm(
        conj.(reverse(s.ladders)),
        conj(s.α),
        s.ϕ,
        nothing,nothing
    )
    return s_conj
end

import Base.+
import Base.-
"""
    +(S1::OpTerm, S2::OpTerm)
Add two operators `S1` and `S2`, return the matrix of the new operator.
#TODO the return type should be struct `OpTerm`
"""
function +(S1::OpTerm, S2::OpTerm)
    matrix!(S1)
    matrix!(S2)
    return S1.α * S1.matrix + S2.α * S2.matrix
end
"""
    +(S::OpTerm, A::Union{Number, UniformScaling, AbstractMatrix})
Add the operator `S` and a matrix `A`, return the matrix of the new operator.
"""
function +(S::OpTerm, A::Union{AbstractMatrix})
    matrix!(S)
    return S.α * S.matrix + A
end
"""
    +(A::Union{Number, UniformScaling, AbstractMatrix}, S::OpTerm)
Add a matrix `A` and the operator `S`, return the matrix of the new operator.
"""
function +(A::Union{AbstractMatrix}, S::OpTerm)
    matrix!(S)
    return A + S.α * S.matrix
end
"""
    -(S1::OpTerm, S2::OpTerm)
Subtract two operators `S1` and `S2`, return the matrix of the new operator.
#TODO the return type should be struct `OpTerm`
"""
function -(S1::OpTerm, S2::OpTerm)
    matrix!(S1)
    matrix!(S2)
    return S1.α * S1.matrix - S2.α * S2.matrix
end
"""
    -(S::OpTerm, A::Union{Number, UniformScaling, AbstractMatrix})
Subtract the operator `S` and a matrix `A`, return the matrix of the new operator.
"""
function -(S::OpTerm, A::Union{AbstractMatrix})
    matrix!(S)
    return S.α * S.matrix - A
end
"""
    -(A::Union{Number, UniformScaling, AbstractMatrix}, S::OpTerm)
Subtract a matrix `A` and the operator `S`, return the matrix of the new operator.
"""
function -(A::Union{AbstractMatrix}, S::OpTerm)
    matrix!(S)
    return A - S.α * S.matrix
end