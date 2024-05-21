"""
    struct Ladder(site::Int, flag::Bool)
A ladder operator at site `site` and flavor `flr` with creation/annihilation flag `flag`.
`flag`: true for creation / false for annihilation
"""
struct Ladder
    site::Int
    flr::Int
    flag::Bool
end
Ladder(sf::Tuple{Int,Bool}) = Ladder(sf[1], sf[2], sf[3])
uuLadder(s::Int; f::Int=1) = Ladder(s, f, true)
ddLadder(s::Int; f::Int=1) = Ladder(s, f, false)

"""
    Ladders(ladder...)::Tuple{Ladder}
`Ladders` takes a variable number of `Ladder` instances and returns them as a tuple.
"""
Ladders(ladder...) = (ladder...,)

"""
    Base.conj(c::Ladder)::Ladder
Hermitian conjugate a `Ladder` operator. 
Overloading the `conj` function for `Ladder` to return a new `Ladder` with the `flag` negated. 
"""
Base.conj(c::Ladder) = Ladder(c.site, c.flr, ~c.flag)

"""
    Base.isless(a1::Ladder, a2::Ladder)::Bool
Overloading the `isless` function for `Ladder` to compare two `Ladder` operators. 
This is used to normally order an arbitary operator. 
1. Creation operators > annihilation operators. 
2. Compare the flavor index for the same kind of operators.
3. Compare the site index for the same kind of operators with the same flavor.
e.g., for nflr=1, a⁺ₙ > a⁺ₙ₋₁ > ... > a⁺₁ > aₙ > aₙ₋₁ > ... > a₁
"""
function Base.isless(a1::Ladder, a2::Ladder)
    if a1.flag == a2.flag
        if a1.flr == a2.flr
            return isless(a1.site, a2.site)
        else
            return isless(a1.flr, a2.flr)
        end
    else
        return isless(a1.flag, a2.flag)
    end
end

"""
    normal_order!(S::NTuple{N,Ladder})::NTuple{N,Ladder} where N
Sort a tuple of `Ladder` operators in the normal order (ascending order since the action of S starts with S[1])
and return the sign caused by the permutation (important for fermions).
"""
@inline function normal_order(S::NTuple{N,Ladder})::Tuple{NTuple{N,Ladder}, Int} where N
    Sv = collect(S)
    sign = normal_order!(Sv)
    S_sorted = Tuple(Sv)
    return S_sorted, sign
end
@inline function normal_order!(S::Vector{Ladder})::Int
    perm = sortperm(S)
    S[:] = S[perm]
    sign = levicivita(perm)
    return sign
end


"""
    apply_Ladder!(v::BitVector, a::Ladder, ε::Int)::Int
Apply a ladder operator `a` to a bit vector `v`, modify the vector `v` and return the coefficient.
"""
function apply_Ladder!(v::BitVector, a::Ladder, ε::Int; nflr::Int=1)::Int
    ns = Int(length(v)/nflr)
    aindex = Int((a.flr-1)*ns + a.site)
    if a.flag == v[aindex]
        return 0
    else
        v[aindex] = a.flag
        return ε^count(v[1:(aindex-1)])
    end
end

"""
    apply_Ladders!(vt::BitVector, v::BitVector, S::NTuple{N,Ladder}, ε::Int)::Int where N
Apply a series of ladder operators `S` to a bit vector `v`, modify the vector `vt` and return the coefficient.
The application starts from the first element of `S` and ends at the last element.
"""
function apply_Ladders!(vt::BitVector, v::BitVector, S::NTuple{N,Ladder}, ε::Int; nflr::Int=1)::Int where N
    s = 1
    copy!(vt, v)
    # start with the first element of S
    # So S is expected to be in the ascending order of operators (reversed normal order) 
    # so that annihilation operators are applied first.
    for a ∈ S
        s *= apply_Ladder!(vt, a, ε, nflr=nflr)
        if s == 0
            return 0
        end
    end
    return s
end

# 一个算符总是系列升降算符的乘积
# struct LadderOp
#     seq_original::Vector{Ladder}     # 原始序列
#     # seq_proper::Vector{Ladder}       # 正规序列
#     # perm::Vector{Int}                # 排序perm
#     # s::Int                           # 正规排序相对原始序列的附带符号，非常重要
#     α::UniformScaling # scaling系数
#     M::Union{Nothing,AbstractMatrix} # 矩阵形式
#     IJV::Union{Nothing,AbstractVector}
#     LadderSequence(seq_original, seq_proper, perm, s, α) = new(
#         seq_original, seq_proper, perm, s, α, nothing, nothing
#     )
# end

# function LadderSequence(seq_original, seq_proper, perm, s, α)
#     LadderSequence(seq_original, seq_proper, perm, s, , nothing)
# end

# function LadderSequence(seq_original::Vector{Ladder}, α::UniformScaling = I)
#     perm = sortperm(seq_original)
#     seq_proper = seq_original[perm]
#     s = levicivita(perm)
#     return LadderSequence(seq_original, seq_proper, perm, s, α)
# end
# function LadderSequence(seq::Vector{Tuple{Int,Bool}}, α::UniformScaling = I)
#     seq_original = Ladder.(seq)
#     return LadderSequence(seq_original, α)
# end

# function printLadder(a::Ladder)
#     if a.flag
#         printstyled(" â_$(a.site) ", color=9)
#     else
#         printstyled(" a_$(a.site) ", color=39)
#     end
# end

# function printLadderSequence(S::LadderSequence)
#     begin
#         print("original :   ")
#         for a in reverse(S.seq_original)
#             printLadder(a)
#         end
#     end
#     print("\n")
#     begin
#         print("proper   :  $(S.s==1 ? '+' : '-')")
#         for a in reverse(S.seq_proper)
#             printLadder(a)
#         end
#     end
#     print("\n")
# end